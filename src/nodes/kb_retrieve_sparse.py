# src/nodes/kb_retrieve_sparse.py -- CMN-C1-053
# KBRetrieveSparseNode -- BM25 keyword retrieval
# Issue #18
from __future__ import annotations

import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel
from src.utils.audit import emit_trace_event
from src.utils.serde import encode_list
from src.services.app_config import app_config

# ---------------------------------------------------------------------------
# rank_bm25 is NOT installed in CI.  Guarded at module level.
# ---------------------------------------------------------------------------

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    _BM25Okapi = None


# ---------------------------------------------------------------------------
# Domain-aware tokenizer
# Preserves API path separators: /, _, . so that tokens like
# "/v1/orders", "api_key", "application.json" are not split incorrectly.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9/_.]+")


def _api_tokenize(text: str) -> list[str]:
    """Tokenize text preserving /, _, . characters used in API paths."""
    return _TOKEN_RE.findall(text.lower())


def _shipped_documents() -> list[dict[str, Any]]:
    """The corpus that travels with the template, as retrievable chunks.

    Read through the shared `kb_port.load_kb` door rather than opening the file here: the
    mechanism CoE will settle for loading a KB from outside the image is not decided yet,
    and every template reading it the same way makes that day one patch applied uniformly
    instead of one refactor per repository.

    Returns [] on any failure. A missing corpus must degrade to "found nothing", never to
    an exception on the request path.
    """
    try:
        from pathlib import Path

        from src.services.kb_port import load_kb

        # Resolved from __file__, never from the working directory: the container runs
        # from /app while tests run from the repo root, and a cwd-relative path finds the
        # corpus in exactly one of those.
        corpus = Path(__file__).resolve().parents[2] / "data" / "api_docs.json"
        loaded = load_kb("api_docs", corpus)
    except Exception:  # noqa: BLE001 -- a corpus problem is not a request failure
        return []
    documents = loaded.data.get("documents") if isinstance(loaded.data, dict) else None
    if not isinstance(documents, list):
        return []
    return [d for d in documents if isinstance(d, dict) and d.get("text")]


def _rank_by_overlap(query: str, documents: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Rank documents by how many query terms they contain. Deterministic, no dependency.

    Not a search engine, and not pretending to be one: it exists so a template that ships
    a small corpus can answer at all when rank_bm25 is absent. Scores are normalised to
    [0, 1] so the hybrid merge downstream treats them like any other sparse result.
    """
    terms = {t for t in _api_tokenize(query) if len(t) > 1}
    if not terms:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for document in documents:
        haystack = " ".join(str(document.get(field, "")) for field in ("title", "text", "endpoint", "method"))
        tokens = set(_api_tokenize(haystack))
        hits = len(terms & tokens)
        if hits:
            scored.append((hits / len(terms), document))
    if not scored:
        return []
    scored.sort(key=lambda pair: pair[0], reverse=True)
    # Normalise so the best match is 1.0, exactly as the BM25 branch above does. The raw
    # ratio is on a different scale from BM25 output, and min_score is calibrated for the
    # latter: a correct match scoring 1-of-3 terms came out at 0.33 and was discarded by a
    # 0.65 threshold, so retrieval found the right endpoint and the merge threw it away.
    best = scored[0][0] or 1.0
    scored = [(score / best, document) for score, document in scored]
    # The shape KBRetrieveHybridNode merges on -- chunk_text / source_path / endpoint_ref,
    # not the corpus file's own field names. Returning the raw document merged cleanly and
    # carried no text, so every answer came back "no documentation found" while retrieval
    # had in fact matched.
    return [
        {
            "chunk_text": str(document.get("text", "")),
            "source_path": f"data/api_docs.json#{document.get('id', '')}",
            "endpoint_ref": f"{document.get('endpoint', '')} {document.get('method', '')}".strip(),
            "doc_type": "openapi_spec",
            "last_verified_at": "",
            "score": round(score, 4),
        }
        for score, document in scored[:limit]
    ]


class KBRetrieveSparseNode(FunctionNode):
    """BM25 sparse retrieval node.

    Constructor injection: callers pass *bm25_index* (BM25Okapi or compatible
    mock) and *bm25_docs* (list of chunk dicts) for testability.

    In production, bm25_index is built from the ingested KB corpus.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        bm25_index: Any = None,
        bm25_docs: list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._config = dict(config or {})
        self._bm25_index = bm25_index  # BM25Okapi instance or mock
        # None means NOT SUPPLIED -- fall back to the corpus the template ships. An
        # explicit [] means the caller is saying "no documents", and that must stay
        # empty: a test pinning the not-ingested path would otherwise start finding the
        # shipped corpus and stop testing what it names.
        self._bm25_docs = bm25_docs

    # ------------------------------------------------------------------
    # FunctionNode contract
    # ------------------------------------------------------------------

    def execute(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = self._config or app_config()

        # Short-circuit on upstream error / blocked status
        if state.get("answer_status") in ("blocked", "error"):
            return {}

        query = state.get("normalized_query") or state.get("raw_query", "")
        if not query:
            return {"sparse_chunks": encode_list([])}

        top_k: int = int(cfg.get("retrieval", {}).get("top_k", 5))
        min_score: float = float(cfg.get("retrieval", {}).get("min_score", 0.65))

        # Need both index and docs
        bm25_index = self._bm25_index
        bm25_docs = self._bm25_docs

        if bm25_docs is None:
            # Nothing was injected, so fall back to the corpus shipped with the template.
            # Without this the agent answered the bilingual no-match to every question --
            # correct for a retrieval that found nothing, and useless to a reader. The
            # shipped corpus is a PLACEHOLDER; see data/api_docs.json _provenance.
            bm25_docs = _shipped_documents()

        if bm25_index is None and bm25_docs:
            # A corpus with no index. rank_bm25 is not installed in CI and is not a
            # required dependency, so build a plain term-overlap ranking instead: for a
            # corpus this size it is enough to find the right endpoint, and being
            # deterministic it is testable in a way a missing library is not.
            return {"sparse_chunks": encode_list(_rank_by_overlap(query, bm25_docs, top_k))}

        if bm25_index is None or not bm25_docs:
            # Still nothing: no index and no corpus.
            return {"sparse_chunks": encode_list([])}

        # BM25 not available in CI — check import guard
        if not _BM25_AVAILABLE and not _is_mock(bm25_index):
            return {"sparse_chunks": encode_list([])}

        # Tokenize query
        query_tokens = _api_tokenize(query)
        if not query_tokens:
            return {"sparse_chunks": encode_list([])}

        # BM25 scores (float array)
        try:
            raw_scores = bm25_index.get_scores(query_tokens)
        except Exception:
            return {"sparse_chunks": encode_list([])}

        # Normalize scores to [0, 1] for hybrid merge compatibility
        max_score = float(max(raw_scores)) if len(raw_scores) > 0 else 0.0
        if max_score == 0.0:
            return {"sparse_chunks": encode_list([])}

        # Build ranked list: (score, doc_index)
        scored = sorted(
            ((float(raw_scores[i]) / max_score, i) for i in range(len(raw_scores))),
            key=lambda x: x[0],
            reverse=True,
        )

        sparse_chunks: list[dict[str, Any]] = []
        for norm_score, idx in scored[:top_k]:
            if norm_score < min_score:
                continue
            doc = bm25_docs[idx]
            sparse_chunks.append(
                {
                    "chunk_text": str(doc.get("chunk_text", "")),
                    "score": norm_score,
                    "source_path": str(doc.get("source_path", "")),
                    "endpoint_ref": str(doc.get("endpoint_ref", "")),
                    "doc_type": str(doc.get("doc_type", "openapi_spec")),
                    # Content-freshness signal carried from KB ingestion metadata.
                    "last_verified_at": str(doc.get("last_verified_at", "")),
                }
            )

        updates = {"sparse_chunks": encode_list(sparse_chunks)}
        emit_trace_event("kb_retrieved_sparse", {"chunk_count": len(sparse_chunks), "status": "success"}, state)
        return updates


def _is_mock(obj: Any) -> bool:
    """Return True if obj is a unittest.mock object (for CI test detection)."""
    try:
        from unittest.mock import MagicMock, Mock  # noqa: PLC0415

        return isinstance(obj, (Mock, MagicMock))
    except ImportError:
        return False
