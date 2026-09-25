# src/nodes/kb_retrieve_dense.py -- CMN-C1-053
# KBRetrieveDenseNode -- Qdrant semantic vector search
# Issue #4
from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel
from src.utils.audit import emit_trace_event
from src.utils.serde import encode_list
from src.services.app_config import app_config

# ---------------------------------------------------------------------------
# Heavy-library guard: sentence_transformers and qdrant_client are NOT
# installed in CI.  They are imported lazily inside execute() and guarded
# with try/except so pytest passes without them.
# ---------------------------------------------------------------------------


def _optional(module: str, attr: str) -> Any:
    """The attribute if the optional package is installed, else None.

    Resolved through importlib rather than a try/except around `from x import Y as _Y`:
    that shape binds one name to a CLASS in one branch and to None in the other, and
    mypy 2.1.0 rejects it ("Cannot assign to a type"). Annotating the name first and
    re-importing into it only trades that for "already defined". Fetching the attribute
    gives the name a single `Any` type and one binding, which is what it actually is --
    something that may or may not be there.
    """
    import importlib  # noqa: PLC0415

    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError):
        return None


_SentenceTransformer: Any = _optional("sentence_transformers", "SentenceTransformer")
_QdrantClient: Any = _optional("qdrant_client", "QdrantClient")
_ST_AVAILABLE = _SentenceTransformer is not None
_QDRANT_AVAILABLE = _QdrantClient is not None


class KBRetrieveDenseNode(FunctionNode):
    """Dense vector retrieval node using multilingual-e5-base + Qdrant.

    Constructor injection: callers (and tests) pass *encoder* and *qdrant*
    as optional keyword arguments so no heavy libraries need to be imported at
    import time during CI.
    """

    # Default collection name (override via config["qdrant_collection"])
    _DEFAULT_COLLECTION = "cmn_c1_053_kb"

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        encoder: Any = None,
        qdrant: Any = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._config = dict(config or {})
        self._encoder = encoder  # SentenceTransformer instance (or mock)
        self._qdrant = qdrant  # QdrantClient instance (or mock)

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
            return {"dense_chunks": encode_list([])}

        # Resolve retrieval params from config -> agent.yaml defaults
        top_k: int = int(cfg.get("retrieval", {}).get("top_k", 5))
        min_score: float = float(cfg.get("retrieval", {}).get("min_score", 0.65))
        model_name: str = cfg.get("embedding_model", "intfloat/multilingual-e5-base")
        collection: str = cfg.get("qdrant_collection", self._DEFAULT_COLLECTION)

        # Lazy init encoder (uses injected instance if available)
        encoder = self._encoder
        if encoder is None:
            if not _ST_AVAILABLE or _SentenceTransformer is None:
                # CI fallback: return empty chunks rather than crashing
                return {"dense_chunks": encode_list([])}
            encoder = _SentenceTransformer(model_name)

        # Lazy init Qdrant client
        qdrant = self._qdrant
        if qdrant is None:
            if not _QDRANT_AVAILABLE or _QdrantClient is None:
                return {"dense_chunks": encode_list([])}
            qdrant_url: str = cfg.get("qdrant_url", "http://localhost:6333")
            qdrant = _QdrantClient(url=qdrant_url)

        # Embed query
        # multilingual-e5 expects "query: <text>" prefix for queries
        prefixed_query = f"query: {query}"
        query_vector = encoder.encode(prefixed_query, normalize_embeddings=True).tolist()

        # Search Qdrant
        try:
            results = qdrant.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception:
            # Graceful degradation: empty results, do not propagate
            return {"dense_chunks": encode_list([])}

        # Build chunk list — filter by min_score
        dense_chunks: list[dict[str, Any]] = []
        for hit in results:
            score: float = float(hit.score)
            if score < min_score:
                continue
            payload: dict[str, Any] = hit.payload or {}
            dense_chunks.append(
                {
                    "chunk_text": str(payload.get("chunk_text", "")),
                    "score": score,
                    "source_path": str(payload.get("source_path", "")),
                    "endpoint_ref": str(payload.get("endpoint_ref", "")),
                    "doc_type": str(payload.get("doc_type", "openapi_spec")),
                    # Content-freshness signal carried from KB ingestion metadata.
                    "last_verified_at": str(payload.get("last_verified_at", "")),
                }
            )

        updates = {"dense_chunks": encode_list(dense_chunks)}
        emit_trace_event("kb_retrieved_dense", {"chunk_count": len(dense_chunks), "status": "success"}, state)
        return updates
