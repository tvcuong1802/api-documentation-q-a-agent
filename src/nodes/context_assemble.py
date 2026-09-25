# src/nodes/context_assemble.py -- CMN-C1-053
# ContextAssembleNode -- dedup, group, token-cap for context window
# Issue #5
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel
from src.utils.audit import emit_trace_event
from src.utils.serde import decode_list, encode_list
from src.services.app_config import app_config

# ---------------------------------------------------------------------------
# Similarity deduplication
# Using Jaccard similarity on character trigrams as a cosine proxy.
# No heavy deps required.
# ---------------------------------------------------------------------------


def _trigrams(text: str) -> set[str]:
    """Return set of character trigrams for a string."""
    t = text.lower().strip()
    if len(t) < 3:
        return {t}
    return {t[i : i + 3] for i in range(len(t) - 2)}


def _jaccard_sim(a: str, b: str) -> float:
    """Jaccard similarity on trigrams — cheap cosine proxy."""
    tg_a = _trigrams(a)
    tg_b = _trigrams(b)
    if not tg_a or not tg_b:
        return 0.0
    intersection = len(tg_a & tg_b)
    union = len(tg_a | tg_b)
    return intersection / union if union > 0 else 0.0


# Threshold above which two chunks sharing the same endpoint_ref are
# considered duplicates (Jaccard proxy for cosine > 0.95)
_DEDUP_THRESHOLD = 0.85  # conservative Jaccard ≈ cosine 0.95 neighbourhood


class ContextAssembleNode(FunctionNode):
    """Post-retrieval assembly: dedup, group by doc_type, token-cap.

    Outputs (written back to state):
        reranked_chunks: deduplicated, token-capped list
        retrieval_count: int — len(reranked_chunks) after assembly
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._config = dict(config or {})
        pass

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

        retrieval_cfg: dict[str, Any] = cfg.get("retrieval", {})
        max_context_tokens: int = int(retrieval_cfg.get("max_context_tokens", 3000))

        chunks: list[dict[str, Any]] = decode_list(state.get("reranked_chunks"))

        # ---- Step 1: Dedup chunks sharing same endpoint_ref ----
        chunks = _dedup_by_endpoint(chunks, threshold=_DEDUP_THRESHOLD)

        # ---- Step 2: Group by doc_type ----
        # Priority order: openapi_spec > markdown_guide > code_example
        # Higher-priority groups fill the context first.
        chunks = _group_and_order(chunks)

        # ---- Step 3: Token cap ----
        # Approximate: 1 token ≈ 4 chars (conservative estimate)
        chunks = _apply_token_cap(chunks, max_context_tokens=max_context_tokens)

        # ---- Step 4: Content-freshness check (controlled-snapshot RAG, #1362) ----
        # Distinct from url_check liveness: a live URL does not mean the indexed
        # content is current. Flag chunks whose last_verified_at exceeds the
        # refresh interval so ResponseGenerate can surface a staleness notice.
        refresh_interval_days: int = int(retrieval_cfg.get("refresh_interval_days", 90))
        stale_count = _count_stale_chunks(chunks, refresh_interval_days)

        updates = {
            "reranked_chunks": encode_list(chunks),
            "retrieval_count": len(chunks),
            "content_staleness_detected": stale_count > 0,
            "stale_chunk_count": stale_count,
        }
        # Approximate: 1 token ≈ 4 chars
        total_chars = sum(len(str(c.get("chunk_text", ""))) for c in chunks)
        approx_tokens = total_chars // 4
        emit_trace_event(
            "context_assembled",
            {"chunk_count": len(chunks), "approx_token_count": approx_tokens, "status": "success"},
            state,
        )
        return updates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso(value: str) -> datetime | None:
    """Parse an ISO-8601 date/datetime string to an aware datetime (UTC).

    Returns None if the value is empty or unparseable. Naive datetimes are
    assumed to be UTC.
    """
    s = value.strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _count_stale_chunks(
    chunks: list[dict[str, Any]],
    refresh_interval_days: int,
) -> int:
    """Count chunks whose last_verified_at is older than the refresh interval.

    Conservative: a chunk with a missing/empty/unparseable last_verified_at is
    treated as unknown freshness and NOT counted as stale (KB ingestion is
    expected to populate the field). Only a parseable timestamp older than the
    cutoff counts as content-stale.
    """
    if refresh_interval_days <= 0:
        return 0
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=refresh_interval_days)
    except Exception:  # noqa: BLE001 — never let a clock error break retrieval
        return 0
    stale = 0
    for chunk in chunks:
        dt = _parse_iso(str(chunk.get("last_verified_at", "")))
        if dt is not None and dt < cutoff:
            stale += 1
    return stale


def _dedup_by_endpoint(
    chunks: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Remove near-duplicate chunks that share the same endpoint_ref."""
    kept: list[dict[str, Any]] = []
    for chunk in chunks:
        ep_ref = str(chunk.get("endpoint_ref", ""))
        chunk_text = str(chunk.get("chunk_text", ""))
        is_dup = False
        for existing in kept:
            if str(existing.get("endpoint_ref", "")) != ep_ref:
                continue
            # Same endpoint_ref — check text similarity
            sim = _jaccard_sim(chunk_text, str(existing.get("chunk_text", "")))
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(chunk)
    return kept


_DOC_TYPE_ORDER: dict[str, int] = {
    "openapi_spec": 0,
    "markdown_guide": 1,
    "code_example": 2,
}


def _group_and_order(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group chunks by doc_type and return in priority order."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        doc_type = str(chunk.get("doc_type", "openapi_spec"))
        groups[doc_type].append(chunk)

    ordered: list[dict[str, Any]] = []
    for doc_type in sorted(groups.keys(), key=lambda dt: _DOC_TYPE_ORDER.get(dt, 99)):
        ordered.extend(groups[doc_type])
    return ordered


def _apply_token_cap(
    chunks: list[dict[str, Any]],
    max_context_tokens: int,
) -> list[dict[str, Any]]:
    """Keep chunks until the cumulative char budget (tokens * 4) is met."""
    max_chars = max_context_tokens * 4
    total_chars = 0
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        text_len = len(str(chunk.get("chunk_text", "")))
        if total_chars + text_len > max_chars:
            break
        result.append(chunk)
        total_chars += text_len
    return result
