# tests/unit/test_freshness.py -- CMN-C1-053
# Content-freshness model (controlled-snapshot RAG, scaffold/#1362):
#   - ContextAssembleNode flags chunks whose last_verified_at exceeds the
#     refresh interval (content staleness, distinct from url_check liveness).
#   - ResponseGenerateNode prepends a bilingual staleness banner when flagged.
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from src.nodes.context_assemble import ContextAssembleNode
from src.nodes.response_generate import _STALENESS_BANNER, ResponseGenerateNode
from src.schemas.state import initial_state
from src.utils.serde import encode_list


def _chunk(text: str, last_verified_at: str, endpoint_ref: str) -> dict[str, Any]:
    return {
        "chunk_text": text,
        "score": 0.9,
        "source_path": "kb/api.json",
        "endpoint_ref": endpoint_ref,
        "doc_type": "openapi_spec",
        "last_verified_at": last_verified_at,
    }


_OLD = "2000-01-01"  # well beyond any sane refresh interval
_RECENT = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
_CFG = {"retrieval": {"refresh_interval_days": 90, "max_context_tokens": 3000}}


# ── ContextAssembleNode: staleness detection ────────────────────────────────


def test_stale_chunk_flagged() -> None:
    state = initial_state("q")
    state["reranked_chunks"] = encode_list([_chunk("old doc", _OLD, "/v1/a GET")])
    out = ContextAssembleNode(config=_CFG).execute(state)
    assert out["content_staleness_detected"] is True
    assert out["stale_chunk_count"] == 1


def test_recent_chunk_not_flagged() -> None:
    state = initial_state("q")
    state["reranked_chunks"] = encode_list([_chunk("fresh doc", _RECENT, "/v1/a GET")])
    out = ContextAssembleNode(config=_CFG).execute(state)
    assert out["content_staleness_detected"] is False
    assert out["stale_chunk_count"] == 0


def test_missing_last_verified_at_not_counted() -> None:
    """Unknown freshness (empty field) is conservative — not flagged stale."""
    state = initial_state("q")
    state["reranked_chunks"] = encode_list([_chunk("no metadata", "", "/v1/a GET")])
    out = ContextAssembleNode(config=_CFG).execute(state)
    assert out["content_staleness_detected"] is False
    assert out["stale_chunk_count"] == 0


def test_mixed_chunks_counts_only_stale() -> None:
    state = initial_state("q")
    state["reranked_chunks"] = encode_list(
        [
            _chunk("old", _OLD, "/v1/a GET"),
            _chunk("fresh", _RECENT, "/v1/b GET"),
            _chunk("unknown", "", "/v1/c GET"),
        ]
    )
    out = ContextAssembleNode(config=_CFG).execute(state)
    assert out["stale_chunk_count"] == 1
    assert out["content_staleness_detected"] is True


def test_refresh_interval_zero_disables_check() -> None:
    state = initial_state("q")
    state["reranked_chunks"] = encode_list([_chunk("old", _OLD, "/v1/a GET")])
    out = ContextAssembleNode(config={"retrieval": {"refresh_interval_days": 0}}).execute(state)
    assert out["stale_chunk_count"] == 0
    assert out["content_staleness_detected"] is False


# ── ResponseGenerateNode: staleness banner ──────────────────────────────────


class _FakeLLM:
    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return json.dumps(
            {
                "thought": "t",
                "answer": "Use [endpoint: /v1/a GET] to authenticate.",
                "citations": ["/v1/a GET"],
                "confidence": 0.9,
            }
        )


def _gen_state(stale: bool) -> Any:
    state = initial_state("How do I authenticate?")
    state["normalized_query"] = "How do I authenticate?"
    state["reranked_chunks"] = encode_list([_chunk("doc", _RECENT, "/v1/a GET")])
    state["retrieval_count"] = 1
    state["content_staleness_detected"] = stale
    return state


def test_banner_prepended_when_stale() -> None:
    out = ResponseGenerateNode(llm=_FakeLLM()).execute(_gen_state(True))
    assert out["answer_status"] == "ok"
    assert out["answer_markdown"].startswith(_STALENESS_BANNER)
    assert "Use [endpoint: /v1/a GET]" in out["answer_markdown"]


def test_no_banner_when_fresh() -> None:
    out = ResponseGenerateNode(llm=_FakeLLM()).execute(_gen_state(False))
    assert out["answer_status"] == "ok"
    assert _STALENESS_BANNER not in out["answer_markdown"]
    assert out["answer_markdown"].startswith("Use [endpoint: /v1/a GET]")
