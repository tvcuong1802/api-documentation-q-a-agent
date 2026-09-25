# tests/unit/test_context_assemble_extra.py -- CMN-C1-053
# Extra edge cases for ContextAssembleNode (Issue #23 gap fill)
# NEVER import or mention agenticstar anywhere.
from __future__ import annotations

from typing import Any


from src.nodes.context_assemble import ContextAssembleNode
from src.schemas.state import initial_state
from src.utils.serde import decode_list, encode_list


def _chunk(
    text: str,
    score: float = 0.8,
    source: str = "kb/api.json",
    endpoint_ref: str = "/v1/test GET",
    doc_type: str = "openapi_spec",
) -> dict[str, Any]:
    return {
        "chunk_text": text,
        "score": score,
        "source_path": source,
        "endpoint_ref": endpoint_ref,
        "doc_type": doc_type,
    }


class TestContextAssembleExtra:
    """Extra edge cases not covered in test_context_assemble.py."""

    def test_token_cap_exactly_at_boundary_second_truncated(self) -> None:
        """
        Token cap exactly at boundary: max_context_tokens=50 → max_chars=200.
        Two chunks: first is 100 chars (fits), second is 101 chars (does not fit).
        Only first chunk should be kept.
        """
        node = ContextAssembleNode(config={"retrieval": {"max_context_tokens": 50}})
        state = initial_state("cap boundary test")
        text_a = "a" * 100  # exactly 100 chars — fits
        text_b = "b" * 101  # 101 chars — does not fit (100+101=201 > 200)
        state["reranked_chunks"] = encode_list(
            [
                _chunk(text_a, endpoint_ref="/v1/ep-a GET"),
                _chunk(text_b, endpoint_ref="/v1/ep-b GET"),
            ]
        )
        cfg = {"retrieval": {"max_context_tokens": 50}}  # 50 * 4 = 200 chars
        result = node.execute(state)
        assert result["retrieval_count"] == 1
        assert decode_list(result["reranked_chunks"])[0]["chunk_text"] == text_a

    def test_empty_reranked_chunks_returns_zero_count(self) -> None:
        """Empty reranked_chunks → retrieval_count=0, reranked_chunks=[]."""
        node = ContextAssembleNode()
        state = initial_state("empty input")
        state["reranked_chunks"] = encode_list([])
        result = node.execute(state)
        assert result["retrieval_count"] == 0
        assert decode_list(result["reranked_chunks"]) == []

    def test_all_chunks_same_endpoint_ref_deduped_to_one(self) -> None:
        """
        All chunks share the same endpoint_ref AND have high text similarity.
        Should be deduplicated to 1 chunk.
        """
        node = ContextAssembleNode()
        state = initial_state("dedup to one")
        # Same text, same endpoint_ref — all are duplicates of each other
        text = "GET /v1/orders\nSummary: List all orders\nParameters: limit offset page status"
        state["reranked_chunks"] = encode_list(
            [
                _chunk(text, endpoint_ref="/v1/orders GET"),
                _chunk(text, endpoint_ref="/v1/orders GET"),
                _chunk(text, endpoint_ref="/v1/orders GET"),
            ]
        )
        result = node.execute(state)
        assert result["retrieval_count"] == 1

    def test_two_chunks_same_ref_high_similarity_deduped(self) -> None:
        """
        Two chunks with same endpoint_ref and nearly identical text → deduped to 1.
        Ensures Jaccard threshold works for near-dupes.
        """
        node = ContextAssembleNode()
        state = initial_state("near dup test")
        text_a = "GET /v1/orders endpoint. Returns paginated list of customer orders. Parameters: limit, page."
        text_b = "GET /v1/orders endpoint. Returns paginated list of customer orders. Parameters: limit, page, status."
        state["reranked_chunks"] = encode_list(
            [
                _chunk(text_a, endpoint_ref="/v1/orders GET"),
                _chunk(text_b, endpoint_ref="/v1/orders GET"),
            ]
        )
        result = node.execute(state)
        # High similarity → should be deduped to 1
        assert result["retrieval_count"] == 1

    def test_blocked_status_short_circuits_to_empty(self) -> None:
        """Blocked status → empty dict returned (no processing)."""
        node = ContextAssembleNode()
        state = initial_state("blocked")
        state["answer_status"] = "blocked"
        state["reranked_chunks"] = encode_list([_chunk("some text", endpoint_ref="/v1/x GET")])
        result = node.execute(state)
        assert result == {}

    def test_error_status_short_circuits_to_empty(self) -> None:
        """Error status → empty dict returned (no processing)."""
        node = ContextAssembleNode()
        state = initial_state("error")
        state["answer_status"] = "error"
        state["reranked_chunks"] = encode_list([_chunk("some text", endpoint_ref="/v1/x GET")])
        result = node.execute(state)
        assert result == {}
