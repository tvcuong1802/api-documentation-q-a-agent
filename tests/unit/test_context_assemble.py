# tests/unit/test_context_assemble.py -- CMN-C1-053
# Unit tests for ContextAssembleNode
# Issue #5
from __future__ import annotations

from typing import Any


from src.nodes.context_assemble import ContextAssembleNode
from src.schemas.state import initial_state
from src.utils.serde import decode_list, encode_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_identical_text_same_endpoint_deduped(self) -> None:
        """Two chunks with identical text and same endpoint_ref → deduplicated to one."""
        node = ContextAssembleNode()
        state = initial_state("list orders")
        text = "GET /v1/orders\nSummary: List orders\nParameters: limit, offset"
        state["reranked_chunks"] = encode_list(
            [
                _chunk(text, endpoint_ref="/v1/orders GET"),
                _chunk(text, endpoint_ref="/v1/orders GET"),  # exact dup
            ]
        )

        result = node.execute(state)
        assert result["retrieval_count"] == 1
        assert len(decode_list(result["reranked_chunks"])) == 1

    def test_similar_text_same_endpoint_deduped(self) -> None:
        """Near-duplicate text (high Jaccard) with same endpoint_ref → deduplicated."""
        node = ContextAssembleNode()
        text_a = "GET /v1/orders\nSummary: List all orders for the user\nParameters: status, limit, offset"
        text_b = "GET /v1/orders\nSummary: List all orders for the user\nParameters: status, limit, page"
        state = initial_state("list orders")
        state["reranked_chunks"] = encode_list(
            [
                _chunk(text_a, endpoint_ref="/v1/orders GET"),
                _chunk(text_b, endpoint_ref="/v1/orders GET"),
            ]
        )

        result = node.execute(state)
        # High similarity → deduped
        assert result["retrieval_count"] == 1

    def test_different_endpoint_not_deduped(self) -> None:
        """Chunks for different endpoint_refs are NOT deduplicated regardless of text similarity."""
        node = ContextAssembleNode()
        text = "POST /v1/orders\nSummary: Create order"
        state = initial_state("create order")
        state["reranked_chunks"] = encode_list(
            [
                _chunk(text, endpoint_ref="/v1/orders POST"),
                _chunk(text, endpoint_ref="/v1/auth POST"),  # different endpoint
            ]
        )

        result = node.execute(state)
        assert result["retrieval_count"] == 2

    def test_distinct_text_same_endpoint_not_deduped(self) -> None:
        """Chunks with very different text but same endpoint_ref are kept (low similarity)."""
        node = ContextAssembleNode()
        state = initial_state("auth")
        state["reranked_chunks"] = encode_list(
            [
                _chunk(
                    "GET /v1/orders\nSummary: List all orders\nParameters: limit offset status",
                    endpoint_ref="/v1/orders GET",
                ),
                _chunk(
                    "Authentication is required. Use the Bearer token scheme.",
                    endpoint_ref="/v1/orders GET",
                ),
            ]
        )

        result = node.execute(state)
        # Very different text → both kept
        assert result["retrieval_count"] == 2


# ---------------------------------------------------------------------------
# Grouping by doc_type
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_openapi_spec_comes_first(self) -> None:
        """openapi_spec chunks precede markdown_guide and code_example."""
        node = ContextAssembleNode()
        state = initial_state("api usage")
        state["reranked_chunks"] = encode_list(
            [
                _chunk("Code example block", doc_type="code_example", endpoint_ref="/v1/c GET"),
                _chunk("Markdown guide text", doc_type="markdown_guide", endpoint_ref="/v1/m GET"),
                _chunk("OpenAPI spec chunk", doc_type="openapi_spec", endpoint_ref="/v1/o GET"),
            ]
        )

        result = node.execute(state)
        doc_types = [c["doc_type"] for c in decode_list(result["reranked_chunks"])]
        # openapi_spec must come before markdown_guide, markdown_guide before code_example
        spec_idx = doc_types.index("openapi_spec")
        md_idx = doc_types.index("markdown_guide")
        code_idx = doc_types.index("code_example")
        assert spec_idx < md_idx < code_idx

    def test_same_doc_type_order_preserved(self) -> None:
        """Within the same doc_type, original order is preserved."""
        node = ContextAssembleNode()
        state = initial_state("q")
        state["reranked_chunks"] = encode_list(
            [
                _chunk("Spec chunk 1", score=0.9, doc_type="openapi_spec", endpoint_ref="/v1/a GET"),
                _chunk("Spec chunk 2", score=0.8, doc_type="openapi_spec", endpoint_ref="/v1/b GET"),
                _chunk("Spec chunk 3", score=0.7, doc_type="openapi_spec", endpoint_ref="/v1/c GET"),
            ]
        )

        result = node.execute(state)
        texts = [c["chunk_text"] for c in decode_list(result["reranked_chunks"])]
        assert texts == ["Spec chunk 1", "Spec chunk 2", "Spec chunk 3"]


# ---------------------------------------------------------------------------
# Token cap
# ---------------------------------------------------------------------------


class TestTokenCap:
    def test_token_cap_limits_chunks(self) -> None:
        """Total chars capped at max_context_tokens * 4."""
        node = ContextAssembleNode(config={"retrieval": {"max_context_tokens": 5}})
        state = initial_state("cap test")
        # Each chunk has 100 chars
        chunk_text = "x" * 100
        state["reranked_chunks"] = encode_list([_chunk(chunk_text, endpoint_ref=f"/v1/ep-{i} GET") for i in range(20)])

        # max_context_tokens=5 → max_chars=20 → only 0 chunks (100 > 20)
        cfg = {"retrieval": {"max_context_tokens": 5}}
        result = node.execute(state)
        assert result["retrieval_count"] == 0

    def test_token_cap_allows_fitting_chunks(self) -> None:
        """Chunks that fit within budget are all kept."""
        node = ContextAssembleNode(config={"retrieval": {"max_context_tokens": 1000}})
        state = initial_state("budget test")
        # 3 chunks × 100 chars = 300 chars; budget = 1000 tokens * 4 = 4000 chars
        state["reranked_chunks"] = encode_list([_chunk("y" * 100, endpoint_ref=f"/v1/ep-{i} GET") for i in range(3)])

        cfg = {"retrieval": {"max_context_tokens": 1000}}
        result = node.execute(state)
        assert result["retrieval_count"] == 3

    def test_partial_cap(self) -> None:
        """Only chunks fitting within budget are kept; excess dropped."""
        node = ContextAssembleNode(config={"retrieval": {"max_context_tokens": 100}})
        state = initial_state("partial cap")
        # 5 chunks of 400 chars each; budget = 100 tokens * 4 = 400 chars → only 1 fits
        state["reranked_chunks"] = encode_list([_chunk("z" * 400, endpoint_ref=f"/v1/ep-{i} GET") for i in range(5)])

        cfg = {"retrieval": {"max_context_tokens": 100}}
        result = node.execute(state)
        assert result["retrieval_count"] == 1


# ---------------------------------------------------------------------------
# retrieval_count
# ---------------------------------------------------------------------------


class TestRetrievalCount:
    def test_retrieval_count_matches_output(self) -> None:
        """retrieval_count equals len(reranked_chunks) in the result."""
        node = ContextAssembleNode()
        state = initial_state("count test")
        state["reranked_chunks"] = encode_list(
            [
                _chunk("Chunk A", endpoint_ref="/v1/a GET"),
                _chunk("Chunk B", endpoint_ref="/v1/b GET"),
                _chunk("Chunk C", endpoint_ref="/v1/c GET"),
            ]
        )

        result = node.execute(state)
        assert result["retrieval_count"] == len(decode_list(result["reranked_chunks"]))

    def test_retrieval_count_zero_on_empty(self) -> None:
        """retrieval_count = 0 when no chunks."""
        node = ContextAssembleNode()
        state = initial_state("empty")
        state["reranked_chunks"] = encode_list([])

        result = node.execute(state)
        assert result["retrieval_count"] == 0

    def test_blocked_state_short_circuits(self) -> None:
        """Blocked state → empty dict."""
        node = ContextAssembleNode()
        state = initial_state("blocked")
        state["answer_status"] = "blocked"
        state["reranked_chunks"] = encode_list([_chunk("should not appear", endpoint_ref="/v1/x GET")])

        result = node.execute(state)
        assert result == {}
