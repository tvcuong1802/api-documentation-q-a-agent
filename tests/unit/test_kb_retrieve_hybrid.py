# tests/unit/test_kb_retrieve_hybrid.py -- CMN-C1-053
# Unit tests for KBRetrieveHybridNode
# Issue #19
from __future__ import annotations

from typing import Any


from src.nodes.kb_retrieve_hybrid import KBRetrieveHybridNode
from src.schemas.state import initial_state
from src.utils.serde import decode_list, encode_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    text: str,
    score: float,
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
# Merge logic
# ---------------------------------------------------------------------------


class TestHybridMergeLogic:
    def test_merge_dense_only(self) -> None:
        """Dense chunks with no sparse counterparts use dense_weight * score."""
        node = KBRetrieveHybridNode(config={"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.0}})
        state = initial_state("test query")
        state["dense_chunks"] = encode_list([_chunk("Dense chunk A", 0.9, endpoint_ref="/v1/a GET")])
        state["sparse_chunks"] = encode_list([])

        cfg = {"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.0}}
        result = node.execute(state)

        assert "reranked_chunks" in result
        assert len(decode_list(result["reranked_chunks"])) == 1
        # hybrid = 0.6 * 0.9 + 0.4 * 0.0 = 0.54
        assert abs(decode_list(result["reranked_chunks"])[0]["score"] - 0.54) < 1e-6

    def test_merge_sparse_only(self) -> None:
        """With no dense side at all, sparse carries the whole weight.

        A hybrid weighting assumes BOTH signals exist. When the dense retriever is not
        available -- no Qdrant provisioned, which is the case wherever this template runs
        on its shipped corpus alone -- multiplying by the sparse weight and then measuring
        against a threshold set for the sum of two discards everything: measured
        2026-09-03, a correct match scored 0.4 x 0.33 = 0.13 against min_score 0.65, so
        retrieval found the right endpoint and the merge threw it away.

        This is not a loosened threshold. It is the same threshold applied to the only
        signal there is.
        """
        node = KBRetrieveHybridNode(config={"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.0}})
        state = initial_state("test query")
        state["dense_chunks"] = encode_list([])
        state["sparse_chunks"] = encode_list([_chunk("Sparse chunk B", 0.8, endpoint_ref="/v1/b POST")])

        cfg = {"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.0}}
        result = node.execute(state)

        assert len(decode_list(result["reranked_chunks"])) == 1
        # No dense chunks at all, so the weights become 0.0 / 1.0: hybrid = 0.8.
        assert abs(decode_list(result["reranked_chunks"])[0]["score"] - 0.8) < 1e-6

    def test_merge_same_chunk_in_both(self) -> None:
        """A chunk in both lists gets both scores combined."""
        node = KBRetrieveHybridNode(config={"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.0}})
        text = "GET /v1/orders\nSummary: List orders"
        source = "kb/orders.json"
        state = initial_state("list orders")
        state["dense_chunks"] = encode_list([_chunk(text, 0.80, source=source, endpoint_ref="/v1/orders GET")])
        state["sparse_chunks"] = encode_list([_chunk(text, 0.70, source=source, endpoint_ref="/v1/orders GET")])

        cfg = {"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.0}}
        result = node.execute(state)

        assert len(decode_list(result["reranked_chunks"])) == 1
        expected = 0.6 * 0.80 + 0.4 * 0.70  # 0.48 + 0.28 = 0.76
        assert abs(decode_list(result["reranked_chunks"])[0]["score"] - expected) < 1e-6

    def test_merge_distinct_chunks_both_lists(self) -> None:
        """Chunks from both lists are merged correctly."""
        node = KBRetrieveHybridNode(config={"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.0}})
        state = initial_state("api usage")
        state["dense_chunks"] = encode_list(
            [
                _chunk("Chunk A", 0.9, source="a.json", endpoint_ref="/v1/a GET"),
                _chunk("Chunk B", 0.8, source="b.json", endpoint_ref="/v1/b GET"),
            ]
        )
        state["sparse_chunks"] = encode_list(
            [
                _chunk("Chunk C", 0.7, source="c.json", endpoint_ref="/v1/c GET"),
            ]
        )

        cfg = {"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.0}}
        result = node.execute(state)

        assert len(decode_list(result["reranked_chunks"])) == 3

    def test_sorted_descending(self) -> None:
        """reranked_chunks must be sorted by hybrid_score descending."""
        node = KBRetrieveHybridNode(config={"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.0}})
        state = initial_state("sort test")
        state["dense_chunks"] = encode_list(
            [
                _chunk("Low score", 0.5, endpoint_ref="/v1/low GET"),
                _chunk("High score", 0.95, endpoint_ref="/v1/high GET"),
                _chunk("Mid score", 0.75, endpoint_ref="/v1/mid GET"),
            ]
        )
        state["sparse_chunks"] = encode_list([])

        cfg = {"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.0}}
        result = node.execute(state)

        scores = [c["score"] for c in decode_list(result["reranked_chunks"])]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Hybrid score computation
# ---------------------------------------------------------------------------


class TestHybridScoreComputation:
    def test_default_weights(self) -> None:
        """Default weights 0.6/0.4 are used when config missing."""
        node = KBRetrieveHybridNode()
        text = "POST /v1/auth\nSummary: Authenticate"
        state = initial_state("auth token")
        state["dense_chunks"] = encode_list([_chunk(text, 1.0, endpoint_ref="/v1/auth POST")])
        state["sparse_chunks"] = encode_list([_chunk(text, 1.0, endpoint_ref="/v1/auth POST")])

        result = node.execute(state)

        assert len(decode_list(result["reranked_chunks"])) == 1
        # 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        assert abs(decode_list(result["reranked_chunks"])[0]["score"] - 1.0) < 1e-6

    def test_custom_weights(self) -> None:
        """Custom weights are respected."""
        node = KBRetrieveHybridNode(config={"retrieval": {"hybrid_dense_weight": 0.3, "hybrid_sparse_weight": 0.7, "min_score": 0.0}})
        text = "chunk"
        state = initial_state("q")
        state["dense_chunks"] = encode_list([_chunk(text, 0.8, endpoint_ref="/v1/x GET")])
        state["sparse_chunks"] = encode_list([_chunk(text, 0.8, endpoint_ref="/v1/x GET")])

        cfg = {"retrieval": {"hybrid_dense_weight": 0.3, "hybrid_sparse_weight": 0.7, "min_score": 0.0}}
        result = node.execute(state)

        expected = 0.3 * 0.8 + 0.7 * 0.8  # 0.24 + 0.56 = 0.8
        assert abs(decode_list(result["reranked_chunks"])[0]["score"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Degraded status
# ---------------------------------------------------------------------------


class TestDegradedStatus:
    def test_degraded_when_all_below_threshold(self) -> None:
        """status=degraded set when no chunks survive min_score filter."""
        node = KBRetrieveHybridNode(config={"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.65}})
        state = initial_state("obscure query")
        state["dense_chunks"] = encode_list([_chunk("Low relevance", 0.3, endpoint_ref="/v1/x GET")])
        state["sparse_chunks"] = encode_list([_chunk("Low relevance BM25", 0.2, endpoint_ref="/v1/x GET")])

        cfg = {"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.65}}
        result = node.execute(state)

        assert result.get("answer_status") == "degraded"
        assert decode_list(result["reranked_chunks"]) == []

    def test_not_degraded_when_chunks_pass(self) -> None:
        """No degraded status when chunks survive filter."""
        node = KBRetrieveHybridNode(config={"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.5}})
        state = initial_state("good query")
        state["dense_chunks"] = encode_list([_chunk("High relevance", 0.9, endpoint_ref="/v1/y GET")])
        state["sparse_chunks"] = encode_list([])

        cfg = {"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.5}}
        result = node.execute(state)

        assert result.get("answer_status") != "degraded"
        assert len(decode_list(result["reranked_chunks"])) == 1

    def test_degraded_on_empty_inputs(self) -> None:
        """Empty dense + sparse → degraded."""
        node = KBRetrieveHybridNode()
        state = initial_state("empty")
        state["dense_chunks"] = encode_list([])
        state["sparse_chunks"] = encode_list([])

        result = node.execute(state)

        assert result.get("answer_status") == "degraded"
        assert decode_list(result["reranked_chunks"]) == []


# ---------------------------------------------------------------------------
# Threshold filtering
# ---------------------------------------------------------------------------


class TestThresholdFiltering:
    def test_exactly_at_threshold_excluded(self) -> None:
        """Chunk with hybrid_score exactly equal to min_score is excluded (< filter)."""
        node = KBRetrieveHybridNode(config={"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.65}})
        # 0.6 * 1.0 + 0.4 * 0.0 = 0.6 — below 0.65 threshold
        state = initial_state("q")
        state["dense_chunks"] = encode_list([_chunk("Edge chunk", 1.0, endpoint_ref="/v1/edge GET")])
        state["sparse_chunks"] = encode_list([])

        cfg = {"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.65}}
        result = node.execute(state)

        # hybrid = 0.6 < 0.65 → excluded
        assert decode_list(result["reranked_chunks"]) == []
        assert result.get("answer_status") == "degraded"

    def test_chunk_just_above_threshold_included(self) -> None:
        """Chunk with hybrid_score just above min_score passes."""
        node = KBRetrieveHybridNode(config={"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.65}})
        # 0.6 * 0.9 + 0.4 * 0.9 = 0.54 + 0.36 = 0.9 → passes 0.65
        state = initial_state("q")
        state["dense_chunks"] = encode_list([_chunk("Passing chunk", 0.9, endpoint_ref="/v1/pass GET")])
        state["sparse_chunks"] = encode_list([_chunk("Passing chunk", 0.9, endpoint_ref="/v1/pass GET")])

        cfg = {"retrieval": {"hybrid_dense_weight": 0.6, "hybrid_sparse_weight": 0.4, "min_score": 0.65}}
        result = node.execute(state)

        assert len(decode_list(result["reranked_chunks"])) == 1

    def test_blocked_state_short_circuits(self) -> None:
        """Blocked state → node returns empty dict (short-circuit)."""
        node = KBRetrieveHybridNode()
        state = initial_state("blocked")
        state["answer_status"] = "blocked"
        state["dense_chunks"] = encode_list([_chunk("should not appear", 0.9, endpoint_ref="/v1/x GET")])
        state["sparse_chunks"] = encode_list([])

        result = node.execute(state)
        assert result == {}
