# tests/unit/test_kb_retrieve_sparse.py -- CMN-C1-053
# Unit tests for KBRetrieveSparseNode
# Issue #18
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


from src.nodes.kb_retrieve_sparse import KBRetrieveSparseNode, _api_tokenize
from src.schemas.state import initial_state
from src.utils.serde import decode_list


# ---------------------------------------------------------------------------
# Helpers: mock BM25Okapi
# Uses plain Python list — no numpy required in CI.
# ---------------------------------------------------------------------------


def _make_bm25(scores: list[float]) -> MagicMock:
    """Create a mock BM25Okapi returning a fixed score vector (no numpy)."""
    bm25 = MagicMock()
    bm25.get_scores.return_value = scores  # plain list — node uses max() and iteration
    return bm25


def _make_docs(n: int) -> list[dict[str, Any]]:
    """Generate n synthetic docs matching the BM25 index."""
    return [
        {
            "chunk_text": f"GET /v1/endpoint-{i}\nSummary: Endpoint {i}",
            "source_path": f"kb/api_{i}.json",
            "endpoint_ref": f"/v1/endpoint-{i} GET",
            "doc_type": "openapi_spec",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# API tokenizer tests
# ---------------------------------------------------------------------------


class TestApiTokenizer:
    def test_preserves_slash_in_paths(self) -> None:
        """/ in API paths is preserved as part of the token."""
        tokens = _api_tokenize("/v1/orders")
        assert "/v1/orders" in tokens

    def test_preserves_underscore(self) -> None:
        """_ in identifiers is preserved."""
        tokens = _api_tokenize("api_key")
        assert "api_key" in tokens

    def test_preserves_dot(self) -> None:
        """Dot in media types is preserved."""
        tokens = _api_tokenize("application.json")
        assert "application.json" in tokens

    def test_lowercases_tokens(self) -> None:
        """Tokenizer outputs lowercase."""
        tokens = _api_tokenize("GET /V1/Orders")
        assert "get" in tokens
        assert "/v1/orders" in tokens

    def test_handles_mixed_text(self) -> None:
        """Mixed natural language and API paths tokenized correctly."""
        tokens = _api_tokenize("List all orders GET /v1/orders?status=pending")
        assert "get" in tokens
        assert "/v1/orders" in tokens
        assert "status" in tokens

    def test_empty_string(self) -> None:
        """Empty string → empty token list."""
        tokens = _api_tokenize("")
        assert tokens == []


# ---------------------------------------------------------------------------
# BM25 scoring tests
# ---------------------------------------------------------------------------


class TestBM25Scoring:
    def test_top_k_respected(self) -> None:
        """Only top_k chunks are returned even when more docs exist."""
        n = 10
        # First doc gets highest score, rest get lower
        scores = [0.9 if i == 0 else 0.5 for i in range(n)]
        bm25 = _make_bm25(scores)
        docs = _make_docs(n)
        node = KBRetrieveSparseNode(bm25_index=bm25, bm25_docs=docs)
        state = initial_state("list orders")
        cfg = {"retrieval": {"top_k": 3, "min_score": 0.0}}

        result = node.execute(state)
        assert len(decode_list(result["sparse_chunks"])) <= 3

    def test_normalized_score_in_range(self) -> None:
        """Normalized scores are in [0, 1] range."""
        scores = [0.0, 5.0, 10.0, 2.0]
        bm25 = _make_bm25(scores)
        docs = _make_docs(4)
        node = KBRetrieveSparseNode(bm25_index=bm25, bm25_docs=docs)
        state = initial_state("query")
        cfg = {"retrieval": {"top_k": 10, "min_score": 0.0}}

        result = node.execute(state)
        for chunk in decode_list(result["sparse_chunks"]):
            assert 0.0 <= chunk["score"] <= 1.0

    def test_highest_raw_score_gets_normalized_1(self) -> None:
        """The highest-scoring doc gets normalized score of 1.0."""
        scores = [3.0, 10.0, 5.0]
        bm25 = _make_bm25(scores)
        docs = _make_docs(3)
        node = KBRetrieveSparseNode(bm25_index=bm25, bm25_docs=docs)
        state = initial_state("query")
        cfg = {"retrieval": {"top_k": 5, "min_score": 0.0}}

        result = node.execute(state)
        # doc[1] has raw score 10.0 → normalized = 1.0
        top_chunk = decode_list(result["sparse_chunks"])[0]
        assert abs(top_chunk["score"] - 1.0) < 1e-6
        assert top_chunk["chunk_text"] == docs[1]["chunk_text"]

    def test_min_score_filter_applied(self) -> None:
        """Chunks below min_score are filtered out."""
        scores = [10.0, 5.0, 1.0]  # normalized: 1.0, 0.5, 0.1
        bm25 = _make_bm25(scores)
        docs = _make_docs(3)
        node = KBRetrieveSparseNode(bm25_index=bm25, bm25_docs=docs)
        state = initial_state("query")
        cfg = {"retrieval": {"top_k": 10, "min_score": 0.65}}

        result = node.execute(state)
        # Only score 1.0 passes (0.5 < 0.65)
        assert len(decode_list(result["sparse_chunks"])) == 1
        assert abs(decode_list(result["sparse_chunks"])[0]["score"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_index_and_no_documents_returns_empty(self) -> None:
        """An explicit empty document list means no documents, and stays empty.

        `None` would mean "not supplied" and fall back to the corpus the template ships.
        The distinction matters: without it this test would quietly start finding the
        shipped corpus and stop testing the path it names.
        """
        node = KBRetrieveSparseNode(bm25_index=None, bm25_docs=[])
        state = initial_state("query")
        result = node.execute(state)
        assert decode_list(result["sparse_chunks"]) == []

    def test_no_index_falls_back_to_the_shipped_corpus(self) -> None:
        """Nothing injected at all: answer from the corpus that travels with the template.

        Without this the agent replied "no relevant documentation found" to every
        question -- correct for a retrieval that found nothing, and useless to a reader.
        """
        node = KBRetrieveSparseNode(bm25_index=None, bm25_docs=None)
        result = node.execute(initial_state("retrieve multiple records"))
        chunks = decode_list(result["sparse_chunks"])
        assert chunks, "the shipped corpus was not consulted"
        assert all(c.get("chunk_text") for c in chunks), "chunks carried no text to answer from"

    def test_empty_docs_list_returns_empty(self) -> None:
        """Index provided but empty docs list → empty result."""
        bm25 = _make_bm25([])
        node = KBRetrieveSparseNode(bm25_index=bm25, bm25_docs=[])
        state = initial_state("query")
        result = node.execute(state)
        assert decode_list(result["sparse_chunks"]) == []

    def test_all_zero_scores_returns_empty(self) -> None:
        """All zero BM25 scores → max_score=0 → empty result."""
        bm25 = _make_bm25([0.0, 0.0, 0.0])
        docs = _make_docs(3)
        node = KBRetrieveSparseNode(bm25_index=bm25, bm25_docs=docs)
        state = initial_state("no match")
        result = node.execute(state)
        assert decode_list(result["sparse_chunks"]) == []

    def test_empty_query_returns_empty(self) -> None:
        """Empty normalized_query → sparse_chunks=[]."""
        bm25 = _make_bm25([1.0, 0.5])
        docs = _make_docs(2)
        node = KBRetrieveSparseNode(bm25_index=bm25, bm25_docs=docs)
        state = initial_state("")
        state["normalized_query"] = ""
        result = node.execute(state)
        assert decode_list(result["sparse_chunks"]) == []

    def test_blocked_state_short_circuits(self) -> None:
        """status=blocked → empty dict, BM25 not called."""
        bm25 = _make_bm25([1.0])
        docs = _make_docs(1)
        node = KBRetrieveSparseNode(bm25_index=bm25, bm25_docs=docs)
        state = initial_state("blocked")
        state["answer_status"] = "blocked"

        result = node.execute(state)
        assert result == {}
        bm25.get_scores.assert_not_called()

    def test_bm25_exception_returns_empty(self) -> None:
        """If BM25 raises, node returns empty sparse_chunks gracefully."""
        bm25 = MagicMock()
        bm25.get_scores.side_effect = RuntimeError("BM25 failure")
        docs = _make_docs(3)
        node = KBRetrieveSparseNode(bm25_index=bm25, bm25_docs=docs)
        state = initial_state("error query")

        result = node.execute(state)
        assert decode_list(result["sparse_chunks"]) == []


# ---------------------------------------------------------------------------
# Chunk schema
# ---------------------------------------------------------------------------


class TestChunkSchema:
    def test_chunk_has_required_fields(self) -> None:
        """Each sparse_chunk has: chunk_text, score, source_path, endpoint_ref, doc_type."""
        bm25 = _make_bm25([1.0])
        docs = _make_docs(1)
        node = KBRetrieveSparseNode(bm25_index=bm25, bm25_docs=docs)
        state = initial_state("schema check")
        cfg = {"retrieval": {"top_k": 5, "min_score": 0.0}}

        result = node.execute(state)
        assert len(decode_list(result["sparse_chunks"])) == 1
        chunk = decode_list(result["sparse_chunks"])[0]
        assert "chunk_text" in chunk
        assert "score" in chunk
        assert "source_path" in chunk
        assert "endpoint_ref" in chunk
        assert "doc_type" in chunk

    def test_chunk_fields_are_serializable(self) -> None:
        """All chunk fields are msgpack-serializable (str, float)."""
        bm25 = _make_bm25([1.0])
        docs = _make_docs(1)
        node = KBRetrieveSparseNode(bm25_index=bm25, bm25_docs=docs)
        state = initial_state("serializable")
        cfg = {"retrieval": {"top_k": 5, "min_score": 0.0}}

        result = node.execute(state)
        chunk = decode_list(result["sparse_chunks"])[0]
        assert isinstance(chunk["chunk_text"], str)
        assert isinstance(chunk["score"], float)
        assert isinstance(chunk["source_path"], str)
        assert isinstance(chunk["endpoint_ref"], str)
        assert isinstance(chunk["doc_type"], str)
