# tests/unit/test_kb_retrieve_dense.py -- CMN-C1-053
# Unit tests for KBRetrieveDenseNode
# Issue #4
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


from src.nodes.kb_retrieve_dense import KBRetrieveDenseNode
from src.schemas.state import initial_state
from src.utils.serde import decode_list


# ---------------------------------------------------------------------------
# Helpers: mock SentenceTransformer and QdrantClient
# No numpy required — use a list-wrapper with a .tolist() method.
# ---------------------------------------------------------------------------


class _FakeVector:
    """Minimal vector stub with .tolist() for CI (no numpy)."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeHit:
    """Minimal Qdrant hit stub."""

    def __init__(self, score: float, payload: dict[str, Any]) -> None:
        self.score = score
        self.payload = payload


def _make_encoder(vector: list[float] | None = None) -> MagicMock:
    """Create a mock encoder with deterministic encode() output (no numpy)."""
    enc = MagicMock()
    vec = vector or [0.1] * 768
    enc.encode.return_value = _FakeVector(vec)
    return enc


def _make_qdrant(hits: list[_FakeHit]) -> MagicMock:
    """Create a mock QdrantClient returning the given hits."""
    client = MagicMock()
    client.search.return_value = hits
    return client


# ---------------------------------------------------------------------------
# Score filtering
# ---------------------------------------------------------------------------


class TestScoreFiltering:
    def test_chunks_below_min_score_excluded(self) -> None:
        """Chunks with score < min_score are not included in dense_chunks."""
        hits = [
            _FakeHit(
                0.9,
                {
                    "chunk_text": "High score",
                    "source_path": "a.json",
                    "endpoint_ref": "/v1/a GET",
                    "doc_type": "openapi_spec",
                },
            ),
            _FakeHit(
                0.6,
                {
                    "chunk_text": "Low score",
                    "source_path": "b.json",
                    "endpoint_ref": "/v1/b GET",
                    "doc_type": "openapi_spec",
                },
            ),
            _FakeHit(
                0.3,
                {
                    "chunk_text": "Very low",
                    "source_path": "c.json",
                    "endpoint_ref": "/v1/c GET",
                    "doc_type": "openapi_spec",
                },
            ),
        ]
        node = KBRetrieveDenseNode(encoder=_make_encoder(), qdrant=_make_qdrant(hits))
        state = initial_state("get orders")
        cfg = {"retrieval": {"top_k": 5, "min_score": 0.65}}

        result = node.execute(state)

        assert len(decode_list(result["dense_chunks"])) == 1
        assert decode_list(result["dense_chunks"])[0]["chunk_text"] == "High score"

    def test_all_chunks_above_threshold_kept(self) -> None:
        """All chunks above min_score are included."""
        hits = [
            _FakeHit(
                0.95,
                {
                    "chunk_text": "Chunk A",
                    "source_path": "a.json",
                    "endpoint_ref": "/v1/a GET",
                    "doc_type": "openapi_spec",
                },
            ),
            _FakeHit(
                0.85,
                {
                    "chunk_text": "Chunk B",
                    "source_path": "b.json",
                    "endpoint_ref": "/v1/b GET",
                    "doc_type": "openapi_spec",
                },
            ),
            _FakeHit(
                0.75,
                {
                    "chunk_text": "Chunk C",
                    "source_path": "c.json",
                    "endpoint_ref": "/v1/c GET",
                    "doc_type": "openapi_spec",
                },
            ),
        ]
        node = KBRetrieveDenseNode(encoder=_make_encoder(), qdrant=_make_qdrant(hits))
        state = initial_state("list endpoints")
        cfg = {"retrieval": {"top_k": 5, "min_score": 0.65}}

        result = node.execute(state)
        assert len(decode_list(result["dense_chunks"])) == 3

    def test_score_exactly_at_threshold_included(self) -> None:
        """Score exactly equal to min_score is included (>= boundary: score < min_score skips)."""
        hits = [
            _FakeHit(
                0.65,
                {
                    "chunk_text": "Edge case",
                    "source_path": "a.json",
                    "endpoint_ref": "/v1/a GET",
                    "doc_type": "openapi_spec",
                },
            ),
        ]
        node = KBRetrieveDenseNode(encoder=_make_encoder(), qdrant=_make_qdrant(hits))
        state = initial_state("edge case")
        cfg = {"retrieval": {"top_k": 5, "min_score": 0.65}}

        result = node.execute(state)
        # 0.65 < 0.65 is False → NOT skipped → included
        assert len(decode_list(result["dense_chunks"])) == 1
        assert abs(decode_list(result["dense_chunks"])[0]["score"] - 0.65) < 1e-9


# ---------------------------------------------------------------------------
# Empty results handling
# ---------------------------------------------------------------------------


class TestEmptyResults:
    def test_empty_qdrant_results(self) -> None:
        """Empty Qdrant response → dense_chunks=[], no exception."""
        node = KBRetrieveDenseNode(encoder=_make_encoder(), qdrant=_make_qdrant([]))
        state = initial_state("obscure query")
        result = node.execute(state)
        assert decode_list(result["dense_chunks"]) == []

    def test_all_results_below_threshold(self) -> None:
        """All results below threshold → empty list."""
        hits = [
            _FakeHit(
                0.1,
                {
                    "chunk_text": "Irrelevant",
                    "source_path": "x.json",
                    "endpoint_ref": "/v1/x GET",
                    "doc_type": "openapi_spec",
                },
            ),
        ]
        node = KBRetrieveDenseNode(encoder=_make_encoder(), qdrant=_make_qdrant(hits))
        state = initial_state("no match")
        cfg = {"retrieval": {"top_k": 5, "min_score": 0.65}}
        result = node.execute(state)
        assert decode_list(result["dense_chunks"]) == []

    def test_qdrant_exception_handled_gracefully(self) -> None:
        """If Qdrant raises, node returns empty dense_chunks, not an exception."""
        qdrant = MagicMock()
        qdrant.search.side_effect = RuntimeError("Connection refused")
        node = KBRetrieveDenseNode(encoder=_make_encoder(), qdrant=qdrant)
        state = initial_state("connection error")

        result = node.execute(state)
        assert decode_list(result["dense_chunks"]) == []

    def test_empty_query_returns_empty(self) -> None:
        """Empty normalized_query → dense_chunks=[]."""
        node = KBRetrieveDenseNode(encoder=_make_encoder(), qdrant=_make_qdrant([]))
        state = initial_state("")
        state["normalized_query"] = ""

        result = node.execute(state)
        assert decode_list(result["dense_chunks"]) == []


# ---------------------------------------------------------------------------
# Short-circuit behavior
# ---------------------------------------------------------------------------


class TestShortCircuit:
    def test_blocked_state_short_circuits(self) -> None:
        """status=blocked → empty dict returned, Qdrant not called."""
        qdrant = MagicMock()
        node = KBRetrieveDenseNode(encoder=_make_encoder(), qdrant=qdrant)
        state = initial_state("blocked query")
        state["answer_status"] = "blocked"

        result = node.execute(state)
        assert result == {}
        qdrant.search.assert_not_called()

    def test_error_state_short_circuits(self) -> None:
        """status=error → empty dict returned."""
        qdrant = MagicMock()
        node = KBRetrieveDenseNode(encoder=_make_encoder(), qdrant=qdrant)
        state = initial_state("error query")
        state["answer_status"] = "error"

        result = node.execute(state)
        assert result == {}
        qdrant.search.assert_not_called()


# ---------------------------------------------------------------------------
# Chunk schema validation
# ---------------------------------------------------------------------------


class TestChunkSchema:
    def test_returned_chunks_have_required_fields(self) -> None:
        """Each dense_chunk has: chunk_text, score, source_path, endpoint_ref, doc_type."""
        hits = [
            _FakeHit(
                0.85,
                {
                    "chunk_text": "GET /v1/items",
                    "source_path": "items.json",
                    "endpoint_ref": "/v1/items GET",
                    "doc_type": "openapi_spec",
                },
            ),
        ]
        node = KBRetrieveDenseNode(encoder=_make_encoder(), qdrant=_make_qdrant(hits))
        state = initial_state("items endpoint")
        cfg = {"retrieval": {"top_k": 5, "min_score": 0.65}}

        result = node.execute(state)
        chunk = decode_list(result["dense_chunks"])[0]
        assert "chunk_text" in chunk
        assert "score" in chunk
        assert "source_path" in chunk
        assert "endpoint_ref" in chunk
        assert "doc_type" in chunk

    def test_chunk_fields_are_strings(self) -> None:
        """All string fields in chunks are str type (msgpack-serializable)."""
        hits = [
            _FakeHit(
                0.9,
                {
                    "chunk_text": b"bytes should not appear",  # bytes in payload → must coerce
                    "source_path": None,  # None → must coerce to ""
                    "endpoint_ref": 42,  # int → must coerce to str
                    "doc_type": "openapi_spec",
                },
            ),
        ]
        node = KBRetrieveDenseNode(encoder=_make_encoder(), qdrant=_make_qdrant(hits))
        state = initial_state("coerce test")
        cfg = {"retrieval": {"top_k": 5, "min_score": 0.65}}

        result = node.execute(state)
        chunk = decode_list(result["dense_chunks"])[0]
        assert isinstance(chunk["chunk_text"], str)
        assert isinstance(chunk["source_path"], str)
        assert isinstance(chunk["endpoint_ref"], str)
        assert isinstance(chunk["doc_type"], str)
