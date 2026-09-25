# tests/unit/test_state.py -- CMN-C1-053 Issue #17
from __future__ import annotations
import uuid
import msgpack
import pytest
from src.schemas.state import initial_state
from src.utils.serde import decode_list, encode_list


def _rt(obj):
    return msgpack.unpackb(msgpack.packb(obj, use_bin_type=True), raw=False)


def test_initial_state_all_fields():
    s = initial_state("How do I authenticate?")
    expected = {
        "raw_query",
        "invocation_id",
        "normalized_query",
        "detected_language",
        "query_entities",
        "dense_chunks",
        "sparse_chunks",
        "reranked_chunks",
        "retrieval_count",
        "answer_markdown",
        "cited_endpoints",
        "code_example",
        "content_staleness_detected",
        "stale_chunk_count",
        "s1_blocked",
        "s2_violation",
        "s3_violation",
        "credential_leak_detected",
        "answer_status",
        "error",
        "duration_ms",
    }
    assert set(s.keys()) == expected


def test_raw_query_preserved():
    q = "What are the rate limits for /orders?"
    assert initial_state(q)["raw_query"] == q


def test_invocation_id_is_valid_uuid():
    uuid.UUID(initial_state("q")["invocation_id"])


def test_custom_invocation_id():
    assert initial_state("q", invocation_id="abc")["invocation_id"] == "abc"


def test_defaults():
    s = initial_state("query")
    assert s["normalized_query"] == ""
    assert s["detected_language"] == "unknown"
    # State-safety: structured list fields are JSON-serialized to str (default '[]').
    assert s["query_entities"] == "[]"
    assert s["dense_chunks"] == "[]"
    assert s["sparse_chunks"] == "[]"
    assert s["reranked_chunks"] == "[]"
    assert decode_list(s["dense_chunks"]) == []
    assert s["retrieval_count"] == 0
    assert s["answer_markdown"] == ""
    assert s["cited_endpoints"] == []
    assert s["code_example"] is None
    assert s["s1_blocked"] is False
    assert s["s2_violation"] is False
    assert s["s3_violation"] is False
    assert s["credential_leak_detected"] is False
    assert s["answer_status"] == "ok"
    assert s["error"] is None
    assert s["duration_ms"] == 0


def test_fresh_state_msgpack_roundtrip():
    s = initial_state("msgpack test")
    r = _rt(dict(s))
    assert r["raw_query"] == "msgpack test"
    assert r["answer_status"] == "ok"
    assert r["code_example"] is None


def test_populated_state_msgpack_roundtrip():
    s = initial_state("List all endpoints")
    s["normalized_query"] = "list all endpoints"
    s["detected_language"] = "en"
    s["query_entities"] = encode_list([{"term": "endpoints", "en_term": "endpoints"}])
    s["dense_chunks"] = encode_list(
        [
            {
                "chunk_text": "GET /v1/items",
                "score": 0.87,
                "source_path": "api.yaml",
                "endpoint_ref": "/v1/items GET",
                "doc_type": "openapi_spec",
            }
        ]
    )
    s["reranked_chunks"] = s["dense_chunks"]
    s["retrieval_count"] = 1
    s["answer_markdown"] = "Use GET /v1/items to list all items."
    s["cited_endpoints"] = ["/v1/items GET"]
    s["code_example"] = "curl https://api.example.com/v1/items"
    s["duration_ms"] = 342
    r = _rt(dict(s))
    assert r["answer_markdown"] == "Use GET /v1/items to list all items."
    assert r["cited_endpoints"] == ["/v1/items GET"]
    assert r["retrieval_count"] == 1
    # The list fields survive msgpack as plain str and decode back to the original list.
    assert isinstance(r["dense_chunks"], str)
    assert decode_list(r["dense_chunks"])[0]["endpoint_ref"] == "/v1/items GET"


def test_no_dataclass_in_state():
    import dataclasses

    s = initial_state("check")
    for k, v in s.items():
        assert not dataclasses.is_dataclass(v), f"state[{k!r}] is dataclass"


def test_code_example_none_roundtrip():
    s = initial_state("q")
    assert _rt(dict(s))["code_example"] is None


@pytest.mark.parametrize("answer_status", ["ok", "blocked", "degraded", "error"])
def test_status_roundtrip(answer_status):
    s = initial_state("q")
    s["answer_status"] = answer_status
    assert _rt(dict(s))["answer_status"] == answer_status


def test_large_chunks_msgpack():
    s = initial_state("bulk")
    s["reranked_chunks"] = encode_list(
        [
            {
                "chunk_text": f"chunk {i} " * 50,
                "score": 0.7,
                "source_path": f"doc_{i}.yaml",
                "endpoint_ref": f"/v1/item/{i} GET",
                "doc_type": "openapi_spec",
            }
            for i in range(20)
        ]
    )
    s["retrieval_count"] = 20
    assert len(decode_list(_rt(dict(s))["reranked_chunks"])) == 20
