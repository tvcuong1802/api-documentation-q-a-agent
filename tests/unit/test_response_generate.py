# tests/unit/test_response_generate.py -- CMN-C1-053
# Unit tests for ResponseGenerateNode (Issues #6, #20)
from __future__ import annotations

import json
from unittest.mock import MagicMock


from src.nodes.response_generate import ResponseGenerateNode
from src.schemas.state import CmnApiDocQaState, initial_state
from src.utils.serde import encode_list


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_state_with_chunks(chunks: list[dict] | None = None) -> CmnApiDocQaState:
    """Return a state with reranked_chunks (JSON-serialized str) and a normalized_query."""
    chunks = chunks or []
    state = initial_state("How do I authenticate?")
    state["normalized_query"] = "How do I authenticate?"
    state["reranked_chunks"] = encode_list(chunks)
    state["retrieval_count"] = len(chunks)
    return state


def _valid_chunks() -> list[dict]:
    return [
        {
            "chunk_text": "POST /v1/auth — Authenticate using client credentials.",
            "score": 0.9,
            "source_path": "openapi_spec/auth.yaml",
            "endpoint_ref": "/v1/auth POST",
            "doc_type": "openapi_spec",
        }
    ]


def _valid_answer_json(**overrides: object) -> str:
    data = {
        "thought": "The user wants to know about authentication.",
        "answer": "Use `POST /v1/auth` with your client credentials.\n\nSee [endpoint: /v1/auth POST].",
        "citations": ["/v1/auth POST"],
        "confidence": 0.92,
    }
    data.update(overrides)  # type: ignore[arg-type]
    return json.dumps(data)


# ── Test: valid JSON parsed, primitives written to state ─────────────────────


def test_valid_json_parsed_primitives_written_to_state() -> None:
    """Valid LLM output: primitives written to state; no Pydantic object in state."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = _valid_answer_json()

    node = ResponseGenerateNode(llm=mock_llm)
    state = _make_state_with_chunks(_valid_chunks())

    result = node.execute(state)

    assert result["answer_markdown"] == (
        "Use `POST /v1/auth` with your client credentials.\n\nSee [endpoint: /v1/auth POST]."
    )
    assert result["cited_endpoints"] == ["/v1/auth POST"]
    assert result["answer_status"] == "ok"

    # Ensure no Pydantic / non-primitive objects in state
    state.update(result)
    for key, value in state.items():
        assert isinstance(value, (str, int, bool, list, dict, type(None))), (
            f"state[{key!r}] is not a primitive: {type(value)}"
        )


# ── Test: retry on JSONDecodeError ───────────────────────────────────────────


def test_retry_on_json_decode_error_succeeds_on_second_attempt() -> None:
    """First LLM call returns bad JSON; second returns valid JSON — parse succeeds."""
    mock_llm = MagicMock()
    # First call: invalid JSON; second call: valid JSON
    mock_llm.generate.side_effect = [
        "NOT VALID JSON {{{",
        _valid_answer_json(),
    ]

    node = ResponseGenerateNode(llm=mock_llm)
    state = _make_state_with_chunks(_valid_chunks())

    result = node.execute(state)

    assert result["answer_status"] == "ok"
    assert result["cited_endpoints"] == ["/v1/auth POST"]
    # LLM was called at least twice (initial + one retry for parse)
    assert mock_llm.generate.call_count >= 2


# ── Test: degraded fallback when all retries exhausted ───────────────────────


def test_degraded_fallback_when_all_retries_exhausted() -> None:
    """All LLM retries return bad JSON — fallback to raw text, answer_status='degraded'."""
    mock_llm = MagicMock()
    # All calls return invalid JSON
    mock_llm.generate.return_value = "INVALID {{{ NOT JSON"

    node = ResponseGenerateNode(llm=mock_llm)
    state = _make_state_with_chunks(_valid_chunks())

    result = node.execute(state)

    assert result["answer_status"] == "degraded"
    assert result["cited_endpoints"] == []
    # answer_markdown should be the raw fallback text
    assert "INVALID" in result["answer_markdown"] or result["answer_markdown"] == "INVALID {{{ NOT JSON"


# ── Test: empty LLM response -> answer_status='degraded' ────────────────────────────


def test_empty_llm_response_sets_status_degraded() -> None:
    """Empty string from LLM -> answer_status='degraded'."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = ""

    node = ResponseGenerateNode(llm=mock_llm)
    state = _make_state_with_chunks(_valid_chunks())

    result = node.execute(state)

    assert result["answer_status"] == "degraded"
    assert result["answer_markdown"] == ""
    assert result["cited_endpoints"] == []


# ── Test: thought_trace and cited_endpoints extracted correctly ───────────────


def test_thought_trace_and_cited_endpoints_extracted_correctly() -> None:
    """Parsed thought and citations are correctly extracted as primitives."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = _valid_answer_json(
        thought="Step 1: Check context. Step 2: Cite endpoint.",
        citations=["/v1/auth POST", "/v1/token GET"],
    )

    node = ResponseGenerateNode(llm=mock_llm)
    state = _make_state_with_chunks(_valid_chunks())

    result = node.execute(state)

    assert result["answer_status"] == "ok"
    assert "/v1/auth POST" in result["cited_endpoints"]
    assert "/v1/token GET" in result["cited_endpoints"]
    # All citations must be plain strings
    for citation in result["cited_endpoints"]:
        assert isinstance(citation, str)


# ── Test: returns primitive dict only ────────────────────────────────────────


def test_returns_primitive_dict_only() -> None:
    """Result dict must contain only msgpack-serializable primitive values."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = _valid_answer_json()

    node = ResponseGenerateNode(llm=mock_llm)
    state = _make_state_with_chunks(_valid_chunks())

    result = node.execute(state)

    for key, value in result.items():
        assert isinstance(value, (str, int, bool, list, dict, type(None))), (
            f"result[{key!r}] is not a primitive: {type(value)}"
        )
    # Especially: no Pydantic BaseModel in result
    for value in result.values():
        assert not hasattr(value, "__fields__"), "Pydantic model found in result"
        assert not hasattr(value, "model_fields"), "Pydantic model found in result"


# ── Test: degraded on empty reranked_chunks ───────────────────────────────────


def test_degraded_on_empty_reranked_chunks() -> None:
    """Empty reranked_chunks -> canned no-context response, answer_status='degraded'."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = _valid_answer_json()

    node = ResponseGenerateNode(llm=mock_llm)
    state = _make_state_with_chunks([])

    result = node.execute(state)

    assert result["answer_status"] == "degraded"
    assert "No relevant documentation" in result["answer_markdown"]
    assert result["cited_endpoints"] == []
    # LLM should NOT have been called (we degrade before calling LLM)
    mock_llm.generate.assert_not_called()


# ── Test: short-circuit on blocked status ────────────────────────────────────


def test_short_circuit_on_blocked_status() -> None:
    """Node short-circuits (returns {}) when state status is 'blocked'."""
    mock_llm = MagicMock()

    node = ResponseGenerateNode(llm=mock_llm)
    state = _make_state_with_chunks(_valid_chunks())
    state["answer_status"] = "blocked"

    result = node.execute(state)

    assert result == {}
    mock_llm.generate.assert_not_called()


# ── Test: multiple doc_type groups in prompt ─────────────────────────────────


def test_multiple_doc_type_groups_handled() -> None:
    """Chunks with different doc_types are all included in the prompt."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = _valid_answer_json()

    chunks = [
        {
            "chunk_text": "POST /v1/auth — Auth endpoint.",
            "score": 0.9,
            "source_path": "openapi_spec/auth.yaml",
            "endpoint_ref": "/v1/auth POST",
            "doc_type": "openapi_spec",
        },
        {
            "chunk_text": "Authentication guide: use client credentials.",
            "score": 0.8,
            "source_path": "guides/auth.md",
            "endpoint_ref": "",
            "doc_type": "markdown_guide",
        },
    ]

    node = ResponseGenerateNode(llm=mock_llm)
    state = _make_state_with_chunks(chunks)

    result = node.execute(state)

    assert result["answer_status"] == "ok"
    # Verify the prompt built contained both doc_type groups
    call_args = mock_llm.generate.call_args[0][0]
    assert "OPENAPI_SPEC" in call_args
    assert "MARKDOWN_GUIDE" in call_args
