# tests/unit/test_response_generate_extra.py -- CMN-C1-053
# Extra edge cases for ResponseGenerateNode (Issue #23 gap fill)
# NEVER import or mention agenticstar anywhere.
from __future__ import annotations

import json
from unittest.mock import MagicMock


from src.nodes.response_generate import ResponseGenerateNode
from src.schemas.state import initial_state
from src.utils.serde import encode_list


def _make_state_with_chunks(chunks):
    chunks = chunks or []
    state = initial_state("How do I list orders?")
    state["normalized_query"] = "How do I list orders?"
    state["reranked_chunks"] = encode_list(chunks)
    state["retrieval_count"] = len(chunks)
    return state


def _valid_chunks():
    return [
        {
            "chunk_text": "GET /v1/orders — List orders with pagination.",
            "score": 0.9,
            "source_path": "openapi_spec/orders.yaml",
            "endpoint_ref": "/v1/orders GET",
            "doc_type": "openapi_spec",
        }
    ]


class TestResponseGenerateExtra:
    """Extra edge cases not covered in test_response_generate.py."""

    def test_canned_no_context_response_when_empty_chunks(self) -> None:
        """
        When reranked_chunks=[], the node returns the canned no-context response
        without calling the LLM.
        """
        mock_llm = MagicMock()
        node = ResponseGenerateNode(llm=mock_llm)
        state = _make_state_with_chunks([])

        result = node.execute(state)

        # LLM must NOT be called
        mock_llm.generate.assert_not_called()
        assert result["answer_status"] == "degraded"
        assert result["cited_endpoints"] == []
        assert result["code_example"] is None
        # Answer should contain a no-context message
        assert "No relevant documentation" in result["answer_markdown"]

    def test_all_retries_exhausted_llm_exception_returns_degraded(self) -> None:
        """
        When LLM raises an exception on every attempt (all _MAX_RETRIES+1 calls),
        the node returns status=degraded with an error field.
        """
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM timeout on every retry")

        node = ResponseGenerateNode(llm=mock_llm)
        state = _make_state_with_chunks(_valid_chunks())

        result = node.execute(state)

        assert result["answer_status"] == "degraded"
        assert result["cited_endpoints"] == []
        assert result["answer_markdown"] == ""
        # error field should be populated
        assert "error" in result
        assert result["error"] is not None
        # LLM was called at least once
        assert mock_llm.generate.call_count >= 1

    def test_include_code_examples_false_returns_no_code(self) -> None:
        """When include_code_examples=False, code_example is None even if answer has code block."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(
            {
                "thought": "User wants orders.",
                "answer": "Use this endpoint:\n```python\nimport requests\nrequests.get('/v1/orders')\n```",
                "citations": ["/v1/orders GET"],
                "confidence": 0.9,
            }
        )

        node = ResponseGenerateNode(llm=mock_llm, config={"include_code_examples": False})
        state = _make_state_with_chunks(_valid_chunks())

        result = node.execute(state)

        assert result["answer_status"] == "ok"
        assert result["code_example"] is None  # disabled

    def test_code_example_extracted_when_enabled(self) -> None:
        """When include_code_examples=True (default), code block extracted from answer."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(
            {
                "thought": "User wants orders.",
                "answer": "Example:\n```python\nrequests.get('/v1/orders')\n```",
                "citations": ["/v1/orders GET"],
                "confidence": 0.9,
            }
        )

        node = ResponseGenerateNode(llm=mock_llm, config={"include_code_examples": True})
        state = _make_state_with_chunks(_valid_chunks())

        result = node.execute(state)

        assert result["answer_status"] == "ok"
        assert result["code_example"] is not None
        assert "requests.get" in result["code_example"]

    def test_short_circuit_on_error_status(self) -> None:
        """Node short-circuits (returns {}) when state status is 'error'."""
        mock_llm = MagicMock()
        node = ResponseGenerateNode(llm=mock_llm)
        state = _make_state_with_chunks(_valid_chunks())
        state["answer_status"] = "error"

        result = node.execute(state)

        assert result == {}
        mock_llm.generate.assert_not_called()

    def test_json_with_extra_fields_parsed_ok(self) -> None:
        """LLM response with extra fields beyond schema is handled gracefully."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(
            {
                "thought": "Some reasoning.",
                "answer": "The /v1/orders endpoint lists orders.",
                "citations": ["/v1/orders GET"],
                "confidence": 0.85,
                "extra_field": "this should be ignored",
            }
        )

        node = ResponseGenerateNode(llm=mock_llm)
        state = _make_state_with_chunks(_valid_chunks())

        result = node.execute(state)

        assert result["answer_status"] == "ok"
        assert result["cited_endpoints"] == ["/v1/orders GET"]
