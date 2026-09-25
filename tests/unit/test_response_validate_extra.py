# tests/unit/test_response_validate_extra.py -- CMN-C1-053
# Extra edge cases for ResponseValidateNode (Issue #23 gap fill)
# NEVER import or mention agenticstar anywhere.
from __future__ import annotations

import pytest

from framework.errors import SecurityViolationError
from src.nodes.response_validate import ResponseValidateNode
from src.schemas.state import initial_state
from src.utils.serde import encode_list


def _make_state(**kwargs):
    state = initial_state("test query")
    # reranked_chunks is a JSON-serialized str field (state-safety); encode a list kwarg.
    if isinstance(kwargs.get("reranked_chunks"), list):
        kwargs["reranked_chunks"] = encode_list(kwargs["reranked_chunks"])
    state.update(kwargs)
    return state


class TestResponseValidateExtra:
    """Extra edge cases not covered in test_response_validate.py."""

    def test_jwt_three_part_base64_detected_as_credential_leak(self) -> None:
        """
        JWT 3-part base64url pattern (header.payload.sig) is detected as credential leak.
        Each part >= 10 chars, separated by dots.
        """
        node = ResponseValidateNode()
        # Realistic-looking JWT
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        state = _make_state(
            answer_markdown=f"Use this token: {jwt}",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        with pytest.raises(SecurityViolationError, match="Credential leak"):
            node.execute(state)
        assert state["credential_leak_detected"] is True
        assert state["s3_violation"] is True

    def test_empty_generated_answer_passes_credential_scan(self) -> None:
        """
        Empty answer_markdown string has nothing to scan — passes credential check.
        Also passes grounding check (no citations).
        """
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        result = node.execute(state)
        assert result["credential_leak_detected"] is False
        assert result["s3_violation"] is False

    def test_whitespace_only_answer_passes_credential_scan(self) -> None:
        """Whitespace-only answer_markdown passes credential scan."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="   \n\t  ",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        result = node.execute(state)
        assert result["credential_leak_detected"] is False

    def test_none_answer_markdown_passes_scan(self) -> None:
        """None answer_markdown (normalized to '') passes credential scan without error."""
        node = ResponseValidateNode()
        state = _make_state(
            cited_endpoints=[],
            reranked_chunks=[],
        )
        state["answer_markdown"] = ""  # initial_state gives empty string
        result = node.execute(state)
        assert result["credential_leak_detected"] is False
        assert result["s3_violation"] is False

    def test_sk_key_pattern_detected(self) -> None:
        """sk- prefix key pattern (>= 8 chars after prefix) is detected as credential leak."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Your API key is sk-ABCDEF12345678abcdef1234.",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        with pytest.raises(SecurityViolationError, match="Credential leak"):
            node.execute(state)
        assert state["credential_leak_detected"] is True

    def test_authorization_header_with_long_token_detected(self) -> None:
        """Authorization: header with long token value is detected."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Pass Authorization: Token myLongSecretToken1234567890 in your header.",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        with pytest.raises(SecurityViolationError, match="Credential leak"):
            node.execute(state)
        assert state["credential_leak_detected"] is True

    def test_chunk_text_only_does_not_ground(self) -> None:
        """
        Endpoint grounding must NOT match via chunk_text only (Loophole A / negative mention).
        Even if the endpoint string appears in chunk_text, if endpoint_ref is empty, it raises SecurityViolationError.
        """
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Use the /v1/orders GET endpoint.",
            cited_endpoints=["/v1/orders GET"],
            reranked_chunks=[
                {
                    "chunk_text": "Do not call /v1/orders GET — it is deprecated.",
                    "score": 0.9,
                    "source_path": "kb/orders.json",
                    "endpoint_ref": "",  # no endpoint_ref
                    "doc_type": "openapi_spec",
                }
            ],
        )
        with pytest.raises(SecurityViolationError, match="Hallucinated endpoint"):
            node.execute(state)
        assert state["s3_violation"] is True

    def test_prefix_substring_fails_grounding(self) -> None:
        """
        A citation of a path prefix (e.g. '/v1') must NOT pass grounding against '/v1/orders GET' (Loophole B).
        """
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Refer to the /v1 API.",
            cited_endpoints=["/v1"],
            reranked_chunks=[
                {
                    "chunk_text": "The API is here.",
                    "score": 0.9,
                    "source_path": "kb/orders.json",
                    "endpoint_ref": "/v1/orders GET",
                    "doc_type": "openapi_spec",
                }
            ],
        )
        with pytest.raises(SecurityViolationError, match="Hallucinated endpoint"):
            node.execute(state)
        assert state["s3_violation"] is True

    def test_concatenated_citation_fails_grounding(self) -> None:
        """
        A citation that concatenates a valid and invalid endpoint must NOT pass grounding (Loophole C).
        """
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Refer to /v1/orders GET /v1/admin/secret.",
            cited_endpoints=["/v1/orders GET /v1/admin/secret"],
            reranked_chunks=[
                {
                    "chunk_text": "The API is here.",
                    "score": 0.9,
                    "source_path": "kb/orders.json",
                    "endpoint_ref": "/v1/orders GET",
                    "doc_type": "openapi_spec",
                }
            ],
        )
        with pytest.raises(SecurityViolationError, match="Hallucinated endpoint"):
            node.execute(state)
        assert state["s3_violation"] is True

    def test_credential_check_runs_before_grounding_check(self) -> None:
        """
        Credential scan runs before grounding check.
        If both violations exist, SecurityViolationError should mention credential leak.
        """
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Use Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 and /v1/nonexistent.",
            cited_endpoints=["/v1/nonexistent DELETE"],  # also ungrounded
            reranked_chunks=[],  # no grounding
        )
        with pytest.raises(SecurityViolationError, match="Credential leak"):
            node.execute(state)
        # Credential check fires first → credential_leak_detected set
        assert state["credential_leak_detected"] is True

    def test_trailing_punctuation_stripped_successfully(self) -> None:
        """
        Trailing punctuation (like period, comma, colon, semicolon) in LLM citation strings
        is stripped before validation to avoid false negatives (R4-02).
        """
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Use the /v1/orders GET endpoint.",
            cited_endpoints=["/v1/orders GET.", "[endpoint: /v1/orders GET];"],
            reranked_chunks=[
                {
                    "chunk_text": "The API is here.",
                    "score": 0.9,
                    "source_path": "kb/orders.json",
                    "endpoint_ref": "/v1/orders GET",
                    "doc_type": "openapi_spec",
                }
            ],
        )
        result = node.execute(state)
        assert result["s3_violation"] is False
