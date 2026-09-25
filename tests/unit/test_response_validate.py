# tests/unit/test_response_validate.py -- CMN-C1-053
# Unit tests for ResponseValidateNode (Issues #7, #14)
# NEVER import or mention agenticstar anywhere.
from __future__ import annotations

import pytest

from framework.errors import SecurityViolationError
from src.nodes.response_validate import ResponseValidateNode
from src.schemas.state import initial_state
from src.utils.serde import encode_list


def _make_state(**kwargs):
    """Build a state dict using initial_state factory, then override with kwargs.

    reranked_chunks is a JSON-serialized str field (state-safety); a list kwarg
    is encoded so callers can keep passing plain chunk lists.
    """
    state = initial_state("test query")
    if isinstance(kwargs.get("reranked_chunks"), list):
        kwargs["reranked_chunks"] = encode_list(kwargs["reranked_chunks"])
    state.update(kwargs)
    return state


def _make_chunks(endpoint_refs: list[str]) -> list[dict]:
    """Build minimal reranked_chunks with given endpoint_refs."""
    return [
        {
            "chunk_text": f"Documentation for {ref}",
            "score": 0.9,
            "source_path": f"/docs/{ref.replace(' ', '_')}",
            "endpoint_ref": ref,
            "doc_type": "openapi_spec",
        }
        for ref in endpoint_refs
    ]


class TestResponseValidateNodeGrounding:
    """Tests for endpoint grounding check (Issue #7)."""

    def test_clean_answer_with_grounded_citations_passes(self):
        """cited_endpoints grounded in reranked_chunks → no exception, s3_violation=False."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Use the /users GET endpoint to list users.",
            cited_endpoints=["/users GET"],
            reranked_chunks=_make_chunks(["/users GET"]),
        )
        result = node.execute(state)
        assert result["s3_violation"] is False
        assert result["credential_leak_detected"] is False
        assert result["answer_status"] == "ok"

    def test_hallucinated_endpoint_raises_security_violation(self):
        """Cited endpoint not in any chunk → SecurityViolationError, s3_violation=True."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Use the /admin/delete POST endpoint.",
            cited_endpoints=["/admin/delete POST"],
            reranked_chunks=_make_chunks(["/users GET"]),  # no /admin/delete
        )
        with pytest.raises(SecurityViolationError, match="Hallucinated endpoint"):
            node.execute(state)
        assert state["s3_violation"] is True

    def test_empty_cited_endpoints_passes_grounding_check(self):
        """No citations → vacuously grounded, no exception."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="The API supports OAuth2.",
            cited_endpoints=[],
            reranked_chunks=_make_chunks(["/users GET"]),
        )
        result = node.execute(state)
        assert result["s3_violation"] is False

    def test_multiple_endpoints_all_grounded_passes(self):
        """Multiple cited_endpoints, all grounded → passes."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Use /users GET and /users POST.",
            cited_endpoints=["/users GET", "/users POST"],
            reranked_chunks=_make_chunks(["/users GET", "/users POST"]),
        )
        result = node.execute(state)
        assert result["s3_violation"] is False

    def test_one_hallucinated_among_grounded_raises(self):
        """One hallucinated endpoint among valid ones → SecurityViolationError."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Use /users GET and /secret/admin DELETE.",
            cited_endpoints=["/users GET", "/secret/admin DELETE"],
            reranked_chunks=_make_chunks(["/users GET"]),
        )
        with pytest.raises(SecurityViolationError, match="Hallucinated endpoint"):
            node.execute(state)
        assert state["s3_violation"] is True


class TestResponseValidateNodeCredentialScan:
    """Tests for credential leak scan (Issue #14)."""

    def test_bearer_token_raises_security_violation(self):
        """Bearer token in answer → SecurityViolationError, credential_leak_detected=True."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Use this token: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        with pytest.raises(SecurityViolationError, match="Credential leak"):
            node.execute(state)
        assert state["credential_leak_detected"] is True
        assert state["s3_violation"] is True

    def test_api_key_pattern_detected(self):
        """api_key= pattern in answer → SecurityViolationError."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Pass api_key=s3cr3tpassword123 in headers.",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        with pytest.raises(SecurityViolationError, match="Credential leak"):
            node.execute(state)
        assert state["credential_leak_detected"] is True

    def test_sk_key_detected(self):
        """sk- key pattern in answer → SecurityViolationError."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Your API key is sk-ABCDEFabcdef1234567890.",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        with pytest.raises(SecurityViolationError, match="Credential leak"):
            node.execute(state)
        assert state["credential_leak_detected"] is True

    def test_jwt_pattern_detected(self):
        """JWT 3-part base64.base64.base64 pattern → SecurityViolationError."""
        node = ResponseValidateNode()
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        state = _make_state(
            answer_markdown=f"Token: {jwt}",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        with pytest.raises(SecurityViolationError, match="Credential leak"):
            node.execute(state)
        assert state["credential_leak_detected"] is True

    def test_authorization_header_detected(self):
        """Authorization: header with token → SecurityViolationError."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Set Authorization: Token myverylongsecrettoken123456 in request.",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        with pytest.raises(SecurityViolationError, match="Credential leak"):
            node.execute(state)
        assert state["credential_leak_detected"] is True

    def test_clean_answer_no_credentials_passes(self):
        """No credentials in answer → passes credential scan."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="Use OAuth2 for authentication. See /auth/token POST endpoint.",
            cited_endpoints=["/auth/token POST"],
            reranked_chunks=_make_chunks(["/auth/token POST"]),
        )
        result = node.execute(state)
        assert result["credential_leak_detected"] is False
        assert result["s3_violation"] is False


class TestResponseValidateNodeNonSuppressible:
    """Tests that S-3 runs even when status is blocked or degraded."""

    def test_runs_even_when_status_blocked(self):
        """S-3 non-suppressible: executes even when answer_status='blocked'."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_status="blocked",
            answer_markdown="",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        # Should not raise — clean pass even on blocked status
        result = node.execute(state)
        assert result["s3_violation"] is False
        assert result["answer_status"] == "blocked"  # preserves blocked status

    def test_runs_even_when_status_degraded(self):
        """S-3 non-suppressible: executes even when answer_status='degraded'."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_status="degraded",
            answer_markdown="No relevant documentation found.",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        result = node.execute(state)
        assert result["s3_violation"] is False
        assert result["answer_status"] == "degraded"  # preserves degraded status

    def test_blocked_with_credential_still_raises(self):
        """Even when blocked, credential in answer still triggers S-3 violation."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_status="blocked",
            answer_markdown="Bearer secrettoken12345678 is your key.",
            cited_endpoints=[],
            reranked_chunks=[],
        )
        with pytest.raises(SecurityViolationError, match="Credential leak"):
            node.execute(state)
        assert state["credential_leak_detected"] is True


class TestResponseValidateNodeReturnsPrimitives:
    """Tests that execute() returns only primitives-only dict."""

    def test_returns_primitives_only_dict(self):
        """Return value must be dict with only primitive values."""
        node = ResponseValidateNode()
        state = _make_state(
            answer_markdown="The /users GET endpoint returns a list.",
            cited_endpoints=["/users GET"],
            reranked_chunks=_make_chunks(["/users GET"]),
        )
        result = node.execute(state)
        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str), f"Key {key!r} is not str"
            assert isinstance(value, (str, int, float, bool, type(None))), (
                f"Value for {key!r} is not a primitive: {type(value)}"
            )
