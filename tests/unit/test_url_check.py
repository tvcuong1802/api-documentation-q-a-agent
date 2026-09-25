# tests/unit/test_url_check.py -- CMN-C1-053
# Unit tests for UrlCheckNode (Issue #26)
# NEVER import or mention agenticstar anywhere.
from __future__ import annotations

# The node encodes its result as a JSON string: a nested list[dict] is not msgpack-safe
# in the LangGraph checkpoint, which the proof-of-boundary state test enforces. Reading it
# back through decode_list keeps these tests asserting the same values as before.
from src.utils.serde import decode_list

import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from src.nodes.url_check import UrlCheckNode, _CIRCUIT_BREAKER_THRESHOLD
from src.schemas.state import initial_state


def _make_state(**kwargs):
    """Build a state dict using initial_state factory, then override with kwargs."""
    state = initial_state("test query")
    state.update(kwargs)
    return state


class TestUrlCheckNodeDisabled:
    """Tests for disabled-by-default behavior."""

    def test_disabled_by_default_no_requests_made(self):
        """When url_liveness_check not set, returns empty results without any HTTP request."""
        node = UrlCheckNode()
        state = _make_state(
            answer_markdown="See https://example.com/api/users for details.",
        )
        with patch("src.nodes.url_check.urllib.request.urlopen") as mock_urlopen:
            result = node.execute(state)
        mock_urlopen.assert_not_called()
        assert decode_list((result["url_check_results"])) == []

    def test_disabled_via_config_false(self):
        """Explicit url_liveness_check=False → disabled."""
        node = UrlCheckNode(config={"url_liveness_check": False})
        state = _make_state(
            answer_markdown="See https://example.com/docs",
        )
        with patch("src.nodes.url_check.urllib.request.urlopen") as mock_urlopen:
            result = node.execute(state)
        mock_urlopen.assert_not_called()
        assert decode_list(result["url_check_results"]) == []

    def test_no_urls_in_answer_returns_empty(self):
        """No URLs in answer_markdown → empty results (even when enabled)."""
        node = UrlCheckNode(config={"url_liveness_check": True})
        state = _make_state(
            answer_markdown="Use the /users GET endpoint for listing users.",
        )
        with patch("src.nodes.url_check.urllib.request.urlopen") as mock_urlopen:
            result = node.execute(state)
        mock_urlopen.assert_not_called()
        assert decode_list(result["url_check_results"]) == []


class TestUrlCheckNodeEnabled:
    """Tests for enabled URL liveness check."""

    def _mock_live_response(self, status: int = 200):
        """Create a mock HTTP response with given status code."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_enabled_live_url_returns_live_true(self):
        """When enabled and URL returns 200, result has live=True."""
        node = UrlCheckNode(config={"url_liveness_check": True})
        state = _make_state(
            answer_markdown="See https://api.example.com/users for details.",
        )
        mock_resp = self._mock_live_response(200)
        with patch("src.nodes.url_check.urllib.request.urlopen", return_value=mock_resp):
            result = node.execute(state)

        assert len(decode_list(result["url_check_results"])) == 1
        check = decode_list(result["url_check_results"])[0]
        assert check["url"] == "https://api.example.com/users"
        assert check["live"] is True
        assert check["status_code"] == 200

    def test_http_error_handled_gracefully_no_raise(self):
        """HTTP error (e.g. 500) is handled gracefully — no exception propagated."""
        node = UrlCheckNode(config={"url_liveness_check": True})
        state = _make_state(
            answer_markdown="See https://api.example.com/broken",
        )
        http_error = urllib.error.HTTPError(
            url="https://api.example.com/broken",
            code=500,
            msg="Internal Server Error",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,  # type: ignore[arg-type]
        )
        with patch("src.nodes.url_check.urllib.request.urlopen", side_effect=http_error):
            # Should NOT raise
            result = node.execute(state)

        assert len(decode_list(result["url_check_results"])) == 1
        check = decode_list(result["url_check_results"])[0]
        assert check["url"] == "https://api.example.com/broken"
        assert check["live"] is False
        assert check["status_code"] == 500

    def test_connection_error_handled_gracefully(self):
        """Network error handled gracefully — live=False, status_code=None."""
        node = UrlCheckNode(config={"url_liveness_check": True})
        state = _make_state(
            answer_markdown="See https://unreachable.example.com/api",
        )
        with patch(
            "src.nodes.url_check.urllib.request.urlopen",
            side_effect=OSError("Connection refused"),
        ):
            result = node.execute(state)

        assert len(decode_list(result["url_check_results"])) == 1
        check = decode_list(result["url_check_results"])[0]
        assert check["live"] is False
        assert check["status_code"] is None

    def test_returns_primitives_dict(self):
        """Result must be a dict with only primitives."""
        node = UrlCheckNode(config={"url_liveness_check": True})
        state = _make_state(
            answer_markdown="See https://example.com/docs",
        )
        mock_resp = self._mock_live_response(200)
        with patch("src.nodes.url_check.urllib.request.urlopen", return_value=mock_resp):
            result = node.execute(state)

        assert isinstance(result, dict)
        checks = decode_list(result["url_check_results"])
        assert isinstance(checks, list)
        for item in checks:
            assert isinstance(item, dict)
            for k, v in item.items():
                assert isinstance(k, str)
                assert isinstance(v, (str, bool, int, float, type(None)))


class TestUrlCheckNodeCircuitBreaker:
    """Tests for the circuit breaker behavior."""

    def test_circuit_breaker_stops_after_threshold_consecutive_failures(self):
        """After _CIRCUIT_BREAKER_THRESHOLD consecutive failures, stops making requests."""
        node = UrlCheckNode(config={"url_liveness_check": True})
        # More URLs than the circuit breaker threshold
        num_urls = _CIRCUIT_BREAKER_THRESHOLD + 3
        urls = [f"https://fail{i}.example.com/api" for i in range(num_urls)]
        answer = " ".join(f"See {url}" for url in urls)
        state = _make_state(answer_markdown=answer)

        call_count = 0

        def mock_urlopen_fail(req, timeout=None):
            nonlocal call_count
            call_count += 1
            raise OSError("Connection refused")

        with patch("src.nodes.url_check.urllib.request.urlopen", side_effect=mock_urlopen_fail):
            result = node.execute(state)

        # Should have stopped after CIRCUIT_BREAKER_THRESHOLD failures
        assert call_count == _CIRCUIT_BREAKER_THRESHOLD
        assert len(decode_list(result["url_check_results"])) == _CIRCUIT_BREAKER_THRESHOLD

    def test_circuit_breaker_resets_on_success(self):
        """Successful request resets the consecutive failure counter."""
        node = UrlCheckNode(config={"url_liveness_check": True})
        # Pattern: fail, fail, SUCCESS, fail, fail — breaker should not trip
        urls = [
            "https://fail1.example.com",
            "https://fail2.example.com",
            "https://success.example.com",
            "https://fail3.example.com",
            "https://fail4.example.com",
        ]
        answer = " ".join(urls)
        state = _make_state(answer_markdown=answer)

        call_idx = 0
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        def mock_urlopen_mixed(req, timeout=None):
            nonlocal call_idx
            url = req.full_url if hasattr(req, "full_url") else str(req)
            call_idx += 1
            if "success" in url:
                return mock_resp
            raise OSError("fail")

        with patch("src.nodes.url_check.urllib.request.urlopen", side_effect=mock_urlopen_mixed):
            result = node.execute(state)

        # All 5 should have been attempted (no premature breaker trip)
        assert len(decode_list(result["url_check_results"])) == 5

    def test_enabled_via_state_flag_override(self):
        """State url_liveness_check flag enables the check at runtime."""
        node = UrlCheckNode()  # No config — disabled by default
        state = _make_state(
            answer_markdown="See https://example.com/api",
        )
        # Runtime override via state
        state["url_liveness_check"] = True  # type: ignore[typeddict-unknown-key]

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("src.nodes.url_check.urllib.request.urlopen", return_value=mock_resp):
            result = node.execute(state)

        assert len(decode_list(result["url_check_results"])) == 1
        assert decode_list(result["url_check_results"])[0]["live"] is True


class TestUrlCheckNodeSSRF:
    """Tests verifying the Server-Side Request Forgery (SSRF) protections."""

    @pytest.mark.parametrize(
        "ssrf_url",
        [
            "http://127.0.0.1/admin",
            "http://192.168.1.100/status",
            "http://10.0.0.1/metadata",
            "http://169.254.169.254/metadata",
            "http://100.64.0.1/status",
        ],
    )
    def test_ssrf_blocked_on_ip_ranges(self, ssrf_url: str):
        """IP address hostnames belonging to loopback or private ranges are blocked directly."""
        node = UrlCheckNode(config={"url_liveness_check": True})
        state = _make_state(answer_markdown=f"Go to {ssrf_url}")

        with patch("src.nodes.url_check.urllib.request.urlopen") as mock_urlopen:
            result = node.execute(state)

        mock_urlopen.assert_not_called()
        assert len(decode_list(result["url_check_results"])) == 1
        check = decode_list(result["url_check_results"])[0]
        assert check["url"] == ssrf_url
        assert check["live"] is False
        assert check["ssrf_blocked"] is True

    def test_ssrf_blocked_on_localhost_dns(self):
        """Hostname resolving to loopback via DNS is blocked."""
        node = UrlCheckNode(config={"url_liveness_check": True})
        state = _make_state(answer_markdown="Check http://localhost/config")

        with patch("src.nodes.url_check.socket.gethostbyname", return_value="127.0.0.1"):
            with patch("src.nodes.url_check.urllib.request.urlopen") as mock_urlopen:
                result = node.execute(state)

        mock_urlopen.assert_not_called()
        assert len(decode_list(result["url_check_results"])) == 1
        check = decode_list(result["url_check_results"])[0]
        assert check["live"] is False
        assert check["ssrf_blocked"] is True

    def test_ssrf_blocked_on_dns_failure(self):
        """DNS resolution failure defaults to blocking the URL for safety."""
        node = UrlCheckNode(config={"url_liveness_check": True})
        state = _make_state(answer_markdown="Visit http://nonexistent-domain.xyz/docs")

        with patch("src.nodes.url_check.socket.gethostbyname", side_effect=Exception("DNS Error")):
            with patch("src.nodes.url_check.urllib.request.urlopen") as mock_urlopen:
                result = node.execute(state)

        mock_urlopen.assert_not_called()
        assert len(decode_list(result["url_check_results"])) == 1
        check = decode_list(result["url_check_results"])[0]
        assert check["live"] is False
        assert check["ssrf_blocked"] is True
