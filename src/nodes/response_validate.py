# src/nodes/response_validate.py -- CMN-C1-053
# ResponseValidateNode: S-3 mandatory output gate — always executes.
# Issues: #7 (endpoint grounding), #14 (credential leak scan)
#
# Method contract: execute(self, state: dict[str, Any], config: dict | None) -> dict
# S-3 is NON-SUPPRESSIBLE: runs regardless of state["answer_status"].
# NEVER import or mention agenticstar anywhere.
from __future__ import annotations

import re
from typing import Any, ClassVar

from framework.errors import SecurityViolationError
from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel
from src.utils.audit import emit_trace_event
from src.utils.serde import decode_list


# ── Compiled credential patterns (module-level, compiled once) ──────────────
# Patterns from §3.8 of docs/02_design.md:
#   Bearer <token>      — e.g. "Bearer eyJ...<truncated>"
#   api_key=<value>     — e.g. "api_key=s3cr3t"
#   sk-<key>            — e.g. "sk-ABCDEF123..."  (OpenAI-style)
#   Authorization: <value> — e.g. "Authorization: Token abc123"
#   JWT (3-part base64.base64.base64)
_RE_BEARER = re.compile(
    r"Bearer\s+[A-Za-z0-9._\-]{10,}",
    re.IGNORECASE,
)
_RE_API_KEY = re.compile(
    r"api[_\-]?key\s*[=:]\s*[A-Za-z0-9._\-]{6,}",
    re.IGNORECASE,
)
_RE_SK_KEY = re.compile(
    r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{7,}",
)
_RE_AUTHORIZATION = re.compile(
    r"Authorization\s*:\s*\S+\s+[A-Za-z0-9._\-]{10,}",
    re.IGNORECASE,
)
# JWT: three base64url segments separated by dots (header.payload.sig). Anchored on eyJ.
_RE_JWT = re.compile(
    r"\beyJ[A-Za-z0-9_\-]{7,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b",
)

_CREDENTIAL_PATTERNS: list[re.Pattern[str]] = [
    _RE_BEARER,
    _RE_API_KEY,
    _RE_SK_KEY,
    _RE_AUTHORIZATION,
    _RE_JWT,
]


def _contains_credential(text: str) -> bool:
    """Return True if text matches any hardcoded-credential pattern."""
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            return True
    return False


_RE_ENDPOINT_EXTRACT = re.compile(
    r"^\s*(?:\[endpoint:\s*)?(/[^\s\]]+\s+(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS))(?:\])?\s*$",
    re.IGNORECASE,
)


def _normalize_citation(ep: str) -> str:
    """Extract bare /path METHOD from [endpoint: /path METHOD] or plain /path METHOD."""
    ep_clean = ep.strip().rstrip(".,;:")
    m = _RE_ENDPOINT_EXTRACT.match(ep_clean)
    return m.group(1).strip().lower() if m else ep_clean.lower()


def _check_endpoint_grounding(
    cited_endpoints: list[str],
    reranked_chunks: list[dict[str, Any]],
) -> bool:
    """
    Return True if all cited endpoints are grounded in at least one reranked chunk's endpoint_ref.

    We normalize both the citations and the chunk endpoint_refs before comparison.
    This avoids the negative-mention loophole by ignoring chunk_text.
    An empty cited_endpoints list is always grounded (vacuously true).
    """
    if not cited_endpoints:
        return True

    # Build grounding set from endpoint_ref ONLY (not chunk_text — avoids negative mention loophole)
    grounding_normalized = {
        chunk.get("endpoint_ref", "").strip().lower()
        for chunk in reranked_chunks
        if chunk.get("endpoint_ref", "").strip()
    }

    for endpoint in cited_endpoints:
        ep_normalized = _normalize_citation(endpoint)
        if not ep_normalized:
            continue
        if ep_normalized not in grounding_normalized:
            return False

    return True


class ResponseValidateNode(FunctionNode):
    """
    Eighth node in the CMN-C1-053 pipeline — S-3 mandatory output gate.

    NON-SUPPRESSIBLE: execute() ALWAYS runs regardless of state["answer_status"].
    Two checks (both mandatory per §3.8):

    1. Endpoint grounding (#7):
       Every entry in cited_endpoints must appear in at least one chunk's
       source_path, endpoint_ref, or chunk_text from reranked_chunks.
       Violation → s3_violation=True, raise SecurityViolationError.

    2. Credential leak scan (#14):
       Scan answer_markdown for hardcoded credentials (Bearer, api_key=,
       sk-, Authorization: header, JWT 3-part token).
       Violation → credential_leak_detected=True, s3_violation=True, raise SecurityViolationError.

    On clean pass: return {"s3_violation": False, "credential_leak_detected": False,
                           "status": <current_status>}
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._config = config or {}

    def execute(
        self,
        state: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        S-3 gate: always runs, even when answer_status='blocked' or 'degraded'.

        Order of checks:
        1. Credential scan (most critical — prevents secret exfiltration)
        2. Endpoint grounding (prevents hallucinated citations)

        Both checks run independently; credential scan is checked first.
        A violation in either raises SecurityViolationError immediately.
        """
        answer_markdown: str = state.get("answer_markdown", "") or ""
        cited_endpoints: list[str] = state.get("cited_endpoints", []) or []
        reranked_chunks: list[dict[str, Any]] = decode_list(state.get("reranked_chunks"))
        current_status: str = state.get("answer_status", "ok") or "ok"

        # ── Check 1: Credential leak scan (#14) ──────────────────────────────
        if _contains_credential(answer_markdown):
            state["credential_leak_detected"] = True
            state["s3_violation"] = True
            emit_trace_event(
                "response_validated",
                {
                    "s3_violation": True,
                    "credential_leak_detected": True,
                    "answer_status": "blocked",
                    "reason": "Credential leak detected in generated answer",
                },
                state,
            )
            raise SecurityViolationError("Credential leak detected in generated answer")

        # ── Check 2: Endpoint grounding (#7) ─────────────────────────────────
        if not _check_endpoint_grounding(cited_endpoints, reranked_chunks):
            state["s3_violation"] = True
            emit_trace_event(
                "response_validated",
                {
                    "s3_violation": True,
                    "credential_leak_detected": False,
                    "answer_status": "blocked",
                    "reason": "Hallucinated endpoint citation detected",
                },
                state,
            )
            raise SecurityViolationError("Hallucinated endpoint citation detected")

        # ── Clean pass ────────────────────────────────────────────────────────
        updates = {
            "s3_violation": False,
            "credential_leak_detected": False,
            "answer_status": current_status,
        }
        emit_trace_event(
            "response_validated",
            {"s3_violation": False, "credential_leak_detected": False, "answer_status": current_status},
            state,
        )
        return updates
