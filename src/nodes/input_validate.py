# src/nodes/input_validate.py -- CMN-C1-053
# InputValidateNode: S-1 trust gate, token budget, S-2 PII scrub, injection block.
# Issues: #3 (S-1 + token budget), #13 (PII scrub), #29 (injection sanitization)
#
# Method contract: execute(self, state, config) -- per an internal implementation note 2026-05-22.
# Never use _invoke_impl on node classes.
from __future__ import annotations

import re
import unicodedata
from typing import Any, ClassVar

from framework.errors import SecurityViolationError
from framework.nodes.function_node import FunctionNode
from src.utils.audit import emit_trace_event
from framework.schemas.invocation_context import TrustLevel


# ── Module-level compiled regexes (compiled once at import) ──────────────────

# PII patterns (S-2 scrubbing)
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_RE_PHONE_JP = re.compile(r"(?:(?:\+81|0)[-\s]?)(?:\d{1,4}[-\s]?\d{2,4}[-\s]?\d{3,4})")
_RE_PHONE_INTL = re.compile(r"\+\d{1,3}[-\s.]\(?\d{1,4}\)?[-\s.]\d{1,4}[-\s.]\d{1,9}")
# Mock credential / API key fragments appearing in queries
_RE_BEARER = re.compile(r"[Bb]earer\s+[A-Za-z0-9._\-]{10,}")
_RE_API_KEY_PARAM = re.compile(r"api[_\-]?key\s*[=:]\s*[A-Za-z0-9._\-]{8,}", re.IGNORECASE)
_RE_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{15,}")

# Injection attack patterns (S-2 / issue #29)
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # Prompt injection -- system prompt override attempts
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|above)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:an?\s+)?(?:different|new|other|evil)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:a\s+)?(?:an?\s+)?(?:DAN|jailbreak|unrestricted)", re.IGNORECASE),
    re.compile(r"(?:system|assistant|user)\s*:\s*(?:you\s+are|ignore)", re.IGNORECASE),
    re.compile(r"<\s*(?:system|instruction|prompt)\s*>", re.IGNORECASE),
    # Bash / shell injection sequences
    re.compile(r"(?:;|&&|\|\||`|\$\()\s*(?:rm|curl|wget|bash|sh|python|nc|cat\s+/etc)"),
    re.compile(r"\$\{IFS\}"),
    re.compile(r"(?:\.\.[/\\]){2,}"),  # path traversal
    # Adversarial role / privilege escalation markers
    re.compile(r"\[INST\]|\[SYS\]|###\s*System", re.IGNORECASE),
    re.compile(r"jailbreak|dan\s+mode|developer\s+mode\s+enabled", re.IGNORECASE),
]

_REDACTED = "[REDACTED]"

#: What this agent needs before it can say anything. It answers questions FROM the API
#: documentation, so with no question there is nothing to look up.
#:
#: This path used to raise SecurityViolationError. An empty message is not an attack, it
#: is an incomplete request -- and the raise cost the reader the answer twice over: the
#: framework wrote a traceback into `error_log`, which is what `get_output`'s flip tests
#: for, so the bilingual sentence the envelope had already composed was delivered under
#: `status: error`, and `normalize_terminal_output()` raises on that. Measured 2026-09-17:
#: the reader saw "agent failed" and never the sentence.
#:
#: Bilingual unconditionally: an empty message carries no script to read a language from,
#: so nobody knows who is on the other side, and a notice the reader cannot read is not a
#: notice.
_NO_QUESTION_EN = (
    "No question was received, so nothing was looked up — this is not a finding that the "
    "documentation has no answer. Send the question in a sentence or two: which API, "
    "endpoint or behaviour you need. The answer will be assembled from the API "
    "documentation that matches it."
)
_NO_QUESTION_JA = (
    "ご質問が届いていないため、検索は実施していません（該当する記載がないという結果では"
    "ありません）。どの API・エンドポイント・挙動についてお知りになりたいかを一〜二文で"
    "ご記載ください。一致する API ドキュメントから回答を構成します。"
)
NO_QUESTION_SENTENCE = f"{_NO_QUESTION_EN}\n\n{_NO_QUESTION_JA}"


def _normalize_fullwidth(text: str) -> str:
    """Convert full-width ASCII alphanumerics/punctuation to half-width via NFKC."""
    return unicodedata.normalize("NFKC", text)


def _scrub_pii(text: str) -> str:
    """Replace PII and credential patterns with [REDACTED]. Returns scrubbed string."""
    text = _RE_EMAIL.sub(_REDACTED, text)
    text = _RE_PHONE_JP.sub(_REDACTED, text)
    text = _RE_PHONE_INTL.sub(_REDACTED, text)
    text = _RE_BEARER.sub(_REDACTED, text)
    text = _RE_API_KEY_PARAM.sub(_REDACTED, text)
    text = _RE_SK_KEY.sub(_REDACTED, text)
    return text


def _detect_injection(text: str) -> bool:
    """Return True if any injection pattern matches."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


class InputValidateNode(FunctionNode):
    """
    First node in the CMN-C1-053 pipeline.

    Enforces (S-2 input sanitization):
      - Token budget (max_query_length from config)
      - S-2 PII scrubbing (email, phone, API key/Bearer patterns)
      - Injection detection (prompt override, bash, path traversal, jailbreak)

    Returns answer_status="blocked" for an empty message -- an incomplete request, not
    an attack -- together with `validation_error`, the sentence the reader acts on.
    Raises SecurityViolationError only for a real S-2 violation (over budget,
    injection). On success, writes sanitized text to state["normalized_query"].

    S-1 trust enforcement is owned by the backbone `BaseNode.__call__` on the
    pre_process slot orchestrator (ADR-006) — not re-checked inline here (new-gen
    `execute()` does not receive `config`/`invocation_context`).
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._max_query_length: int = (config or {}).get("security", {}).get("max_query_length", 2000)

    def execute(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute S-2 input validation pipeline:
          1. Token budget check
          2. Full-width NFKC normalisation
          3. S-2 PII scrubbing
          4. Injection detection
        Returns partial state dict; caller merges into full state.
        (S-1 trust is enforced upstream by the backbone on the slot orchestrator.)
        """
        # S-1 (trust gate) is enforced by the backbone on the slot orchestrator;
        # this node owns S-2 (input sanitization) only.
        # ── Token budget ──────────────────────────────────────────────────────────
        raw_query = state.get("user_input") or state.get("raw_query", "")
        if not isinstance(raw_query, str) or not raw_query.strip():
            # RETURNED, not raised. Two separate reasons, and the second is why the
            # mutation that used to sit here was dead code:
            #  * an empty message is an incomplete request, not a security violation.
            #    `s2_violation` is the S-2 record and this path does not belong in it.
            #  * `state[...] = ...` immediately before a `raise` does not survive.
            #    Measured 2026-09-17: at the envelope, `answer_status` and
            #    `s2_violation` both read None on this path -- the framework catches the
            #    exception and the node's partial update is never merged. Only
            #    `error_log`, which the framework itself writes, came through.
            #
            # `answer_status="blocked"` is what MainNode already short-circuits on, so
            # no retrieval and no generation run on an empty query.
            updates = {
                "answer_status": "blocked",
                "s1_blocked": False,
                "s2_violation": False,
                "validation_error": NO_QUESTION_SENTENCE,
            }
            emit_trace_event(
                "input_validated",
                {"answer_status": "blocked", "reason": "no question in the message"},
                state,
            )
            return updates
        if len(raw_query) > self._max_query_length:
            state["answer_status"] = "blocked"
            state["s2_violation"] = True
            emit_trace_event(
                "input_validated",
                {
                    "answer_status": "blocked",
                    "reason": f"Query length {len(raw_query)} exceeds max_query_length {self._max_query_length}",
                    "s1_blocked": False,
                    "s2_violation": True,
                },
                state,
            )
            raise SecurityViolationError(
                f"Query length {len(raw_query)} exceeds max_query_length {self._max_query_length}"
            )

        # ── Full-width normalisation ──────────────────────────────────────────────
        # Normalise before PII/injection scan so that full-width evasion attempts
        # (e.g. ｉｇｎｏｒｅ　ｐｒｅｖｉｏｕｓ) are caught by the regex patterns.
        normalized = _normalize_fullwidth(raw_query).strip()

        # ── S-2: PII scrubbing ────────────────────────────────────────────────────
        # Scrub from the normalised copy; immediately discard the unmasked
        # intermediate so it does not linger in local scope beyond this point.
        scrubbed = _scrub_pii(normalized)
        del normalized  # discard unmasked intermediate per security requirement (#13)

        # ── Injection detection ───────────────────────────────────────────────────
        if _detect_injection(scrubbed):
            state["answer_status"] = "blocked"
            state["s2_violation"] = True
            emit_trace_event(
                "input_validated",
                {
                    "answer_status": "blocked",
                    "reason": "Injection pattern detected in query",
                    "s1_blocked": False,
                    "s2_violation": True,
                },
                state,
            )
            raise SecurityViolationError("Injection pattern detected in query")

        # ── Success: write sanitized query to state ───────────────────────────────
        # Only the scrubbed string reaches downstream nodes.
        # raw_query is preserved in state for audit trace only.
        updates = {
            "normalized_query": scrubbed,
            "answer_status": "ok",
            "s1_blocked": False,
            "s2_violation": False,
        }
        emit_trace_event("input_validated", updates, state)
        return updates
