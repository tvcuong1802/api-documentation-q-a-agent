# src/nodes/response_generate.py -- CMN-C1-053
# ResponseGenerateNode: LLM answer generation with CoT, citation, and structured output parsing.
# Issues: #6 (CoT + citation), #20 (Pydantic parse + fallback)
#
# Method contract: execute(self, state, config) -- per an internal implementation note 2026-05-22.
# Never use _invoke_impl on node classes.
# NEVER import or mention agenticstar anywhere.
from __future__ import annotations

import json
import logging
import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel
from src.utils.audit import emit_trace_event
from src.utils.serde import decode_list
from src.services.app_config import app_config

# Fixed bilingual text, not a translation chosen from the detected language. A query that
# retrieves nothing is also the case where language detection is least reliable -- and on
# the rejected-input path there may be no readable query at all. Guessing wrong hands the
# reader a refusal they cannot read, which is the same as no refusal.
_NO_DOCS_ANSWER = (
    "No relevant documentation found for your query.\nご質問に一致するドキュメントは見つかりませんでした。"
)


logger = logging.getLogger(__name__)

# ── Pydantic availability check (guarded import) ────────────────────────────
# pydantic is NOT in pyproject.toml production deps — guard for CI compatibility.
# Used only internally to parse LLM output; never written to state.
try:
    from pydantic import BaseModel as _PydanticBaseModel, ValidationError as _ValidationError

    class ApiDocAnswer(_PydanticBaseModel):
        """Intermediate parsing model — NEVER written to State."""

        thought: str
        answer: str
        citations: list[str]
        confidence: float

    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    _PydanticBaseModel = None  # type: ignore
    _ValidationError = None  # type: ignore

    class ApiDocAnswer:  # type: ignore[no-redef]
        """Fallback dataclass when pydantic is not available (CI stub)."""

        def __init__(self, thought: str, answer: str, citations: list[str], confidence: float) -> None:
            self.thought = thought
            self.answer = answer
            self.citations = citations
            self.confidence = confidence


# ── LLM client (constructor-injected) ───────────────────────────────────────
# New-gen contract: the production LLM client is dependency-injected via the
# node constructor (`llm=`). No framework.llm import (non-public / ci_stub-only).
# When no client is injected (CI / local), a deterministic stub is used.


# ── Constants ────────────────────────────────────────────────────────────────

_CANNED_NO_CONTEXT = json.dumps(
    {
        "thought": "No relevant documentation chunks were retrieved. Cannot answer.",
        "answer": _NO_DOCS_ANSWER,
        "citations": [],
        "confidence": 0.0,
    }
)

_DEFAULT_SYSTEM_PROMPT = (
    "You are an API documentation assistant. "
    "Answer developer questions about API endpoints, parameters, authentication, and usage. "
    "Support both Japanese and English queries. "
    "Always cite using [endpoint: /path METHOD] format. "
    "Always cite endpoints using the OpenAPI path template form (e.g., /v1/orders/{orderId} GET) — never substitute real values into path parameters. "
    "Never fabricate endpoint paths not present in the provided context."
)

# Regex to extract <thought>...</thought> from raw LLM output (fallback path)
_RE_THOUGHT = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)

# Content-staleness banner (bilingual). Prepended to the answer when
# ContextAssembleNode flags retrieved chunks older than the refresh interval.
# This is the content-freshness signal — distinct from url_check liveness (#1362).
_STALENESS_BANNER = (
    "> ⚠️ **Freshness notice / 鮮度に関する注意:** some cited documentation was last "
    "verified beyond the refresh interval and may be out of date — please confirm "
    "against the official source. 一部の参照ドキュメントは最終確認から更新間隔を超えて"
    "おり、内容が古い可能性があります。公式ドキュメントでご確認ください。"
)


# ── Stub LLM for CI when neither framework.llm nor openai/litellm is available ──


class _StubLLM:
    """The reply when no model is available. It says so, and asserts nothing else.

    It used to answer "stub answer" with confidence 0.5 -- a sentence shaped like an
    answer, carrying no signal that no model had run. On the Marketplace that was EVERY
    request, because the client came from the constructor and the runner passes none. A
    reader has no way to tell that apart from a real answer, which is the expensive
    failure: not a crash, a confident wrong answer.

    Bilingual because this branch is also where the language decision is least reliable,
    and confidence 0.0 so the validator downstream cannot promote it.
    """

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return json.dumps(
            {
                "thought": "No written answer could be produced, so the retrieved documentation was not read.",
                "answer": (
                    "No relevant documentation found for your query: this answer could not be "
                    "generated right now, so nothing was read from the documentation.\n"
                    "ご質問に一致するドキュメントは見つかりませんでした。現在この回答を生成できず、"
                    "ドキュメントは読まれていません。"
                ),
                "citations": [],
                "confidence": 0.0,
            }
        )


def _build_llm_client(injected_llm: Any | None, state: dict[str, Any] | None = None) -> Any:
    """The injected client, else one built from this invocation's secrets, else the stub.

    The stub was the only fallback, and the Marketplace runner constructs the graph as
    `agent_cls()` with no arguments -- so on every real request the client was the stub
    and the agent answered "stub answer" while reporting success. The secrets exist only
    per invocation, which is why this takes `state` and is called from execute() rather
    than from __init__.
    """
    if injected_llm is not None:
        return injected_llm
    try:
        from src.services.llm_provider import build_llm_client

        built = build_llm_client(dict(state or {}))
    except Exception:  # noqa: BLE001 -- no client available is a documented degradation
        built = None
    return built if built is not None else _StubLLM()


# ── Context building helpers ─────────────────────────────────────────────────


def _group_chunks_by_doc_type(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group reranked_chunks by doc_type field. Unknown types go to 'other'."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        doc_type = chunk.get("doc_type", "other") or "other"
        groups.setdefault(doc_type, []).append(chunk)
    return groups


def _build_context_section(chunks: list[dict[str, Any]], doc_type: str) -> str:
    """Render a group of chunks into a labelled context block."""
    lines = [f"[{doc_type.upper()} CONTEXT]"]
    for i, chunk in enumerate(chunks, start=1):
        endpoint_ref = chunk.get("endpoint_ref", "")
        chunk_text = chunk.get("chunk_text", "")
        if endpoint_ref:
            lines.append(f"  [{i}] {endpoint_ref}: {chunk_text}")
        else:
            lines.append(f"  [{i}] {chunk_text}")
    return "\n".join(lines)


def _build_prompt(
    system_prompt: str,
    normalized_query: str,
    reranked_chunks: list[dict[str, Any]],
) -> str:
    """
    Build the full LLM prompt from system_prompt + grouped context + query.

    Instructs the LLM to:
      - reason inside <thought>...</thought> tags (CoT)
      - cite endpoints using [endpoint: /path METHOD] format
      - return structured JSON: {"thought":..., "answer":..., "citations":[...], "confidence":...}
    """
    grouped = _group_chunks_by_doc_type(reranked_chunks)
    context_blocks = []
    for doc_type in ("openapi_spec", "markdown_guide", "code_example", "other"):
        if doc_type in grouped:
            context_blocks.append(_build_context_section(grouped[doc_type], doc_type))
    # Any remaining unknown doc_types
    for doc_type, chunks in grouped.items():
        if doc_type not in ("openapi_spec", "markdown_guide", "code_example", "other"):
            context_blocks.append(_build_context_section(chunks, doc_type))

    context_str = "\n\n".join(context_blocks) if context_blocks else "(No context provided)"

    return (
        f"{system_prompt}\n\n"
        "INSTRUCTIONS:\n"
        "1. Reason step-by-step inside <thought>...</thought> tags before answering.\n"
        "2. Cite every endpoint you reference using the format: [endpoint: /path METHOD]\n"
        "3. Return your response as valid JSON only, with this schema:\n"
        '   {"thought": "<your reasoning>", "answer": "<markdown answer>", '
        '"citations": ["<citation1>", ...], "confidence": <0.0-1.0>}\n'
        "4. Never fabricate endpoint paths not present in the context below.\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"USER QUERY: {normalized_query}\n\n"
        "RESPONSE (JSON only):"
    )


# ── JSON parsing helpers ─────────────────────────────────────────────────────


def _extract_json_from_text(text: str) -> str:
    """
    Try to extract a JSON object from text that may contain leading/trailing content.
    Returns the first {...} block found, or the original text if no block found.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _parse_llm_output(raw: str) -> ApiDocAnswer:
    """
    Parse LLM output into ApiDocAnswer.
    1. Try direct JSON parse + Pydantic/dataclass validation.
    2. Try extracting JSON block from raw text.
    Raises json.JSONDecodeError or ValueError if both fail.
    """
    # Attempt 1: direct JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt 2: extract JSON block
        extracted = _extract_json_from_text(raw)
        data = json.loads(extracted)  # raises JSONDecodeError if still invalid

    # Build ApiDocAnswer (Pydantic if available, else plain class)
    if _PYDANTIC_AVAILABLE:
        return ApiDocAnswer(**data)  # raises pydantic.ValidationError on schema mismatch
    else:
        return ApiDocAnswer(
            thought=str(data.get("thought", "")),
            answer=str(data.get("answer", "")),
            citations=list(data.get("citations", [])),
            confidence=float(data.get("confidence", 0.0)),
        )


# ── Main node ────────────────────────────────────────────────────────────────


class ResponseGenerateNode(FunctionNode):
    """
    Seventh node in the CMN-C1-053 pipeline.

    Builds a CoT prompt, calls the LLM, parses the structured JSON response via
    ApiDocAnswer (Pydantic or plain dataclass), extracts primitives to state.

    Never writes the Pydantic/dataclass object to state — only primitives.
    Retries up to 2 times on transient errors (JSONDecodeError, LLM timeout).
    Degrades gracefully on final failure or empty context.
    """

    _MAX_RETRIES = 2

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        llm: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        # An explicit config wins outright; the file is read only when the caller
        # supplied none. `{}` means "no settings", not "load the defaults".
        cfg = dict(config) if config is not None else app_config()
        # Kept for tests and callers that inject one; the request-time client is
        # resolved in execute(), where an invocation context exists.
        self._injected_llm = llm
        self._llm = _build_llm_client(llm)
        self._system_prompt: str = cfg.get("system_prompt", _DEFAULT_SYSTEM_PROMPT).strip()
        self._include_code_examples: bool = cfg.get("include_code_examples", True)

    def execute(
        self,
        state: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        Execute LLM generation:
          1. Short-circuit if status is 'blocked' or 'error'
          2. Handle empty context (degraded — no hallucination)
          3. Build prompt from system_prompt + grouped context
          4. Call LLM with retry logic (up to 2 retries)
          5. Parse ApiDocAnswer — extract primitives to state
          6. Fallback to raw LLM text on final parse failure
        """
        # Resolved here, not in __init__: the runner builds the graph with no arguments
        # and the secrets exist only per invocation.
        client = _build_llm_client(self._injected_llm, state)
        # ── Short-circuit on blocked/error ─────────────────────────────────────
        current_status = state.get("answer_status", "ok")
        if current_status in ("blocked", "error"):
            return {}

        reranked_chunks: list[dict[str, Any]] = decode_list(state.get("reranked_chunks"))
        normalized_query: str = state.get("normalized_query", "") or state.get("raw_query", "")

        # ── Degraded path: empty context ────────────────────────────────────────
        if not reranked_chunks:
            return {
                "answer_markdown": _NO_DOCS_ANSWER,
                "cited_endpoints": [],
                "code_example": None,
                "answer_status": "degraded",
            }

        # ── Build prompt ────────────────────────────────────────────────────────
        prompt = _build_prompt(self._system_prompt, normalized_query, reranked_chunks)

        # ── LLM call with retry ─────────────────────────────────────────────────
        raw_output: str = ""

        for attempt in range(self._MAX_RETRIES + 1):
            try:
                raw_output = client.generate(prompt)
                break  # success — exit retry loop
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, exc)
                if attempt == self._MAX_RETRIES:
                    # All retries exhausted — degrade with empty answer
                    logger.error("All LLM retries exhausted: %s", exc)
                    # `logger` is stderr; the platform's audit trail is not. Measured
                    # 2026-09-15: with a dead key this degraded correctly and the event
                    # tail was IDENTICAL to a healthy run, so an operator had no way to
                    # tell a degraded answer from a real one -- the whole reason the
                    # degrade is allowed to be silent to the reader is that it is NOT
                    # silent to the operator. The record carries trace_id/correlation_id
                    # from state, which is what ties this line to the request.
                    emit_trace_event(
                        "response_model_unavailable",
                        {"reason": type(exc).__name__, "attempts": attempt + 1},
                        state,
                    )
                    return {
                        "answer_markdown": "",
                        "cited_endpoints": [],
                        "code_example": None,
                        "answer_status": "degraded",
                        "error": str(exc),
                    }

        # ── Empty LLM response ──────────────────────────────────────────────────
        if not raw_output or not raw_output.strip():
            return {
                "answer_markdown": "",
                "cited_endpoints": [],
                "code_example": None,
                "answer_status": "degraded",
            }

        # ── Parse with retry ────────────────────────────────────────────────────
        parsed: ApiDocAnswer | None = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                parsed = _parse_llm_output(raw_output)
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Parse attempt %d failed: %s", attempt + 1, exc)
                if attempt < self._MAX_RETRIES:
                    # Retry the LLM call on parse failure (transient / malformed output)
                    try:
                        raw_output = client.generate(prompt)
                    except Exception as llm_exc:  # noqa: BLE001
                        logger.warning("LLM retry on parse failure %d failed: %s", attempt + 1, llm_exc)
                        # Continue to next attempt with existing raw_output
                else:
                    # Final failure — fallback to degraded raw text
                    logger.error("All parse retries exhausted: %s", exc)
                    return {
                        "answer_markdown": raw_output,
                        "cited_endpoints": [],
                        "code_example": None,
                        "answer_status": "degraded",
                    }

        if parsed is None:
            # Should not reach here, but guard for type safety
            return {
                "answer_markdown": raw_output,
                "cited_endpoints": [],
                "code_example": None,
                "answer_status": "degraded",
            }

        # ── Extract primitives from ApiDocAnswer — NEVER write Pydantic obj to state ──
        answer_markdown: str = str(parsed.answer)
        # Surface the content-staleness notice (set by ContextAssembleNode) so the
        # reader is warned when a controlled-snapshot chunk exceeds the refresh interval.
        if state.get("content_staleness_detected") and answer_markdown.strip():
            answer_markdown = f"{_STALENESS_BANNER}\n\n{answer_markdown}"
        # Citations: ensure list[str] of primitives only
        cited_endpoints: list[str] = [str(c) for c in (parsed.citations or [])]
        # Code example: extract from answer if include_code_examples enabled
        code_example: str | None = _extract_code_example(answer_markdown) if self._include_code_examples else None

        updates = {
            "answer_markdown": answer_markdown,
            "cited_endpoints": cited_endpoints,
            "code_example": code_example,
            "answer_status": "ok",
        }
        emit_trace_event(
            "response_generated",
            {
                "citation_count": len(cited_endpoints),
                "confidence": float(getattr(parsed, "confidence", 0.0)),
                "status": "success",
            },
            state,
        )
        return updates


def _extract_code_example(answer_markdown: str) -> str | None:
    """Extract first fenced code block from answer_markdown, or None if absent."""
    match = re.search(r"```(?:\w+)?\n(.*?)```", answer_markdown, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
