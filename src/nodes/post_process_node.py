"""
src/nodes/post_process_node.py — CMN-C1-053

PostProcessNode (slot: post_process). New-gen FunctionNode. NON-SUPPRESSIBLE.
Always executes (incl. blocked / error / degraded paths). Folds the terminal
output passes:
  1. ResponseValidateNode — S-3 output gate (grounding / credential-leak / safety);
                            always runs, even on blocked requests
  2. UrlCheckNode         — optional URL-liveness pass (no-op when disabled)

Sub-nodes are DI'd and called via execute() (another template fold pattern). This node is
the sole graph sink — no path bypasses the S-3 gate. Config is constructor-injected.
"""

from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel

from src.nodes.response_validate import ResponseValidateNode
from src.nodes.url_check import UrlCheckNode
from src.utils.audit import emit_trace_event


class PostProcessNode(FunctionNode):
    """Post-process slot — non-suppressible S-3 response gate then URL-liveness check."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        node_config: dict[str, Any] | None = None,
        response_validate_node: ResponseValidateNode | None = None,
        url_check_node: UrlCheckNode | None = None,
    ) -> None:
        super().__init__()
        cfg = node_config or {}
        self._response_validate = response_validate_node or ResponseValidateNode(config=cfg)
        self._url_check = url_check_node or UrlCheckNode(config=cfg)

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        # S-3 gate runs unconditionally (even on blocked/error paths).
        merged = {**state, **self._response_validate.execute(state)}
        # URL liveness — best-effort, opt-in; no-op when disabled.
        merged = {**merged, **self._url_check.execute(merged)}
        emit_trace_event("post_process_complete", {"answer_status": merged.get("answer_status", "ok")}, state)
        # Partial-update dict; surface `result` (JSON-safe primitives) as invoke() output.
        return {
            "answer_markdown": merged.get("answer_markdown", ""),
            "answer_status": merged.get("answer_status", "ok"),
            "result": merged.get("answer_markdown", ""),
        }
