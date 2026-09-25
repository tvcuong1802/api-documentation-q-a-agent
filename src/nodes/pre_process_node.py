"""
src/nodes/pre_process_node.py — CMN-C1-053

PreProcessNode (slot: pre_process). New-gen FunctionNode that folds the two
input-stage passes into a single slot (no GraphNode wrapper, Cat-1 pattern):
  1. InputValidateNode  — S-2 input sanitization (token budget, NFKC, PII scrub,
                          injection detection); may set answer_status="blocked" / raise
  2. QueryNormalizeNode — bilingual normalization + entity extraction

Sub-nodes are DI'd and called via execute() (another template fold pattern). Config is
constructor-injected (the backbone __call__ does not thread config to execute()).
Early-exit: if InputValidateNode sets answer_status="blocked", normalization is skipped
(the backbone routes onward; the post_process S-3 gate still runs).
"""

from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel

from src.nodes.input_validate import InputValidateNode
from src.nodes.query_normalize import QueryNormalizeNode
from src.utils.audit import emit_trace_event


class PreProcessNode(FunctionNode):
    """Pre-process slot — input validation (S-2) then query normalization."""

    # S-1 trust gate (ADR-006): enforced by BaseNode.__call__() before execute().
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        node_config: dict[str, Any] | None = None,
        input_validate_node: InputValidateNode | None = None,
        query_normalize_node: QueryNormalizeNode | None = None,
    ) -> None:
        super().__init__()
        cfg = node_config or {}
        self._input_validate = input_validate_node or InputValidateNode(config=cfg)
        self._query_normalize = query_normalize_node or QueryNormalizeNode(
            vocab_path=cfg.get("vocab_path", "config/bilingual_vocab.json")
        )

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        # 1. Input validation (S-2). Raises SecurityViolationError on hard violation;
        #    may set answer_status="blocked".
        work = {**state, **self._input_validate.execute(state)}
        # Early-exit: a blocked request skips normalization (S-3 still runs in post_process).
        if work.get("answer_status") == "blocked":
            emit_trace_event("pre_process_blocked", {"answer_status": "blocked"}, state)
            # `validation_error` travels with the block. Returning the band alone dropped
            # the sentence one layer above the node that composed it -- the intake wrote
            # it, this early-exit returned a dict without it, and the envelope that reads
            # it to decide the INPUT was the problem saw nothing. The reader then got
            # "agent failed" over a sentence that existed. A value that crosses layers is
            # only carried where every layer carries it.
            blocked: dict[str, Any] = {"answer_status": "blocked"}
            sentence = work.get("validation_error")
            if isinstance(sentence, str) and sentence.strip():
                blocked["validation_error"] = sentence
            return blocked
        # 2. Query normalization (bilingual + entity extraction).
        work = {**work, **self._query_normalize.execute(work)}
        emit_trace_event("pre_process_complete", {"answer_status": work.get("answer_status", "ok")}, state)
        # Partial-update dict only (wheel contract — node_history is accumulate-reducer).
        return {k: work[k] for k in ("answer_status", "normalized_query", "s1_blocked", "s2_violation") if k in work}
