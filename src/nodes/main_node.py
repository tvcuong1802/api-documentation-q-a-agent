"""
src/nodes/main_node.py — CMN-C1-053

MainNode (slot: main). New-gen FunctionNode that folds the retrieval + generation
passes into a single slot (no GraphNode wrapper, Cat-1 pattern):
  1. KBRetrieveDenseNode   — dense vector retrieval
  2. KBRetrieveSparseNode  — sparse / BM25 retrieval
  3. KBRetrieveHybridNode  — fuse + rerank dense+sparse
  4. ContextAssembleNode   — dedup, token-cap, content-staleness flagging
  5. ResponseGenerateNode  — grounded Markdown answer + citations

Sub-nodes are DI'd and called via execute() (another template fold pattern). Config is
constructor-injected. Retrieval backends (encoder/qdrant/bm25) and the LLM are
injected via the sub-node constructors in production; default None → empty
retrieval / deterministic stub in CI. The slot short-circuits when the request was
blocked upstream or an error is already set (S-3 still runs in post_process).
"""

from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel

from src.nodes.kb_retrieve_dense import KBRetrieveDenseNode
from src.nodes.kb_retrieve_sparse import KBRetrieveSparseNode
from src.nodes.kb_retrieve_hybrid import KBRetrieveHybridNode
from src.nodes.context_assemble import ContextAssembleNode
from src.nodes.response_generate import ResponseGenerateNode
from src.utils.audit import emit_trace_event


class MainNode(FunctionNode):
    """Main slot — retrieve (dense/sparse/hybrid) → assemble context → generate answer."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        node_config: dict[str, Any] | None = None,
        encoder: Any = None,
        qdrant: Any = None,
        bm25_index: Any = None,
        bm25_docs: Any = None,
        llm: Any = None,
        kb_dense_node: KBRetrieveDenseNode | None = None,
        kb_sparse_node: KBRetrieveSparseNode | None = None,
        kb_hybrid_node: KBRetrieveHybridNode | None = None,
        context_assemble_node: ContextAssembleNode | None = None,
        response_generate_node: ResponseGenerateNode | None = None,
    ) -> None:
        super().__init__()
        cfg = node_config or {}
        self._kb_dense = kb_dense_node or KBRetrieveDenseNode(encoder=encoder, qdrant=qdrant)
        self._kb_sparse = kb_sparse_node or KBRetrieveSparseNode(bm25_index=bm25_index, bm25_docs=bm25_docs)
        self._kb_hybrid = kb_hybrid_node or KBRetrieveHybridNode()
        self._context_assemble = context_assemble_node or ContextAssembleNode()
        self._response_generate = response_generate_node or ResponseGenerateNode(llm=llm, config=cfg)

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        from framework.schemas.agent_status import AgentStatus

        # Short-circuit: blocked input or an upstream error → no retrieval/generation.
        if state.get("answer_status") == "blocked" or state.get("error"):
            emit_trace_event("main_short_circuit", {"answer_status": state.get("answer_status", "ok")}, state)
            # Blocked is still a completed advisory run (fail-closed), not a control
            # error — route to post_process so the S-3 output gate runs. Only a real
            # error field routes to finalize.
            update: dict[str, Any] = {}
            if not state.get("error"):
                update["status"] = AgentStatus.SUCCESS.value
            return update
        work = dict(state)
        work = {**work, **self._kb_dense.execute(work)}
        work = {**work, **self._kb_sparse.execute(work)}
        work = {**work, **self._kb_hybrid.execute(work)}
        work = {**work, **self._context_assemble.execute(work)}
        work = {**work, **self._response_generate.execute(work)}
        emit_trace_event("main_complete", {"answer_status": work.get("answer_status", "ok")}, state)
        # Partial-update dict; set control status=SUCCESS so route() reaches
        # post_process (any non-SUCCESS control status → finalize). The domain
        # outcome lives in answer_status, never in the framework control status.
        out = {
            k: work[k]
            for k in (
                "answer_markdown",
                "answer_status",
                "citations",
                "normalized_query",
                "retrieved_chunks",
                "assembled_context",
                "confidence",
            )
            if k in work
        }
        if work.get("error"):
            out["error"] = work["error"]
        else:
            out["status"] = AgentStatus.SUCCESS.value
        return out
