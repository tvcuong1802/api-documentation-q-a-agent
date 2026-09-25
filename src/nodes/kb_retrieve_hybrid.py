# src/nodes/kb_retrieve_hybrid.py -- CMN-C1-053
# KBRetrieveHybridNode -- merge dense + sparse results with hybrid scoring
# Issue #19
from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel
from src.utils.audit import emit_trace_event
from src.utils.serde import decode_list, encode_list
from src.services.app_config import app_config


class KBRetrieveHybridNode(FunctionNode):
    """Merge dense_chunks + sparse_chunks using weighted hybrid scoring.

    Formula (defaults from config/agent.yaml):
        hybrid_score = 0.6 * dense_score + 0.4 * sparse_score

    Deduplication key: (source_path, chunk_text) — same chunk appearing in
    both lists is merged; missing score treated as 0.0.

    Outputs:
        reranked_chunks: list[dict] sorted descending by hybrid_score,
                         filtered at min_score threshold
        status: "degraded"  when no chunks survive the threshold
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._config = dict(config or {})
        pass

    # ------------------------------------------------------------------
    # FunctionNode contract
    # ------------------------------------------------------------------

    def execute(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = self._config or app_config()

        # Short-circuit on upstream error / blocked status
        if state.get("answer_status") in ("blocked", "error"):
            return {}

        retrieval_cfg: dict[str, Any] = cfg.get("retrieval", {})
        dense_weight: float = float(retrieval_cfg.get("hybrid_dense_weight", 0.6))
        sparse_weight: float = float(retrieval_cfg.get("hybrid_sparse_weight", 0.4))
        min_score: float = float(retrieval_cfg.get("min_score", 0.65))

        dense_chunks: list[dict[str, Any]] = decode_list(state.get("dense_chunks"))
        sparse_chunks: list[dict[str, Any]] = decode_list(state.get("sparse_chunks"))

        # Build merged index keyed by (source_path, chunk_text)
        # Value: {"dense_score": float, "sparse_score": float, **chunk_fields}
        merged: dict[tuple[str, str], dict[str, Any]] = {}

        for chunk in dense_chunks:
            key = (str(chunk.get("source_path", "")), str(chunk.get("chunk_text", "")))
            merged[key] = {
                "chunk_text": chunk.get("chunk_text", ""),
                "source_path": chunk.get("source_path", ""),
                "endpoint_ref": chunk.get("endpoint_ref", ""),
                "doc_type": chunk.get("doc_type", "openapi_spec"),
                "last_verified_at": chunk.get("last_verified_at", ""),
                "dense_score": float(chunk.get("score", 0.0)),
                "sparse_score": 0.0,
            }

        for chunk in sparse_chunks:
            key = (str(chunk.get("source_path", "")), str(chunk.get("chunk_text", "")))
            if key in merged:
                merged[key]["sparse_score"] = float(chunk.get("score", 0.0))
            else:
                merged[key] = {
                    "chunk_text": chunk.get("chunk_text", ""),
                    "source_path": chunk.get("source_path", ""),
                    "endpoint_ref": chunk.get("endpoint_ref", ""),
                    "doc_type": chunk.get("doc_type", "openapi_spec"),
                    "last_verified_at": chunk.get("last_verified_at", ""),
                    "dense_score": 0.0,
                    "sparse_score": float(chunk.get("score", 0.0)),
                }

        # A hybrid weighting assumes BOTH signals exist. When the dense retriever is not
        # available -- no Qdrant provisioned, which is the case wherever this template
        # runs on its shipped corpus alone -- every result is multiplied by the sparse
        # weight and then measured against a threshold set for the sum of two. Measured
        # 2026-09-03: a correct match scored 0.4 x 0.33 = 0.13 against min_score 0.65, so
        # retrieval found the right endpoint and the merge discarded it, and the agent
        # answered "no relevant documentation found" to every question.
        #
        # With no dense side, sparse carries the whole weight. That is not a loosened
        # threshold: it is the same threshold applied to the only signal there is.
        if not dense_chunks and sparse_chunks:
            dense_weight, sparse_weight = 0.0, 1.0

        # Compute hybrid scores and filter
        reranked: list[dict[str, Any]] = []
        for entry in merged.values():
            hybrid_score = dense_weight * entry["dense_score"] + sparse_weight * entry["sparse_score"]
            if hybrid_score < min_score:
                continue
            reranked.append(
                {
                    "chunk_text": entry["chunk_text"],
                    "score": hybrid_score,
                    "source_path": entry["source_path"],
                    "endpoint_ref": entry["endpoint_ref"],
                    "doc_type": entry["doc_type"],
                    "last_verified_at": entry.get("last_verified_at", ""),
                }
            )

        # Sort descending by hybrid_score
        reranked.sort(key=lambda c: c["score"], reverse=True)

        result: dict[str, Any] = {"reranked_chunks": encode_list(reranked)}

        # Degraded status when no chunks survive
        if not reranked:
            result["answer_status"] = "degraded"

        emit_trace_event(
            "kb_retrieved_hybrid",
            {"merged_count": len(reranked), "status": result.get("answer_status", "success")},
            state,
        )
        return result
