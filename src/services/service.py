"""
src/services/service.py — CMN-C1-053

Service-layer placeholder for scaffold-structure compliance. The domain logic for
this template lives in the node classes (src/nodes/), and the production retrieval
backends (encoder / Qdrant client / BM25 index) and LLM client are dependency-injected
into the graph at construction time (see src/graph/graph.py). No agenticstar imports.
"""

from __future__ import annotations

from typing import Any


def build_retrieval_backends(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a holder for production retrieval backends.

    Placeholder: in production this wires the embedding encoder, the Qdrant client,
    and the BM25 index from `config`; in CI/local these stay None and the nodes fall
    back to empty retrieval (degraded path). Kept as a thin seam so backend wiring
    has a single home without coupling the graph to a specific vendor SDK.
    """
    cfg = config or {}
    return {
        "encoder": cfg.get("encoder"),
        "qdrant": cfg.get("qdrant"),
        "bm25_index": cfg.get("bm25_index"),
        "bm25_docs": cfg.get("bm25_docs"),
    }
