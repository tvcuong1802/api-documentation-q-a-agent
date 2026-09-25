# src/state.py -- CMN-C1-053
# CmnApiDocQaState -- flat TypedDict state schema
# Rules (the framework contract para4):
#   - Flat TypedDict only -- no Pydantic, no dataclass
#   - All fields msgpack-serializable (str | int | bool | list | dict | None)
#   - No JWT, API keys, credentials, or InvocationContext stored here

from __future__ import annotations
import uuid
from framework.schemas.agent_state import AgentState


class CmnApiDocQaState(AgentState):
    # Removed 2026-09-04: duration_ms/invocation_id/raw_query were declared and written by nothing. `raw_query` was read as a fallback in four nodes, so that fallback was always empty -- it looked like a second source and was a dead branch.
    # Input
    #: What UrlCheckNode found, JSON-encoded (default "[]"). Two reasons for the string:
    #: LangGraph drops any key the schema does not name, silently -- the node ran,
    #: produced results, and nothing downstream could see them -- and a nested list[dict]
    #: is not msgpack-safe in the checkpoint, which the proof-of-boundary state test
    #: enforces. Encode on write / decode on read via src.utils.serde.
    url_check_results: str

    # Pre-process: InputValidateNode + QueryNormalizeNode
    normalized_query: str
    detected_language: str  # 'ja' | 'en' | 'unknown'
    # State-safety: structured list fields are JSON-serialized to str (flat-TypedDict
    # rule — no nested list[dict] persisted to the checkpoint). Encode on write /
    # decode on read via src.utils.serde. See docs/04_security_review.md.
    query_entities: str  # JSON str of [{term: str, en_term: str}, ...]

    # Retrieval: KBRetrieveDenseNode
    dense_chunks: str  # JSON str of [{chunk_text, score, source_path, endpoint_ref, doc_type}]

    # Retrieval: KBRetrieveSparseNode
    sparse_chunks: str  # JSON str; same schema as dense_chunks

    # Retrieval: KBRetrieveHybridNode -> ContextAssembleNode
    reranked_chunks: str  # JSON str; merged, deduplicated, token-capped
    retrieval_count: int

    # Generation: ResponseGenerateNode
    answer_markdown: str
    cited_endpoints: list[str]  # ['/path METHOD', ...]
    code_example: str | None

    # Content freshness: ContextAssembleNode (controlled-snapshot RAG, see #1362)
    # Distinct from url_check liveness (URL up) — this is content-staleness:
    # a retrieved chunk whose last_verified_at exceeds the refresh interval.
    content_staleness_detected: bool  # any retrieved chunk older than refresh interval
    stale_chunk_count: int

    # Security
    s1_blocked: bool
    s2_violation: bool
    s3_violation: bool
    credential_leak_detected: bool

    # Control
    answer_status: str  # 'ok' | 'blocked' | 'degraded' | 'error'
    error: str | None
    #: The SENDER'S sentence when their input is what stopped the run. `answer_status`
    #: and `error_log` beside it are for an operator -- a band and a traceback; this is
    #: the half a person can act on, and the shared envelope reads it to decide that the
    #: INPUT was the problem (`refused_the_message`) rather than the agent.
    #: DECLARED because LangGraph merges only the fields this schema names: undeclared,
    #: the intake writes it and it is dropped before the envelope looks, with no error
    #: anywhere, and the reader gets "agent failed" over a sentence that was composed
    #: for them. Measured 2026-09-17 on the empty-message path.
    validation_error: str


def initial_state(raw_query: str, invocation_id: str = "") -> CmnApiDocQaState:
    """Factory for CmnApiDocQaState. All tests MUST use this -- never raw dict."""
    return CmnApiDocQaState(
        raw_query=raw_query,
        invocation_id=invocation_id or str(uuid.uuid4()),
        normalized_query="",
        detected_language="unknown",
        query_entities="[]",
        dense_chunks="[]",
        sparse_chunks="[]",
        reranked_chunks="[]",
        retrieval_count=0,
        answer_markdown="",
        cited_endpoints=[],
        code_example=None,
        content_staleness_detected=False,
        stale_chunk_count=0,
        s1_blocked=False,
        s2_violation=False,
        s3_violation=False,
        credential_leak_detected=False,
        answer_status="ok",
        error=None,
        duration_ms=0,
    )
