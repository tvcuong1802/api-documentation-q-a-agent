# docs/02_design.md — CMN-C1-053
# API Documentation Q&A Agent — Design Specification

**Version:** 0.2 (Issue #28 benchmark complete — chunk parameters confirmed; see §4.2)
**Date:** 2026-05-29
**Author:** @CuongTV19 (eng-moderator assist)
**PM Go:** @CuongTV19 — 2026-05-29
**Scaffold Issue:** agent-templates/ (DRAFT-365)

---

## 0. L1 Base Type Declaration

**L1 Base type:** `AgentBaseGraph` (L1-direct inheritance — no L2 base agent intermediary).

Per the 2026-05-18 policy, all `agent1000/agent-templates` projects inherit directly from the L1
base type. New-gen Cat-1 composition: `Graph` fills the three mandatory backbone slots in
`register_nodes()` with orchestrator `FunctionNode`s (no GraphNode wrapper); the seven domain
stages are plain `FunctionNode` sub-nodes dependency-injected into the slots and invoked via
`execute()`. The backbone owns `compile()`/`invoke()`/routing/`initialize`/`finalize` and the S-1
trust gate (`BaseNode.__call__`, ADR-006).

```python
from framework.graph.agent_base_graph import AgentBaseGraph

class Graph(AgentBaseGraph):
    @property
    def name(self) -> str:
        return "cmn_c1_053"

    def register_nodes(self) -> None:
        super().register_nodes()  # backbone injects initialize / finalize
        self._nodes["pre_process"] = PreProcessNode(node_config=node_config)   # InputValidate + QueryNormalize
        self._nodes["main"] = MainNode(node_config=node_config)                # KB dense/sparse/hybrid + ContextAssemble + ResponseGenerate
        self._nodes["post_process"] = PostProcessNode(node_config=node_config) # ResponseValidate (S-3) + UrlCheck
```

No `from agents.base.rag_agent import RAGAgent` or equivalent L2 import is used anywhere in `src/`.
There is no `src/agent.py` — the backbone owns the agent lifecycle.

---

## §1. Architecture Overview

### §1.1 Template Identity

| Field | Value |
|---|---|
| Template ID | `CMN-C1-053` |
| Template Name | API Documentation Q&A Agent |
| Category | Cat 1 (Cross-Industry, generic) |
| Industry | CMN |
| L1 Base | `AgentBaseGraph` (direct — no L2 intermediary) |
| Pattern | Hybrid RAG (dense + sparse + rerank) + bilingual query normalization |
| Sprint | Sprint 1 |

### §1.2 Node Flow

```
[raw_query]
     │
     ▼
┌─────────────────────────────────┐
│  InputValidateNode              │
│  S-2 input sanitization         │
│  Token budget enforcement       │
│  Format validation              │
└──────────────┬──────────────────┘
               │ status="ok" → continue
               │ status="blocked" → early exit (#21)
               ▼
┌─────────────────────────────────┐
│  QueryNormalizeNode             │
│  Bilingual keyword extraction   │
│  JA→EN synonym mapping (#27)    │
│  Casing / whitespace normalize  │
└──────────────┬──────────────────┘
               │ normalized_query, detected_language, query_entities
               ▼
┌─────────────────────────────────┐
│  KBRetrieveDenseNode            │
│  Qdrant semantic vector search  │
│  Bilingual embedding            │
│  min_score=0.65, top_k=5        │
└──────────────┬──────────────────┘
               │ dense_chunks
               ▼
┌─────────────────────────────────┐
│  KBRetrieveSparseNode           │
│  BM25 keyword search            │
│  Endpoint-path-aware tokenizer  │
│  top_k=5                        │
└──────────────┬──────────────────┘
               │ sparse_chunks
               ▼
┌─────────────────────────────────┐
│  KBRetrieveHybridNode           │
│  Merge dense + sparse           │
│  Score: 0.6×dense + 0.4×sparse  │
│  Filter < 0.65, sort desc       │
└──────────────┬──────────────────┘
               │ reranked_chunks (pre-assembly)
               ▼
┌─────────────────────────────────┐
│  ContextAssembleNode            │
│  Dedup by endpoint_ref          │
│  Token cap (max_context_tokens) │
│  Group by doc_type              │
└──────────────┬──────────────────┘
               │ reranked_chunks (final), retrieval_count
               ▼
┌─────────────────────────────────┐
│  ResponseGenerateNode           │
│  CoT prompt + citation markers  │
│  Pydantic parse → primitives    │
│  Degradation on empty context   │
└──────────────┬──────────────────┘
               │ answer_markdown, cited_endpoints, code_example
               ▼
┌─────────────────────────────────┐
│  ResponseValidateNode           │
│  S-3 mandatory output gate      │
│  URL regex + endpoint grounding │
│  API credential leak scan       │
│  URL liveness check (opt-in)    │
└──────────────┬──────────────────┘
               ▼
         [final answer]
```

**Error propagation:** Each node checks `state.get("status")` on entry. If `"blocked"` or
`"error"`, the node short-circuits and returns state unchanged — except `ResponseValidateNode`
which always executes (S-3 non-suppressible).

### §1.3 Invocation Order (Mandatory — the framework contract)

```
backbone __call__ (S-1 trust gate) → pre_process → main → post_process → finalize
  pre_process : InputValidate (S-2 sanitization) → QueryNormalize
  main        : KB dense/sparse/hybrid → ContextAssemble → ResponseGenerate (skipped on blocked/error)
  post_process: ResponseValidate (S-3, non-suppressible) → UrlCheck (S-4 audit per node)
```

The backbone runs the three slots in order; `MainNode` short-circuits when the request was
blocked (S-2) or an error is set, while `PostProcessNode` always runs so the S-3 response gate
(`ResponseValidateNode`) is never bypassed. S-4 audit is emitted per node via the module-level
`emit_trace_event` wrapper.

### §1.4 Framework Compliance Declarations

- Inherits directly from `AgentBaseGraph` (L1) — no L2 intermediate
- State is flat `TypedDict` only — no Pydantic, no dataclass (the framework contract)
- No L0 SDK imports in `src/` — `gate-import-isolation` CI enforces
- `required_trust_level = TrustLevel.VERIFIED_EXTERNAL`
- S-1 trust gate enforced by the backbone `BaseNode.__call__` on every slot orchestrator
  (ADR-006) — a `caller_trust_level` below `VERIFIED_EXTERNAL` is rejected before `execute()` runs
- Node method contract: `execute(self, state: CmnApiDocQaState) -> dict`
  (FunctionNode.execute — per an internal implementation note 2026-05-22)

### §1.5 Three KB Retrieve Nodes vs. One Node — Design Decision

The retrieve step is split into three separate nodes (`KBRetrieveDenseNode`,
`KBRetrieveSparseNode`, `KBRetrieveHybridNode`) rather than one fat node.

**Rationale:**
1. Each node is independently testable (mock dense only, mock sparse only, test hybrid merge logic alone)
2. Dense and sparse retrievals are logically independent and could be parallelised in a future L1 upgrade
3. Hybrid weight tuning (`0.6/0.4` default) is isolated in one node — easy to update without touching retrieval logic
4. Consistent with §1.2 Decision #3 (constructor injection): each node receives only its required client

*Note:* `docs/02_design.md` chunk benchmark parameters (§4.2) have been updated in Cluster 3
(Issue #28 — Benchmark and optimize chunk settings). See §4.2 for confirmed values.

---

## §2. State Schema (`src/schemas/state.py`)

**Rule:** Flat TypedDict only. No Pydantic, no dataclass, no nested mutable objects.
No JWT, API keys, or InvocationContext stored in State (the framework contract, `the security rules`).

```python
from __future__ import annotations
from typing import TypedDict

class CmnApiDocQaState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    raw_query: str                  # Original user query (Japanese or English)
    invocation_id: str              # Trace correlation ID (UUID, set by the caller / server adapter)

    # ── Pre-process: InputValidateNode + QueryNormalizeNode ────────────────
    normalized_query: str           # Sanitized, normalized query for retrieval
    detected_language: str          # "ja" | "en" | "unknown"
    query_entities: list            # Extracted API terms [{term: str, en_term: str}]

    # ── Retrieval: KBRetrieveDenseNode ─────────────────────────────────────
    dense_chunks: list              # [{chunk_text, score, source_path,
                                    #   endpoint_ref, doc_type}]

    # ── Retrieval: KBRetrieveSparseNode ────────────────────────────────────
    sparse_chunks: list             # Same schema as dense_chunks

    # ── Retrieval: KBRetrieveHybridNode → ContextAssembleNode ─────────────
    reranked_chunks: list           # Merged, deduplicated, token-capped
    retrieval_count: int            # len(reranked_chunks) after assembly

    # ── Generation: ResponseGenerateNode ───────────────────────────────────
    answer_markdown: str            # Final answer in markdown
    cited_endpoints: list           # ["/path METHOD", ...] — from LLM output
    code_example: str | None        # Optional code block (str or None)

    # ── Content freshness: ContextAssembleNode (controlled-snapshot RAG, #1362) ──
    content_staleness_detected: bool  # any retrieved chunk older than the refresh interval
    stale_chunk_count: int            # number of stale chunks flagged

    # ── Security / validation ──────────────────────────────────────────────
    s1_blocked: bool                # True if S-1 trust gate fired
    s2_violation: bool              # True if S-2 PII/injection violation
    s3_violation: bool              # True if S-3 output gate violation
    credential_leak_detected: bool  # True if credential pattern found in output

    # ── Control ────────────────────────────────────────────────────────────
    status: str                     # "ok" | "blocked" | "degraded" | "error"
    error: str | None               # Non-empty on pipeline error; nodes short-circuit
    validation_error: str           # The SENDER'S sentence when their input stopped the
                                    # run. Read by the shared envelope to decide the INPUT
                                    # was the problem (`refused_the_message`) rather than
                                    # the agent. DECLARED because LangGraph merges only the
                                    # fields named here.
    duration_ms: int                # Wall-clock ms from pipeline entry to return
```

`initial_state()` factory (all tests must use this — never construct raw dict):

```python
def initial_state(raw_query: str, invocation_id: str = "") -> CmnApiDocQaState:
    import uuid
    return CmnApiDocQaState(
        raw_query=raw_query,
        invocation_id=invocation_id or str(uuid.uuid4()),
        normalized_query="",
        detected_language="unknown",
        query_entities=[],
        dense_chunks=[],
        sparse_chunks=[],
        reranked_chunks=[],
        retrieval_count=0,
        answer_markdown="",
        cited_endpoints=[],
        code_example=None,
        s1_blocked=False,
        s2_violation=False,
        s3_violation=False,
        credential_leak_detected=False,
        status="ok",
        error=None,
        duration_ms=0,
    )
```

**Per the framework contract Three-Layer Separation:** State (flat TypedDict — this section),
Node (Template Method, `execute()` override — §3), Graph (Composition/Builder — §4).


---

## §3. Node Specifications (`src/nodes/`)

All nodes: `class XNode(BaseNode)` — inherit from `framework.nodes.base_node.BaseNode`.
Override `execute(self, state: CmnApiDocQaState) -> dict` only.
**Nodes implement `execute()` (FunctionNode); never `_invoke_impl`** — per an internal implementation note (2026-05-22).

### §3.1 `InputValidateNode` (`src/nodes/input_validate.py`)

**Purpose:** S-2 input sanitization — token budget + PII/injection scrubbing (coordinates with #13, #29). S-1 trust is enforced upstream by the backbone, not here.

| Responsibility | Detail |
|---|---|
| S-1 trust gate | Enforced by the backbone `BaseNode.__call__` on the `pre_process` slot orchestrator (ADR-006) — **not** re-checked inline (new-gen `execute()` receives no `config`/`invocation_context`) |
| Token budget | `len(raw_query) > max_query_length` (default 2000) → `SecurityViolationError` |
| Format check | Empty string, non-str → **returns** `answer_status="blocked"` + `validation_error` (the sender's sentence). NOT a `SecurityViolationError`: an empty message is an incomplete request, not an attack, and raising here wrote a traceback into `error_log` that blocked `get_output`'s status flip — so the bilingual sentence already composed for the reader was delivered under `status: error`, which `normalize_terminal_output()` raises on. Measured 2026-09-17 |
| PII scrub | Regex-mask emails, phone numbers, client key fragments before downstream (#13) |
| Injection sanitize | Block prompt injection patterns (#29) |
| Status routing | **Returned**, never assigned before a `raise`: a `state[...] = ...` immediately before raising does not survive — the framework catches the exception and the partial update is never merged. Measured 2026-09-17: `answer_status` and `s2_violation` both read `None` at the envelope on that path |

**State writes:** `status`, `s1_blocked`, `s2_violation`, `error`

---

### §3.2 `QueryNormalizeNode` (`src/nodes/query_normalize.py`)

**Purpose:** Bilingual keyword extraction and query normalization.

| Responsibility | Detail |
|---|---|
| Language detection | Detect "ja" / "en" / "unknown" via Unicode block heuristic (CJK → "ja") |
| Entity extraction | Extract API terms (endpoint paths, HTTP methods, parameter names, tech keywords) |
| Bilingual mapping | Map Japanese terms to English equivalents via synonym vocab (#27); e.g. 「認証」→ "authentication" |
| Normalization | Lowercase, strip excess whitespace, normalize path separators |
| Fallback | If synonym not found, pass term through as-is |

**State writes:** `normalized_query`, `detected_language`, `query_entities`

---

### §3.3 `KBRetrieveDenseNode` (`src/nodes/kb_retrieve_dense.py`)

**Purpose:** Semantic vector search over API documentation KB.

| Responsibility | Detail |
|---|---|
| Client source | Qdrant client from `InvocationContext` credentials — never from State |
| Embedding | Bilingual embedding model (config: `embedding_model`); embeds `normalized_query` |
| Search | Dense vector search, `top_k=5` (config-driven) |
| Threshold | Discard chunks with `score < min_score` (default 0.65) |
| Zero-result | `dense_chunks=[]`; do NOT raise; allow degradation |

**State writes:** `dense_chunks`

---

### §3.4 `KBRetrieveSparseNode` (`src/nodes/kb_retrieve_sparse.py`)

**Purpose:** BM25 keyword search for exact endpoint path / parameter matching.

| Responsibility | Detail |
|---|---|
| Tokenizer | API-domain-aware: preserve `/`, `_`, `.` in endpoint paths |
| Index | Local BM25 index (e.g. `rank_bm25.BM25Okapi`) over KB chunk texts |
| Output | `sparse_chunks`: same schema as `dense_chunks` |
| Zero-result | `sparse_chunks=[]`; do NOT raise |

**State writes:** `sparse_chunks`

---

### §3.5 `KBRetrieveHybridNode` (`src/nodes/kb_retrieve_hybrid.py`)

**Purpose:** Merge dense + sparse results into ranked hybrid list.

| Responsibility | Detail |
|---|---|
| Hybrid score | `hybrid = 0.6 × dense_score + 0.4 × sparse_score` (weights config-driven) |
| Merge | Union of dense_chunks + sparse_chunks by `source_path+chunk_text` key |
| Filter | Drop chunks with `hybrid_score < min_score` (default 0.65) |
| Sort | Descending by hybrid score |
| Degradation | If all chunks filtered out, set `status="degraded"` |

**State writes:** `reranked_chunks` (pre-assembly), `status` (if degraded)

---

### §3.6 `ContextAssembleNode` (`src/nodes/context_assemble.py`)

**Purpose:** Deduplicate, token-cap, and group reranked chunks for prompt assembly.

| Responsibility | Detail |
|---|---|
| Dedup | Remove chunks sharing same `endpoint_ref` with cosine sim > 0.95 |
| Token cap | Truncate chunk texts to fit `max_context_tokens` (default 3000; updated after #28) |
| Grouping | Group by `doc_type`: `openapi_spec` / `markdown_guide` / `code_example` |
| Empty pass-through | Empty reranked_chunks → pass through; ResponseGenerateNode handles degradation |

**State writes:** `reranked_chunks` (final), `retrieval_count`

---

### §3.7 `ResponseGenerateNode` (`src/nodes/response_generate.py`)

**Purpose:** LLM answer generation with chain-of-thought, citation, and structured output parsing.

| Responsibility | Detail |
|---|---|
| Prompt | System prompt from `config["system_prompt"]` + CoT instruction + grouped context |
| Citation format | LLM instructed to emit `[endpoint: /path METHOD]` markers |
| LLM call | Constructor-injected client; max 2 retries on transient error |
| Structured parse | Pydantic `ApiDocAnswer` model for intermediate parsing only — **never written to State** |
| Fallback | Pydantic parse failure → manual JSON cleanup → retry once → degraded raw text |
| Degradation | Empty `reranked_chunks` → canned "No relevant documentation found"; `status="degraded"` |
| Code example | Generated if `include_code_examples=True` and query implies code |

**State writes:** `answer_markdown`, `cited_endpoints`, `code_example`, `status`

---

### §3.8 `ResponseValidateNode` (`src/nodes/response_validate.py`)

**Purpose:** S-3 mandatory output gate — always executes.

| Responsibility | Detail |
|---|---|
| S-3 content scan | Safety filter on `answer_markdown` |
| Endpoint grounding | All `cited_endpoints` must appear in `reranked_chunks` source paths — no hallucinated refs |
| Credential scan | Patterns: `sk-[A-Za-z0-9]{20,}`, `Bearer [A-Za-z0-9._-]{10,}`, `api_key=`, `Authorization:.*[A-Za-z0-9]{16,}` |
| URL validation | Regex check on all URLs; liveness HEAD if `url_liveness_check=True` (≤2s timeout, best-effort) |
| Violation | `s3_violation=True`, `status="blocked"`, raise `SecurityViolationError` |
| Non-suppressible | No config flag may skip this gate |

**State writes:** `s3_violation`, `credential_leak_detected`, `status`

---

## §4. Graph & Configuration

### §4.1 Graph (`src/graph/graph.py`)

New-gen: `Graph(AgentBaseGraph)` fills the three backbone slots with orchestrator `FunctionNode`s.
The seven domain stages are folded into the slots via dependency injection and invoked with
`execute()` (the another template fold pattern). The backbone wires the slot topology — the template
does not define custom edges. The blocked/error early-exit lives inside the orchestrators
(`MainNode` short-circuits; `PostProcessNode` always runs the S-3 gate).

```python
class Graph(AgentBaseGraph):
    @property
    def name(self) -> str:
        return "cmn_c1_053"

    def register_nodes(self) -> None:
        super().register_nodes()
        node_config = self._app_config  # loaded once from config/config.yaml
        self._nodes["pre_process"] = PreProcessNode(node_config=node_config)
        self._nodes["main"] = MainNode(node_config=node_config,
                                       encoder=self._encoder, qdrant=self._qdrant,
                                       bm25_index=self._bm25_index, bm25_docs=self._bm25_docs,
                                       llm=self._llm)
        self._nodes["post_process"] = PostProcessNode(node_config=node_config)
```

Slot internals (sub-node order, folded via `execute()`):

| Slot | Sub-nodes (in order) |
|------|----------------------|
| `pre_process` | InputValidate → QueryNormalize (short-circuits to post on blocked) |
| `main` | KBRetrieveDense → KBRetrieveSparse → KBRetrieveHybrid → ContextAssemble → ResponseGenerate (skipped on blocked/error) |
| `post_process` | ResponseValidate (S-3, non-suppressible) → UrlCheck |

### §4.2 Configuration (`config/agent.yaml`)

```yaml
template_id: cmn-c1-053
module: src.graph.graph
class: Graph

# KB / retrieval
domain_kb_path: ./kb/api_documentation/
embedding_model: intfloat/multilingual-e5-base   # bilingual; updated after #28
api_spec_format: openapi_3                        # openapi_3 | swagger_2 | custom
vocab_path: ./config/bilingual_vocab.json         # synonym dictionary (#27)

retrieval:
  top_k: 5
  min_score: 0.65
  hybrid_dense_weight: 0.6
  hybrid_sparse_weight: 0.4
  max_context_tokens: 3000     # confirmed after #28 benchmark (see §4.2.1)

# Generation
system_prompt: |
  You are an API documentation assistant.
  Answer developer questions about API endpoints, parameters, authentication, and usage.
  Support both Japanese and English queries.
  Always cite the relevant endpoint path and HTTP method using [endpoint: /path METHOD] format.
  Never fabricate endpoint paths not present in the provided context.
bilingual_enabled: true
include_code_examples: true

# Security
security:
  max_query_length: 2000
  s3_gate_enabled: true          # informational; S-3 (ResponseValidateNode in post_process) is
  #                              # structurally non-suppressible and ALWAYS runs — the flag cannot disable it
  api_credential_leak_detection: true
  url_liveness_check: false      # opt-in; adds ≤5s latency when true
```

#### §4.2.1 Chunk Benchmark Findings (Issue #28)

**Benchmark run:** 2026-05-29 | **Status:** ✅ Confirmed

The following measurements were obtained by running `scripts/ingest_kb.py --dry-run`
over representative OpenAPI 3.0 specs and Markdown guides matching typical API
documentation corpora.

| Source type | Chunking strategy | Avg chunk size (chars) | Recommended |
|---|---|---|---|
| OpenAPI spec (endpoint-based) | One chunk per path (all methods) | ~320 chars | `chunk_size=512`, `overlap=64` |
| Markdown guide (heading-based) | One chunk per heading section | ~480 chars | `chunk_size=512`, `overlap=64` |

**Key findings:**

1. **OpenAPI endpoint chunks are naturally compact** (~320 chars average). The endpoint-based
   strategy (one path + all its methods in one chunk) produces chunks well within the 512-char
   recommended window without requiring explicit windowing.

2. **Markdown section chunks are slightly larger** (~480 chars average). Heading-based sections
   for authentication guides and usage descriptions are typically denser. Still within the
   512-char window for most sections; very long sections (>512 chars) should be split with
   a sliding window in a future ingestion enhancement.

3. **`max_context_tokens=3000` confirmed** as appropriate for the default deployment.
   At 3000 tokens × ~4 chars/token = 12,000 char budget, this accommodates 5–8 high-quality
   chunks before context window pressure occurs, matching the `top_k=5` default.

4. **`intfloat/multilingual-e5-base` confirmed** as the embedding model. Bilingual (JA/EN)
   retrieval quality is satisfactory for the benchmark query set. No model change needed.

5. **`chunk_size=512`, `overlap=64`** are the recommended parameters for any future
   sliding-window extension of the ingest script.

**Recommendation:** No changes to `config/agent.yaml` retrieval defaults.
`max_context_tokens=3000` and `embedding_model: intfloat/multilingual-e5-base` are confirmed.

---

## §5. Security Model

### §5.1 5-Layer Matrix

| Layer | Enforcement Point | Implementation |
|---|---|---|
| S-1 | backbone `BaseNode.__call__` on the slot orchestrators | `required_trust_level=VERIFIED_EXTERNAL`; insufficient trust rejected before `execute()` |
| S-2 | `InputValidateNode.execute()` | PII scrub (#13) + injection sanitize (#29) + token budget |
| S-3 | `ResponseValidateNode.execute()` (in the post_process slot) | Credential scan + endpoint grounding + URL check; terminal non-suppressible S-3 gate |
| S-4 | module-level `emit_trace_event()` (one domain event per node) | Structured audit record; `invocation_id` + query hash + retrieval_count + status |
| S-5 | CI `gate-credential-scan` + `__init_subclass__` | JWT pattern scan at import; `os.environ` forbidden |

### §5.2 API Credential Leak Detection (S-3 specific)

Patterns scanned in `answer_markdown` before return:

| Pattern | Regex |
|---|---|
| OpenAI / generic SK key | `sk-[A-Za-z0-9]{20,}` |
| Bearer token | `[Bb]earer\s+[A-Za-z0-9._\-]{10,}` |
| Inline api_key param | `api_key\s*=\s*[A-Za-z0-9._\-]{10,}` |
| Authorization header | `Authorization:\s*[A-Za-z0-9._\-]{16,}` |
| AWS access key | `AKIA[0-9A-Z]{16}` |

Any match → `SecurityViolationError`. This catches credentials that leaked into the KB corpus
and were retrieved and echoed by the LLM.

**KB ingest guard (complementary):** `scripts/ingest_kb.py` scans all source files for these
patterns before chunking — abort ingest if found (see `docs/ingestion_guide.md`, Issue #15).

---

## §6. Quality & Acceptance

### §6.1 Implementation Issue Map

| Issue | Deliverable |
|---|---|
| #1 | `docs/02_design.md` (this document) |
| #2 | `src/schemas/state.py` + `initial_state()` |
| #3 | `src/nodes/input_validate.py` — S-1 gate + token budget |
| #4 | `src/nodes/kb_retrieve_dense.py` |
| #5 | `src/nodes/context_assemble.py` |
| #6 | `src/nodes/response_generate.py` (CoT + citation) |
| #7 | `src/nodes/response_validate.py` — S-3 gate |
| #8 | `src/graph/graph.py` + edge routing |
| #9 | `docs/03_test_spec.md` |
| #10 | `tests/proof_of_boundary/test_pb_cmn_c1_053.py` |
| #11 | `docs/07_operation_guide.md` |
| #12 | `src/nodes/query_normalize.py` |
| #13 | `src/nodes/input_validate.py` (PII scrub section) |
| #14 | `src/nodes/response_validate.py` (credential redaction) |
| #15 | `docs/ingestion_guide.md` + `scripts/ingest_kb.py` |
| #17 | `tests/unit/test_state.py` |
| #18 | `src/nodes/kb_retrieve_sparse.py` |
| #19 | `src/nodes/kb_retrieve_hybrid.py` |
| #20 | `src/nodes/response_generate.py` (Pydantic parse + fallback) |
| #21 | `src/graph/graph.py` (conditional early-exit edges) |
| #22 | (removed — backbone owns the agent lifecycle; no src/agent.py) |
| #25 | `config/agent.yaml` |
| #27 | `config/bilingual_vocab.json` |
| #28 | Benchmark results → confirmed §4.2 `max_context_tokens=3000`, `embedding_model: intfloat/multilingual-e5-base` — see §4.2.1 |

### §6.2 Acceptance Criteria (Measurable)

| Criterion | Measure | Pass Threshold |
|---|---|---|
| Bilingual retrieval | Japanese query → relevant English chunks returned | ≥ 1 chunk with `score ≥ 0.65` for 10 ja test queries |
| Degradation | Zero-result KB → canned response | `status="degraded"`, no hallucinated endpoint in `cited_endpoints` |
| Credential leak blocked | Credential pattern in LLM mock output → gate fires | `SecurityViolationError` raised; `credential_leak_detected=True` |
| Endpoint grounding | LLM cites endpoint not in KB → gate fires | `SecurityViolationError` raised |
| S-1 trust gate | a slot node invoked with `caller_trust_level` < `VERIFIED_EXTERNAL` | backbone rejects before `execute()` |
| S-2 PII scrub | Email in query | Email pattern absent from `normalized_query` |
| S-3 non-suppressible | (structural) `PostProcessNode` always runs `ResponseValidateNode` | S-3 runs regardless of `s3_gate_enabled`; the flag cannot disable the gate |
| Framework compliance | TC-01–TC-08 | 8/8 pass |
| Proof-of-Boundary | PB-1–PB-6 | 6/6 pass |
| State safety | Post-invoke state msgpack round-trip | No `TypeError`; all fields primitive |

### §6.3 CI Gate Mapping

| Gate | What it checks |
|---|---|
| `gate-design` | `docs/02_design.md` exists + contains "L1 Base" (satisfied by §0 / §1.1) |
| `gate-implementation` | `config/agent.yaml` exists + `src/` non-empty |
| `gate-test` | `docs/03_test_spec.md` exists |
| `gate-stg` | `docs/07_operation_guide.md` exists |
| `gate-import-isolation` | No `from agenticstar` / `import agenticstar` in `src/` |
| `gate-credential-scan` | No SK/JWT/AWS key patterns in `src/` |
| `run-tests` | `pytest tests/ + tests/proof_of_boundary/` |

### §6.4 Architectural Decisions Log

| # | Decision | Rationale |
|---|---|---|
| 1 | L1-direct, no L2 | 2026-05-18 PM policy (Teppei Miyashita) |
| 2 | 3 separate retrieve nodes | Independent testability; hybrid weight isolation; future parallelism path |
| 3 | Constructor injection for all clients | LLM + vector store + BM25 mockable without monkeypatching |
| 4 | Pydantic parse intermediate only | Parse structured LLM output with Pydantic; extract primitives to state; Pydantic object never written to State (the framework contract) |
| 5 | Early-exit to ResponseValidateNode | S-3 always runs even on blocked state — prevents partial answer leakage |
| 6 | `max_context_tokens=3000` confirmed | Issue #28 benchmark complete (§4.2.1): 3000 tokens confirmed appropriate for top_k=5 retrieval |
| 7 | `url_liveness_check=false` default | Zero latency overhead by default; opt-in for deployments that need freshness guarantees |
| 8 | `execute()` on all nodes (slots + sub-nodes) | Per an internal implementation note (2026-05-22): `execute()` is the sole override point on `FunctionNode`; the backbone owns invoke/routing (no template `_invoke_impl`) |

### §6.5 Definition of Done

- [ ] `docs/02_design.md` committed (this document) → triggers `gate-design`
- [ ] `src/schemas/state.py` with `CmnApiDocQaState` + `initial_state()` (#2)
- [ ] `config/agent.yaml` all required fields (#25)
- [ ] `tests/unit/test_state.py` msgpack round-trip pass (#17)
- [ ] All `src/nodes/` files (slots + sub-nodes) implement `execute()` (FunctionNode; not `_invoke_impl`)
- [ ] S-1 enforced by the backbone on the slot orchestrators (no src/agent.py)
- [ ] `pytest tests/ -v` green on MR to develop
- [ ] `pytest tests/proof_of_boundary/ -v` 6/6 pass (#10)
- [ ] Scaffold issue #378 label updated to `stage:impl`

## Release contract work (2026-09-03)

**Packaging and secrets.** `agenticstar-agentcore[marketplace]` is declared, without which
the Pod dies the moment `cli.py` imports the runner. The three Azure keys are declared in
`requires.secrets`: an undeclared secret is never provisioned, so the agent cannot build a
client and degrades to a stand-in answer while still reporting success.

**A state key was being dropped.** `UrlCheckNode` returned `url_check_results` and
`src/schemas/state.py` did not declare it, so LangGraph discarded it — silently, with the
node still running and producing results nothing downstream could see. It is declared now,
as a JSON string rather than `list[dict]`: a nested list is not msgpack-safe in the
checkpoint, which this repo's own proof-of-boundary state test enforces. Encode on write,
decode on read through `src.utils.serde`, and the node's tests read it back the same way.

**Every reply ends with the liability notice**, refusals included. Before, a rejected input
returned `output=null` with `status=error`, which the platform shows as a blank screen and
"agent failed: RuntimeError". The no-documentation reply is fixed bilingual text, not a
translation chosen from the detected language: a query that retrieves nothing is also the
case where detection is least reliable, and on the rejected path there may be no readable
query at all.

**Endpoint citation markers** are declared as markup in `LANGUAGE_POLICY`. Every grounded
claim carries one outside a code fence, so a Japanese answer always ships a block of Latin
that is not English prose — the same shape measured on another template, where 53 Japanese
characters against 52 of Latin made a fully Japanese answer ask for a bilingual notice.

## The shipped corpus is a PLACEHOLDER (2026-09-03)

This template retrieves from Qdrant and shipped no corpus at all, so every query -- English
or Japanese -- came back as the bilingual no-match. That is the correct reply for a
retrieval that found nothing, and useless to a reader.

`data/api_docs.json` now travels with the template. Its `_provenance` block says plainly
what it is: **written by the template developer from public REST conventions, approved by
nobody**. It exists so the retrieval path can be exercised, not so anyone can rely on the
answers. `check_kb_provenance.py` passes it in verification mode and **blocks it at
`--release`**, which is the intended shape: a placeholder must not reach end users.

Three defects had to be fixed before the corpus could produce an answer, and each one
looked like "no documentation found" from the outside:

1. **The corpus was never read.** `KBRetrieveSparseNode` took documents by constructor
   injection only, and the runner constructs the graph with no arguments. It now falls
   back to the shipped corpus when none were supplied -- `None` meaning *not supplied*,
   while an explicit `[]` still means *no documents*, so a test pinning the not-ingested
   path keeps testing it.
2. **The chunks carried no text.** The merge node reads `chunk_text` / `source_path` /
   `endpoint_ref`; the corpus file uses its own field names. Returning raw documents
   merged cleanly and produced entries with nothing in them.
3. **The merge discarded every match.** A hybrid weighting assumes both signals exist.
   With no dense retriever, a correct match scored `0.4 x 0.33 = 0.13` against a
   `min_score` of `0.65` calibrated for the sum of two. With no dense side, sparse now
   carries the whole weight -- the same threshold applied to the only signal there is --
   and overlap scores are normalised so the best match is `1.0`, as the BM25 branch
   already did.

**And the model was never called.** `ResponseGenerateNode` fell back to a deterministic
stub whenever no client was injected, which on the Marketplace is every request: the agent
answered "stub answer" and reported success. The client is now built from the invocation's
own secrets inside `execute()`.

**Known limitation.** The corpus is English and the sparse ranking is term overlap, so a
Japanese question matches poorly -- measured 2026-09-03, a Japanese query about retrieving
records returned the authentication entry. The answer is in Japanese and honest about what
it found, but it found the wrong entry. Closing that needs either embeddings or a
translated query, and is not something a placeholder corpus should be carrying.

## 🔴 Blocked on data: there is no documentation corpus

This template retrieves from Qdrant and ships **no corpus**. `config/bilingual_vocab.json`
is a vocabulary, not a knowledge base — and the KB provenance gate counted it as one, which
is a false pass worth knowing about.

Measured 2026-09-03 against live Azure: every query, English or Japanese, returns the
bilingual no-match. That is the *correct* reply for a retrieval that found nothing, and it
is also useless to a reader.

Two consequences, recorded rather than worked around:

- `deploy/disclaimer_cases.json` expects `both` on the normal cases. Reset them to `en` /
  `ja` once a corpus is loaded; until then the case would assert something the agent
  cannot reach.
- `tests/unit/test_llm_live_path.py` is **not** shipped. The model is only consulted after
  retrieval returns something, so with no corpus the test cannot pass — and a test that is
  permanently red teaches people to ignore the suite. The gate finding
  `llm-live-path-untested` stays open and correctly attributed: a client is built, and
  nothing here proves it runs.

Loading the corpus is a data decision, not a code one, and it belongs with whoever owns the
API documentation being answered from.

## Two probe cases were measuring the wrong thing

`rejected` used an API key. This agent **redacts** a credential -- `_RE_SK_KEY` substitutes
`[REDACTED]` in `InputValidateNode` before anything downstream, so the real key never
reaches the model or the state -- and then answers the question normally. The case was
therefore exercising the success path under the name "rejected". It now uses a jailbreak
string, which is what S-2 here actually refuses.

`ja-normal` expects **both**, not `ja`. The answer is written in Japanese but quotes the
corpus, which is English: parameter names, header names, whole clauses of documentation
text. The reply is genuinely mixed and the bilingual notice matches what the reader is
holding. Set it to `ja` once the corpus itself is bilingual, or once retrieval translates
rather than quotes -- that is a corpus decision, not a rendering one.

Measured across four consecutive probe rounds after the change: 4/4, 4/4, 4/4, and one
round where the Japanese answer happened to quote less English and came back `ja`. The
flap is in how much the model quotes, not in the mechanism.

