"""
src/graph/graph.py — CMN-C1-053

New-gen graph composition for the API Documentation Q&A Agent. Graph(AgentBaseGraph)
fills the three mandatory slots in register_nodes() (no GraphNode wrapper — Cat-1
pattern, design §0):
  pre_process  → PreProcessNode   (input validation S-2 + query normalization)
  main         → MainNode         (dense/sparse/hybrid retrieval + context + generate)
  post_process → PostProcessNode  (S-3 response gate + URL-liveness; sole sink)
The framework owns compile()/invoke()/routing; this class adds no custom execute/invoke.
App config (config/config.yaml) is loaded once and injected into the slot orchestrators'
constructors (the backbone __call__ does not thread config to execute()).
Production retrieval backends (encoder/qdrant/bm25) and the LLM are injected here.
"""

from __future__ import annotations

import re

from pathlib import Path
from typing import Any

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus

from src.nodes.pre_process_node import PreProcessNode
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode
from src.schemas.state import CmnApiDocQaState

_CONFIG_PATH = Path("config/config.yaml")


def _load_app_config() -> dict[str, Any]:
    """Load config.yaml (security/retrieval/generation). Best-effort; defaults on absence."""
    cfg: dict[str, Any] = {}
    try:
        import yaml

        if _CONFIG_PATH.exists():
            cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — missing/unreadable config → node defaults apply
        pass
    return cfg


class Graph(AgentBaseGraph):
    """Level-1 Cat-1 agent — CMN-C1-053 (API Documentation Q&A)."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._app_config = _load_app_config()
        self._encoder = (config or {}).get("encoder")
        self._qdrant = (config or {}).get("qdrant")
        self._bm25_index = (config or {}).get("bm25_index")
        self._bm25_docs = (config or {}).get("bm25_docs")
        self._llm = (config or {}).get("llm")
        super().__init__(config)

    @property
    def name(self) -> str:
        return "cmn_c1_053"

    @property
    def state_schema(self) -> type:
        return CmnApiDocQaState

    def register_nodes(self) -> None:
        super().register_nodes()  # injects initialize / finalize (real SDK)
        self._nodes = getattr(self, "_nodes", None) or {}
        node_config = self._app_config
        self._nodes["pre_process"] = PreProcessNode(node_config=node_config)
        self._nodes["main"] = MainNode(
            node_config=node_config,
            encoder=self._encoder,
            qdrant=self._qdrant,
            bm25_index=self._bm25_index,
            bm25_docs=self._bm25_docs,
            llm=self._llm,
        )
        self._nodes["post_process"] = PostProcessNode(node_config=node_config)

    def get_output(self, state: Any) -> dict[str, Any]:
        """The framework envelope, with a liability trailer on the payload.

        The envelope shape is kept: the Marketplace runner and the Stage-5 evidence
        script both read it as a dict (`status`, `output`). Only the payload changes.
        """
        envelope = _with_disclaimer(dict(super().get_output(state)), state)
        # This agent records its outcome in its OWN key, `answer_status` ("ok" / "degraded"
        # / "blocked"), and never set the framework's `status` -- so the envelope left
        # InitializeNode's `pending` in place. `normalize_terminal_output()` raises on any
        # status but SUCCESS, so every single answer reached the reader as "agent failed",
        # including the 391-character one measured here. `answer_status` is the agent's
        # judgement and stays the record; `status` is the transport, and it has to say
        # "there is something to read" whenever there is.
        # The condition is "is there something to read", not "did the pipeline reach a
        # stage that sets `answer_status`". An empty message never reaches that stage, so
        # the earlier version left `pending` on a path whose body is a good bilingual
        # "tell me what to look at" -- written, formatted, and then raised on.
        # `answer_status` stays the agent's own record; it is not the transport.
        if str(envelope.get("output") or "").strip() and not (
            state.get("error_log") if isinstance(state, dict) else None
        ):
            envelope["status"] = AgentStatus.SUCCESS.value
        return envelope


def _decided_language(state: Any) -> str | None:
    """The language this reader wants, or None when the message has none to read.

    `answer_language` is the decision the intake call makes from the message as
    received. Falling back to the script of the message covers the paths where S-2
    rejected the input in the gate, so `execute()` never ran: the agent still knows WHO
    it is talking to even when it will not act on WHAT they said.

    Returning None means bilingual, and that is now reserved for its one honest case --
    punctuation, digits, an empty line. Passing None unconditionally, as this file used
    to, gave every reader whose request failed an English block with a Japanese
    translation stapled underneath, which reads as an agent that never worked out which
    of the two it was for.
    """
    if not isinstance(state, dict):
        return None
    code = str(state.get("answer_language") or "").strip().lower()[:2]
    if code in ("en", "ja"):
        return code
    for key in ("user_input", "raw_query", "validated_input", "normalized_query"):
        value = state.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        japanese = sum(1 for ch in value if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff")
        # Latin WORDS, not characters: a Japanese message quoting an API key or a long id
        # used to read as English -- one 26-character token outweighed fourteen kana.
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z']*", value) if 2 <= len(w) <= 20]
        latin = sum(len(w) for w in words)
        if japanese and japanese * 2 >= latin:
            return "ja"
        return "en" if latin else None
    return None


REFUSED_NOTICE = "This request was refused.\nこのリクエストは拒否されました。"


def _refused_before_answering(state: Any) -> bool:
    """True when the S-1 trust gate refused the caller, so there is no answer to dress.

    Every other failure gets a trailer and a sentence a reader can act on. This one gets
    the refusal and nothing else -- no scope line, no guidance. A caller who is not
    permitted to invoke the agent must not be told what it is for, and `output: None` is
    equally wrong: the platform renders it as a blank screen under "agent failed", so the
    reader is told nothing, not even that they were refused. Two repo tests pulled in
    opposite directions on this and both were right about half of it.

    Read from `error_log`, which BaseNode.__call__ writes -- framework behaviour, the
    same in every repo, rather than a per-agent convention that would drift.
    """
    if not isinstance(state, dict):
        return False
    return any("S-1 trust gate denied" in str(e) for e in (state.get("error_log") or ()))


def _with_disclaimer(envelope: dict[str, Any], state: Any) -> dict[str, Any]:
    """Delegate to the shared envelope: one door for the whole fleet.

    This used to be a private copy. Keeping it private meant every change to the reply
    contract cost a refactor here instead of a file copy, and it silently missed the
    improvements the shared one gained -- the language decision, the single-closing-
    sentence rule, the blob check.

    Only the scope wording stays local, because only a reader of THIS agent can write it.
    """
    from src.services.agent_scope import SCOPE_EN, SCOPE_JA
    from src.services.output_envelope import with_disclaimer

    return with_disclaimer(envelope, state, scope_en=SCOPE_EN, scope_ja=SCOPE_JA)
