# src/nodes/query_normalize.py -- CMN-C1-053
# QueryNormalizeNode: bilingual language detection, entity extraction, synonym mapping.
# Issue: #12
#
# Method contract: execute(self, state, config) -- per an internal implementation note 2026-05-22.
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel
from src.utils.audit import emit_trace_event
from src.utils.serde import encode_list


# ── CJK Unicode block range for language detection ────────────────────────────
# Covers CJK Unified Ideographs, Hiragana, Katakana, and common CJK extensions.
_CJK_RANGES: list[tuple[int, int]] = [
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs (main block)
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0xFF65, 0xFF9F),  # Halfwidth Katakana
]

# ── API term patterns to protect from normalisation ───────────────────────────
# These are extracted before synonym mapping and restored afterwards.
_RE_ENDPOINT_PATH = re.compile(r"/[a-zA-Z0-9_\-/.{}]+")  # /v1/orders, /api/{id}
_RE_HTTP_METHOD = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b")
_RE_PARAM_NAME = re.compile(r"\b([a-z][a-zA-Z0-9_]{1,30})\b")  # camelCase or snake_case params


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _detect_language(text: str) -> str:
    """Return 'ja' if CJK characters are present, 'en' otherwise."""
    cjk_count = sum(1 for ch in text if _is_cjk_char(ch))
    return "ja" if cjk_count > 0 else "en"


def _extract_api_entities(text: str) -> list[dict[str, Any]]:
    """
    Extract protected API entities from the query text.
    Returns list of {term, en_term} dicts; en_term == term (no translation for API terms).
    """
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in _RE_ENDPOINT_PATH.finditer(text):
        t = m.group()
        if t not in seen:
            entities.append({"term": t, "en_term": t})
            seen.add(t)
    for m in _RE_HTTP_METHOD.finditer(text):
        t = m.group()
        if t not in seen:
            entities.append({"term": t, "en_term": t})
            seen.add(t)
    return entities


def _load_vocab(vocab_path: str) -> dict[str, str]:
    """
    Load bilingual vocabulary from JSON file.
    Completely driven by config -- new synonyms require only a JSON edit, no code change.
    Returns empty dict on missing/malformed file (graceful degradation).
    """
    path = Path(vocab_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Strip the _comment key if present; return only str->str mappings
        return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, str)}
    except (json.JSONDecodeError, OSError):
        return {}


def _apply_synonym_mapping(text: str, vocab: dict[str, str]) -> str:
    """
    Replace Japanese terms in text with English equivalents from vocab.
    Longer terms are matched first (greedy longest-match) to avoid partial replacements.
    """
    if not vocab:
        return text
    # Sort by length descending so longer terms take priority
    for ja_term in sorted(vocab, key=len, reverse=True):
        if ja_term in text:
            text = text.replace(ja_term, vocab[ja_term])
    return text


class QueryNormalizeNode(FunctionNode):
    """
    Second node in the CMN-C1-053 pipeline.

    Performs:
      1. Language detection (CJK heuristic -> 'ja' or 'en')
      2. API entity extraction (endpoint paths, HTTP methods) -- protected from mapping
      3. Japanese -> English synonym mapping via config/bilingual_vocab.json
      4. Casing / whitespace normalisation

    The vocabulary dictionary is fully config-driven: add entries to
    config/bilingual_vocab.json to extend coverage with zero code changes.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, vocab_path: str = "./config/bilingual_vocab.json") -> None:
        super().__init__()
        self._vocab_path = vocab_path
        # Vocab loaded lazily on first execute (allows config to be set after init)
        self._vocab: dict[str, str] | None = None

    def _get_vocab(self) -> dict[str, str]:
        if self._vocab is None:
            self._vocab = _load_vocab(self._vocab_path)
        return self._vocab

    def execute(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalise query for downstream retrieval.
        Expects state["normalized_query"] already set by InputValidateNode.
        Falls back to state["raw_query"] if normalized_query is empty.
        """
        query = state.get("normalized_query") or state.get("raw_query", "")

        # ── Language detection ────────────────────────────────────────────────────
        detected_language = _detect_language(query)

        # ── Entity extraction (API terms protected from synonym mapping) ──────────
        entities = _extract_api_entities(query)

        # ── Synonym mapping (Japanese only; English queries pass through) ─────────
        if detected_language == "ja":
            vocab = self._get_vocab()
            query = _apply_synonym_mapping(query, vocab)

        # ── Casing / whitespace normalisation ─────────────────────────────────────
        # Lowercase (after synonym mapping so vocab keys match original casing)
        query = query.strip().lower()
        # Collapse multiple spaces
        query = re.sub(r"\s{2,}", " ", query)

        updates = {
            "normalized_query": query,
            "detected_language": detected_language,
            "query_entities": encode_list(entities),
        }
        emit_trace_event(
            "query_normalized", {"detected_language": detected_language, "entity_count": len(entities)}, state
        )
        return updates
