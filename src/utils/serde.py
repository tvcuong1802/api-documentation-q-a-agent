"""
src/utils/serde.py — CMN-C1-053

JSON (de)serialization helpers for State fields that hold structured data.

State-safety: the retrieval list fields (`query_entities`, `dense_chunks`,
`sparse_chunks`, `reranked_chunks`) are persisted to the LangGraph checkpoint as
JSON strings (not `list[dict[str, Any]]`) so that no `Any`-typed,
non-msgpack-serializable value can corrupt the checkpoint. Nodes encode on write
and decode on read via these helpers.
"""

from __future__ import annotations

import json
from typing import Any


def encode_list(value: list[dict[str, Any]] | None) -> str:
    """Serialize a list of dicts to a JSON string. None -> '[]'."""
    return json.dumps(value or [], ensure_ascii=False)


def decode_list(value: Any) -> list[dict[str, Any]]:
    """Decode a JSON string back to a list of dicts. Tolerates None / already-list.

    Returns [] for empty/None/invalid input so consuming nodes can always iterate.
    """
    if value is None or value == "":
        return []
    if isinstance(value, list):
        # Defensive: a caller may still pass a live list (in-process orchestration).
        return value
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return []
    return decoded if isinstance(decoded, list) else []


def encode_obj(value: dict[str, Any] | None) -> str:
    """Serialize a dict (object) to a JSON string. None -> '{}'."""
    return json.dumps(value or {}, ensure_ascii=False)


def decode_obj(value: Any) -> dict[str, Any]:
    """Decode a JSON string back to a dict. Tolerates None / '' / already-dict.

    Returns {} for empty/None/invalid input so consuming nodes can always access keys.
    """
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        # Defensive: a caller may still pass a live dict (in-process orchestration).
        return value
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}
