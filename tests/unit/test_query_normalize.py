# tests/unit/test_query_normalize.py -- CMN-C1-053
# Unit tests for QueryNormalizeNode (Issue #12)
from __future__ import annotations

import json
import tempfile


from src.nodes.query_normalize import (
    QueryNormalizeNode,
    _detect_language,
    _extract_api_entities,
    _apply_synonym_mapping,
)
from src.schemas.state import initial_state
from src.utils.serde import decode_list


# ── Language detection ────────────────────────────────────────────────────────


def test_detect_language_japanese_hiragana() -> None:
    assert _detect_language("認証のエンドポイントを教えてください") == "ja"


def test_detect_language_japanese_katakana() -> None:
    assert _detect_language("レート制限はどうなっていますか？") == "ja"


def test_detect_language_english() -> None:
    assert _detect_language("How do I use the /v1/orders endpoint?") == "en"


def test_detect_language_mixed_favours_ja() -> None:
    # Mixed query with CJK characters -> ja
    assert _detect_language("/orders エンドポイントの使い方") == "ja"


def test_detect_language_empty() -> None:
    assert _detect_language("") == "en"


# ── Entity extraction ──────────────────────────────────────────────────────────


def test_extract_endpoint_path() -> None:
    entities = _extract_api_entities("How do I call /v1/orders?")
    terms = [e["term"] for e in entities]
    assert "/v1/orders" in terms


def test_extract_http_method() -> None:
    entities = _extract_api_entities("Use POST to create a resource")
    terms = [e["term"] for e in entities]
    assert "POST" in terms


def test_extract_multiple_entities() -> None:
    entities = _extract_api_entities("GET /v1/items returns a list")
    terms = [e["term"] for e in entities]
    assert "GET" in terms
    assert "/v1/items" in terms


def test_no_entities_in_plain_query() -> None:
    entities = _extract_api_entities("what is authentication?")
    # No endpoint paths or HTTP methods in this query
    assert all(e["term"] not in ("GET", "POST", "PUT", "DELETE") for e in entities)


def test_entity_en_term_equals_term() -> None:
    # API entities are not translated
    entities = _extract_api_entities("GET /v1/orders")
    for e in entities:
        assert e["term"] == e["en_term"]


# ── Synonym mapping ────────────────────────────────────────────────────────────


def test_synonym_mapping_basic() -> None:
    vocab = {"認証": "authentication", "取得": "get"}
    result = _apply_synonym_mapping("認証のトークンを取得する方法", vocab)
    assert "authentication" in result
    assert "get" in result
    assert "認証" not in result


def test_synonym_mapping_longest_match_wins() -> None:
    # "レート制限" should map to "rate limit", not split into "レート" + "制限"
    vocab = {"レート制限": "rate limit", "制限": "limit"}
    result = _apply_synonym_mapping("レート制限を確認したい", vocab)
    assert "rate limit" in result
    # "制限" alone should not also appear as "limit" since the longer match consumed it
    # (longest-match means "レート制限" -> "rate limit", no residual "制限")
    assert "limit" not in result.replace("rate limit", "")


def test_synonym_mapping_empty_vocab_passthrough() -> None:
    text = "認証エラーを確認"
    assert _apply_synonym_mapping(text, {}) == text


def test_synonym_mapping_no_match_passthrough() -> None:
    vocab = {"認証": "authentication"}
    text = "エラーコードを確認"
    assert _apply_synonym_mapping(text, vocab) == text


# ── QueryNormalizeNode.execute() ──────────────────────────────────────────────


def _node_with_vocab(vocab: dict[str, str]) -> QueryNormalizeNode:
    """Create a node backed by a temp vocab file."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(vocab, tmp)
    tmp.flush()
    tmp.close()
    node = QueryNormalizeNode(vocab_path=tmp.name)
    return node


def test_execute_english_query_normalised() -> None:
    node = QueryNormalizeNode(vocab_path="./config/bilingual_vocab.json")
    state = initial_state("How do I use  the /v1/orders  endpoint?")
    state["normalized_query"] = state["raw_query"]
    result = node.execute(state)
    assert result["detected_language"] == "en"
    # Double spaces collapsed
    assert "  " not in result["normalized_query"]
    # Lowercased
    assert result["normalized_query"] == result["normalized_query"].lower()


def test_execute_japanese_query_mapped() -> None:
    vocab = {"認証": "authentication", "エラー": "error"}
    node = _node_with_vocab(vocab)
    state = initial_state("認証エラーの原因は？")
    state["normalized_query"] = state["raw_query"]
    result = node.execute(state)
    assert result["detected_language"] == "ja"
    assert "authentication" in result["normalized_query"]
    assert "error" in result["normalized_query"]


def test_execute_sets_query_entities() -> None:
    node = QueryNormalizeNode(vocab_path="./config/bilingual_vocab.json")
    state = initial_state("GET /v1/orders returns a list")
    state["normalized_query"] = state["raw_query"]
    result = node.execute(state)
    terms = [e["term"] for e in decode_list(result["query_entities"])]
    assert "GET" in terms
    assert "/v1/orders" in terms


def test_execute_missing_vocab_file_graceful() -> None:
    node = QueryNormalizeNode(vocab_path="./config/nonexistent_vocab.json")
    state = initial_state("認証エラーの対処法")
    state["normalized_query"] = state["raw_query"]
    # Should not raise -- graceful degradation on missing vocab
    result = node.execute(state)
    assert result["detected_language"] == "ja"
    assert isinstance(result["normalized_query"], str)


def test_execute_falls_back_to_raw_query_if_normalized_empty() -> None:
    node = QueryNormalizeNode(vocab_path="./config/bilingual_vocab.json")
    state = initial_state("GET /v1/items")
    # normalized_query not set -> falls back to raw_query
    result = node.execute(state)
    assert "/v1/items" in result["normalized_query"]


def test_execute_returns_primitive_dict() -> None:
    node = QueryNormalizeNode(vocab_path="./config/bilingual_vocab.json")
    state = initial_state("What is the rate limit?")
    state["normalized_query"] = state["raw_query"]
    result = node.execute(state)
    assert isinstance(result, dict)
    for v in result.values():
        assert isinstance(v, (str, int, bool, list, dict, type(None)))


def test_vocab_extensible_without_code_change() -> None:
    """New synonyms in vocab file are picked up without touching node code."""
    vocab = {"新しい用語": "new term", "テスト": "test"}
    node = _node_with_vocab(vocab)
    state = initial_state("新しい用語のテスト方法")
    state["normalized_query"] = state["raw_query"]
    result = node.execute(state)
    assert "new term" in result["normalized_query"]
    assert "test" in result["normalized_query"]
