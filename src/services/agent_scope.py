"""What THIS agent's output must not be used for, and how it reads a request.

Kept in its own module so the wording lives beside the agent it describes, while the
MECHANISM stays byte-identical fleet-wide in ``disclaimer.py`` and ``input_intake.py``.
A generic "AI-generated draft" is true of every template here and tells a reader nothing
they can act on; this names the decisions the output must not stand in for.
"""

from __future__ import annotations

SCOPE_EN = (
    "Reference only: an answer assembled from the API documentation that matched your "
    "question. It is not a specification, and it does not guarantee the endpoint behaves "
    "this way in your tenant or version. Confirm against the official documentation "
    "before you build on it."
)

SCOPE_JA = (
    "参考情報です。ご質問に一致した API ドキュメントから構成した回答であり、仕様書ではなく、"
    "お客様のテナントやバージョンで同じ挙動になることを保証するものではありません。実装前に"
    "公式ドキュメントでご確認ください。"
)


LANGUAGE_POLICY: dict[str, object] = {
    # Every grounded claim is stamped with the endpoint it came from, outside code
    # fences, so a Japanese answer always ships a block of Latin that is not English
    # prose. Same shape as another template, where 53 Japanese characters against 52 of Latin
    # made a fully Japanese answer ask for a bilingual notice.
    "ignore_patterns": (r"\[endpoint:[^\]]*\]", r"\[[A-Za-z0-9_.\-/]+\]"),
}


# What the intake call reads out of a free-form question. No `fields` are declared:
# retrieval runs on the normalized query and nothing here consumes a structured field.
# What the call adds is the language of the answer -- a question typed in romanised
# Japanese is entirely Latin, and reading the characters gets that reader wrong.
INTAKE_POLICY: dict[str, object] = {
    "languages": ("en", "ja"),
    "default_language": "en",
    "fields": {},
    "capabilities": (
        "Answer a question about a documented API from its own documentation",
        "Show a request example for an endpoint",
        "Point to the documentation behind a specific behaviour or field",
    ),
    "examples": (
        {
            "message": "kono API de fukusuu no rekoodo wo shutoku suru houhou wo oshiete kudasai",
            "expect": {"language": "ja", "fields": {}, "fits": "yes", "suggestion": None},
        },
        {
            "message": "この API で複数のレコードを取得するにはどうすればよいですか？",
            "expect": {"language": "ja", "fields": {}, "fits": "yes", "suggestion": None},
        },
        {
            # An English question naming a Japanese product: the product is not the language.
            "message": "How do I paginate results from the kintone records endpoint?",
            "expect": {"language": "en", "fields": {}, "fits": "yes", "suggestion": None},
        },
        {
            "message": "Call this endpoint for me and delete the records it returns.",
            "expect": {"language": "en", "fields": {}, "fits": "no", "suggestion": 2},
        },
    ),
}
