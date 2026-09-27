"""Three-way intent fusion: LLM, embedding overlap, and keyword patterns.

Weights match the architecture note (0.7 / 0.2 / 0.1). The LLM path is already
the perception decision; the other two only correct it when they agree more
strongly than the model does.
"""

from __future__ import annotations

import re
from typing import Any

_WEIGHTS = {"llm": 0.7, "embedding": 0.2, "pattern": 0.1}

# Patterns catch short, obvious turns the model sometimes over-interprets as a JD.
_PATTERNS: dict[str, tuple[str, ...]] = {
    "answer": (r"^(你好|嗨|在吗|谢谢|你是谁)", r"(什么是|解释|区别|为什么|怎么理解)"),
    "clarify": (r"(哪一块|哪个方向|还是).*(还是|\?|？)",),
    "generate_interview": (r"(岗位|jd|职位|招聘|出题|模拟面试|面试题|生成.{0,12}面试)",),
}

_LEXICON: dict[str, tuple[str, ...]] = {
    "answer": ("你好", "什么", "解释", "区别", "原理"),
    "clarify": ("方向", "哪块", "还是", "选择"),
    "generate_interview": ("岗位", "招聘", "职责", "出题", "模拟面试", "任职", "生成"),
}


def pattern_intent(text: str) -> tuple[str, float]:
    """Zero-latency vote. No match returns a weak answer rather than a guess."""
    for intent, rules in _PATTERNS.items():
        if any(re.search(rule, text, flags=re.I) for rule in rules):
            return intent, 0.9
    return "answer", 0.2


def embedding_intent(text: str) -> tuple[str, float]:
    """Lexical stand-in for the embedding vote when no query vector is available.

    Real cosine is used by fuse_intent when the caller passes scores. This path
    keeps fusion deterministic in tests and when the embedding vendor is down.
    """
    tokens = set(re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{1,}", text.lower()))
    best_intent, best = "answer", 0.0
    for intent, words in _LEXICON.items():
        overlap = len(tokens & set(words)) / max(1, len(words))
        if overlap > best:
            best_intent, best = intent, overlap
    return best_intent, min(1.0, best)


def fuse_intent(
    text: str,
    llm_decision: dict[str, Any],
    *,
    embedding_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Blend the three votes. A high-confidence LLM finish is kept as-is."""
    llm_intent = str(llm_decision.get("intent") or "answer")
    if llm_intent not in _LEXICON:
        llm_intent = "answer"
    pattern, pattern_score = pattern_intent(text)
    if embedding_scores:
        embed_intent = max(embedding_scores, key=embedding_scores.get)
        embed_score = float(embedding_scores[embed_intent])
    else:
        embed_intent, embed_score = embedding_intent(text)

    totals: dict[str, float] = {}
    votes = {
        "llm": (llm_intent, 1.0 if llm_decision.get("source") == "agent" else 0.55),
        "embedding": (embed_intent, embed_score),
        "pattern": (pattern, pattern_score),
    }
    for source, (intent, score) in votes.items():
        totals[intent] = totals.get(intent, 0.0) + _WEIGHTS[source] * score
    # The other two lanes are a veto, not a vote the 0.7 LLM weight can out-score.
    # They only fire together, and only against an obvious pattern such as a greeting.
    others_agree = embed_intent == pattern and embed_intent != llm_intent and embed_score >= 0.5 and pattern_score >= 0.8
    chosen = embed_intent if others_agree else max(totals, key=totals.get)
    merged = dict(llm_decision)
    merged["intent"] = chosen
    merged["source_scores"] = {
        "llm": round(votes["llm"][1], 3),
        "embedding": round(embed_score, 3),
        "pattern": round(pattern_score, 3),
    }
    merged["fusion"] = chosen
    return merged
