"""Evidence context selection with source ids and deterministic token budgets."""

from __future__ import annotations

import os
from typing import Any, Sequence

from app.materials.knowledge import token_estimate


def build_context(
    selected_chunks: Sequence[dict[str, Any]],
    *,
    max_chunks: int = 8,
    max_tokens: int = 5000,
    max_chunk_chars: int = 1200,
    include_source_id: bool = True,
) -> dict[str, Any]:
    """Format selected chunks as stable ``[Sx]`` evidence blocks.

    Chunks are consumed in ranking order.  The selector stops before the token
    budget and records truncation rather than silently allowing an oversized
    prompt.  A single oversized first chunk is clipped so the result remains
    useful and deterministic.
    """

    output: list[dict[str, Any]] = []
    context_parts: list[str] = []
    used_tokens = 0
    truncated = False
    for source_index, original in enumerate(list(selected_chunks)[: max(0, int(max_chunks))], start=1):
        item = dict(original)
        text = str(item.get("text") or "")[: max(1, int(max_chunk_chars))]
        if len(str(original.get("text") or "")) > len(text):
            truncated = True
        remaining = max(0, int(max_tokens) - used_tokens)
        if remaining <= 0:
            truncated = True
            break
        estimate = token_estimate(text)
        if estimate > remaining:
            # A rough token-to-character bound avoids adding a whole oversized chunk.
            text = _clip_to_tokens(text, remaining)
            estimate = token_estimate(text)
            truncated = True
        if not text.strip():
            continue
        source_id = f"S{source_index}"
        item["source_id"] = source_id
        item["text"] = text
        item["token_estimate"] = estimate
        output.append(item)
        used_tokens += estimate
        heading = f"[{source_id}] " if include_source_id else ""
        filename = str(item.get("filename") or item.get("material_id") or "source")
        ordinal = item.get("ordinal", 0)
        context_parts.append(f"{heading}{filename} (chunk {ordinal})\n{text}")

    return {
        "selected_chunks": output,
        "context_text": "\n\n".join(context_parts),
        "diagnostics": {
            "selected_count": len(output),
            "token_count": used_tokens,
            "context_truncated": truncated,
            "max_chunks": max_chunks,
            "max_tokens": max_tokens,
        },
    }


def select_context(selected_chunks: Sequence[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Alias matching the graph node name."""

    return build_context(selected_chunks, **kwargs)


def format_context(selected_chunks: Sequence[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for prompt builders."""

    return build_context(selected_chunks, **kwargs)


def _clip_to_tokens(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    # The knowledge tokenizer is mixed Chinese/word based; walk token spans to preserve text boundaries.
    from app.materials.knowledge import tokenize

    tokens = tokenize(text)
    kept: list[str] = []
    count = 0
    for token in tokens:
        if token.isspace():
            kept.append(token)
            continue
        if count >= budget:
            break
        kept.append(token)
        count += 1
    return "".join(kept).strip()


__all__ = ["build_context", "format_context", "select_context"]
