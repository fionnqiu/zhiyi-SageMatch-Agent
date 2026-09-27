"""Score fusion, candidate governance, and provider-optional reranking."""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from app.integrations.llm import UsageBudgetError


def normalize_scores(candidates: Sequence[dict[str, Any]], *, key: str = "retrieval_score") -> list[dict[str, Any]]:
    """Min-max normalize one route while preserving all retrieval metadata."""

    values = [float(item.get(key, item.get("score", 0.0)) or 0.0) for item in candidates]
    if not values:
        return []
    high, low = max(values), min(values)
    span = high - low
    output: list[dict[str, Any]] = []
    for item, value in zip(candidates, values, strict=True):
        copy = dict(item)
        normalized = (value - low) / span if span else (1.0 if high > 0 else 0.0)
        copy["normalized_score"] = round(max(0.0, min(1.0, normalized)), 6)
        output.append(copy)
    return output


def fuse_candidates(
    lexical: Sequence[dict[str, Any]],
    vector: Sequence[dict[str, Any]],
    *,
    strategy: str = "weighted",
    lexical_weight: float = 0.35,
    rrf_k: int = 60,
    candidate_k: int | None = 40,
) -> list[dict[str, Any]]:
    """Fuse routes using one explicit strategy: weighted scores or reciprocal rank fusion."""

    strategy = str(strategy or "weighted").lower()
    if strategy not in {"weighted", "rrf"}:
        raise ValueError("fusion strategy must be weighted or rrf")
    lexical_norm = normalize_scores(lexical)
    vector_norm = normalize_scores(vector)
    by_id: dict[str, dict[str, Any]] = {}
    scores: defaultdict[str, float] = defaultdict(float)
    route_count: defaultdict[str, int] = defaultdict(int)
    weight = max(0.0, min(1.0, float(lexical_weight)))
    for route, items, route_weight in (
        ("lexical", lexical_norm, weight),
        ("vector", vector_norm, 1.0 - weight),
    ):
        for rank, item in enumerate(items, start=1):
            key = _identity(item)
            by_id.setdefault(key, dict(item))
            by_id[key]["retrieval_source"] = _merge_sources(by_id[key].get("retrieval_source"), route)
            route_count[key] += 1
            if strategy == "rrf":
                scores[key] += 1.0 / (max(1, int(rrf_k)) + rank)
            else:
                scores[key] += float(item.get("normalized_score", 0.0)) * route_weight
    max_score = max(scores.values(), default=0.0)
    fused: list[dict[str, Any]] = []
    for key, item in by_id.items():
        raw = scores[key]
        normalized = raw / max_score if max_score else 0.0
        item["retrieval_score"] = round(max(0.0, min(1.0, normalized)), 6)
        item["fused_score"] = item["retrieval_score"]
        item["score"] = item["retrieval_score"]
        item["retrieval_routes"] = route_count[key]
        fused.append(item)
    fused.sort(key=lambda item: (-item["retrieval_score"], str(item.get("chunk_id") or "")))
    return fused[: max(0, int(candidate_k))] if candidate_k is not None else fused


def govern_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    max_chunks_per_material: int = 4,
    max_materials: int | None = None,
    max_candidates: int | None = None,
    merge_adjacent: bool = True,
) -> list[dict[str, Any]]:
    """Deduplicate chunks, merge adjacent ordinals, and enforce diversity limits."""

    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        item = dict(candidate)
        key = _identity(item)
        old = unique.get(key)
        if old is None or _score(item) > _score(old):
            unique[key] = item
    ranked = sorted(unique.values(), key=lambda item: (-_score(item), str(item.get("chunk_id") or "")))
    if merge_adjacent:
        ranked = _merge_adjacent(ranked)
    material_counts: defaultdict[str, int] = defaultdict(int)
    material_order: list[str] = []
    selected: list[dict[str, Any]] = []
    for item in ranked:
        material = str(item.get("material_id") or item.get("filename") or "")
        if material not in material_order:
            if max_materials is not None and len(material_order) >= max(0, int(max_materials)):
                continue
            material_order.append(material)
        if material_counts[material] >= max(0, int(max_chunks_per_material)):
            continue
        material_counts[material] += 1
        selected.append(item)
        if max_candidates is not None and len(selected) >= max(0, int(max_candidates)):
            break
    return selected


async def rerank_candidates(
    query: str,
    candidates: Sequence[dict[str, Any]],
    *,
    reranker: Callable[[str, list[dict[str, Any]]], Any] | None = None,
    top_n: int = 8,
    batch_size: int = 16,
    timeout_seconds: float = 8.0,
    fail_open: bool = True,
) -> dict[str, Any]:
    """Apply an injected reranker, falling back to coarse ranking on failure."""

    coarse = list(candidates)[: max(0, int(top_n))]
    diagnostics: dict[str, Any] = {
        "rerank_used": False,
        "rerank_fallback": False,
        "fallback_reason": "",
        "provider_status": "unconfigured" if reranker is None else "configured",
    }
    if reranker is None or not candidates:
        diagnostics["fallback_reason"] = "provider_unavailable" if reranker is None else "empty_candidates"
        return {"candidates": coarse, "retrieval_status": "ok" if coarse else "empty", "diagnostics": diagnostics}

    try:
        batches = [list(candidates)[i : i + max(1, int(batch_size))] for i in range(0, len(candidates), max(1, int(batch_size)))]
        rescored: list[dict[str, Any]] = []
        for batch in batches:
            raw = reranker(query, batch)
            if inspect.isawaitable(raw):
                raw = await asyncio.wait_for(raw, timeout=max(0.01, float(timeout_seconds)))
            rescored.extend(_apply_reranker_scores(batch, raw))
        rescored.sort(key=lambda item: (-_score(item), str(item.get("chunk_id") or "")))
        diagnostics.update({"rerank_used": True, "provider_status": "ok"})
        return {"candidates": rescored[: max(0, int(top_n))], "retrieval_status": "ok", "diagnostics": diagnostics}
    except UsageBudgetError:
        # Rerank fail-open handles provider outages, never graph budget denial.
        raise
    except Exception as exc:  # noqa: BLE001 - fail-open is an explicit contract
        diagnostics.update(
            {
                "rerank_fallback": bool(fail_open),
                "fallback_reason": "rerank_fallback",
                "provider_status": "error",
                "error_type": type(exc).__name__,
            }
        )
        if fail_open:
            return {"candidates": coarse, "retrieval_status": "degraded", "diagnostics": diagnostics}
        return {"candidates": [], "retrieval_status": "failed", "diagnostics": diagnostics}


async def rerank(query: str, candidates: Sequence[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Short alias used by graph nodes."""

    return await rerank_candidates(query, candidates, **kwargs)


def _apply_reranker_scores(batch: list[dict[str, Any]], raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get("scores") or raw.get("results") or raw.get("data") or []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("reranker returned no scores")
    scores: list[float] = []
    for value in raw:
        if isinstance(value, dict):
            value = value["score"] if "score" in value else value.get("relevance_score")
        scores.append(float(value))
    if len(scores) != len(batch):
        raise ValueError("reranker score count does not match candidate count")
    return [{**item, "rerank_score": max(0.0, min(1.0, score)), "retrieval_score": max(0.0, min(1.0, score)), "score": max(0.0, min(1.0, score))} for item, score in zip(batch, scores, strict=True)]


def _merge_adjacent(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in items:
        if merged and _adjacent(merged[-1], item):
            target = merged[-1]
            target["text"] = "\n".join(value for value in (target.get("text", ""), item.get("text", "")) if value)
            target["merged_chunk_ids"] = list(dict.fromkeys([*(target.get("merged_chunk_ids") or [target.get("chunk_id")]), item.get("chunk_id")]))
            target["ordinal_end"] = item.get("ordinal")
            target["retrieval_score"] = max(_score(target), _score(item))
            target["score"] = target["retrieval_score"]
        else:
            copy = dict(item)
            copy.setdefault("merged_chunk_ids", [copy.get("chunk_id")])
            merged.append(copy)
    return merged


def _adjacent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return str(left.get("material_id") or "") == str(right.get("material_id") or "") and int(right.get("ordinal") or 0) == int(left.get("ordinal_end", left.get("ordinal", 0))) + 1


def _identity(item: dict[str, Any]) -> str:
    return str(item.get("chunk_id") or f"{item.get('material_id', '')}:{item.get('ordinal', 0)}:{item.get('text', '')}")


def _score(item: dict[str, Any]) -> float:
    return float(item.get("retrieval_score", item.get("rerank_score", item.get("score", 0.0))) or 0.0)


def _merge_sources(current: Any, route: str) -> str:
    values = [part for part in str(current or "").split(",") if part]
    if route not in values:
        values.append(route)
    return ",".join(values)


__all__ = [
    "fuse_candidates",
    "govern_candidates",
    "normalize_scores",
    "rerank",
    "rerank_candidates",
]
