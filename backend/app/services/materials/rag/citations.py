"""Deterministic source citation validation and one-shot repair decisions."""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Sequence


_SOURCE_RE = re.compile(r"\[S(\d+)\]", re.IGNORECASE)


def extract_citation_ids(answer: str) -> list[str]:
    """Extract unique source ids in first-appearance order."""

    seen: set[str] = set()
    ids: list[str] = []
    for match in _SOURCE_RE.finditer(str(answer or "")):
        source_id = f"S{int(match.group(1))}"
        if source_id not in seen:
            seen.add(source_id)
            ids.append(source_id)
    return ids


def validate_citations(
    answer: str,
    selected_chunks: Sequence[dict[str, Any]],
    *,
    citations: Sequence[dict[str, Any] | str] | None = None,
    require_citation: bool = True,
    min_cited_sources: int = 1,
) -> dict[str, Any]:
    """Check citation existence, context membership, and optional metadata consistency."""

    available_map = {_source_id(item, index): item for index, item in enumerate(selected_chunks, start=1)}
    available = set(available_map)
    answer_ids = extract_citation_ids(answer)
    structured = _structured_ids(citations or [])
    cited_ids = list(dict.fromkeys([*answer_ids, *structured]))
    invalid = [source_id for source_id in cited_ids if source_id not in available]
    metadata_errors: list[str] = []
    for entry in citations or []:
        if not isinstance(entry, dict):
            continue
        source_id = _source_id(entry, 0)
        if source_id == "INVALID_SOURCE_ID":
            metadata_errors.append("invalid_source_id")
            continue
        selected = available_map.get(source_id)
        if selected is None:
            continue
        for key in ("chunk_id", "material_id", "filename"):
            if entry.get(key) not in (None, "") and str(entry.get(key)) != str(selected.get(key) or ""):
                metadata_errors.append(f"{source_id}:{key}")
    missing = max(0, int(min_cited_sources) - len(set(cited_ids) & available))
    valid = not invalid and not metadata_errors and (not require_citation or (len(cited_ids) >= int(min_cited_sources) and not missing))
    # Source coverage is a diagnostic only; claim coverage needs answer-point
    # annotations and cannot be inferred from the count of cited documents.
    source_coverage = len(set(cited_ids) & available) / len(available) if available else 0.0
    return {
        "valid": valid,
        "citations": [{"source_id": source_id} for source_id in cited_ids if source_id in available],
        "cited_source_ids": cited_ids,
        "available_source_ids": sorted(available, key=lambda value: int(value[1:])),
        "invalid_source_ids": invalid,
        "metadata_errors": metadata_errors,
        "missing_sources": missing,
        "citation_validity": 1.0 if valid else 0.0,
        "source_coverage": round(source_coverage, 4),
        "citation_coverage": None,
        "evidence_coverage": "ok" if valid else "failed",
        "repair_needed": not valid,
    }


def validate_citation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Singular alias retained for older graph node imports."""

    return validate_citations(*args, **kwargs)


async def repair_citations(
    answer: str,
    selected_chunks: Sequence[dict[str, Any]],
    *,
    repair_provider: Callable[[str, str], Any] | None = None,
    require_citation: bool = True,
) -> dict[str, Any]:
    """Attempt at most one repair, then return the final structural decision."""

    initial = validate_citations(answer, selected_chunks, require_citation=require_citation)
    if initial["valid"] or not selected_chunks:
        initial["repair_attempted"] = False
        return initial
    candidate_answer = str(answer or "")
    try:
        if repair_provider is None:
            # A source label alone cannot establish that the answer is grounded.
            # Keep the failed validation and let the caller show a degraded state.
            initial["repair_attempted"] = False
            initial["fallback_reason"] = "repair_provider_unavailable"
            return initial
        else:
            repaired = repair_provider(candidate_answer, _render_sources(selected_chunks))
            if inspect.isawaitable(repaired):
                repaired = await repaired
            candidate_answer = str(repaired or candidate_answer)
        final = validate_citations(candidate_answer, selected_chunks, require_citation=require_citation)
        final.update({"repair_attempted": True, "repaired_answer": candidate_answer})
        return final
    except Exception as exc:  # noqa: BLE001 - failed repair must be observable, not fatal
        initial.update(
            {
                "repair_attempted": True,
                "repair_failed": True,
                "repair_error_type": type(exc).__name__,
                "evidence_coverage": "failed",
            }
        )
        return initial


def citation_gate(answer: str, selected_chunks: Sequence[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Short synchronous gate for nodes that do not need a repair call."""

    return validate_citations(answer, selected_chunks, **kwargs)


def _structured_ids(entries: Sequence[dict[str, Any] | str]) -> list[str]:
    ids: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            ids.extend(extract_citation_ids(entry))
        elif isinstance(entry, dict):
            value = entry.get("source_id") or entry.get("source") or entry.get("id")
            if value:
                match = re.fullmatch(r"S([1-9][0-9]*)", str(value), re.IGNORECASE)
                if match:
                    ids.append(f"S{int(match.group(1))}")
    return list(dict.fromkeys(ids))


def _source_id(item: dict[str, Any], index: int) -> str:
    value = item.get("source_id") or item.get("source")
    if value:
        match = re.fullmatch(r"S([1-9][0-9]*)", str(value), re.IGNORECASE)
        if match:
            return f"S{int(match.group(1))}"
        return "INVALID_SOURCE_ID"
    return f"S{index}"


def _render_sources(chunks: Sequence[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{_source_id(item, i)}] {item.get('text', '')}" for i, item in enumerate(chunks, start=1))


__all__ = [
    "citation_gate",
    "extract_citation_ids",
    "repair_citations",
    "validate_citation",
    "validate_citations",
]
