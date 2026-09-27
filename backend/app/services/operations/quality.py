"""Provider-independent RAG metrics and calibrated quality gates."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from typing import Any


def recall_at_k(retrieved: Sequence[Any], relevant: Iterable[Any], k: int | None = None) -> float:
    expected = _ids(relevant)
    if not expected:
        return 1.0 if not _ids(retrieved[:k] if k else retrieved) else 0.0
    found = _ids(retrieved[:k] if k else retrieved) & expected
    return round(len(found) / len(expected), 4)


def precision_at_k(retrieved: Sequence[Any], relevant: Iterable[Any], k: int | None = None) -> float:
    values = list(retrieved[:k] if k else retrieved)
    denominator = k if k is not None else len(values)
    if denominator <= 0:
        return 0.0
    expected = _ids(relevant)
    # A repeated candidate consumes a rank slot but cannot add another relevant hit.
    hits = {_item_id(item) for item in values} & expected
    return round(len(hits) / denominator, 4)


def mean_reciprocal_rank(retrieved: Sequence[Any], relevant: Iterable[Any], k: int | None = None) -> float:
    expected = _ids(relevant)
    for rank, item in enumerate(retrieved[:k] if k else retrieved, start=1):
        if _item_id(item) in expected:
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_at_k(retrieved: Sequence[Any], relevant: Iterable[Any], k: int | None = None) -> float:
    values = list(retrieved[:k] if k else retrieved)
    expected = _ids(relevant)
    if not expected:
        return 1.0 if not values else 0.0
    seen: set[str] = set()
    dcg = 0.0
    for rank, item in enumerate(values, start=1):
        item_id = _item_id(item)
        if item_id in expected and item_id not in seen:
            dcg += 1.0 / math.log2(rank + 1)
            seen.add(item_id)
    ideal_hits = min(len(expected), k if k is not None else len(values))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return round(dcg / ideal, 4) if ideal else 0.0


def hit_rate(retrieved: Sequence[Any], relevant: Iterable[Any], k: int | None = None) -> float:
    expected = _ids(relevant)
    values = _ids(retrieved[:k] if k else retrieved)
    if not expected:
        return 1.0 if not values else 0.0
    return 1.0 if expected & values else 0.0


def effective_candidate_rate(candidates: Sequence[Any], relevant: Iterable[Any]) -> float:
    """Measure the share of candidates that can support an expected material/chunk."""

    values = list(candidates)
    if not values:
        return 0.0
    expected = _ids(relevant)
    # Duplicate relevant chunks do not increase the effective share of a shortlist.
    return round(len({_item_id(item) for item in values} & expected) / len(values), 4)


def evaluate_retrieval(dataset: Sequence[dict[str, Any]], *, k: int = 10) -> dict[str, Any]:
    """Aggregate fixed-set retrieval metrics from versioned case mappings."""

    rows = list(dataset)
    if not rows:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "hit_rate": 0.0, "effective_candidate_rate": 0.0, "case_count": 0}
    per_case: list[dict[str, float]] = []
    for row in rows:
        retrieved = row.get("retrieved") or row.get("candidates") or []
        chunk_targets = row.get("expected_chunks") or []
        material_targets = row.get("expected_materials") or []
        relevant = row.get("relevant") or chunk_targets or material_targets
        # A chunk target must never be satisfied by a matching material id, and
        # material-level cases must compare against material ids explicitly.
        id_key = "material_id" if material_targets and not chunk_targets and not row.get("relevant") else "chunk_id"
        if not row.get("relevant"):
            retrieved = [item.get(id_key, "") if isinstance(item, dict) else item for item in retrieved]
        per_case.append(
            {
                "recall_at_k": recall_at_k(retrieved, relevant, k),
                "precision_at_k": precision_at_k(retrieved, relevant, k),
                "mrr": mean_reciprocal_rank(retrieved, relevant, k),
                "ndcg_at_k": ndcg_at_k(retrieved, relevant, k),
                "hit_rate": hit_rate(retrieved, relevant, k),
                "effective_candidate_rate": effective_candidate_rate(retrieved[:k], relevant),
            }
        )
    keys = per_case[0].keys()
    aggregate = {key: round(statistics.fmean(row[key] for row in per_case), 4) for key in keys}
    aggregate["case_count"] = len(per_case)
    aggregate["k"] = k
    aggregate["cases"] = per_case
    return aggregate


def evaluate_citations(
    answers: Sequence[dict[str, Any]],
    *,
    require_citation: bool = True,
) -> dict[str, Any]:
    """Aggregate deterministic citation validity and coverage signals."""

    if not answers:
        return {"citation_validity": None, "citation_coverage": None, "case_count": 0}
    validity = [1.0 if bool(row.get("valid")) else 0.0 for row in answers]
    coverage = [float(row["citation_coverage"]) for row in answers if row.get("citation_coverage") is not None]
    return {
        "citation_validity": round(statistics.fmean(validity), 4),
        "citation_coverage": round(statistics.fmean(coverage), 4) if coverage else None,
        "required": require_citation,
        "case_count": len(answers),
    }


def quality_gate(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    """Apply lower bounds and explicit ``max_`` upper bounds, failing closed."""

    failures: dict[str, dict[str, float | None]] = {}
    observed: dict[str, float | None] = {}
    for name, threshold in thresholds.items():
        upper_bound = name.startswith("max_")
        metric_name = name[4:] if upper_bound else name
        value = _number(metrics.get(metric_name))
        required = _number(threshold)
        observed[name] = value
        if value is None or required is None or (value > required if upper_bound else value < required):
            failures[name] = {"observed": value, "required": required}
    return {"passed": not failures, "failures": failures, "observed": observed, "thresholds": dict(thresholds)}


def run_quality_gate(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    """Named alias for CI and admin callers."""

    return quality_gate(metrics, thresholds)


def _ids(values: Iterable[Any]) -> set[str]:
    return {_item_id(value) for value in values if _item_id(value)}


def _item_id(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("chunk_id", "id", "material_id", "filename", "source_id"):
            if value.get(key) not in (None, ""):
                return str(value[key])
    return str(value).strip() if value not in (None, "") else ""


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


# Common short names make the metric contract easy to consume in dashboards.
recall = recall_at_k
precision = precision_at_k
mrr = mean_reciprocal_rank
ndcg = ndcg_at_k

__all__ = [
    "effective_candidate_rate",
    "evaluate_citations",
    "evaluate_retrieval",
    "hit_rate",
    "mean_reciprocal_rank",
    "mrr",
    "ndcg",
    "ndcg_at_k",
    "precision",
    "precision_at_k",
    "quality_gate",
    "recall",
    "recall_at_k",
    "run_quality_gate",
]


def evaluate_answer_evidence(answers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate reviewed claim labels without treating citations as proof of truth.

    Each claim must have an explicit supported boolean from a human or a
    separately recorded evidence review. Missing labels leave the result
    unverified, so a retrieval-only fixture cannot pass an answer quality gate.
    """

    reviewed = [row for row in answers if isinstance(row.get("claims"), list)]
    claims = [claim for row in reviewed for claim in row["claims"] if isinstance(claim, dict)]
    labels = [claim.get("supported") for claim in claims]
    # A zero-claim answer needs an explicit abstention label. Otherwise an
    # unchecked answer could hide behind a fully reviewed neighboring case.
    unmarked_empty = any(not row["claims"] and row.get("abstained") is not True for row in reviewed)
    if len(reviewed) != len(answers) or unmarked_empty or not claims or any(type(label) is not bool for label in labels):
        return {"groundedness": None, "unsupported_claim_rate": None,
                "reviewed_claim_count": 0, "answer_case_count": len(answers)}
    supported = sum(labels)
    return {"groundedness": round(supported / len(claims), 4),
            "unsupported_claim_rate": round((len(claims) - supported) / len(claims), 4),
            "reviewed_claim_count": len(claims), "answer_case_count": len(answers)}


def evaluate_system_metrics(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate operational evidence used by the stage-five quality gate.

    Optional boolean rates use only explicitly observed rows. This keeps a
    missing SSE or recovery probe distinguishable from a measured failure.
    """

    rows = list(runs)
    latencies = [_number(row.get("latency_ms")) for row in rows]
    latency_values = [value for value in latencies if value is not None]
    node_values: dict[str, list[float]] = {}
    logged_provider_outcomes: list[bool] = []
    case_provider_outcomes: list[bool] = []
    logged_model_tokens: list[float | None] = []
    logged_model_costs: list[float | None] = []
    for row in rows:
        calls = row.get("provider_calls")
        if isinstance(calls, list):
            logged_provider_outcomes.extend(value for value in calls if type(value) is bool)
            logged_model_tokens.append(_number(row.get("logged_model_token_count")))
            logged_model_costs.append(_number(row.get("logged_model_cost")))
        elif type(row.get("provider_failed")) is bool:
            # Fixed-set fixtures can still supply one outcome per case.
            case_provider_outcomes.append(row["provider_failed"])
        for node, raw in (row.get("node_latency_ms") or {}).items():
            value = _number(raw)
            if value is not None:
                node_values.setdefault(str(node), []).append(value)
    # Call logs and case observations have different denominators; use one
    # source for the reported rate so an unlogged case cannot dilute a call.
    provider_outcomes = logged_provider_outcomes or case_provider_outcomes
    has_logged_model_calls = bool(logged_provider_outcomes)
    return {
        "p50_latency_ms": _percentile(latency_values, 0.50),
        "p95_latency_ms": _percentile(latency_values, 0.95),
        "node_latency_ms": {
            node: {"p50": _percentile(values, 0.50), "p95": _percentile(values, 0.95)}
            for node, values in sorted(node_values.items())
        },
        "provider_failure_rate": round(sum(provider_outcomes) / len(provider_outcomes), 4)
        if provider_outcomes else None,
        "provider_failure_rate_basis": "logged_model_calls" if has_logged_model_calls else "case_observations",
        "logged_model_call_count": len(logged_provider_outcomes),
        "logged_model_token_count": int(sum(logged_model_tokens))
        if logged_model_tokens and all(value is not None for value in logged_model_tokens) else None,
        "logged_model_estimated_cost": round(sum(logged_model_costs), 6)
        if logged_model_costs and all(value is not None for value in logged_model_costs) else None,
        "fallback_rate": _boolean_rate(rows, "fallback"),
        "sse_completion_rate": _boolean_rate(rows, "sse_completed"),
        "checkpoint_recovery_rate": _boolean_rate(rows, "checkpoint_recovered"),
        "token_count": int(sum(_number(row.get("token_count")) or 0 for row in rows))
        if rows and all(_number(row.get("token_count")) is not None for row in rows) else None,
        "estimated_cost": round(sum(_number(row.get("cost")) or 0 for row in rows), 6)
        if rows and all(_number(row.get("cost")) is not None for row in rows) else None,
        "run_count": len(rows),
    }


def _boolean_rate(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    """Return a measured rate while preserving an unobserved value as None."""

    observed = [row[key] for row in rows if type(row.get(key)) is bool]
    return round(sum(bool(value) for value in observed) / len(observed), 4) if observed else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    """Use linear interpolation so small fixed sets have stable percentiles."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)


__all__.extend(["evaluate_answer_evidence", "evaluate_system_metrics"])
