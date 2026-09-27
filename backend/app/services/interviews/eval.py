"""Admin quality eval: question-pack coverage and scoring consistency."""

from __future__ import annotations

import statistics
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session, selectinload

from app.materials import knowledge
from app.models import EvalRun, Interview
from app.services.shared.common import audit, new_id
from app.services.interviews.interview import SCORE_DIMENSIONS, get_interview, score_band, write_report
from app.services.materials.recall import recall_snippets
from app.services.operations.quality import (
    evaluate_answer_evidence,
    evaluate_citations,
    evaluate_retrieval,
    evaluate_system_metrics,
    quality_gate,
)
from app.services.materials.rag.retrievers import lexical_retrieve
from app.services.chat.session import generate_question_pack


async def run_question_eval(db: Session, job_text: str) -> EvalRun:
    hits = await recall_snippets(db, job_text)
    payload = await generate_question_pack(db, job_text, hits)
    questions = payload.get("questions") or []
    stems = [q.get("stem") or "" for q in questions]
    dup = duplicate_rate(stems)
    # A pack is usable only when every item can be asked out loud. Leftover choices fail the run.
    spoken = sum(1 for q in questions if str(q.get("kind") or "open") in {"open", "scenario"} and not q.get("options"))
    coverage = float(payload.get("coverage") or 0.9)
    metrics = {
        "coverage": coverage,
        "duplicate_rate": dup,
        "question_count": len(questions),
        "spoken_count": spoken,
        "usable": 8 <= len(questions) <= 12 and spoken == len(questions),
    }
    run = EvalRun(
        id=new_id(),
        kind="question",
        status="done",
        input_text=job_text[:2000],
        metrics=metrics,
        detail={"job_title": payload.get("job_title"), "stems": stems[:8]},
    )
    db.add(run)
    audit(db, "eval.question", run.id, metrics)
    db.commit()
    db.refresh(run)
    return run


async def run_score_eval(db: Session, interview_id: str | None, repeats: int = 5) -> EvalRun:
    interview = None
    if interview_id:
        interview = get_interview(db, interview_id)
    if interview is None:
        interview = (
            db.query(Interview)
            .options(selectinload(Interview.turns), selectinload(Interview.report))
            .filter(Interview.status == "ended")
            .order_by(Interview.ended_at.desc())
            .first()
        )
    if interview is None:
        raise ValueError("没有已结束的面试可供评分评测")
    transcript = [{"role": t.role, "content": t.content} for t in interview.turns]
    scores: list[float] = []
    dimension_scores: dict[str, list[float]] = {key: [] for key in SCORE_DIMENSIONS}
    for _ in range(max(2, min(repeats, 5))):
        recap = await write_report(db, interview.title, transcript)
        scores.append(float(recap.get("score") if recap.get("score") is not None else 0))
        raw_dimensions = recap.get("dimensions") or {}
        for key in SCORE_DIMENSIONS:
            value = raw_dimensions.get(key)
            dimension_scores[key].append(float(value.get("score", 0)) if isinstance(value, dict) else 0.0)
    sigma = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    dimension_sigma = {key: round(statistics.pstdev(values), 3) for key, values in dimension_scores.items()}
    # Consistency passes only when the total and every fixed dimension stay within two points.
    stable = 1.0 if sigma <= 2 and all(value <= 2 for value in dimension_sigma.values()) else 0.0
    metrics = {
        "n": len(scores),
        "scores": scores,
        "sigma": round(sigma, 3),
        "kendall_tau": round(stable, 3),
        "mean": round(sum(scores) / len(scores), 2),
        "dimension_scores": dimension_scores,
        "dimension_sigma": dimension_sigma,
        "band_consistency": round(sum(score_band(value) == score_band(scores[0]) for value in scores) / len(scores), 3),
    }
    run = EvalRun(
        id=new_id(),
        kind="score",
        status="done",
        input_text=interview.id,
        metrics=metrics,
        detail={"title": interview.title},
    )
    db.add(run)
    audit(db, "eval.score", interview.id, metrics)
    db.commit()
    db.refresh(run)
    return run


def list_eval_runs(db: Session) -> list[EvalRun]:
    return db.query(EvalRun).order_by(EvalRun.created_at.desc()).limit(40).all()


def duplicate_rate(stems: list[str]) -> float:
    if len(stems) < 2:
        return 0.0
    dup = 0
    for i, a in enumerate(stems):
        sa = set(knowledge.terms(a))
        for b in stems[i + 1 :]:
            sb = set(knowledge.terms(b))
            if not sa or not sb:
                continue
            if len(sa & sb) / len(sa | sb) > 0.6:
                dup += 1
    pairs = len(stems) * (len(stems) - 1) / 2
    return round(dup / pairs, 4) if pairs else 0.0


def run_fixed_rag_eval(path: Path, *, k: int = 3) -> dict:
    """Run the versioned synthetic lexical baseline used by the offline gate."""

    dataset = json.loads(path.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    corpus = dataset["materials"]
    rows = []
    for case in cases:
        retrieved = lexical_retrieve(case["question"], corpus, top_k=k)
        rows.append({**case, "retrieved": retrieved})
    retrieval = evaluate_retrieval(rows, k=k)
    citations = evaluate_citations(dataset.get("answer_reviews") or [])
    evidence = evaluate_answer_evidence(dataset.get("answer_reviews") or [])
    system = evaluate_system_metrics(dataset.get("system_runs") or [])
    # The lexical fixture has no provider answer or runtime observations.
    # Preserve their missing values in the report; never gate on fabricated
    # answer labels or synthesized timings.
    metrics = {
        **retrieval,
        **{key: value for key, value in citations.items() if key != "case_count"},
        **evidence,
        **system,
    }
    return {
        "dataset_version": dataset["dataset_version"],
        "material_index_version": dataset["material_index_version"],
        "model_config": dataset["model_config"],
        "run_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "metric_groups": {
            "retrieval": retrieval,
            "citations": citations,
            "answer_evidence": evidence,
            "system": system,
        },
        "gate": quality_gate(metrics, dataset["thresholds"]),
    }


def collect_system_run_metrics(db: Session, run_ids: list[str]) -> dict:
    """Compute observed operational metrics from persisted production run IDs.

    Missing provider, stream, or checkpoint recovery observations stay unset.
    This is a read-only aggregate and never infers successful recovery merely
    because a checkpoint owner row exists.
    """

    from app.models.platform.audit import LlmCallLog
    from app.models.platform.runtime import GraphRun, NodeRun
    from app.models.platform.stream import StreamRun
    from app.services.chat.stream_events import stream_completion_outcomes

    ids = list(dict.fromkeys(str(value) for value in run_ids if value))
    if not ids:
        return evaluate_system_metrics([])
    graphs = db.query(GraphRun).filter(GraphRun.id.in_(ids)).all()
    nodes = db.query(NodeRun).filter(NodeRun.run_id.in_(ids)).all()
    providers = db.query(LlmCallLog).filter(LlmCallLog.run_id.in_(ids)).all()
    streams = db.query(StreamRun).filter(StreamRun.id.in_(ids)).all()
    by_graph = {row.id: row for row in graphs}
    rows: dict[str, dict] = {run_id: {"node_latency_ms": {}} for run_id in by_graph}
    for graph in graphs:
        row = rows[graph.id]
        if graph.created_at and graph.completed_at:
            start = graph.created_at
            end = graph.completed_at
            if start.tzinfo is None and end.tzinfo is not None:
                end = end.replace(tzinfo=None)
            elif start.tzinfo is not None and end.tzinfo is None:
                start = start.replace(tzinfo=None)
            duration_ms = (end - start).total_seconds() * 1000
            if duration_ms >= 0:
                row["latency_ms"] = duration_ms
        diagnostics = graph.diagnostics or {}
        if diagnostics.get("fallback_reason") is not None:
            row["fallback"] = bool(diagnostics["fallback_reason"])
        if type(diagnostics.get("checkpoint_recovered")) is bool:
            row["checkpoint_recovered"] = diagnostics["checkpoint_recovered"]
        for key in ("token_count", "cost"):
            if diagnostics.get(key) is not None:
                row[key] = diagnostics[key]
    for node in nodes:
        if node.run_id in rows and node.duration_ms is not None:
            # Multiple attempts contribute to the same run's node duration.
            values = rows[node.run_id]["node_latency_ms"]
            values[node.node] = values.get(node.node, 0.0) + node.duration_ms
    for provider in providers:
        if provider.run_id in rows:
            row = rows[provider.run_id]
            # Provider failure rate counts calls, while graph diagnostics count runs.
            row.setdefault("provider_calls", []).append(provider.status != "ok")
            row.setdefault("provider_token_values", []).append(provider.total_tokens)
            row.setdefault("provider_cost_values", []).append(provider.estimated_cost)
    for row in rows.values():
        if "provider_calls" not in row:
            continue
        # Graph diagnostics include embedding calls absent from LlmCallLog.
        # Keep logged model usage separate so it cannot replace that total.
        tokens = row.pop("provider_token_values")
        costs = row.pop("provider_cost_values")
        row["logged_model_token_count"] = sum(tokens) if all(value is not None for value in tokens) else None
        row["logged_model_cost"] = sum(costs) if all(value is not None for value in costs) else None
    for run_id, completed in stream_completion_outcomes(db, streams).items():
        if run_id in rows:
            rows[run_id]["sse_completed"] = completed
    return evaluate_system_metrics(list(rows.values()))
