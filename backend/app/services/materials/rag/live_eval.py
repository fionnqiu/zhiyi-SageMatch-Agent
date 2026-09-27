"""Controlled six-stage RAG evaluation against a versioned, real material index."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import inspect

from app.materials import knowledge
from app.integrations import llm
from app.agents.providers.router import choose_provider
from app.core.db import SessionLocal
from app.models import Material, MaterialChunk, RoleBinding
from app.core.rag_config import get_rag_config
from app.services.operations.llm_gateway import complete
from app.services.operations.quality import evaluate_answer_evidence, evaluate_citations, evaluate_retrieval, evaluate_system_metrics, quality_gate
from app.services.materials.rag.citations import repair_citations, validate_citations
from app.services.materials.rag.context import build_context
from app.services.materials.rag.stages import StagedRetrieval
from app.services.materials.recall import embed_or_empty


STAGES = (
    "baseline", "query_rewrite", "coarse_dedup", "reranker",
    "citation_constraint", "claim_coverage",
)
REQUIRED_CASE_CATEGORIES = frozenset({
    "synonym", "multi_turn_ellipsis", "bilingual", "version_constraint",
    "multi_material_conflict", "unanswerable", "cross_chunk",
})


def index_version(db: Any) -> tuple[str, set[str], set[str]]:
    """Hash ready material content so labels cannot silently target a changed index."""
    rows = (db.query(MaterialChunk, Material).join(Material, Material.id == MaterialChunk.material_id)
            .filter(Material.status == "ready").all())
    digest = hashlib.sha256()
    chunks: set[str] = set()
    materials: set[str] = set()
    for chunk, material in sorted(rows, key=lambda row: (row[1].id, row[0].ordinal, row[0].id)):
        chunks.add(chunk.id)
        materials.add(material.id)
        payload = [material.id, material.filename, chunk.id, chunk.ordinal, chunk.text,
                   chunk.embedding_model, chunk.embedding]
        digest.update(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}", chunks, materials


def load_dataset(path: Path, version: str, chunks: set[str], materials: set[str], *, require_coverage: bool = True) -> dict[str, Any]:
    """Require real index IDs and explicit labels before spending provider calls."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("material_index_version") != version:
        raise ValueError("dataset material_index_version does not match the ready material index")
    if not isinstance(data.get("dataset_version"), str) or not data["dataset_version"].strip():
        raise ValueError("dataset_version is required")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset needs at least one case")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("each case needs a question")
        # The fixed-set contract permits material-only labels; normalize that
        # omission so later retrieval scoring never invents chunk ground truth.
        if "expected_chunks" not in case:
            case["expected_chunks"] = []
        for field, available in (("expected_materials", materials), ("expected_chunks", chunks)):
            labels = case.get(field)
            if not isinstance(labels, list) or any(not isinstance(value, str) or value not in available for value in labels):
                raise ValueError(f"case has missing or unknown {field}")
        if (not isinstance(case.get("answer_points"), list)
                or any(not isinstance(point, str) or not point.strip() for point in case["answer_points"])
                or type(case.get("must_cite")) is not bool):
            raise ValueError("each case needs answer_points and must_cite")
        history = case.get("history", [])
        if not isinstance(history, list) or any(
            not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"}
            or not isinstance(turn.get("content"), str) for turn in history
        ):
            raise ValueError("history must contain user/assistant text turns")
        if case.get("category") == "multi_turn_ellipsis" and not history:
            raise ValueError("multi_turn_ellipsis case needs history")
    if require_coverage:
        categories = {case.get("category") for case in cases}
        missing = REQUIRED_CASE_CATEGORIES - categories
        if missing:
            raise ValueError("fixed dataset is missing required categories: " + ", ".join(sorted(missing)))
    thresholds = data.get("thresholds")
    if thresholds is not None and (not isinstance(thresholds, dict) or any(
        not isinstance(name, str) or not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(value)
        for name, value in thresholds.items()
    )):
        raise ValueError("thresholds must map metric names to numbers")
    return data


def attach_claim_reviews(report: dict[str, Any], reviews: dict[str, Any]) -> None:
    """Bind human claim labels to exact generated answers before scoring them."""
    supplied_stages = reviews.get("stages") if isinstance(reviews, dict) else None
    if supplied_stages is None and isinstance(reviews, dict) and "cases" in reviews:
        supplied_stages = {"claim_coverage": reviews["cases"]}
    if not isinstance(supplied_stages, dict) or not supplied_stages:
        raise ValueError("claim review stages are required")
    if any(stage not in report["stages"] for stage in supplied_stages):
        raise ValueError("claim review names an unknown stage")
    validated: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]], bool, list[str] | None]]] = {}
    for stage, supplied in supplied_stages.items():
        rows = report["stages"][stage]["cases"]
        if not isinstance(supplied, list) or len(supplied) != len(rows):
            raise ValueError("claim review case count does not match generated answers")
        patches = []
        for row, review in zip(rows, supplied, strict=True):
            if not isinstance(review, dict) or review.get("question") != row["question"]:
                raise ValueError("claim review question does not match generated answer")
            answer_hash = hashlib.sha256(row["answer"].encode("utf-8")).hexdigest()
            if review.get("answer_sha256") != answer_hash:
                raise ValueError("claim review answer hash does not match generated answer")
            claims = review.get("claims")
            if not isinstance(claims, list) or (not claims and review.get("abstained") is not True) or any(
                not isinstance(claim, dict) or not isinstance(claim.get("text"), str) or not claim["text"].strip()
                or type(claim.get("supported")) is not bool for claim in claims
            ):
                raise ValueError("claim reviews need explicit supported labels or an abstention")
            if claims and review.get("abstained") is True:
                raise ValueError("abstention cannot include factual claims")
            covered = review.get("covered_answer_points")
            expected = row.get("answer_points") or []
            if covered is not None and (not isinstance(covered, list) or any(
                not isinstance(point, str) or point not in expected for point in covered
            )):
                raise ValueError("covered_answer_points must reference labeled answer points")
            patches.append((row, claims, review.get("abstained") is True, covered))
        validated[stage] = patches
    # Apply only after every answer hash and claim label has passed validation.
    for stage, patches in validated.items():
        for row, claims, abstained, covered in patches:
            row["claims"] = claims
            row["abstained"] = abstained
            row["covered_answer_points"] = covered
        cases = report["stages"][stage]["cases"]
        evidence = evaluate_answer_evidence(cases)
        labeled_points = [row for row in cases if row.get("answer_points")]
        evidence["answer_point_coverage"] = (
            round(sum(len(set(row["covered_answer_points"])) for row in labeled_points)
                  / sum(len(set(row["answer_points"])) for row in labeled_points), 4)
            if labeled_points and all(row.get("covered_answer_points") is not None for row in labeled_points)
            else None
        )
        unanswerable = [row for row in cases if not row.get("expected_materials") and not row.get("expected_chunks")]
        evidence["uncertainty_expression_rate"] = (
            round(sum(row["abstained"] for row in unanswerable) / len(unanswerable), 4)
            if unanswerable else None
        )
        report["stages"][stage]["answer_evidence"] = evidence
        # Structural citation validity is measured by the runtime. Claim
        # coverage needs a separate explicit human label for each assertion.
        all_claims = [claim for _, claims, _, _ in patches for claim in claims]
        if all_claims and all(type(claim.get("cited")) is bool for claim in all_claims):
            report["stages"][stage]["citations"]["citation_coverage"] = round(
                sum(claim["cited"] for claim in all_claims) / len(all_claims), 4
            )
    report["semantic_review_status"] = "reviewed" if set(validated) == set(report["stages"]) else "partial"
    _refresh_final_gate(report)


def _refresh_final_gate(report: dict[str, Any]) -> None:
    """Gate only the final stage using versioned, explicitly supplied bounds."""

    thresholds = report.get("thresholds")
    if not thresholds:
        report["gate"] = None
        return
    final = report["stages"]["claim_coverage"]
    metrics = {
        **final["retrieval"],
        **{key: value for key, value in final["citations"].items() if key != "case_count"},
        **final["answer_evidence"],
        **final["system"],
    }
    report["gate"] = quality_gate(metrics, thresholds)
    # Retrieval can meet every requested numeric bound while the generated
    # answers remain unreviewed, so a real-run pass requires completed review.
    if report.get("semantic_review_status") != "reviewed":
        report["gate"]["passed"] = False
        report["gate"]["failures"]["semantic_review"] = {
            "observed": None, "required": 1.0,
        }


def verify_reviewed_report(db: Any, dataset_path: Path, run_path: Path,
                           review_path: Path, reviewed_path: Path) -> None:
    """Recompute the reviewed gate against the current index without provider calls."""
    version, chunks, materials = index_version(db)
    dataset = load_dataset(dataset_path, version, chunks, materials)
    raw = json.loads(run_path.read_text(encoding="utf-8"))
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    if (raw.get("material_index_version") != version
            or raw.get("dataset_version") != dataset["dataset_version"]
            or raw.get("thresholds") != dataset.get("thresholds")
            or set(raw.get("stages", {})) != set(STAGES)):
        raise ValueError("evaluation report does not match the current fixed dataset")
    # A gate calibrated with different retrieval settings or model identities
    # cannot authorize the current private startup, even on an unchanged index.
    if raw.get("retrieval_config") != safe_config_snapshot():
        raise ValueError("RAG configuration changed since the evaluation run")
    if raw.get("model_config") != configured_model_snapshot(db):
        raise ValueError("RAG model binding changed since the evaluation run")
    labels = ("question", "expected_materials", "expected_chunks", "answer_points", "must_cite")
    if any(
        [{key: row.get(key) for key in labels} for row in raw["stages"][stage].get("cases", [])]
        != [{key: case.get(key) for key in labels} for case in dataset["cases"]]
        for stage in STAGES
    ):
        raise ValueError("evaluation case labels do not match the fixed dataset")
    try:
        k = raw["retrieval_config"]["recall"]["top_k"]
        if type(k) is not int or k < 1:
            raise ValueError("invalid retrieval k")
        for stage in STAGES:
            summary = raw["stages"][stage]
            cases = summary["cases"]
            for row in cases:
                recorded = row["citation_validation"]
                available = recorded["available_source_ids"]
                if (not isinstance(available, list)
                        or available != [f"S{index}" for index in range(1, len(available) + 1)]
                        or len(available) > len(row["retrieved"])):
                    raise ValueError("evaluation source IDs do not match retrieved cases")
                recalculated = validate_citations(
                    row["answer"], [{"source_id": source_id} for source_id in available],
                    require_citation=row["must_cite"],
                )
                structural_keys = (
                    "valid", "citations", "cited_source_ids", "available_source_ids",
                    "invalid_source_ids", "metadata_errors", "missing_sources",
                    "citation_validity", "source_coverage", "citation_coverage",
                    "evidence_coverage", "repair_needed",
                )
                if any(recorded.get(key) != recalculated[key] for key in structural_keys):
                    raise ValueError("evaluation citation validation does not match the answer")
            citations = evaluate_citations([
                {"valid": row["citation_validation"]["valid"],
                 "citation_coverage": row["citation_validation"]["citation_coverage"]}
                for row in cases
            ])
            if (summary["retrieval"] != evaluate_retrieval(cases, k=k)
                    or summary["citations"] != citations
                    or summary["answer_evidence"] != evaluate_answer_evidence(cases)
                    or summary["system"] != evaluate_system_metrics(cases)):
                raise ValueError("evaluation metrics do not match case records")
    except (KeyError, TypeError) as exc:
        raise ValueError("evaluation report is missing metric inputs") from exc
    # The review hashes bind each label to the original answer. Full report
    # equality also catches edited metrics or a stale gate in the published copy.
    attach_claim_reviews(raw, json.loads(review_path.read_text(encoding="utf-8")))
    if raw != reviewed:
        raise ValueError("reviewed report differs from the recomputed result")
    if raw.get("semantic_review_status") != "reviewed" or not (raw.get("gate") or {}).get("passed"):
        raise ValueError("reviewed quality gate did not pass")


def provider_preflight(db: Any) -> dict[str, str]:
    """Check required adapters without exposing stored credentials in output."""
    cfg = get_rag_config()
    choice = choose_provider(db, "analyst")
    from app.core.config import get_settings
    settings = get_settings()
    provider = choice.provider
    model = (choice.binding.model if choice.binding and choice.binding.model else None) or settings.llm_model
    if not ((provider.api_key if provider else "") or settings.llm_api_key) or not model:
        raise ValueError("analyst answer provider is not configured")
    if not cfg.query_rewrite.enabled or not cfg.query_rewrite.model or not cfg.query_rewrite.base_url or not (os.getenv("SAGEMATCH_RAG_QUERY_REWRITE_API_KEY") or settings.sagematch_rag_query_rewrite_api_key):
        raise ValueError("query rewrite provider is not configured")
    if not cfg.rerank.enabled or not cfg.rerank.model or not cfg.rerank.base_url or not (os.getenv("SAGEMATCH_RAG_RERANK_API_KEY") or settings.sagematch_rag_rerank_api_key):
        raise ValueError("reranker provider is not configured")
    if not cfg.embedding.enabled or not cfg.embedding.model or not cfg.embedding.base_url or not cfg.embedding.api_key:
        raise ValueError("embedding provider is not configured")
    return {"answer_model": model, "rewrite_model": cfg.query_rewrite.model,
            "rerank_model": cfg.rerank.model, "embedding_model": cfg.embedding.model if cfg.embedding.enabled else ""}


def configured_model_snapshot(db: Any) -> dict[str, str]:
    """Read stable model identities without making startup depend on breaker state."""
    from app.core.config import get_settings

    cfg = get_rag_config()
    binding = db.query(RoleBinding).filter(RoleBinding.role == "analyst").one_or_none()
    answer_model = (binding.model if binding and binding.model else None) or get_settings().llm_model
    return {"answer_model": answer_model, "rewrite_model": cfg.query_rewrite.model,
            "rerank_model": cfg.rerank.model, "embedding_model": cfg.embedding.model if cfg.embedding.enabled else ""}


def safe_config_snapshot() -> dict[str, Any]:
    """Record reproducible RAG settings while excluding injected credentials."""

    # Endpoint URLs can carry credentials in userinfo or query parameters, so
    # the report records tuning/model choices but never raw provider URLs.
    return get_rag_config().model_dump(exclude={
        "embedding": {"api_key", "base_url"},
        "query_rewrite": {"api_key", "base_url"},
        "rerank": {"api_key", "base_url"},
    })


def runtime_schema_preflight(db: Any) -> None:
    """Reject an unmigrated runtime before provider routing can query new columns."""
    inspector = inspect(db.get_bind())
    required = {"probe_lease_until", "probe_owner"}
    columns = {column["name"] for column in inspector.get_columns("provider_health")}
    if not required <= columns:
        raise ValueError("runtime schema migration required for provider_health")


async def _baseline_retrieval(db: Any, query: str) -> dict[str, Any]:
    """Measure the pre-optimization single-list scorer on the ready index."""
    started = time.monotonic()
    rows = (db.query(MaterialChunk, Material).join(Material, Material.id == MaterialChunk.material_id)
            .filter(Material.status == "ready").all())
    packed = [{"chunk_id": chunk.id, "material_id": material.id,
               "filename": material.filename, "ordinal": chunk.ordinal,
               "text": chunk.text, "embedding": chunk.embedding}
              for chunk, material in rows]
    vectors, model = await embed_or_empty(db, [query])
    if len(vectors) != 1 or not vectors[0] or not model:
        raise RuntimeError("baseline embedding provider did not return a query vector")
    cfg = get_rag_config()
    candidates = knowledge.recall(query, packed, k=cfg.recall.top_k, query_vec=vectors[0])
    context = build_context(
        candidates, max_chunks=min(cfg.recall.top_k, cfg.context.max_chunks),
        max_tokens=cfg.context.max_tokens, max_chunk_chars=cfg.context.max_chunk_chars,
        include_source_id=cfg.context.include_source_id,
    )
    selected = context["selected_chunks"]
    truncated = context["diagnostics"].get("context_truncated", False)
    return {
        "selected_chunks": selected, "context_text": context["context_text"],
        "search_queries": [query], "retrieval_status": "ok" if selected else "empty",
        "diagnostics": {
            **context["diagnostics"], "embedding_model": model,
            "retrieval_source": "legacy_single_list",
            "rewrite_used": False, "rerank_used": False,
            "fallback_reasons": ["context_truncated"] if truncated else [],
            "stage_latency_ms": {"baseline_retrieval": round((time.monotonic() - started) * 1000, 2)},
        },
    }


async def evaluate_case(db: Any, case: dict[str, Any], stage: str) -> dict[str, Any]:
    """Exercise progressively enabled production stages and retain answer evidence."""
    cfg = get_rag_config()
    query = case["question"]
    started = time.monotonic()
    if stage == "baseline":
        retrieval = await _baseline_retrieval(db, query)
    else:
        run = StagedRetrieval(db, query, history=case.get("history"))
        run.cfg = cfg.model_copy(deep=True)
        run.cfg.query_rewrite.enabled = True
        run.cfg.rerank.enabled = STAGES.index(stage) >= 3
        await run.query_rewrite()
        await run.retrieve_parallel()
        run.candidate_fusion()
        if STAGES.index(stage) >= 2:
            run.deduplicate_and_diversify()
        else:
            run.governed = run.fused
        await run.rerank()
        retrieval = run.context_select()
    diagnostics = retrieval["diagnostics"]
    # This evaluation is fail-closed: a fallback would make the stage comparison misleading.
    if (diagnostics.get("lexical_failed") or diagnostics.get("vector_failed")
            or diagnostics.get("rewrite_failed") or diagnostics.get("rerank_fallback")
            or diagnostics.get("fallback_reasons") or retrieval["retrieval_status"] in {"failed", "degraded"}):
        raise RuntimeError(f"{stage} retrieval provider failed or degraded")
    if stage in STAGES[1:] and not run.embedding_model:
        raise RuntimeError(f"{stage} embedding provider did not return a query vector")
    if stage in STAGES[3:] and retrieval["selected_chunks"] and not diagnostics.get("rerank_used"):
        raise RuntimeError(f"{stage} reranker did not run")
    context = retrieval["context_text"] or "（知识库没有相关证据）"
    cite_instruction = ("每个有依据的事实在句末标注可用来源编号 [Sx]。没有依据时明确说不知道，不能编造。"
                        if STAGES.index(stage) >= 4 else "请直接回答；若证据不足则说明不确定。")
    answer_started = time.monotonic()
    answer = await complete(db, "analyst", "你是知识问答助手。只依据提供的材料回答。" + cite_instruction,
                            f"知识片段：\n{context}\n\n问题：{query}", temperature=0.0)
    answer_ms = round((time.monotonic() - answer_started) * 1000, 2)
    claim_revision_ms = None
    if stage == "claim_coverage":
        # This is a distinct intervention to compare with citation constraints;
        # its output still needs independent human claim review for quality scores.
        revision_started = time.monotonic()
        answer = await complete(
            db, "analyst",
            "逐条核对初稿中的事实断言与给定证据。删除无法由证据支持的断言；证据不足时明确表达不确定。"
            "只使用给定来源编号标注保留的断言，只返回修订后的正文。",
            f"知识片段：\n{context}\n\n问题：{query}\n\n初稿：\n{answer}",
            temperature=0.0,
        )
        claim_revision_ms = round((time.monotonic() - revision_started) * 1000, 2)
    selected = retrieval["selected_chunks"]
    # Measure the same citation requirement at every stage; only the prompt
    # constraint and repair policy change at the citation_constraint stage.
    citation = validate_citations(answer, selected, require_citation=case["must_cite"])
    if STAGES.index(stage) >= 4 and selected and not citation["valid"]:
        async def repair(text: str, sources: str) -> str:
            return await complete(db, "analyst", "只修正引用，删去无证据断言；只返回正文。",
                                  f"原回答：\n{text}\n\n可用证据：\n{sources}", temperature=0.0)
        citation = await repair_citations(answer, selected, repair_provider=repair, require_citation=case["must_cite"])
        answer = citation.get("repaired_answer", answer) if citation["valid"] else answer
    retrieved = [{"chunk_id": chunk_id, "material_id": item.get("material_id")}
                 for item in selected for chunk_id in (item.get("merged_chunk_ids") or [item.get("chunk_id")])]
    return {
        "stage": stage, "question": query, "answer": answer,
        "retrieved": retrieved, "expected_chunks": case["expected_chunks"],
        "expected_materials": case["expected_materials"], "answer_points": case["answer_points"],
        "must_cite": case["must_cite"], "citation_validation": citation,
        "claims": None,  # Explicit review labels are needed before groundedness is measurable.
        "latency_ms": round((time.monotonic() - started) * 1000, 2),
        "node_latency_ms": {**diagnostics.get("stage_latency_ms", {}), "answer": answer_ms,
                            **({"claim_revision": claim_revision_ms} if claim_revision_ms is not None else {})},
        "provider_failed": False, "fallback": bool(diagnostics.get("fallback_reasons")),
        "retrieval_status": retrieval["retrieval_status"],
        "search_queries": retrieval["search_queries"],
        "rewrite_used": bool(diagnostics.get("rewrite_used")),
        "rerank_used": bool(diagnostics.get("rerank_used")),
        "cited_source_ids": citation["cited_source_ids"],
    }


async def run_evaluation(db: Any, dataset: dict[str, Any], version: str, models: dict[str, str]) -> dict[str, Any]:
    """Run all six configurations over the same labeled corpus and questions."""
    results: dict[str, Any] = {}
    for stage in STAGES:
        cases = []
        for case in dataset["cases"]:
            # Keep each question's provider usage separate; unavailable usage
            # or pricing stays unknown instead of becoming an estimated zero.
            with llm.usage_budget_scope() as usage:
                measured = await evaluate_case(db, case, stage)
            snapshot = usage.snapshot()
            measured["token_count"] = snapshot["total_tokens"]
            measured["cost"] = snapshot["estimated_cost"]
            cases.append(measured)
        retrieval = evaluate_retrieval(cases, k=get_rag_config().recall.top_k)
        citations = evaluate_citations([{"valid": row["citation_validation"]["valid"],
                                        "citation_coverage": row["citation_validation"]["citation_coverage"]}
                                       for row in cases])
        results[stage] = {"cases": cases, "retrieval": retrieval, "citations": citations,
                          "answer_evidence": evaluate_answer_evidence(cases),
                          "system": evaluate_system_metrics(cases)}
    report = {"dataset_version": dataset["dataset_version"], "material_index_version": version,
            "model_config": models, "retrieval_config": safe_config_snapshot(),
            "run_at": datetime.now(timezone.utc).isoformat(),
            "stages": results, "semantic_review_status": "unreviewed",
            "thresholds": dataset.get("thresholds")}
    _refresh_final_gate(report)
    return report


def main() -> int:
    """CLI fails without a real corpus and providers; stdout contains no credentials."""
    parser = argparse.ArgumentParser(description="Run controlled RAG evaluation against the ready material index")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review", type=Path, help="human claim labels bound to generated answer hashes")
    parser.add_argument("--report", type=Path, help="existing run to annotate without repeating provider calls")
    parser.add_argument("--verify-reviewed", type=Path, help="verify a reviewed report against --dataset, --report, and --review")
    parser.add_argument("--index-version", action="store_true", help="print current ready index fingerprint and exit")
    args = parser.parse_args()
    if args.verify_reviewed is not None:
        db = SessionLocal()
        try:
            if args.dataset is None or args.report is None or args.review is None:
                raise ValueError("--verify-reviewed requires --dataset, --report, and --review")
            verify_reviewed_report(db, args.dataset, args.report, args.review, args.verify_reviewed)
            print("reviewed quality gate verified against the current ready index")
            return 0
        except ValueError as exc:
            print(f"evaluation failed: {exc}")
            return 1
        except Exception as exc:
            print(f"evaluation failed: {type(exc).__name__}")
            return 1
        finally:
            db.rollback()
            db.close()
    if args.report is not None:
        try:
            if args.review is None or args.output is None:
                raise ValueError("--report requires --review and --output")
            report = json.loads(args.report.read_text(encoding="utf-8"))
            attach_claim_reviews(report, json.loads(args.review.read_text(encoding="utf-8")))
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"wrote reviewed report to {args.output}")
            return 1 if report.get("gate") is not None and not report["gate"]["passed"] else 0
        except ValueError as exc:
            print(f"evaluation failed: {exc}")
            return 1
        except Exception as exc:
            print(f"evaluation failed: {type(exc).__name__}")
            return 1
    db = SessionLocal()
    try:
        version, chunks, materials = index_version(db)
        if args.index_version:
            print(version)
            return 0
        if args.dataset is None or args.output is None:
            raise ValueError("--dataset and --output are required")
        if not chunks:
            raise ValueError("ready material index is empty")
        dataset = load_dataset(args.dataset, version, chunks, materials)
        runtime_schema_preflight(db)
        models = provider_preflight(db)
        result = asyncio.run(run_evaluation(db, dataset, version, models))
        if args.review is not None:
            attach_claim_reviews(result, json.loads(args.review.read_text(encoding="utf-8")))
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(STAGES)} stages to {args.output}")
        return 1 if result.get("gate") is not None and not result["gate"]["passed"] else 0
    except ValueError as exc:
        print(f"evaluation failed: {exc}")
        return 1
    except Exception as exc:
        # Provider exceptions and DSNs can contain secrets. Show only the type here.
        print(f"evaluation failed: {type(exc).__name__}")
        return 1
    finally:
        db.rollback()
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
