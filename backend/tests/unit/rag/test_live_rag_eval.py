"""Fail-closed checks for the controlled, real-corpus RAG evaluation entry point."""

import json
import asyncio
import copy
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.operations.quality import evaluate_answer_evidence, evaluate_citations, evaluate_retrieval, evaluate_system_metrics
from app.services.materials.rag.citations import validate_citations
from app.services.materials.rag.live_eval import (
    _refresh_final_gate, attach_claim_reviews, configured_model_snapshot, evaluate_case, index_version, load_dataset, main, provider_preflight, run_evaluation, runtime_schema_preflight, safe_config_snapshot, verify_reviewed_report,
)


def _index_db(embedding: list[float]) -> MagicMock:
    """Represent a ready row without creating a provider or synthetic quality result."""
    chunk = SimpleNamespace(id="chunk-1", ordinal=0, text="Evidence", embedding_model="embed-v1", embedding=embedding)
    material = SimpleNamespace(id="material-1", filename="evidence.md")
    db = MagicMock()
    db.query.return_value.join.return_value.filter.return_value.all.return_value = [(chunk, material)]
    return db


def test_index_fingerprint_changes_when_embedding_changes() -> None:
    original, chunks, materials = index_version(_index_db([0.1, 0.2]))
    changed, _, _ = index_version(_index_db([0.1, 0.3]))

    assert original != changed
    assert chunks == {"chunk-1"}
    assert materials == {"material-1"}


def test_dataset_rejects_stale_or_unknown_real_index_labels(tmp_path: Path) -> None:
    version, chunks, materials = index_version(_index_db([0.1, 0.2]))
    dataset = {
        "dataset_version": "reviewed-v1", "material_index_version": version,
        "cases": [{"question": "Evidence?", "expected_materials": ["material-1"],
                   "expected_chunks": ["chunk-1"], "answer_points": ["Evidence"], "must_cite": True}],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    assert load_dataset(path, version, chunks, materials, require_coverage=False)["dataset_version"] == "reviewed-v1"
    with pytest.raises(ValueError, match="required categories"):
        load_dataset(path, version, chunks, materials)

    dataset["material_index_version"] = "synthetic-v1"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    with pytest.raises(ValueError, match="material_index_version"):
        load_dataset(path, version, chunks, materials, require_coverage=False)

    dataset["material_index_version"] = version
    dataset["cases"][0]["expected_chunks"] = ["missing"]
    path.write_text(json.dumps(dataset), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_chunks"):
        load_dataset(path, version, chunks, materials, require_coverage=False)


def test_dataset_accepts_optional_expected_chunks(tmp_path: Path) -> None:
    """Material-only labels remain valid without inventing chunk ground truth."""
    version, chunks, materials = index_version(_index_db([0.1, 0.2]))
    path = tmp_path / "material-only.json"
    path.write_text(json.dumps({
        "dataset_version": "material-only-v1", "material_index_version": version,
        "cases": [{"question": "Evidence?", "expected_materials": ["material-1"],
                   "answer_points": ["Evidence"], "must_cite": True}],
    }), encoding="utf-8")

    loaded = load_dataset(path, version, chunks, materials, require_coverage=False)
    assert loaded["cases"][0]["expected_chunks"] == []


def test_provider_preflight_fails_before_calls_when_rewrite_is_unconfigured() -> None:
    cfg = SimpleNamespace(query_rewrite=SimpleNamespace(enabled=False, model="", base_url=""))
    choice = SimpleNamespace(provider=SimpleNamespace(api_key="present"),
                             binding=SimpleNamespace(model="answer-model"))
    with patch("app.services.materials.rag.live_eval.get_rag_config", return_value=cfg), patch(
        "app.services.materials.rag.live_eval.choose_provider", return_value=choice
    ):
        with pytest.raises(ValueError, match="query rewrite provider"):
            provider_preflight(MagicMock())


def test_gate_model_snapshot_uses_binding_without_health_routing() -> None:
    """A temporary breaker opening cannot invalidate the approved model identity."""
    from app.core.rag_config import RagConfig

    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = SimpleNamespace(model="primary")
    with patch("app.services.materials.rag.live_eval.get_rag_config", return_value=RagConfig()), patch(
        "app.core.config.get_settings", return_value=SimpleNamespace(llm_model="fallback")
    ), patch("app.services.materials.rag.live_eval.choose_provider", side_effect=AssertionError("health route used")):
        assert configured_model_snapshot(db)["answer_model"] == "primary"


def test_config_snapshot_does_not_include_injected_embedding_key() -> None:
    from app.core.rag_config import RagConfig

    config = RagConfig()
    config.embedding.api_key = "private-key"
    config.embedding.base_url = "https://private-key@example.test/v1?token=private-key"
    with patch("app.services.materials.rag.live_eval.get_rag_config", return_value=config):
        snapshot = safe_config_snapshot()
    assert snapshot["recall"]["fusion"] == "weighted"
    assert "api_key" not in snapshot["embedding"]
    assert "base_url" not in snapshot["embedding"]


def test_runtime_schema_preflight_reports_migration_without_db_details() -> None:
    inspector = MagicMock()
    inspector.get_columns.return_value = [{"name": "provider_id"}]
    with patch("app.services.materials.rag.live_eval.inspect", return_value=inspector):
        with pytest.raises(ValueError, match="runtime schema migration required for provider_health"):
            runtime_schema_preflight(MagicMock())


def test_claim_reviews_bind_to_exact_answer() -> None:
    import hashlib

    report = {"stages": {"claim_coverage": {"cases": [
        {"question": "Q", "answer": "A [S1]", "claims": None}
    ], "answer_evidence": {}}}, "semantic_review_status": "unreviewed"}
    reviews = {"cases": [{"question": "Q", "answer_sha256": hashlib.sha256(b"A [S1]").hexdigest(),
                          "claims": [{"text": "A", "supported": True}]}]}
    attach_claim_reviews(report, reviews)
    assert report["stages"]["claim_coverage"]["answer_evidence"]["groundedness"] == 1.0
    assert report["semantic_review_status"] == "reviewed"

    reviews["cases"][0]["answer_sha256"] = "stale"
    with pytest.raises(ValueError, match="answer hash"):
        attach_claim_reviews(report, reviews)


def test_reviewed_report_verifier_recomputes_gate_and_rejects_tampering(tmp_path: Path) -> None:
    """A published gate must match the answer-bound review and current index."""
    import app.services.materials.rag.live_eval as live_eval

    question = "Evidence?"
    answer = "Supported [S1]"
    dataset = {"dataset_version": "fixed-v1", "thresholds": {"groundedness": 0.95},
               "cases": [{"question": question, "expected_materials": ["material-1"],
                          "expected_chunks": [], "answer_points": [], "must_cite": True}]}
    case = {"question": question, "answer": answer, "answer_points": [],
            "expected_materials": ["material-1"], "expected_chunks": [],
            "must_cite": True, "retrieved": [{"chunk_id": "chunk-1", "material_id": "material-1"}],
            "citation_validation": validate_citations(answer, [{"source_id": "S1"}]),
            "claims": None, "latency_ms": 10.0}
    raw = {"dataset_version": "fixed-v1", "material_index_version": "index-v1",
           "model_config": {},
           "thresholds": dataset["thresholds"], "semantic_review_status": "unreviewed",
           "retrieval_config": {"recall": {"top_k": 10}},
           "stages": {"claim_coverage": {
               "cases": [case],
               "retrieval": evaluate_retrieval([case], k=10),
               "citations": evaluate_citations([case["citation_validation"]]),
               "answer_evidence": evaluate_answer_evidence([case]),
               "system": evaluate_system_metrics([case]),
           }}}
    reviews = {"cases": [{"question": question,
                          "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                          "claims": [{"text": "Supported", "supported": True}]}]}
    reviewed = copy.deepcopy(raw)
    attach_claim_reviews(reviewed, reviews)
    paths = [tmp_path / name for name in ("dataset.json", "run.json", "review.json", "reviewed.json")]
    for path, value in zip(paths, (dataset, raw, reviews, reviewed), strict=True):
        path.write_text(json.dumps(value), encoding="utf-8")
    with patch.object(live_eval, "STAGES", ("claim_coverage",)), patch.object(
        live_eval, "index_version", return_value=("index-v1", set(), set())
    ), patch.object(live_eval, "load_dataset", return_value=dataset), patch.object(
        live_eval, "safe_config_snapshot", return_value=raw["retrieval_config"]
    ) as current_config, patch.object(live_eval, "configured_model_snapshot", return_value={}):
        verify_reviewed_report(object(), *paths)
        current_config.return_value = {"recall": {"top_k": 5}}
        with pytest.raises(ValueError, match="RAG configuration changed"):
            verify_reviewed_report(object(), *paths)
        current_config.return_value = raw["retrieval_config"]
        with patch.object(live_eval, "configured_model_snapshot", return_value={"answer_model": "changed"}):
            with pytest.raises(ValueError, match="model binding changed"):
                verify_reviewed_report(object(), *paths)
        reviewed["gate"]["passed"] = False
        paths[3].write_text(json.dumps(reviewed), encoding="utf-8")
        with pytest.raises(ValueError, match="differs"):
            verify_reviewed_report(object(), *paths)
        changed_metrics = copy.deepcopy(raw)
        changed_metrics["stages"]["claim_coverage"]["retrieval"]["hit_rate"] = 0.0
        paths[1].write_text(json.dumps(changed_metrics), encoding="utf-8")
        with pytest.raises(ValueError, match="metrics"):
            verify_reviewed_report(object(), *paths)
        changed_citation = copy.deepcopy(raw)
        changed_citation["stages"]["claim_coverage"]["cases"][0]["citation_validation"]["valid"] = False
        changed_citation["stages"]["claim_coverage"]["citations"] = evaluate_citations(
            [changed_citation["stages"]["claim_coverage"]["cases"][0]["citation_validation"]]
        )
        changed_citation_reviewed = copy.deepcopy(changed_citation)
        attach_claim_reviews(changed_citation_reviewed, reviews)
        paths[1].write_text(json.dumps(changed_citation), encoding="utf-8")
        paths[3].write_text(json.dumps(changed_citation_reviewed), encoding="utf-8")
        with pytest.raises(ValueError, match="citation validation"):
            verify_reviewed_report(object(), *paths)
        # Editing both report copies must still fail against the fixed labels.
        changed = copy.deepcopy(raw)
        changed["stages"]["claim_coverage"]["cases"][0]["expected_materials"] = []
        changed_reviewed = copy.deepcopy(changed)
        attach_claim_reviews(changed_reviewed, reviews)
        paths[1].write_text(json.dumps(changed), encoding="utf-8")
        paths[3].write_text(json.dumps(changed_reviewed), encoding="utf-8")
        with pytest.raises(ValueError, match="case labels"):
            verify_reviewed_report(object(), *paths)
    with patch.object(live_eval, "index_version", return_value=("index-v2", set(), set())), patch.object(
        live_eval, "load_dataset", return_value=dataset
    ):
        with pytest.raises(ValueError, match="current fixed dataset"):
            verify_reviewed_report(object(), *paths)


def test_real_evaluation_records_measured_tokens_without_inventing_cost() -> None:
    """The six stage report uses provider counters and leaves unknown pricing blank."""
    from app.integrations import llm
    from app.core.rag_config import RagConfig

    async def measured_case(_db, case, stage):
        with llm.model_request_budget("system", case["question"], 10, "unpriced-model"):
            llm.record_provider_usage({"prompt_tokens": 2, "completion_tokens": 3,
                                       "total_tokens": 5}, "unpriced-model")
        return {"question": case["question"], "stage": stage, "retrieved": [],
                "expected_materials": [], "expected_chunks": [], "claims": None,
                "citation_validation": {"valid": True, "citation_coverage": None},
                "latency_ms": 1.0}

    with patch("app.services.materials.rag.live_eval.evaluate_case", new=measured_case), patch(
        "app.services.materials.rag.live_eval.get_rag_config", return_value=RagConfig()
    ), patch("app.services.materials.rag.live_eval.safe_config_snapshot", return_value={}):
        report = asyncio.run(run_evaluation(object(), {
            "dataset_version": "fixture-v1", "cases": [{"question": "Q"}],
        }, "index-v1", {}))
    for stage in report["stages"].values():
        assert stage["cases"][0]["token_count"] == 5
        assert stage["cases"][0]["cost"] is None
        assert stage["system"]["token_count"] == 5
        assert stage["system"]["estimated_cost"] is None


def test_unanswerable_case_accepts_explicit_abstention_review() -> None:
    import hashlib

    answer = "材料不足，无法核实。"
    report = {"stages": {"claim_coverage": {"cases": [
        {"question": "No evidence?", "answer": answer, "claims": None,
         "expected_materials": [], "expected_chunks": []}
    ], "answer_evidence": {}}}, "semantic_review_status": "unreviewed"}
    reviews = {"cases": [{"question": "No evidence?", "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                          "claims": [], "abstained": True}]}
    attach_claim_reviews(report, reviews)
    assert report["semantic_review_status"] == "reviewed"
    assert report["stages"]["claim_coverage"]["answer_evidence"]["groundedness"] is None
    assert report["stages"]["claim_coverage"]["answer_evidence"]["uncertainty_expression_rate"] == 1.0


def test_claim_review_rejects_abstention_with_factual_claims() -> None:
    answer = "Unverified claim"
    report = {"stages": {"claim_coverage": {"cases": [{
        "question": "No evidence?", "answer": answer, "claims": None,
        "expected_materials": [], "expected_chunks": [],
    }], "answer_evidence": {}}}, "semantic_review_status": "unreviewed"}
    reviews = {"cases": [{
        "question": "No evidence?", "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "claims": [{"text": "Unverified claim", "supported": False}], "abstained": True,
    }]}

    with pytest.raises(ValueError, match="abstention"):
        attach_claim_reviews(report, reviews)


def test_stage_reviews_support_complete_comparison_and_atomic_validation() -> None:
    import hashlib

    report = {"stages": {
        stage: {"cases": [{"question": "Q", "answer": answer, "claims": None,
                           "answer_points": ["A"]}],
                "answer_evidence": {}, "citations": {"citation_coverage": None}}
        for stage, answer in (("baseline", "A"), ("claim_coverage", "A [S1]"))
    }, "semantic_review_status": "unreviewed"}
    reviews = {"stages": {
        stage: [{"question": "Q", "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                 "claims": [{"text": "A", "supported": supported, "cited": supported}],
                 "covered_answer_points": ["A"] if supported else []}]
        for stage, answer, supported in (("baseline", "A", False), ("claim_coverage", "A [S1]", True))
    }}
    stale = {"stages": {key: [dict(value[0])] for key, value in reviews["stages"].items()}}
    stale["stages"]["claim_coverage"][0]["answer_sha256"] = "stale"
    with pytest.raises(ValueError, match="answer hash"):
        attach_claim_reviews(report, stale)
    assert report["stages"]["baseline"]["cases"][0]["claims"] is None

    attach_claim_reviews(report, reviews)
    assert report["semantic_review_status"] == "reviewed"
    assert report["stages"]["baseline"]["answer_evidence"]["unsupported_claim_rate"] == 1.0
    assert report["stages"]["claim_coverage"]["answer_evidence"]["groundedness"] == 1.0
    assert report["stages"]["baseline"]["citations"]["citation_coverage"] == 0.0
    assert report["stages"]["claim_coverage"]["citations"]["citation_coverage"] == 1.0
    assert report["stages"]["baseline"]["answer_evidence"]["answer_point_coverage"] == 0.0
    assert report["stages"]["claim_coverage"]["answer_evidence"]["answer_point_coverage"] == 1.0


def test_baseline_measures_missing_required_citation_before_constraint_stage() -> None:
    from app.core.rag_config import RagConfig

    retrieval = {
        "selected_chunks": [{"source_id": "S1", "chunk_id": "c1", "material_id": "m1", "text": "Evidence"}],
        "context_text": "[S1] Evidence", "search_queries": ["Q"],
        "diagnostics": {"fallback_reasons": [], "stage_latency_ms": {}},
        "retrieval_status": "ok",
    }
    pipeline = MagicMock()
    pipeline.query_rewrite = AsyncMock()
    pipeline.retrieve_parallel = AsyncMock()
    pipeline.rerank = AsyncMock()
    pipeline.context_select.return_value = retrieval
    with patch("app.services.materials.rag.live_eval._baseline_retrieval", new=AsyncMock(return_value=retrieval)), patch(
        "app.services.materials.rag.live_eval.get_rag_config", return_value=RagConfig()
    ), patch("app.services.materials.rag.live_eval.complete", new=AsyncMock(return_value="Answer without citation")):
        result = asyncio.run(evaluate_case(object(), {
            "question": "Q", "expected_chunks": ["c1"], "expected_materials": ["m1"],
            "answer_points": ["Evidence"], "must_cite": True,
        }, "baseline"))
    assert result["citation_validation"]["valid"] is False


def test_baseline_uses_legacy_rank_without_staged_fusion() -> None:
    from app.core.rag_config import RagConfig

    cfg = RagConfig()
    cfg.embedding.lexical_weight = 0.8
    cfg.recall.top_k = 2
    chunk_lexical = SimpleNamespace(id="lex", ordinal=0, text="target", embedding=[0.0, 1.0])
    chunk_vector = SimpleNamespace(id="vec", ordinal=0, text="other", embedding=[1.0, 0.0])
    material = SimpleNamespace(id="material", filename="source.md")
    db = MagicMock()
    db.query.return_value.join.return_value.filter.return_value.all.return_value = [
        (chunk_lexical, material), (chunk_vector, material),
    ]
    with patch("app.services.materials.rag.live_eval.StagedRetrieval", side_effect=AssertionError("staged fusion used")), patch(
        "app.services.materials.rag.live_eval.get_rag_config", return_value=cfg
    ), patch("app.materials.knowledge.get_rag_config", return_value=cfg), patch(
        "app.services.materials.rag.live_eval.embed_or_empty", new=AsyncMock(return_value=([[1.0, 0.0]], "embed-v1"))
    ) as embedding, patch("app.services.materials.rag.live_eval.complete", new=AsyncMock(return_value="Answer [S1]")):
        result = asyncio.run(evaluate_case(db, {
            "question": "target", "expected_chunks": ["lex"], "expected_materials": ["material"],
            "answer_points": ["target"], "must_cite": True,
        }, "baseline"))

    embedding.assert_awaited_once_with(db, ["target"])
    assert result["retrieved"][0]["chunk_id"] == "lex"
    assert result["search_queries"] == ["target"]
    assert result["rewrite_used"] is False
    assert result["rerank_used"] is False
    assert result["citation_validation"]["valid"] is True


def test_baseline_fails_closed_when_query_embedding_is_missing() -> None:
    from app.services.materials.rag.live_eval import _baseline_retrieval

    db = MagicMock()
    db.query.return_value.join.return_value.filter.return_value.all.return_value = []
    with patch("app.services.materials.rag.live_eval.embed_or_empty", new=AsyncMock(return_value=([], ""))):
        with pytest.raises(RuntimeError, match="baseline embedding provider"):
            asyncio.run(_baseline_retrieval(db, "Q"))


def test_claim_coverage_stage_revises_claims_before_citation_validation() -> None:
    from app.core.rag_config import RagConfig

    retrieval = {
        "selected_chunks": [{"source_id": "S1", "chunk_id": "c1", "material_id": "m1", "text": "Verified"}],
        "context_text": "[S1] Verified", "search_queries": ["Q"],
        "diagnostics": {"fallback_reasons": [], "stage_latency_ms": {}, "rerank_used": True},
        "retrieval_status": "ok",
    }
    pipeline = MagicMock()
    pipeline.query_rewrite = AsyncMock()
    pipeline.retrieve_parallel = AsyncMock()
    pipeline.rerank = AsyncMock()
    pipeline.context_select.return_value = retrieval
    pipeline.embedding_model = "embed-v1"
    answers = AsyncMock(side_effect=["Unverified claim [S1]", "Verified [S1]"])
    with patch("app.services.materials.rag.live_eval.StagedRetrieval", return_value=pipeline), patch(
        "app.services.materials.rag.live_eval.get_rag_config", return_value=RagConfig()
    ), patch("app.services.materials.rag.live_eval.complete", new=answers):
        result = asyncio.run(evaluate_case(object(), {
            "question": "Q", "expected_chunks": ["c1"], "expected_materials": ["m1"],
            "answer_points": ["Verified"], "must_cite": True,
        }, "claim_coverage"))

    assert answers.await_count == 2
    assert "Unverified claim" in answers.await_args_list[1].args[3]
    assert result["answer"] == "Verified [S1]"
    assert result["citation_validation"]["valid"] is True
    assert result["claims"] is None


def test_real_report_gate_waits_for_reviewed_answer_evidence() -> None:
    import hashlib

    answer = "Supported [S1]"
    report = {
        "stages": {"claim_coverage": {
            "cases": [{"question": "Q", "answer": answer, "claims": None}],
            "retrieval": {"recall_at_k": 1.0},
            "citations": {"citation_validity": 1.0, "citation_coverage": None},
            "answer_evidence": {"groundedness": None}, "system": {},
        }},
        "thresholds": {"groundedness": 0.9, "max_unsupported_claim_rate": 0.1},
        "semantic_review_status": "unreviewed",
    }
    _refresh_final_gate(report)
    assert report["gate"]["passed"] is False
    attach_claim_reviews(report, {"cases": [{
        "question": "Q", "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "claims": [{"text": "Supported", "supported": True, "cited": True}],
    }]})
    assert report["gate"]["passed"] is True


def test_retrieval_only_threshold_does_not_pass_unreviewed_report() -> None:
    report = {
        "stages": {"claim_coverage": {
            "retrieval": {"recall_at_k": 1.0}, "citations": {},
            "answer_evidence": {"groundedness": None}, "system": {},
        }},
        "thresholds": {"recall_at_k": 0.9},
        "semantic_review_status": "unreviewed",
    }

    _refresh_final_gate(report)

    assert report["gate"]["passed"] is False
    assert "semantic_review" in report["gate"]["failures"]


def test_review_cli_writes_failed_gate_and_returns_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answer = "Unsupported answer"
    report = {
        "stages": {"claim_coverage": {
            "cases": [{"question": "Q", "answer": answer, "claims": None}],
            "retrieval": {"recall_at_k": 1.0}, "citations": {},
            "answer_evidence": {"groundedness": None}, "system": {},
        }},
        "thresholds": {"groundedness": 0.9},
        "semantic_review_status": "unreviewed",
    }
    review = {"cases": [{
        "question": "Q", "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "claims": [{"text": "Unsupported answer", "supported": False}],
    }]}
    report_path = tmp_path / "report.json"
    review_path = tmp_path / "review.json"
    output_path = tmp_path / "reviewed.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    review_path.write_text(json.dumps(review), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["live_eval", "--report", str(report_path),
                                    "--review", str(review_path), "--output", str(output_path)])

    assert main() == 1
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["semantic_review_status"] == "reviewed"
    assert saved["gate"]["passed"] is False
