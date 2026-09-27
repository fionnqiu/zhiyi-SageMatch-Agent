"""Response mappers shared by the session and interview routers."""

from __future__ import annotations

from app import schemas, services
from app.services.interviews.interview import SCORE_DIMENSIONS
from app.models import Interview, ProviderConfig


def session_detail(session) -> schemas.ChatSessionDetail:
    return schemas.ChatSessionDetail(
        id=session.id,
        title=session.title,
        job_title=session.job_title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[schemas.ChatMessageOut.model_validate(m, from_attributes=True) for m in session.messages],
    )


def interview_out(item: Interview) -> schemas.InterviewOut:
    question = services.current_question(item)
    report = None
    if item.report:
        report = schemas.ReportOut(
            id=item.report.id,
            score=item.report.score,
            review=item.report.review,
            issues=[schemas.ReportIssue.model_validate(i) for i in (item.report.issues or [])],
        dimensions=(
                {
                    key: schemas.ReportDimension.model_validate(item.report.dimensions[key])
                    for key in SCORE_DIMENSIONS
                    if key in item.report.dimensions
                }
                if isinstance(item.report.dimensions, dict) and item.report.dimensions
            else None
        ),
            scoring_status=item.report.scoring_status or (
                "unavailable"
                if isinstance(item.report.dimensions, dict)
                and item.report.dimensions
                and all("评分模型当前不可用" in value.get("evidence", "") for value in item.report.dimensions.values())
                else "invalid"
                if isinstance(item.report.dimensions, dict)
                and item.report.dimensions
                and all("模型未返回完整" in value.get("evidence", "") for value in item.report.dimensions.values())
                else "valid"
                if isinstance(item.report.dimensions, dict) and item.report.dimensions
                else "legacy"
            ),
            created_at=item.report.created_at,
        )
    return schemas.InterviewOut(
        id=item.id,
        title=item.title,
        status="ready" if item.status == "abandoned" else item.status,
        current_question_index=item.current_question_index,
        started_at=item.started_at,
        ended_at=item.ended_at,
        elapsed_seconds=item.elapsed_seconds,
        tags=_tag_list(item.tags),
        summary=item.summary,
        score=item.report.score if item.report else None,
        created_at=item.created_at,
        current_question=question_out(question) if question else None,
        report=report,
    )


def interview_detail(item: Interview) -> schemas.InterviewDetail:
    base = interview_out(item)
    return schemas.InterviewDetail(
        **base.model_dump(),
        turns=[schemas.InterviewTurnOut.model_validate(t, from_attributes=True) for t in item.turns],
    )


def question_out(question) -> schemas.QuestionOut:
    return schemas.QuestionOut(
        id=question.id,
        ordinal=question.ordinal,
        stem=question.stem,
        options=_option_dicts(question.options),
        explanation=question.explanation,
        generated_by=question.generated_by,
    )


def _tag_list(raw: object) -> list[str]:
    """A pack once stored the whole focus line as one string. Split it instead of failing the list."""
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        parts = [part.strip() for part in raw.replace("、", ",").split(",")]
        return [part for part in parts if part]
    return []


def _option_dicts(raw: object) -> list[dict[str, str]]:
    """Older packs stored each option as one string. The live page expects key and text."""
    if not isinstance(raw, list):
        return []
    options: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            key = str(item.get("key") or "")
            text = str(item.get("text") or "")
            if key or text:
                options.append({"key": key, "text": text})
            continue
        text = str(item).strip()
        if not text:
            continue
        # "A. 正文" keeps the letter. A bare sentence gets the next letter.
        key, _, rest = text.partition(".")
        if len(key) == 1 and key.isalpha() and rest.strip():
            options.append({"key": key, "text": rest.strip()})
        else:
            options.append({"key": chr(ord("A") + index), "text": text})
    return options


def provider_out(row: ProviderConfig) -> schemas.ProviderOut:
    return schemas.ProviderOut(
        id=row.id,
        name=row.name,
        protocol=row.protocol,
        base_url=row.base_url,
        capability=row.capability,
        status=row.status,
        latency_ms=row.latency_ms,
        models=row.models or [],
        notes=row.notes or "",
        key_masked=services.mask_key(row.api_key or ""),
        has_key=bool(row.api_key),
    )


def report_text(interview: Interview) -> str:
    report = interview.report
    lines = [
        interview.title,
        f"评分：{report.score if report else '-'}",
        "",
        report.review if report else "",
        "",
        "关键失分点",
    ]
    if report and isinstance(report.dimensions, dict):
        lines.extend(["", "分项评分"])
        for key, label in SCORE_DIMENSIONS.items():
            value = report.dimensions.get(key)
            if not isinstance(value, dict):
                continue
            lines.extend([f"- {label}: {value.get('score', '-')}/25", f"  依据：{value.get('evidence', '')}", f"  建议：{value.get('advice', '')}"])
    for item in (report.issues if report else []) or []:
        lines.extend(["", f"- {item.get('issue')}", f"  {item.get('quote')}", f"  {item.get('advice')}"])
    return "\n".join(lines)
