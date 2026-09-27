"""Admin quality-eval routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import schemas, services
from app.core.db import get_db

router = APIRouter(prefix="/api/admin")


@router.get("/eval", response_model=list[schemas.EvalRunOut])
def admin_eval_runs(db: Session = Depends(get_db)) -> list[schemas.EvalRunOut]:
    return [schemas.EvalRunOut.model_validate(r, from_attributes=True) for r in services.list_eval_runs(db)]


@router.post("/eval/questions", response_model=schemas.EvalRunOut)
async def admin_eval_questions(payload: schemas.EvalQuestionIn, db: Session = Depends(get_db)) -> schemas.EvalRunOut:
    run = await services.run_question_eval(db, payload.job_text)
    return schemas.EvalRunOut.model_validate(run, from_attributes=True)


@router.post("/eval/scores", response_model=schemas.EvalRunOut)
async def admin_eval_scores(payload: schemas.EvalScoreIn, db: Session = Depends(get_db)) -> schemas.EvalRunOut:
    try:
        run = await services.run_score_eval(db, payload.interview_id, payload.repeats)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return schemas.EvalRunOut.model_validate(run, from_attributes=True)
