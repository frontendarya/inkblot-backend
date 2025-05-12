from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .diagnosis_counter.diagnosis_counter import DiagnosisCounter
from .. import crud, schemas
from ..database import SessionLocal

router = APIRouter(tags=["diagnosis"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/", response_model=schemas.DiagnosisOut)
async def count_diagnosis(diagnosis_in: schemas.DiagnosisCounterIn, db: Session = Depends(get_db)):
    db_balance = crud.get_balance_by_user_id(db, user_id=diagnosis_in.user_id)
    if db_balance is None:
        raise HTTPException(status_code=404, detail="You don't have balance")
    elif db_balance < 1:
        raise HTTPException(status_code=400, detail="You don't have enough balance")
    else:
        crud.remove_tokens(db, user_id=diagnosis_in.user_id)
    """
    Посчитать количество ответов и отказов
    СОЗДАТЬ ЗАПИСЬ В ИСТОРИИ
    Получить диагноз (текст в модели (x cast, чтобы подходило в enum), списокм в формулу),
    сохранить в бд
    списать токены (если их недостаточно, то выслать ошибку), сохрнаить в историю
    :param diagnosis_in:
    :param user:
    :param db:
    :return:
    """
    # meaning_model, popular_model
    diagnosis_ids = []

    results = DiagnosisCounter(diagnosis_in).get_diagnosis()
    for result in results:
        db_result = crud.get_diagnosis_by_short_name(db, result.short_name)
        result.id = db_result.id
        result.category = db_result.category
        result.description = db_result.description
        diagnosis_ids.append(db_result.id)

    history_db = crud.create_history(db, diagnosis_in.user_id, diagnosis_ids)
    return {"diagnosis": results,
            "history_id": history_db.id}
