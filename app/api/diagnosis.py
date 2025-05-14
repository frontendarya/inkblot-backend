import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .diagnosis_counter.diagnosis_counter import DiagnosisCounter
from .. import crud, schemas
from app.models.base import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/diagnosis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["diagnosis"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/", response_model=schemas.DiagnosisOut)
async def count_diagnosis(diagnosis_in: schemas.DiagnosisCounterIn, db: Session = Depends(get_db)):
    logger.info(f"Starting diagnosis for user_id: {diagnosis_in.user_id}")

    # Проверка баланса
    db_balance = crud.get_balance_by_user_id(db, user_id=diagnosis_in.user_id)
    if db_balance is None:
        logger.error(f"No balance found for user_id: {diagnosis_in.user_id}")
        raise HTTPException(status_code=404, detail="You don't have balance")
    elif db_balance < 1:
        logger.warning(f"Insufficient balance for user_id: {diagnosis_in.user_id}, balance: {db_balance}")
        raise HTTPException(status_code=400, detail="You don't have enough balance")
    else:
        logger.info(f"Sufficient balance ({db_balance}) for user_id: {diagnosis_in.user_id}, removing tokens")
        db_balance = crud.remove_tokens(db, user_id=diagnosis_in.user_id)

    # Получение диагноза
    logger.info(f"Calculating diagnosis for user_id: {diagnosis_in.user_id}")
    diagnosis_ids = []

    try:
        results = DiagnosisCounter(diagnosis_in).get_diagnosis()
        logger.info(f"Diagnosis calculation completed for user_id: {diagnosis_in.user_id}")

        for result in results:
            db_result = crud.get_diagnosis_by_short_name(db, result.short_name)
            if not db_result:
                logger.error(f"Diagnosis not found for short_name: {result.short_name}")
                continue

            result.id = db_result.id
            result.category = db_result.category
            result.description = db_result.description
            diagnosis_ids.append(db_result.id)
            logger.debug(f"Processed diagnosis: {result.short_name} (ID: {db_result.id})")

        # Создание истории
        history_db = crud.create_history(db, diagnosis_in.user_id, diagnosis_ids)
        logger.info(f"History record created with ID: {history_db.id} for user_id: {diagnosis_in.user_id}")

        return {
            "diagnosis": results,
            "history_id": history_db.id,
            "count_tokens": db_balance.count_tokens
        }

    except Exception as e:
        logger.error(f"Error during diagnosis processing for user_id: {diagnosis_in.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during diagnosis processing")
