import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app import schemas, crud
from app.models.base import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/history.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["history"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{user_id}", response_model=schemas.HistoryListOut)
async def get_history(user_id: int, db: Session = Depends(get_db)):
    logger.info(f"Attempting to get history for user_id: {user_id}")

    user = crud.get_user_by_id(db, user_id)
    if not user:
        logger.error(f"User not found with id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    history_db = crud.get_all_history_by_user_id(db, user_id)
    logger.info(f"Retrieved {len(history_db)} history records for user_id: {user_id}")

    return {"diagnosis": history_db}


@router.delete("/{user_id}", status_code=200)
async def delete_history(user_id: int, db: Session = Depends(get_db)):
    logger.info(f"Attempting to delete history for user_id: {user_id}")

    db_user = crud.get_user_by_id(db, user_id)
    if db_user is None:
        logger.error(f"User not found with id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    try:
        deleted_count = crud.delete_history(db, db_user)
        logger.info(f"Successfully deleted {deleted_count} history records for user_id: {user_id}")
        return Response(status_code=200)

    except Exception as e:
        logger.error(f"Error deleting history for user_id: {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while deleting history")