from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app import schemas, crud
from ..database import SessionLocal

router = APIRouter(tags=["history"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{user_id}", response_model=schemas.HistoryListOut)
async def get_history(user_id: int, db: Session = Depends(get_db)):
    user = crud.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    history_db = crud.get_all_history_by_user_id(db, user_id)
    return {"diagnosis": history_db}


@router.delete("/{user_id}", status_code=200)
async def delete_history(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_id(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    crud.delete_history(db, db_user)
    return Response(status_code=200)
