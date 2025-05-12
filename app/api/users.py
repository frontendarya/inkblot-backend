from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from .. import crud, schemas
from ..database import SessionLocal

router = APIRouter(tags=["users"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{user_id}", response_model=schemas.UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_id(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db_balance = crud.get_balance_by_user_id(db, db_user.id)
    return {"id": db_user.id,
            "username": db_user.username,
            "count_tokens": db_balance.count_tokens}


@router.patch("/edit-profile", response_model=schemas.UserOut)
def edit(new_user: schemas.UserEdit, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_id(db, new_user.id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db_user = crud.update_user(db, new_user)
    db_balance = crud.get_balance_by_user_id(db, db_user.id)
    return {"id": db_user.id,
            "username": db_user.username,
            "count_tokens": db_balance.count_tokens}


@router.delete("/{user_id}", status_code=200)
def delete(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_id(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    crud.delete_user(db, user_id)
    return Response(status_code=200)
