from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from .. import crud, schemas
from ..database import SessionLocal

router = APIRouter(tags=["messages"], default_response_class=Response)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{user_id}", response_model=schemas.MessagesOut)
async def get_messages(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_id(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db_msg = crud.get_messages(db, user_to_id=user_id)
    if db_msg is None:
        return {"messages": "У вас пока нет сообщений"}
    return {"messages": db_msg}


@router.post("/send-message/", status_code=200)
async def send_message(message: schemas.MessageCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, message.recipient_username)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    crud.create_message(db, history_id=message.history_id, user_to_id=db_user.id)
    crud.add_tokens(db, db_user.id)
    return Response(status_code=200, content="Сообщение успешно отправлено")


@router.delete("/{message_id}", status_code=200)
async def delete(message_id: int, user: schemas.UserIn, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_id(db, user.id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    crud.delete_message(db, message_id)
    return Response(status_code=200)
