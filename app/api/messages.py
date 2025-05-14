import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from .. import crud, schemas
from app.models.base import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/messages.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["messages"], default_response_class=Response)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{user_id}", response_model=schemas.MessagesOut)
async def get_messages(user_id: int, db: Session = Depends(get_db)):
    logger.info(f"Attempting to get messages for user_id: {user_id}")
    db_user = crud.get_user_by_id(db, user_id)
    if db_user is None:
        logger.error(f"User not found with id: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    db_msg = crud.get_messages_by_user_to_id(db, user_to_id=user_id)
    logger.info(f"Successfully retrieved {len(db_msg)} messages for user_id: {user_id}")
    return {"messages": db_msg}


@router.post("/send-message/", status_code=200)
async def send_message(message: schemas.MessageCreate, db: Session = Depends(get_db)):
    logger.info(
        f"Attempting to send message to user: {message.recipient_username} "
        f"from history_id: {message.history_id}"
    )
    db_user = crud.get_user_by_username(db, message.recipient_username)
    if db_user is None:
        logger.error(f"Recipient not found: {message.recipient_username}")
        raise HTTPException(status_code=404, detail="User not found")

    crud.create_message(db, history_id=message.history_id, user_to_id=db_user.id)
    crud.add_tokens(db, db_user.id)
    logger.info(
        f"Message successfully sent to user_id: {db_user.id} "
        f"from history_id: {message.history_id}"
    )
    return Response(status_code=200, content="Сообщение успешно отправлено")


@router.delete("/{message_id}", status_code=200)
async def delete(message_id: int, user: schemas.UserIn, db: Session = Depends(get_db)):
    logger.info(f"Attempting to delete message_id: {message_id} by user_id: {user.id}")
    db_user = crud.get_user_by_id(db, user.id)
    if db_user is None:
        logger.error(f"User not found with id: {user.id}")
        raise HTTPException(status_code=404, detail="User not found")

    crud.delete_message(db, message_id)
    logger.info(f"Message {message_id} successfully deleted by user_id: {user.id}")
    return Response(status_code=200)