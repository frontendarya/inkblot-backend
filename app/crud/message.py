from datetime import datetime

from sqlalchemy.orm import Session

from .. import models
from ..models import Message


def create_message(db: Session, history_id: int, user_to_id: int):
    message = Message(history_id=history_id, user_to_id=user_to_id, created_at=datetime.now())
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_messages_by_user_to_id(db: Session, user_to_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.Message).filter(models.Message.user_to_id == user_to_id).offset(skip).limit(limit).all()


def get_message_by_id(db: Session, message_id: int):
    return db.query(models.Message).filter(models.Message.id == message_id).first()


def delete_message(db: Session, item_id: int):
    db_item = get_message_by_id(db, item_id)
    if db_item:
        db.delete(db_item)
        db.commit()
        return db_item
    return None
