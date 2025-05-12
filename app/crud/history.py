from datetime import datetime
from typing import List

from sqlalchemy.orm import Session

from .. import models
from ..models import History


# Таблица статичная, в ближайшее время менять не планирую)
# TODO: передавать список id диагнозов
def create_history(db: Session, user_id: int, diagnosis_ids: List[int]) -> History:
    history_item = History(user_id=user_id, diagnosis_ids=diagnosis_ids, created_at=datetime.now())
    db.add(history_item)
    db.commit()
    db.refresh(history_item)
    return history_item


def get_history_by_id(db: Session, item_id: int):
    return db.query(models.History).filter(models.History.id == item_id).first()


def get_all_history_by_user_id(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.History).filter(models.History.user_id == user_id).offset(skip).limit(limit).all()


def delete_history_item(db: Session, item_id: int):
    db_item = get_history_by_id(db, item_id)
    if db_item:
        db.delete(db_item)
        db.commit()
        return db_item
    return None


def delete_history(db: Session, user_id: int):
    db_items = db.query(models.History).filter(models.History.user_id == user_id).all()
    for db_item in db_items:
        if db_item:
            db.delete(db_item)
            db.commit()
            return db_item
    return None
