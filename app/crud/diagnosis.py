from sqlalchemy.orm import Session

from .. import models
from ..models import Diagnosis


# Таблица статичная, в ближайшее время менять не планирую)
def create_diagnosis(db: Session, category: str, short_name: str, description: str):
    message = Diagnosis(category=category, short_name=short_name, description=description)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_all_diagnosis(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Diagnosis).offset(skip).limit(limit).all()


def get_diagnosis_by_id(db: Session, item_id: int):
    return db.query(models.Diagnosis).filter(models.Diagnosis.id == item_id).first()


def get_diagnosis_by_short_name(db: Session, short_name: str):
    return db.query(models.Diagnosis).filter(models.Diagnosis.short_name == short_name).first()


def delete_diagnosis(db: Session, item_id: int):
    db_item = db.query(models.Diagnosis).filter(models.Diagnosis.id == item_id).first()
    if db_item:
        db.delete(db_item)
        db.commit()
        return db_item
    return None
