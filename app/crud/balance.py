from sqlalchemy.orm import Session

from .. import models


# По умолчанию начисляется 5 токенов
def create_balance(db: Session, user_id: int):
    db_item = models.Balance(user_id=user_id, count_tokens=5)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


def get_balance_by_user_id(db: Session, user_id: int):
    return db.query(models.Balance).filter(models.Balance.user_id == user_id).first()


# Пополнить баланс (отправка сообщения)
def add_tokens(db: Session, user_id: int):
    db_item = get_balance_by_user_id(db, user_id)
    if db_item:
        db_item.count_tokens += 3
        db.commit()
        db.refresh(db_item)
        return db_item
    return None


# Снять токены (оплата за тест)
def remove_tokens(db: Session, user_id: int):
    db_item = get_balance_by_user_id(db, user_id)
    if db_item:
        db_item.count_tokens -= 1
        db.commit()
        db.refresh(db_item)
        return db_item
    return None
