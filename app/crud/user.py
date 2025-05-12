from sqlalchemy.orm import Session
from passlib.context import CryptContext

from .. import models, schemas

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_item = models.User(username=user.username, password=hashed_password)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def authenticate_user(db, username: str, password: str):
    user_db = get_user_by_username(db, username)
    if not user_db:
        return False
    if not pwd_context.verify(password, user_db.password):
        return False
    return user_db


# Получение всех user
def get_all_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


# Получение user по ID
def get_user_by_id(db: Session, item_id: int):
    return db.query(models.User).filter(models.User.id == item_id).first()


def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()


def update_user(db: Session, user: schemas.UserEdit):
    db_item = db.query(models.User).filter(models.User.id == user.id).first()
    if db_item:
        db_item.username = user.username_new
        db_item.password = user.password_new
        db.commit()
        db.refresh(db_item)
        return db_item
    return None


def delete_user(db: Session, user_id: int):
    db_item = db.query(models.User).filter(models.User.id == user_id).first()
    if db_item:
        db.delete(db_item)
        db.commit()
        return db_item
    return None
