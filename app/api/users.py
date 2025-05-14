import logging

from fastapi import Depends, HTTPException, APIRouter
from fastapi.responses import Response
from sqlalchemy.orm import Session

from .. import crud, schemas
from app.models.base import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/users.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["user"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{user_id}", response_model=schemas.UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    logger.info(f"Attempting to get user with ID: {user_id}")

    db_user = crud.get_user_by_id(db, user_id)
    if db_user is None:
        logger.error(f"User not found with ID: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    db_balance = crud.get_balance_by_user_id(db, db_user.id)
    logger.info(
        f"Successfully retrieved user {db_user.username} (ID: {db_user.id}) with balance: {db_balance.count_tokens}")

    return {
        "id": db_user.id,
        "username": db_user.username,
        "count_tokens": db_balance.count_tokens
    }


@router.patch("/edit-profile", response_model=schemas.UserOut)
def edit(new_user: schemas.UserEdit, db: Session = Depends(get_db)):
    logger.info(f"Attempting to edit user profile for ID: {new_user.id}")

    db_user = crud.get_user_by_id(db, new_user.id)
    if db_user is None:
        logger.error(f"User not found for editing (ID: {new_user.id})")
        raise HTTPException(status_code=404, detail="User not found")

    try:
        updated_user = crud.update_user(db, new_user)
        db_balance = crud.get_balance_by_user_id(db, updated_user.id)

        logger.info(f"Successfully updated user {updated_user.username} (ID: {updated_user.id})")
        logger.debug(f"Updated user details: {updated_user.__dict__}")

        return {
            "id": updated_user.id,
            "username": updated_user.username,
            "count_tokens": db_balance.count_tokens
        }
    except Exception as e:
        logger.error(f"Error updating user (ID: {new_user.id}): {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while updating user")


@router.delete("/{user_id}", status_code=200)
def delete(user_id: int, db: Session = Depends(get_db)):
    logger.info(f"Attempting to delete user with ID: {user_id}")

    db_user = crud.get_user_by_id(db, user_id)
    if db_user is None:
        logger.error(f"User not found for deletion (ID: {user_id})")
        raise HTTPException(status_code=404, detail="User not found")

    try:
        crud.delete_user(db, user_id)
        logger.info(f"Successfully deleted user (ID: {user_id})")
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Error deleting user (ID: {user_id}): {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while deleting user")
