import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from .. import schemas, crud, deps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/auth.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


@router.post("/login", response_model=schemas.Token)
async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: Session = Depends(deps.get_db)
):
    logger.info(f"Attempting login for username: {form_data.username}")

    user_db = crud.authenticate_user(db, form_data.username, form_data.password)
    if not user_db:
        logger.warning(f"Failed login attempt for username: {form_data.username}")
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=deps.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = deps.create_access_token(
        data={"sub": user_db.username}, expires_delta=access_token_expires
    )

    logger.info(f"Successful login for username: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register", response_model=schemas.UserOut)
def register(user: schemas.UserCreate, db: Session = Depends(deps.get_db)):
    logger.info(f"Attempting to register new user: {user.username}")

    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        logger.warning(f"Registration failed - username already exists: {user.username}")
        raise HTTPException(status_code=400, detail="Username already registered")

    created_user = crud.create_user(db, user)
    logger.info(f"Successfully registered new user: {user.username} with ID: {created_user.id}")

    db_balance = crud.get_balance_by_user_id(db, created_user.id)

    return {
        "id": created_user.id,
        "username": created_user.username,
        "count_tokens": db_balance.count_tokens
    }
