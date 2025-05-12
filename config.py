from pydantic_settings import BaseSettings

DB_HOST = 'localhost'
DB_PORT = '5433'
DB_NAME = 'fast_api'
DB_USER = 'amin'
DB_PASSWORD = 'my_super_password'

DATABASE_URL = f'postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'


class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str = "supersecret"
    DEBUG: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
