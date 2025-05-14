import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, diagnosis, history, messages, users
from app.models.base import Base, engine

"""

inkblot_project/
├── app/
│   ├── api/                 # Роутеры, бизнес-логика
│   │   └── user.py
│   ├── crud/               # Запросы к БД
│   │   └── user.py
│   ├── models/              # Внутренние доменные модели
│   │   └── user.py
│   ├── schemas/             # DTO (вход и выход) pydantic
│   │   └── user.py
│   ├── main.py              # Входная точка
│   ├── base.py          # Креды для подключения к БД
├── config.py                # Настройки
└── requirements.txt

"""

# Инициализация таблиц
Base.metadata.create_all(bind=engine)
app = FastAPI(title="Inkblot Project")

# CORS для фронта
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(diagnosis.router, prefix="/api/v1/diagnosis", tags=["diagnosis"])
app.include_router(history.router, prefix="/api/v1/history", tags=["history"])
app.include_router(messages.router, prefix="/api/v1/messages", tags=["messages"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

# uvicorn app.main:app --reload
# http://localhost:8000/docs
