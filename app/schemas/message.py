from datetime import datetime
from typing import List

from pydantic import BaseModel

from app.schemas.history import HistoryOut


# Отправка сообщения
class MessageCreate(BaseModel):
    recipient_username: str
    history_id: int

    class Config:
        orm_mode = True


class Message(BaseModel):
    id: int
    history: HistoryOut
    created_at: datetime


class MessageOut(BaseModel):
    sender_username: str
    messages: List[Message]

    class Config:
        orm_mode = True


# Отображение отправителей со всеми отправленными диагнозами у пользователя
class MessagesOut(BaseModel):
    messages: List[MessageOut]

    class Config:
        orm_mode = True
