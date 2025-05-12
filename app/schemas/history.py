from datetime import datetime
from typing import List

from pydantic import BaseModel

from app.schemas.diagnosis import DiagnosisOut


class HistoryOut(BaseModel):
    id: int
    user_id: int
    diagnosis: List[DiagnosisOut]
    created_at: datetime

    class Config:
        orm_mode = True


# личная история для пользователя
class HistoryListOut(BaseModel):
    diagnosis: List[HistoryOut]

    class Config:
        orm_mode = True
