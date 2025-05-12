from typing import List

from pydantic import BaseModel


class TestResults(BaseModel):
    card_id: int = Field(default=..., max_length=20, description="Username, не более 20 символов")
    description: str
    form: bool
    is_moving: bool
    part: int
    form_or_color: int
    is_passed: bool

    class Config:
        orm_mode = True


class DiagnosisCounterIn(BaseModel):
    user_id: int
    test_results: List[TestResults]

    class Config:
        orm_mode = True


class DiagnosisCreate(BaseModel):
    category: str
    short_name: str
    description: str

    class Config:
        orm_mode = True

class Diagnosis(BaseModel):
    id: int
    category: str
    short_name: str
    description: str

    class Config:
        orm_mode = True

class DiagnosisOut(BaseModel):
    diagnosis: List[Diagnosis]
    history_id: int

    class Config:
        orm_mode = True
