from pydantic import BaseModel, Field

class UserCreate(BaseModel):
    username: str = Field(default=..., max_length=20, description="Username, не более 20 символов")
    password: str = Field(default=..., description="Пароль")

    class Config:
        orm_mode = True

class UserEdit(BaseModel):
    id: int
    username_new: str
    password_new: str

    class Config:
        orm_mode = True

class UserIn(BaseModel):
    id: int
    username: str
    count_tokens: int

    class Config:
        orm_mode = True

class UserOut(BaseModel):
    id: int
    username: str
    count_tokens: int

    class Config:
        orm_mode = True