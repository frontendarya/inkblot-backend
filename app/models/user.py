from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from app.models.base import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, nullable=False)
    password = Column(String, nullable=False)

    balance = relationship("Balance", back_populates="user")
    message = relationship("Message", back_populates="user")
    history = relationship("History", back_populates="user")
