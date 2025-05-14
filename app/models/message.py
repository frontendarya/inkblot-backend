from datetime import datetime

from sqlalchemy import Column, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship

from app.models.base import Base


class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    history_id = Column(Integer, ForeignKey('history.id'))
    user_to_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime)

    history = relationship("History")
    user = relationship("User")
