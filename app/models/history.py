from datetime import datetime

from sqlalchemy import Column, Integer, ForeignKey, DateTime, ARRAY
from sqlalchemy.orm import relationship

from app.database import Base


class History(Base):
    __tablename__ = 'history'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    diagnosis_ids = Column(ARRAY[Integer])
    created_at = Column(DateTime, default=datetime)

    user = relationship("User", back_populates="history")
