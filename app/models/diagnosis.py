from sqlalchemy import Column, Integer, String

from app.models.base import Base


class Diagnosis(Base):
    __tablename__ = "diagnosis"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    category = Column(String)
    short_name = Column(String)
    description = Column(String)
