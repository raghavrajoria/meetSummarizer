"""Persistence models."""

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class MeetingSession(Base):
    __tablename__ = "meeting_sessions"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    group_name: Mapped[str] = mapped_column(String(300), default="")
    date: Mapped[str] = mapped_column(String(32), default="")
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    media_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
