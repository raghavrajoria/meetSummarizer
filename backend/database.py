"""SQLite database setup for the local IndicMeet service."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from indicmeet.settings import get_settings


_settings = get_settings()
DATA_DIR = _settings.data_dir
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR = _settings.media_dir
DATABASE_URL = _settings.database_url

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite:") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
