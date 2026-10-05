"""SQLite database setup for the local IndicMeet service."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from indicmeet.settings import get_settings


_settings = get_settings()
DATA_DIR = _settings.data_dir
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR = _settings.media_dir
def database_url(value: str) -> str:
    """Use the pinned psycopg 3 driver for ordinary Postgres URLs."""
    if value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://"):]
    return value.replace("postgresql://", "postgresql+psycopg://", 1)


DATABASE_URL = database_url(_settings.database_url)

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
