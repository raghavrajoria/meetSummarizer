"""Alembic creates/adopts SQLite schemas and emits PostgreSQL-compatible DDL."""
from pathlib import Path
import io
import pytest
from alembic import command
from alembic.config import Config
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, select
from backend.database import Base, database_url
from backend.models import MeetingSession

ROOT = Path(__file__).resolve().parents[1]

def config():
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend" / "migrations"))
    return cfg

@pytest.mark.parametrize("adopt_existing", [False, True])
def test_first_migration_and_downgrade_preserve_meetings(tmp_path, monkeypatch, adopt_existing):
    url = f"sqlite:///{(tmp_path / 'migration.sqlite3').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    engine = create_engine(url)
    try:
        if adopt_existing:
            MeetingSession.__table__.create(engine)
            with engine.begin() as connection:
                connection.execute(MeetingSession.__table__.insert(), {
                    "id": "existing", "title": "Existing", "group_name": "", "date": "", "payload": {"segments": []}})
        command.upgrade(config(), "head")
        assert {"meeting_sessions", "jobs", "alembic_version"} <= set(inspect(engine).get_table_names())
        with engine.connect() as connection:
            assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
            if adopt_existing:
                assert connection.scalar(select(MeetingSession.id)) == "existing"
        command.downgrade(config(), "base")
        assert "jobs" not in inspect(engine).get_table_names()
        assert "meeting_sessions" in inspect(engine).get_table_names()
        command.upgrade(config(), "head")
        assert "jobs" in inspect(engine).get_table_names()
    finally:
        engine.dispose()

def test_postgres_driver_and_offline_migration(monkeypatch):
    assert database_url("postgres://user:password@localhost/db") == "postgresql+psycopg://user:password@localhost/db"
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/indicmeet")
    cfg = config()
    cfg.output_buffer = io.StringIO()
    command.upgrade(cfg, "head", sql=True)
    sql = cfg.output_buffer.getvalue()
    assert "CREATE TABLE meeting_sessions" in sql
    assert "CREATE TABLE jobs" in sql
    assert "FOREIGN KEY(meeting_id) REFERENCES meeting_sessions" in sql
    assert "job_status" in sql and "stage_timings JSON" in sql
