"""Serialize migration startup using a Postgres advisory lock."""
import os,subprocess,sys
from sqlalchemy import text
def main():
    from .config_guard import validate_app_configuration
    validate_app_configuration()
    from .database import engine
    with engine.connect() as connection:
        postgres=connection.dialect.name=="postgresql"
        if postgres:connection.execute(text("SELECT pg_advisory_lock(170105)") )
        try:
            from alembic.config import Config
            from alembic import command
            command.upgrade(Config("alembic.ini"),"head")
        finally:
            if postgres:connection.execute(text("SELECT pg_advisory_unlock(170105)"))
    role=os.environ.get("SERVICE_ROLE","api")
    arguments=[sys.executable,"-m","backend.worker"] if role=="worker" else [sys.executable,"-m","uvicorn","backend.main:app","--host","0.0.0.0","--port","8000","--no-access-log"]
    os.execv(sys.executable,arguments)
if __name__=="__main__":main()
