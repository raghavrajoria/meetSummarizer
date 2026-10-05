"""Alembic uses the API's environment settings, without logging a credential URL."""
from alembic import context
from sqlalchemy import create_engine, pool
from backend.database import Base, database_url
from backend import models
from indicmeet.settings import get_settings

url = database_url(get_settings().database_url)
if context.is_offline_mode():
    context.configure(url=url, target_metadata=Base.metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()
