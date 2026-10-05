"""Shared backend fixtures; all patches/overrides are scoped and restored."""
from types import SimpleNamespace
import pytest

@pytest.fixture
def api_env(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend import main
    from backend.database import Base, get_db
    from backend.jobs import DatabaseJobQueue
    from backend.storage import LocalStorage

    monkeypatch.setenv("API_TOKENS", "test-token,second-token")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    monkeypatch.setenv("WORKER_LEASE_SECONDS", "180")
    engine = create_engine(f"sqlite:///{(tmp_path / 'api.sqlite3').as_posix()}",
                           connect_args={"check_same_thread": False})
    sessions = sessionmaker(bind=engine, autoflush=False)
    Base.metadata.create_all(engine)
    from backend import security
    monkeypatch.setattr(security, "SessionLocal", sessions)
    monkeypatch.setenv("MEDIA_SIGNING_SECRET", "test-signing-secret-at-least-32-characters")
    store = LocalStorage(tmp_path / "media")
    queue = DatabaseJobQueue(sessions)
    def database():
        with sessions() as db:
            yield db
    previous = dict(main.app.dependency_overrides)
    main.app.dependency_overrides.update({get_db: database, main.get_storage: lambda: store,
                                          main.get_queue: lambda: queue})
    try:
        with TestClient(main.app) as client:
            yield SimpleNamespace(client=client, sessions=sessions, store=store, queue=queue, engine=engine)
    finally:
        main.app.dependency_overrides.clear()
        main.app.dependency_overrides.update(previous)
        engine.dispose()

@pytest.fixture
def api_client(api_env, monkeypatch):
    from backend import main
    monkeypatch.setattr(main, "probe_media", lambda path: {
        "duration": 1.0, "kind": "audio", "media_type": "audio/wav"})
    api_env.client.headers["Authorization"] = "Bearer test-token"
    return api_env.client

@pytest.fixture
def asr_rows():
    return [{"idx": 0, "start": 0.0, "end": 1.0, "speaker": "SPEAKER_01", "text": "Private transcript",
             "lang": "en", "method": "import", "quality": "accepted", "reasons": [],
             "duration": 1.0, "confidence": None}]
