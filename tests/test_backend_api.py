"""API tests for session listing, retrieval, media ranges, and imports."""

import io
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend import main
from backend.database import Base, get_db
from backend.storage import LocalStorage


@pytest.fixture
def client(tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    store = LocalStorage(tmp_path / "media")

    def test_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[get_db] = test_db
    main.app.dependency_overrides[main.get_storage] = lambda: store
    with TestClient(main.app) as test_client:
        yield test_client
    main.app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def import_payload(client, *, session_id="test-session", media=b"abcdef"):
    payload = {
        "id": session_id,
        "title": "Test meeting",
        "segments": [{"t": 0, "speaker": "SPEAKER_01", "tx": "Hello"}],
    }
    response = client.post(
        "/sessions/import",
        data={"session_json": json.dumps(payload)},
        files={"media": ("meeting.wav", io.BytesIO(media), "audio/wav")},
    )
    assert response.status_code == 201, response.text
    return payload


def test_list_sessions_and_get_one(client):
    assert client.get("/sessions").json() == []
    payload = import_payload(client)
    listed = client.get("/sessions").json()
    assert [item["id"] for item in listed] == [payload["id"]]
    detail = client.get(f"/sessions/{payload['id']}")
    assert detail.status_code == 200
    assert detail.json()["segments"] == payload["segments"]
    assert client.get("/sessions/not-found").status_code == 404


def test_media_range_and_unsatisfied_range(client):
    payload = import_payload(client)
    url = f"/sessions/{payload['id']}/media"
    response = client.get(url, headers={"Range": "bytes=1-3"})
    assert response.status_code == 206
    assert response.content == b"bcd"
    assert response.headers["content-range"] == "bytes 1-3/6"
    assert response.headers["accept-ranges"] == "bytes"
    assert client.get(url, headers={"Range": "bytes=9-10"}).status_code == 416


def test_import_rejects_invalid_payload_and_duplicates(client):
    assert client.post("/sessions/import", data={"session_json": "{"}).status_code == 400
    assert client.post(
        "/sessions/import", data={"session_json": json.dumps({"id": "x", "title": "No segments"})}
    ).status_code == 422
    import_payload(client, media=b"")
    duplicate = client.post(
        "/sessions/import",
        data={"session_json": json.dumps({"id": "test-session", "title": "Again", "segments": []})},
    )
    assert duplicate.status_code == 409
