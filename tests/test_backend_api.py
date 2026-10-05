"""API tests for session listing, retrieval, media ranges, and imports."""

import io
import json

import pytest


@pytest.fixture
def client(api_client):
    return api_client


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
    assert detail.json()["segments"][0]["text_native"] == payload["segments"][0]["tx"]
    assert detail.json()["segments"][0]["segment_id"].startswith("seg_")
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
