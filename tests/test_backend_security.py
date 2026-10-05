"""Bearer, request correlation, and public health endpoint coverage."""
import logging
import pytest
from backend.logging_config import JsonFormatter, request_id

@pytest.mark.parametrize("header", [None, "Bearer wrong", "Basic test-token", "Bearer", "Bearer test-token extra"])
def test_auth_rejects_without_detail(api_env, header):
    headers = {} if header is None else {"Authorization": header}
    response = api_env.client.get("/meetings", headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers["x-request-id"]

@pytest.mark.parametrize("token", ["test-token", "second-token"])
def test_auth_accepts_configured_tokens(api_env, token):
    response = api_env.client.get("/meetings", headers={"Authorization": f"Bearer {token}", "X-Request-ID": "req-123"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"

@pytest.mark.parametrize("method,path", [("GET", "/sessions"), ("GET", "/docs"), ("GET", "/openapi.json"),
    ("POST", "/meetings"), ("GET", "/jobs/x"), ("POST", "/jobs/x/retry"), ("DELETE", "/meetings/x"),
    ("GET", "/meetings/x/media"), ("GET", "/meetings/x/transcript"), ("POST", "/sessions/import")])
def test_auth_covers_routes(api_env, method, path):
    assert api_env.client.request(method, path).status_code == 401

def test_no_configured_token_denies_all(api_env, monkeypatch):
    monkeypatch.setenv("API_TOKENS", ", ,")
    assert api_env.client.get("/meetings", headers={"Authorization": "Bearer test-token"}).status_code == 401

def test_health_and_readiness_are_public(api_env, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda binary: "fake-ffprobe")
    assert api_env.client.get("/healthz").json() == {"status": "ok"}
    assert api_env.client.get("/readyz").json() == {"status": "ready"}
    assert list(api_env.store.root.iterdir()) == []
    monkeypatch.setattr(shutil, "which", lambda binary: None)
    assert api_env.client.get("/readyz").status_code == 503
    assert api_env.client.get("/healthz").status_code == 200

def test_readiness_requires_migration(api_env, monkeypatch):
    import shutil
    from backend.models import Job
    monkeypatch.setattr(shutil, "which", lambda binary: "fake")
    Job.__table__.drop(api_env.engine)
    assert api_env.client.get("/readyz").json() == {"status": "not_ready"}
    assert api_env.client.get("/readyz").status_code == 503

def test_json_formatter_has_request_id_without_exception_text():
    import json
    token = request_id.set("log-request")
    try:
        record = logging.LogRecord("worker", logging.ERROR, "", 1, "job_failed", (), None)
        try:
            raise RuntimeError("Private transcript")
        except RuntimeError:
            import sys
            record.exc_info = sys.exc_info()
        result = JsonFormatter().format(record)
        assert json.loads(result)["request_id"] == "log-request"
        assert "Private transcript" not in result
    finally:
        request_id.reset(token)

def test_auth_compares_every_token_even_after_match(api_env, monkeypatch):
    from backend import security
    original = security.hmac.compare_digest
    calls = []
    def compare(candidate, configured):
        calls.append(configured)
        return original(candidate, configured)
    monkeypatch.setattr(security.hmac, "compare_digest", compare)
    response = api_env.client.get("/meetings", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert calls == [b"test-token", b"second-token"]

def test_unhandled_errors_are_generic_and_correlated(api_env, monkeypatch):
    from backend import main
    api_env.client.headers["Authorization"] = "Bearer test-token"
    def failure(path):
        raise OSError("Private transcript must never leak")
    monkeypatch.setattr(main, "probe_media", failure)
    response = api_env.client.post("/meetings", files={"recording": ("clip.wav", b"audio")},
                                   headers={"X-Request-ID": "error-request"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert response.headers["x-request-id"] == "error-request"
    assert list(api_env.store.root.iterdir()) == []
