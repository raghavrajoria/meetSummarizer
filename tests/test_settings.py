import pytest

from indicmeet.settings import get_settings


def test_settings_read_environment_values(monkeypatch, tmp_path):
    monkeypatch.setenv("INDICMEET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ASR_MODE", "remote")
    monkeypatch.setenv("ASR_SERVICE_URL", "http://asr.example/transcribe")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/meetings")
    monkeypatch.setenv("CORS_ORIGINS", "https://uat.example, https://prod.example")

    settings = get_settings()

    assert settings.data_dir == tmp_path / "data"
    assert settings.asr_mode == "remote"
    assert settings.asr_service_url == "http://asr.example/transcribe"
    assert settings.groq_model == "test-model"
    assert settings.database_url == "postgresql://example/meetings"
    assert settings.cors_origins == ("https://uat.example", "https://prod.example")


def test_legacy_local_asr_mode_remains_selectable(monkeypatch):
    monkeypatch.setenv("ASR_MODE", "local")
    assert get_settings().asr_mode == "local"


def test_settings_reject_unknown_asr_mode(monkeypatch):
    monkeypatch.setenv("ASR_MODE", "gpu")
    with pytest.raises(ValueError, match="ASR_MODE must be 'import', 'remote', or 'local'"):
        get_settings()
