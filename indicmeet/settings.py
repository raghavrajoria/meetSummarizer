"""Environment-backed configuration shared by the IndicMeet modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _value(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value.strip() else default


def load_environment() -> None:
    """Load project .env values without overriding already-exported variables."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip("\"'")
        if name and value:
            os.environ.setdefault(name, value)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    media_dir: Path
    database_url: str
    cors_origins: tuple[str, ...]
    ffprobe_binary: str
    ffmpeg_binary: str
    hf_token: str | None
    pyannote_model: str
    pyannote_device: str
    whisper_model: str
    whisper_compute_type: str
    whisper_beam_size: int
    indicconformer_model: str
    mms_lid_model: str
    asr_device: str
    asr_indic_decoder: str
    asr_max_indic_chunk_seconds: float
    asr_mode: str
    asr_service_url: str | None
    asr_service_token: str | None
    asr_service_timeout_seconds: float
    import_request_timeout_seconds: float
    groq_api_key: str | None
    groq_api_url: str
    groq_model: str
    groq_temperature: float
    groq_request_timeout_seconds: float
    groq_max_wait_seconds: int
    llm_cache_dir: Path
    api_tokens: tuple[str, ...]
    max_upload_mb: float
    media_timeout_seconds: float
    worker_poll_seconds: float
    worker_lease_seconds: float
    retention_days: int


def get_settings() -> Settings:
    load_environment()
    data_dir = Path(_value("INDICMEET_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
    cors = _value(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8001,http://127.0.0.1:8001",
    )
    asr_mode = _value("ASR_MODE", "import").strip().lower()
    if asr_mode not in {"import", "remote", "local"}:
        raise ValueError("ASR_MODE must be 'import', 'remote', or 'local'")
    return Settings(
        data_dir=data_dir,
        media_dir=Path(_value("INDICMEET_MEDIA_DIR", str(data_dir / "media"))).expanduser(),
        database_url=_value("DATABASE_URL", f"sqlite:///{(data_dir / 'meetSummerizer.sqlite3').as_posix()}"),
        cors_origins=tuple(origin.strip() for origin in cors.split(",") if origin.strip()),
        ffprobe_binary=_value("FFPROBE_BINARY", "ffprobe"),
        ffmpeg_binary=_value("FFMPEG_BINARY", "ffmpeg"),
        hf_token=_value("HF_TOKEN", "") or _value("HUGGINGFACE_TOKEN", "") or None,
        pyannote_model=_value("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"),
        pyannote_device=_value("PYANNOTE_DEVICE", "auto").lower(),
        whisper_model=_value("WHISPER_MODEL", "large-v3"),
        whisper_compute_type=_value("WHISPER_COMPUTE_TYPE", "float16"),
        whisper_beam_size=int(_value("WHISPER_BEAM_SIZE", "5")),
        indicconformer_model=_value("INDICCONFORMER_MODEL", "ai4bharat/indic-conformer-600m-multilingual"),
        mms_lid_model=_value("MMS_LID_MODEL", "facebook/mms-lid-256"),
        asr_device=_value("ASR_DEVICE", "cuda:0"),
        asr_indic_decoder=_value("ASR_INDIC_DECODER", "ctc"),
        asr_max_indic_chunk_seconds=float(_value("ASR_MAX_INDIC_CHUNK_SECONDS", "8.0")),
        asr_mode=asr_mode,
        asr_service_url=_value("ASR_SERVICE_URL", "") or None,
        asr_service_token=_value("ASR_SERVICE_TOKEN", "") or None,
        asr_service_timeout_seconds=float(_value("ASR_SERVICE_TIMEOUT_SECONDS", "120")),
        import_request_timeout_seconds=float(_value("IMPORT_REQUEST_TIMEOUT_SECONDS", "120")),
        groq_api_key=_value("GROQ_API_KEY", "") or None,
        groq_api_url=_value("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"),
        groq_model=_value("GROQ_MODEL", "openai/gpt-oss-20b"),
        groq_temperature=float(_value("GROQ_TEMPERATURE", "0.2")),
        groq_request_timeout_seconds=float(_value("GROQ_REQUEST_TIMEOUT_SECONDS", "120")),
        groq_max_wait_seconds=int(_value("GROQ_MAX_WAIT_SECONDS", "120")),
        llm_cache_dir=Path(_value("INDICMEET_LLM_CACHE", str(PROJECT_ROOT / "llm_cache"))).expanduser(),
        api_tokens=tuple(token.strip() for token in _value("API_TOKENS", "").split(",") if token.strip()),
        max_upload_mb=float(_value("MAX_UPLOAD_MB", "512")),
        media_timeout_seconds=float(_value("MEDIA_TIMEOUT_SECONDS", "300")),
        worker_poll_seconds=float(_value("WORKER_POLL_SECONDS", "2")),
        worker_lease_seconds=float(_value("WORKER_LEASE_SECONDS", "180")),
        retention_days=int(_value("RETENTION_DAYS", "0")),
    )
