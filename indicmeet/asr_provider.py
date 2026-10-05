"""Validated import and remote ASR providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from .settings import get_settings


class AsrRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    idx: int = Field(ge=0, strict=True)
    start: float = Field(ge=0, allow_inf_nan=False, strict=True)
    end: float = Field(ge=0, allow_inf_nan=False, strict=True)
    speaker: str = Field(min_length=1)
    text: str
    lang: str = Field(min_length=1)
    method: str = Field(min_length=1)
    quality: Literal["accepted", "review", "rejected"]
    reasons: list[str]
    duration: float = Field(ge=0, allow_inf_nan=False, strict=True)
    confidence: float | None = Field(default=None, allow_inf_nan=False, strict=True)
    words: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def end_not_before_start(self) -> "AsrRow":
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


_ROWS = TypeAdapter(list[AsrRow])


class AsrValidationError(ValueError):
    """ASR input is not a list of rows matching the service contract."""


def validate_asr_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list) and value and "segment_id" in value[0]:
        from .contract import canonicalize
        try:
            return canonicalize(value)
        except ValueError as exc:
            raise AsrValidationError("Invalid canonical ASR response") from exc
    if not isinstance(value, list):
        raise AsrValidationError("ASR JSON must be a list of row objects")
    try:
        rows = _ROWS.validate_python(value)
    except ValidationError as exc:
        problems = []
        for error in exc.errors(include_input=False):
            location = ".".join(str(part) for part in error["loc"])
            problems.append(f"{location}: {error['msg']}")
        raise AsrValidationError("Invalid ASR JSON: " + "; ".join(problems)) from None
    return [row.model_dump(exclude_unset=True) for row in rows]


def validate_asr_file(path: Path) -> list[dict[str, Any]]:
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AsrValidationError(f"Cannot read ASR JSON file {path}: {exc}") from exc
    from .contract import canonicalize
    try:
        return canonicalize(value)
    except (ValueError,TypeError) as exc:
        raise AsrValidationError("Invalid imported transcript") from exc


class AsrProvider(Protocol):
    def transcribe(self, audio_path: Path, *, asr_json_path: Path | None = None, turns: list[dict] | None = None) -> list[dict[str, Any]]: ...


class ImportAsrProvider:
    def transcribe(self, audio_path: Path, *, asr_json_path: Path | None = None, turns: list[dict] | None = None) -> list[dict[str, Any]]:
        del audio_path  # The uploaded recording is retained alongside its imported ASR rows.
        if asr_json_path is None:
            raise ValueError("Import ASR mode requires an asr.json file")
        from .contract import canonicalize
        return canonicalize(validate_asr_file(asr_json_path))


from .remote_asr import RemoteAsrProvider


def get_asr_provider(mode: str | None = None) -> AsrProvider:
    selected = mode or get_settings().asr_mode
    if selected == "import":
        return ImportAsrProvider()
    if selected == "remote":
        return RemoteAsrProvider()
    if selected == "local":
        return LocalGpuAsrProvider()
    raise ValueError("ASR provider mode must be import, remote, or local")

class LocalGpuAsrProvider:
    def transcribe(self,audio_path,*,asr_json_path=None,turns=None):
        from .asr import IndicMeetASR
        from .contract import canonicalize
        if turns is None: raise ValueError("Local GPU provider requires speaker turns")
        return canonicalize(IndicMeetASR().transcribe_turns(str(audio_path),turns))
