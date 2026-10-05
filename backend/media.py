"""Validate real container contents without trusting upload metadata."""

import json
import math
import subprocess
from pathlib import Path

from indicmeet.settings import get_settings


class MediaValidationError(ValueError):
    pass


def probe_media(path: Path) -> dict:
    settings = get_settings()
    try:
        result = subprocess.run([
            settings.ffprobe_binary, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path),
        ], capture_output=True, text=True, timeout=settings.media_timeout_seconds, check=True)
        info = json.loads(result.stdout)
        streams = info.get("streams", [])
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        durations = [info.get("format", {}).get("duration")] + [s.get("duration") for s in audio]
        duration = None
        for value in durations:
            try:
                candidate = float(value)
            except (ValueError, TypeError):
                continue
            if math.isfinite(candidate) and candidate > 0:
                duration = candidate
                break
        if not audio or duration is None:
            raise MediaValidationError("Audio stream and duration required")
        video = any(s.get("codec_type") == "video" for s in streams)
        fmt = info.get("format", {}).get("format_name", "").split(",")
        content_type = "application/octet-stream"
        if "wav" in fmt:
            content_type = "audio/wav"
        elif "mp3" in fmt:
            content_type = "audio/mpeg"
        elif "mov" in fmt or "mp4" in fmt:
            content_type = "video/mp4" if video else "audio/mp4"
        elif "webm" in fmt or "matroska" in fmt:
            content_type = "video/webm" if video else "audio/webm"
        elif "ogg" in fmt:
            content_type = "video/ogg" if video else "audio/ogg"
        elif "flac" in fmt:
            content_type = "audio/flac"
        return {"duration": duration, "kind": "video" if video else "audio", "media_type": content_type}
    except FileNotFoundError:
        raise RuntimeError("Media tool unavailable") from None
    except (subprocess.SubprocessError, ValueError, TypeError, AttributeError, StopIteration, KeyError):
        raise MediaValidationError("Media must contain audio and a duration") from None


def extract_audio(source: Path, destination: Path) -> Path:
    settings = get_settings()
    subprocess.run([settings.ffmpeg_binary, "-nostdin", "-v", "error", "-y", "-i", str(source),
                    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)],
                   capture_output=True, timeout=settings.media_timeout_seconds, check=True)
    return destination
