"""Local orchestration for IndicMeet media, diarization, ASR, and summary stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .asr_provider import get_asr_provider, validate_asr_rows
from .settings import get_settings

GPU_HINT = "run diarization on a GPU, or pass --diar-csv"
CACHE_VERSION_FFPROBE = 1
CACHE_VERSION_AUDIO = 1
CACHE_VERSION_DIARIZATION = 1
CACHE_VERSION_ASR = 1
CACHE_VERSION_SUMMARY = 1
CACHE_VERSION_SESSION = 1
CACHE_VERSION_MEDIA_COPY = 1


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _json_read(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _input_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_cache_key(input_files: list[Path], settings: dict[str, Any], cache_version: int) -> str:
    inputs = [{"sha256": _input_sha256(path)} for path in input_files]
    payload = {"cache_version": cache_version, "inputs": inputs, "settings": settings}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cache_metadata_path(cache: Path) -> Path:
    return cache.with_name(cache.name + ".cache.json")


def _cache_matches(cache: Path, key: str) -> bool:
    metadata = _cache_metadata_path(cache)
    if not cache.exists() or not metadata.exists():
        return False
    try:
        return _json_read(metadata).get("key") == key
    except (OSError, AttributeError, json.JSONDecodeError):
        return False


def _write_cache_metadata(cache: Path, key: str) -> None:
    _json_write(_cache_metadata_path(cache), {"key": key})


def _stage(name: str, source: str, destination: str, started: float) -> None:
    print(f"[{name}] input: {source}")
    print(f"[{name}] output: {destination}")
    print(f"[{name}] elapsed: {time.monotonic() - started:.2f}s")


def _cached_json(
    name: str,
    cache: Path,
    source: str,
    force: bool,
    make: Callable[[], Any],
    *,
    input_files: list[Path],
    settings: dict[str, Any],
    cache_version: int,
) -> Any:
    started = time.monotonic()
    key = _stage_cache_key(input_files, settings, cache_version)
    if _cache_matches(cache, key) and not force:
        result = _json_read(cache)
        print(f"[{name}] cache hit")
    else:
        result = make()
        _json_write(cache, result)
        _write_cache_metadata(cache, key)
    _stage(name, source, str(cache), started)
    return result


def _command(command: list[str], *, stage: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{stage}: required executable {command[0]!r} was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"{stage} failed: {details}") from exc
    return result


def _probe(media: Path, out: Path, force: bool) -> dict[str, Any]:
    cache = out / "ffprobe.json"
    settings = get_settings()

    def make() -> dict[str, Any]:
        result = _command(
            [settings.ffprobe_binary, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(media)],
            stage="ffprobe",
        )
        probe = json.loads(result.stdout)
        streams = probe.get("streams") or []
        audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
        if not audio:
            raise RuntimeError(f"No audio stream found in media file: {media}")
        video = [stream for stream in streams if stream.get("codec_type") == "video"]
        return {"kind": "video" if video else "audio", "streams": streams, "format": probe.get("format", {})}

    return _cached_json("ffprobe", cache, str(media), force, make,
                        input_files=[media], settings={"binary": settings.ffprobe_binary,
                        "args": ["-v", "error", "-show_streams", "-show_format", "-of", "json"]},
                        cache_version=CACHE_VERSION_FFPROBE)


def _extract_audio(media: Path, out: Path, force: bool) -> Path:
    wav = out / "audio_16k_mono.wav"
    cache = out / "audio.json"
    app_settings = get_settings()
    cache_settings = {"binary": app_settings.ffmpeg_binary, "args": ["-vn", "-ac", "1", "-ar", "16000",
                       "-sample_fmt", "s16", "-c:a", "pcm_s16le"]}
    cache_key = _stage_cache_key([media], cache_settings, CACHE_VERSION_AUDIO)
    started = time.monotonic()
    if wav.exists() and _cache_matches(cache, cache_key) and not force:
        print("[ffmpeg] cache hit")
        _stage("ffmpeg", str(media), str(wav), started)
        return wav
    _command(
        [
            app_settings.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-y", "-i", str(media),
            "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-c:a", "pcm_s16le", str(wav),
        ],
        stage="ffmpeg audio extraction",
    )
    _json_write(cache, {"wav": wav.name, "sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"})
    _write_cache_metadata(cache, cache_key)
    _stage("ffmpeg", str(media), str(wav), started)
    return wav


def _read_diar_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"start", "end"}.issubset({field.lower() for field in reader.fieldnames}):
            raise RuntimeError(f"Diarization CSV must have start and end columns: {path}")
        for index, raw in enumerate(reader):
            row = {str(k).lower(): v for k, v in raw.items() if k is not None}
            speaker = row.get("speaker") or row.get("label") or row.get("speaker_id") or "SPEAKER_00"
            rows.append({"start": float(row["start"]), "end": float(row["end"]), "speaker": str(speaker), "idx": index})
    return rows


def _write_diar_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["start", "end", "speaker"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in ("start", "end", "speaker")} for row in rows)


def _gpu_diarize(wav: Path, csv_path: Path) -> list[dict[str, Any]]:
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(f"Diarization dependencies are missing ({exc.name}); {GPU_HINT}") from exc
    app_settings = get_settings()
    token = app_settings.hf_token
    if not torch.cuda.is_available():
        raise RuntimeError("Diarization requires the teammate GPU host; pass --diar-csv")
    try:
        pipeline = Pipeline.from_pretrained(app_settings.pyannote_model, token=token)
        if app_settings.pyannote_device == "cuda" or (app_settings.pyannote_device == "auto" and torch.cuda.is_available()):
            pipeline.to(torch.device("cuda"))
        elif app_settings.pyannote_device not in {"auto", "cpu"}:
            raise RuntimeError("PYANNOTE_DEVICE must be auto, cpu, or cuda")
        annotation = pipeline(str(wav))
    except Exception as exc:
        raise RuntimeError(f"GPU diarization stage failed: {exc}") from exc
    from .speakers import pyannote_turns
    rows = [{"start":t.start,"end":t.end,"speaker":t.speaker,"idx":i} for i,t in enumerate(pyannote_turns(annotation))]
    _write_diar_csv(csv_path, rows)
    return rows


def _attendee_data(path: Path | None) -> tuple[Any, Any]:
    if path is None:
        return None, None
    value = _json_read(path)
    if isinstance(value, dict):
        return value.get("attendees", value.get("participants", [])), value.get("aliases", {})
    return value, None


def _asr_stage(wav: Path, diar_csv: Path, supplied: Path | None, out: Path, force: bool) -> list[dict[str, Any]]:
    cache = out / "asr.json"
    app_settings = get_settings()
    mode = "import" if supplied else app_settings.asr_mode
    if mode == "import" and supplied is None:
        raise RuntimeError("ASR_MODE=import requires --asr-json")
    if mode == "remote" and not supplied and not app_settings.asr_service_url:
        raise RuntimeError("ASR_SERVICE_URL is required when ASR_MODE=remote")

    def make() -> list[dict[str, Any]]:
        if mode == "local":
            try:
                from .asr import IndicMeetASR
                turns = IndicMeetASR.build_turns(str(diar_csv))
                asr_engine = IndicMeetASR()
                result = asr_engine.transcribe_turns(str(wav), turns, out_json=None)
            except ImportError as exc:
                raise RuntimeError(f"ASR dependencies are missing ({exc.name})") from exc
            if not isinstance(result, list):
                raise RuntimeError("ASR stage returned an invalid result; expected a list of segments")
            return validate_asr_rows(result)
        provider = get_asr_provider(mode)
        return provider.transcribe(wav, asr_json_path=supplied)

    source_files = [supplied] if supplied else ([wav, diar_csv] if mode == "local" else [wav])
    stage_settings = {"mode": mode}
    if mode == "remote":
        stage_settings.update({"service_url": app_settings.asr_service_url,
                               "timeout_seconds": app_settings.asr_service_timeout_seconds})
    elif mode == "local":
        stage_settings.update({"whisper_model": app_settings.whisper_model,
                               "whisper_compute_type": app_settings.whisper_compute_type,
                               "whisper_beam_size": app_settings.whisper_beam_size,
                               "indicconformer_model": app_settings.indicconformer_model,
                               "mms_lid_model": app_settings.mms_lid_model,
                               "device": app_settings.asr_device,
                               "decoder": app_settings.asr_indic_decoder,
                               "max_indic_chunk_seconds": app_settings.asr_max_indic_chunk_seconds})
    return _cached_json("ASR", cache, str(supplied or wav), force, make,
                        input_files=source_files, settings=stage_settings,
                        cache_version=CACHE_VERSION_ASR)


def _summary_stage(asr: list[dict[str, Any]], attendees: Any, aliases: Any, supplied: Path | None,
                   out: Path, force: bool) -> dict[str, Any]:
    cache = out / "summary.json"
    app_settings = get_settings()

    def make() -> dict[str, Any]:
        if supplied:
            value = _json_read(supplied)
            if not isinstance(value, dict):
                raise RuntimeError(f"Dry summary JSON must be an object: {supplied}")
            return value
        if not app_settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing; set it in the environment or project .env, or pass --dry-summary")
        from . import summary
        return summary.summarize(asr, attendees, aliases)

    input_files = [out / "asr.json"]
    if supplied:
        input_files.append(supplied)
    stage_settings = {"model": app_settings.groq_model, "endpoint": app_settings.groq_api_url,
                      "temperature": app_settings.groq_temperature,
                      "request_timeout_seconds": app_settings.groq_request_timeout_seconds,
                      "max_wait_seconds": app_settings.groq_max_wait_seconds,
                      "summary_cache_version": CACHE_VERSION_SUMMARY, "strict_review": app_settings.strict_review,
                      "attendees": attendees, "aliases": aliases}
    return _cached_json("summary", cache, str(supplied or "ASR + attendees + aliases"), force, make,
                        input_files=input_files, settings=stage_settings,
                        cache_version=CACHE_VERSION_SUMMARY)


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _session_payload(session_id: str, title: str, kind: str, media_name: str,
                     asr: list[dict[str, Any]], summary: dict[str, Any], attendees: Any) -> dict[str, Any]:
    speaker_ids = list(dict.fromkeys(str(item.get("speaker") or "SPEAKER_00") for item in asr))
    attendee_names = attendees if isinstance(attendees, list) else []
    speakers = []
    for index, speaker_id in enumerate(speaker_ids):
        spans = [item for item in asr if str(item.get("speaker")) == speaker_id]
        speakers.append({
            "id": speaker_id,
            "name": attendee_names[index] if index < len(attendee_names) else speaker_id,
            "role": "Participant" if index < len(attendee_names) else "Unidentified speaker",
            "confidence": "unverified",
            "verified": False,
            "speaker_name_source": "attendee_list" if index < len(attendee_names) else "none",
            "join": _clock(min(float(item.get("start", 0)) for item in spans)),
            "leave": _clock(max(float(item.get("end", item.get("start", 0))) for item in spans)),
        })

    from .contract import canonicalize
    canonical = canonicalize(asr)
    segments = [{**item, "id": item["segment_id"], "t": item["start"], "time": _clock(item["start"]),
                 "lang": item["language"], "tx": item["text_native"],
                 "en": item["text_english"] or (item["text_native"] if item["language"] == "en" else ""),
                 "roman": item["text_roman"], "verified": item["quality"] == "accepted"} for item in canonical]
    by_id = {item["segment_id"]: item for item in segments}

    def evidence(item):
        refs = item.get("source_segment_ids", [])
        return [{"segmentId": ref, "t": by_id[ref]["start"], "time": by_id[ref]["time"]}
                for ref in refs if isinstance(ref, str) and ref in by_id]

    def summary_items(key, label):
        output = []
        for item in summary.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            ev = evidence(item)
            if not ev:
                continue
            output.append({"text": item.get(label) or item.get("text") or item.get("q") or "",
                           "source_segment_ids": [e["segmentId"] for e in ev],
                           "verified": not bool(item.get("unverified")), "evidence": ev,
                           **({"owner": item.get("owner"), "due": item.get("due"),
                               "owner_name_source": item.get("owner_name_source", "inferred" if item.get("owner") else "none"),
                               "owner_source_segment_ids": item.get("owner_source_segment_ids", [])} if "owner" in item else {})})
        return output

    intelligence = {
        "discussed": summary.get("overview", ""),
        "overviewClaims": [{**c, "evidence": evidence(c)} for c in summary.get("overview_claims", []) if evidence(c)],
        "keyDiscussion": summary_items("key_discussion", "point"),
        "decisions": summary_items("decisions", "decision"),
        "actionItems": summary_items("action_items", "task"),
        "followUps": summary_items("follow_ups", "item"),
        "questions": summary_items("questions", "question"),
        "concerns": summary_items("concerns", "concern"),
    }
    return {
        "id": session_id,
        "title": title,
        "group": "",
        "date": "",
        "dateLabel": "Date not recorded",
        "time": "",
        "kind": kind,
        "media": media_name,
        "status": "processed",
        "summary": summary.get("overview", ""),
        "summaryData": summary,
        "participants": speakers,
        "speakers": speakers,
        "transcript": canonical,
        "segments": canonical,
        "intelligence": intelligence,
    }


def _post_import(url: str, payload_path: Path, media_path: Path) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("HTTP import requires requests; install backend/requirements.txt") from exc
    with payload_path.open("r", encoding="utf-8") as session_file, media_path.open("rb") as media_file:
        response = requests.post(
            url,
            data={"session_json": session_file.read()},
            files={"media": (media_path.name, media_file)},
            timeout=get_settings().import_request_timeout_seconds,
        )
    if not response.ok:
        raise RuntimeError(f"Backend import failed ({response.status_code}): {response.text[:500]}")
    return response.json()


def run(args: argparse.Namespace) -> dict[str, Any]:
    media = args.media.expanduser().resolve()
    if not media.is_file():
        raise RuntimeError(f"Media file does not exist: {media}")
    if not args.session_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in args.session_id):
        raise RuntimeError("session-id may contain only letters, digits, dot, underscore, and hyphen")
    out = args.out.expanduser().resolve() if args.out else (get_settings().data_dir / "sessions" / args.session_id).resolve()
    out.mkdir(parents=True, exist_ok=True)
    supplied_diar = args.diar_csv.expanduser().resolve() if args.diar_csv else None
    supplied_asr = args.asr_json.expanduser().resolve() if args.asr_json else None
    supplied_attendees = args.attendees.expanduser().resolve() if args.attendees else None
    dry_summary = args.dry_summary.expanduser().resolve() if args.dry_summary else None

    probe = _probe(media, out, args.force)
    wav = _extract_audio(media, out, args.force)

    diar_cache = out / "diarization.json"
    diar_csv_path = out / "diarization.csv"

    def make_diar() -> list[dict[str, Any]]:
        if supplied_diar:
            rows = _read_diar_csv(supplied_diar)
            shutil.copy2(supplied_diar, diar_csv_path)
            return rows
        rows = _gpu_diarize(wav, diar_csv_path)
        return rows

    diar = _cached_json("diarization", diar_cache, str(supplied_diar or wav), args.force, make_diar,
                        input_files=[supplied_diar or wav],
                        settings={"mode": "import" if supplied_diar else "gpu",
                                  "model": get_settings().pyannote_model,
                                  "device": get_settings().pyannote_device},
                        cache_version=CACHE_VERSION_DIARIZATION)
    if not diar_csv_path.exists():
        _write_diar_csv(diar_csv_path, diar)

    asr = _asr_stage(wav, diar_csv_path, supplied_asr, out, args.force)
    attendees, aliases = _attendee_data(supplied_attendees)
    summary = _summary_stage(asr, attendees, aliases, dry_summary, out, args.force)

    destination_media = out / f"media{media.suffix.lower()}"
    media_cache = out / "media.json"
    media_key = _stage_cache_key([media], {"stored_name": destination_media.name}, CACHE_VERSION_MEDIA_COPY)
    started = time.monotonic()
    if not destination_media.exists() or not _cache_matches(media_cache, media_key) or args.force:
        shutil.copy2(media, destination_media)
        _json_write(media_cache, {"source": str(media), "stored": destination_media.name})
        _write_cache_metadata(media_cache, media_key)
    else:
        print("[media-copy] cache hit")
    _stage("media-copy", str(media), str(destination_media), started)

    session_cache = out / "session.json"
    session = _cached_json(
        "assemble-session", session_cache,
        f"{media} + diarization + ASR + summary",
        args.force,
        lambda: _session_payload(args.session_id, args.title or media.stem, probe["kind"], destination_media.name,
                                 asr, summary, attendees),
        input_files=[diar_cache, out / "asr.json", out / "summary.json", destination_media],
        settings={"session_id": args.session_id, "title": args.title or media.stem,
                  "kind": probe["kind"], "media_name": destination_media.name, "attendees": attendees},
        cache_version=CACHE_VERSION_SESSION,
    )
    if args.import_url:
        started = time.monotonic()
        response = _post_import(args.import_url, session_cache, destination_media)
        _stage("backend-import", f"{session_cache} + {destination_media}", args.import_url, started)
        print(f"[backend-import] result: {response.get('id', 'ok')}")
    return session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--title")
    parser.add_argument("--attendees", type=Path)
    parser.add_argument("--diar-csv", type=Path)
    parser.add_argument("--asr-json", type=Path)
    parser.add_argument("--dry-summary", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--import-url", help="POST the assembled session to this /sessions/import URL")
    parser.add_argument("--skip-review", action="store_true", help="Exclude review segments from enrichment and summary")
    parser.add_argument("--force", action="store_true", help="rerun and replace cached stage outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.skip_review:
        import os
        os.environ["STRICT_REVIEW"] = "true"
    try:
        run(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
