"""Local orchestration for IndicMeet media, diarization, ASR, and summary stages."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
GPU_HINT = "run this stage on Kaggle, or pass --diar-csv"


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _json_read(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


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
) -> Any:
    started = time.monotonic()
    if cache.exists() and not force:
        result = _json_read(cache)
        print(f"[{name}] cache hit")
    else:
        result = make()
        _json_write(cache, result)
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

    def make() -> dict[str, Any]:
        result = _command(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(media)],
            stage="ffprobe",
        )
        probe = json.loads(result.stdout)
        streams = probe.get("streams") or []
        audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
        if not audio:
            raise RuntimeError(f"No audio stream found in media file: {media}")
        video = [stream for stream in streams if stream.get("codec_type") == "video"]
        return {"kind": "video" if video else "audio", "streams": streams, "format": probe.get("format", {})}

    return _cached_json("ffprobe", cache, str(media), force, make)


def _extract_audio(media: Path, out: Path, force: bool) -> Path:
    wav = out / "audio_16k_mono.wav"
    cache = out / "audio.json"
    started = time.monotonic()
    if cache.exists() and wav.exists() and not force:
        print("[ffmpeg] cache hit")
        _stage("ffmpeg", str(media), str(wav), started)
        return wav
    _command(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media),
            "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-c:a", "pcm_s16le", str(wav),
        ],
        stage="ffmpeg audio extraction",
    )
    _json_write(cache, {"wav": wav.name, "sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"})
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
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        annotation = pipeline(str(wav))
    except Exception as exc:
        raise RuntimeError(f"GPU diarization stage failed: {exc}") from exc
    rows = [
        {"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker), "idx": index}
        for index, (segment, _track, speaker) in enumerate(annotation.itertracks(yield_label=True))
    ]
    _write_diar_csv(csv_path, rows)
    return rows


def _attendee_data(path: Path | None) -> tuple[Any, Any]:
    if path is None:
        return None, None
    value = _json_read(path)
    if isinstance(value, dict):
        return value.get("attendees", value.get("participants", [])), value.get("aliases", {})
    return value, None


def _load_asr(path: Path) -> list[dict[str, Any]]:
    value = _json_read(path)
    if isinstance(value, dict):
        value = value.get("asr", value.get("transcript", value.get("segments")))
    if not isinstance(value, list):
        raise RuntimeError(f"ASR JSON must contain an array (or an object with asr/transcript/segments): {path}")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        start = item.get("start", item.get("t", 0))
        end = item.get("end", start)
        speaker = str(item.get("speaker") or item.get("speaker_id") or "SPEAKER_00")
        if not speaker.startswith("SPEAKER_"):
            speaker = f"SPEAKER_{speaker}"
        normalized.append({
            **item,
            "idx": item.get("idx", index),
            "start": float(start),
            "end": float(end),
            "speaker": speaker,
            "text": str(item.get("text", item.get("tx", "")) or ""),
            "lang": item.get("lang", "unknown"),
            "quality": item.get("quality", "accepted"),
        })
    return normalized


def _asr_stage(wav: Path, diar_csv: Path, supplied: Path | None, out: Path, force: bool) -> list[dict[str, Any]]:
    cache = out / "asr.json"

    def make() -> list[dict[str, Any]]:
        if supplied:
            return _load_asr(supplied)
        try:
            from .asr import IndicMeetASR
            turns = IndicMeetASR.build_turns(str(diar_csv))
            asr_engine = IndicMeetASR()
            result = asr_engine.transcribe_turns(str(wav), turns, out_json=None)
        except ImportError as exc:
            raise RuntimeError(f"ASR dependencies are missing ({exc.name}); {GPU_HINT}") from exc
        if not isinstance(result, list):
            raise RuntimeError("ASR stage returned an invalid result; expected a list of segments")
        return _load_asr_from_rows(result)

    # If an ASR source file is supplied, preserve its normalized result in the session cache.
    return _cached_json("ASR", cache, str(supplied or f"{wav} + {diar_csv}"), force, make)


def _load_asr_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, item in enumerate(rows):
        normalized.append({
            **item,
            "idx": item.get("idx", index),
            "start": float(item.get("start", 0)),
            "end": float(item.get("end", item.get("start", 0))),
            "speaker": str(item.get("speaker") or "SPEAKER_00"),
            "text": str(item.get("text") or ""),
            "lang": item.get("lang", "unknown"),
            "quality": item.get("quality", "accepted"),
        })
    return normalized


def _load_dotenv() -> None:
    if os.environ.get("GROQ_API_KEY"):
        return
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key == "GROQ_API_KEY" and value:
            os.environ.setdefault(key, value)


def _summary_stage(asr: list[dict[str, Any]], attendees: Any, aliases: Any, supplied: Path | None,
                   out: Path, force: bool) -> dict[str, Any]:
    cache = out / "summary.json"

    def make() -> dict[str, Any]:
        if supplied:
            value = _json_read(supplied)
            if not isinstance(value, dict):
                raise RuntimeError(f"Dry summary JSON must be an object: {supplied}")
            return value
        _load_dotenv()
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is missing; set it in the environment or project .env, or pass --dry-summary")
        from . import summary
        return summary.summarize(asr, attendees, aliases)

    return _cached_json("summary", cache, str(supplied or "ASR + attendees + aliases"), force, make)


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
            "join": _clock(min(float(item.get("start", 0)) for item in spans)),
            "leave": _clock(max(float(item.get("end", item.get("start", 0))) for item in spans)),
        })
    segment_ids = {item.get("idx", i): f"segment-{i + 1:04d}" for i, item in enumerate(asr)}
    segments = []
    for i, item in enumerate(asr):
        start = float(item.get("start", 0))
        segments.append({
            "id": segment_ids[item.get("idx", i)],
            "t": start,
            "end": float(item.get("end", start)),
            "time": _clock(start),
            "speaker": item.get("speaker", "SPEAKER_00"),
            "lang": item.get("lang", "unknown"),
            "tx": item.get("text", ""),
            "en": item.get("english", "") or (item.get("text", "") if item.get("lang") == "en" else ""),
            "quality": item.get("quality", "accepted"),
            "verified": item.get("quality") == "accepted",
            "reasons": item.get("reasons", []),
        })

    def evidence(item: dict[str, Any]) -> list[dict[str, Any]]:
        refs = item.get("ev", [])
        result = []
        for ref in refs if isinstance(refs, list) else []:
            matched = next((s for s in segments if s["id"] == segment_ids.get(ref) or s["t"] == ref), None)
            if matched:
                result.append({"segmentId": matched["id"], "t": matched["t"], "time": matched["time"]})
        return result

    def summary_items(key: str, label: str) -> list[dict[str, Any]]:
        items = summary.get(key, []) or []
        out_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = item.get(label) or item.get("text") or item.get("point") or item.get("task") or item.get("item") or ""
            ev = evidence(item)
            out_items.append({"text": text, "verified": not bool(item.get("unverified")), "evidence": ev,
                              **({"owner": item.get("owner")} if "owner" in item else {}),
                              **({"due": item.get("due")} if "due" in item else {})})
        return out_items

    intelligence = {
        "discussed": summary.get("overview", ""),
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
        "transcript": asr,
        "segments": segments,
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
            timeout=120,
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
    out = args.out.expanduser().resolve() if args.out else (ROOT / "data" / "sessions" / args.session_id).resolve()
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

    diar = _cached_json("diarization", diar_cache, str(supplied_diar or wav), args.force, make_diar)
    if not diar_csv_path.exists():
        _write_diar_csv(diar_csv_path, diar)

    asr = _asr_stage(wav, diar_csv_path, supplied_asr, out, args.force)
    attendees, aliases = _attendee_data(supplied_attendees)
    summary = _summary_stage(asr, attendees, aliases, dry_summary, out, args.force)

    destination_media = out / f"media{media.suffix.lower()}"
    media_cache = out / "media.json"
    started = time.monotonic()
    if not destination_media.exists() or args.force:
        shutil.copy2(media, destination_media)
        _json_write(media_cache, {"source": str(media), "stored": destination_media.name})
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
    parser.add_argument("--force", action="store_true", help="rerun and replace cached stage outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
