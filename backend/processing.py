"""Worker-only processing; ASR providers remain the Milestone 1 contract."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from indicmeet.asr_provider import ImportAsrProvider, get_asr_provider, validate_asr_file
from indicmeet.pipeline import _attendee_data, _read_diar_csv, _session_payload
from indicmeet import summary
from indicmeet.settings import get_settings

from .media import probe_media, extract_audio


def validate_sidecars(files, store):
    if "asr_json" in files:
        validate_asr_file(store.path(files["asr_json"]))
    if "attendees_json" in files:
        path = store.path(files["attendees_json"])
        if not isinstance(json.loads(path.read_text(encoding="utf-8")), (list, dict)):
            raise ValueError("Invalid attendees")
        attendees, aliases = _attendee_data(path)
        if attendees is not None and (not isinstance(attendees, list) or not all(isinstance(x, str) for x in attendees)):
            raise ValueError("Invalid attendees")
        if aliases is not None and (not isinstance(aliases, dict) or not all(
            isinstance(k, str) and isinstance(v, list) and all(isinstance(x, str) for x in v)
            for k, v in aliases.items())):
            raise ValueError("Invalid aliases")
    if "diarization_csv" in files:
        for row in _read_diar_csv(store.path(files["diarization_csv"])):
            if not math.isfinite(row["start"]) or not math.isfinite(row["end"]) or row["start"] < 0 or row["end"] < row["start"]:
                raise ValueError("Invalid diarization span")


def process_meeting(record, store, stage):
    """stage(name, progress, callable) records both successes and failures."""
    settings = get_settings()
    files = record.payload["source_files"]
    recording = store.path(files["recording"])
    info = stage("probe", 5, lambda: probe_media(recording))
    attendees_path = store.path(files["attendees_json"]) if "attendees_json" in files else None
    attendees, aliases = stage("metadata", 10, lambda: _attendee_data(attendees_path))
    with TemporaryDirectory(prefix="indicmeet-worker-") as temp:
        if "asr_json" in files:
            # Imported transcripts require neither ffmpeg nor GPU processing.
            asr = stage("asr", 40, lambda: ImportAsrProvider().transcribe(
                recording, asr_json_path=store.path(files["asr_json"])))
        else:
            wav = stage("extract_audio", 20, lambda: extract_audio(recording, Path(temp) / "audio.wav"))
            asr = stage("asr", 40, lambda: get_asr_provider(settings.asr_mode).transcribe(wav))
        if "diarization_csv" in files:
            def assign_speakers():
                spans = _read_diar_csv(store.path(files["diarization_csv"]))
                for row in asr:
                    overlaps = [(max(0, min(row["end"], span["end"]) - max(row["start"], span["start"])), span) for span in spans]
                    if overlaps:
                        overlap, span = max(overlaps, key=lambda item: item[0])
                        if overlap > 0:
                            row["speaker"] = span["speaker"]
                            from indicmeet.contract import stable_id
                            row["segment_id"] = stable_id(row["start"], row["speaker"])
                return asr
            asr = stage("diarization", 55, assign_speakers)
        summarized = stage("summary", 70, lambda: summary.summarize(
            asr, attendees, aliases, api_key=settings.groq_api_key, log=lambda *args: None))
        payload = stage("assemble", 90, lambda: _session_payload(
            record.id, record.title, info["kind"], record.payload.get("original_filename", "recording"),
            asr, summarized, attendees))
        key = record.payload["storage_prefix"] + "/transcript.json"
        def write_transcript():
            store.path(key).write_text(json.dumps(asr, ensure_ascii=False, indent=2), encoding="utf-8")
        stage("transcript", 95, write_transcript)
        return {**payload, "transcript_key": key, "group": record.group_name, "date": record.date,
                "dateLabel": record.date or "Date not recorded"}
