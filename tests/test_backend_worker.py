"""Processing selects the existing ASR provider and never runs GPU stages."""
import json
from types import SimpleNamespace
import pytest
from backend import processing, main
from backend.models import MeetingSession
from backend.worker import Worker

def test_uploaded_asr_skips_extraction_and_configured_provider(api_client, api_env, asr_rows, monkeypatch):
    response = api_client.post("/meetings", files={"recording": ("clip.wav", b"audio"),
        "asr_json": ("asr.json", json.dumps(asr_rows).encode()),
        "attendees_json": ("attendees.json", b'{"attendees":["Alice"],"aliases":{}}'),
        "diarization_csv": ("diarization.csv", b"start,end,speaker\n0,1,SPEAKER_02\n")})
    assert response.status_code == 202
    meeting = response.json()
    monkeypatch.setattr(processing, "probe_media", lambda path: {"kind": "audio", "duration": 1.0})
    def forbidden(*args, **kwargs):
        pytest.fail("Uploaded ASR must skip extraction and the configured provider")
    monkeypatch.setattr(processing, "extract_audio", forbidden)
    monkeypatch.setattr(processing, "get_asr_provider", forbidden)
    def summary(rows, attendees, aliases, **kwargs):
        assert attendees == ["Alice"] and aliases == {}
        assert rows[0]["speaker"] == "SPEAKER_02"
        assert kwargs["log"]("Private transcript") is None
        return {"overview": "Summary"}
    monkeypatch.setattr(processing.summary, "summarize", summary)
    worker = Worker(api_env.queue, api_env.sessions, api_env.store)
    assert worker.run_once()
    job = api_client.get(f"/jobs/{meeting['job_id']}").json()
    assert job["status"] == "done"
    assert "extract_audio" not in job["stage_timings"]
    assert {"probe", "metadata", "asr", "diarization", "summary", "assemble", "transcript"} <= set(job["stage_timings"])
    transcript = api_client.get(f"/meetings/{meeting['id']}/transcript").json()
    assert transcript[0]["speaker"] == "SPEAKER_02"
    artifact = api_env.store.path(meeting["id"] + "/transcript.json")
    assert json.loads(artifact.read_text(encoding="utf-8")) == transcript
    assert api_client.delete(f"/meetings/{meeting['id']}").status_code == 204
    assert not artifact.exists()

def test_without_asr_calls_configured_provider(api_client, api_env, asr_rows, monkeypatch):
    monkeypatch.setenv("ASR_MODE", "remote")
    response = api_client.post("/meetings", files={"recording": ("clip.wav", b"audio")})
    assert response.status_code == 202
    selected = []
    monkeypatch.setattr(processing, "probe_media", lambda path: {"kind": "audio"})
    def extract(source, destination):
        destination.write_bytes(b"wav")
        return destination
    monkeypatch.setattr(processing, "extract_audio", extract)
    class Provider:
        def transcribe(self, audio_path, *, asr_json_path=None, turns=None):
            assert audio_path.read_bytes() == b"wav" and asr_json_path is None
            return asr_rows
    def provider(mode):
        selected.append(mode)
        return Provider()
    monkeypatch.setattr(processing, "get_asr_provider", provider)
    monkeypatch.setattr(processing.summary, "summarize", lambda *args, **kwargs: {"overview": "Summary"})
    assert Worker(api_env.queue, api_env.sessions, api_env.store).run_once()
    assert selected == ["remote"]
    job = api_client.get(f"/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "done" and "extract_audio" in job["stage_timings"]
