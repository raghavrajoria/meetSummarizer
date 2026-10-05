import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from indicmeet.asr_provider import (
    AsrValidationError,
    ImportAsrProvider,
    RemoteAsrProvider,
    get_asr_provider,
    validate_asr_rows,
)


ROW = {
    "idx": 0,
    "start": 0.25,
    "end": 1.5,
    "speaker": "SPEAKER_00",
    "text": "નમસ્તે",
    "lang": "gu",
    "method": "indicconformer_gu",
    "quality": "accepted",
    "reasons": [],
    "duration": 1.25,
    "confidence": None,
}


def test_asr_rows_validate_and_keep_optional_words():
    row = {**ROW, "words": [{"start": 0.25, "end": 0.7, "text": "નમસ્તે"}]}
    assert validate_asr_rows([row]) == [row]


@pytest.mark.parametrize(
    ("change", "expected"),
    [({"quality": "accept"}, "0.quality"), ({"end": -1.0}, "0.end"), ({"text": 12}, "0.text"),
     ({"confidence": "unknown"}, "0.confidence"), ({"start": 3.0, "end": 2.0}, "0")],
)
def test_asr_rows_reject_malformed_values(change, expected):
    with pytest.raises(AsrValidationError) as error:
        validate_asr_rows([{**ROW, **change}])
    assert expected in str(error.value)
    assert ROW["text"] not in str(error.value)


def test_import_provider_loads_and_validates_file(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    transcript = tmp_path / "meeting_asr.json"
    transcript.write_text(json.dumps([ROW]), encoding="utf-8")
    assert ImportAsrProvider().transcribe(audio, asr_json_path=transcript) == __import__("indicmeet.contract", fromlist=["canonicalize"]).canonicalize([ROW])


def test_provider_factory_supports_import_and_remote(monkeypatch):
    monkeypatch.setenv("ASR_SERVICE_URL", "http://127.0.0.1:9999/transcribe")
    assert isinstance(get_asr_provider("import"), ImportAsrProvider)
    assert isinstance(get_asr_provider("remote"), RemoteAsrProvider)


def test_import_provider_rejects_missing_asr_file(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio")
    with pytest.raises(ValueError, match="requires an asr.json"):
        ImportAsrProvider().transcribe(audio)


def test_remote_provider_posts_audio_and_validates_response(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        auth = None

        def do_POST(self):
            type(self).auth = self.headers.get("Authorization")
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.dumps([ROW]).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio fixture")
    try:
        provider = RemoteAsrProvider(
            f"http://127.0.0.1:{server.server_port}/transcribe", token="test-token", timeout=2,
        )
        assert provider.transcribe(audio) == __import__("indicmeet.contract", fromlist=["canonicalize"]).canonicalize([ROW])
        assert Handler.auth == "Bearer test-token"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
