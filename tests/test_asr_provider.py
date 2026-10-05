import json

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
    from scripts.fake_asr_server import fake_server
    audio=tmp_path/"meeting.wav";audio.write_bytes(b"audio fixture")
    with fake_server(row=ROW) as (url,handler):
        provider=RemoteAsrProvider(url,token="test-token",timeout=2,poll_interval=.001)
        rows=provider.transcribe(audio)
        assert rows[0]["text_native"]==ROW["text"]
        assert rows[0]["asr"]["model_version"]=="fake-asr-v2"
        assert handler.auth=="Bearer test-token" and handler.posts==1
