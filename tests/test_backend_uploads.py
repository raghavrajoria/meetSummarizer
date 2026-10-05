"""Streaming limits, generated storage names and actual ffprobe validation."""
import asyncio
import json
from types import SimpleNamespace
import pytest
from backend import media, main
from backend.uploads import receive_upload, UploadTooLarge, UploadError
from backend.storage import LocalStorage

class StreamRequest:
    def __init__(self, chunks):
        self.headers = {"content-type": "multipart/form-data; boundary=boundary"}
        self.chunks = chunks
        self.consumed = 0
    async def stream(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

def part(name="recording", filename="clip.wav", body=b"audio"):
    return (f'--boundary\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            'Content-Type: text/plain\r\n\r\n').encode() + body

def test_limit_aborts_mid_stream_and_deletes_partial(tmp_path):
    store = LocalStorage(tmp_path / "media")
    first = part(body=b"already written")
    request = StreamRequest([first, b"x" * 128, b"never consumed", b"\r\n--boundary--\r\n"])
    with pytest.raises(UploadTooLarge):
        asyncio.run(receive_upload(request, store, max_bytes=len(first) + 64,
                                   file_fields={"recording"}, text_fields={"title"}))
    assert request.consumed == 2
    assert list(store.root.iterdir()) == []

def test_malformed_body_cleans_partial_file(tmp_path):
    store = LocalStorage(tmp_path / "media")
    request = StreamRequest([part(body=b"unfinished")])
    with pytest.raises(UploadError):
        asyncio.run(receive_upload(request, store, max_bytes=4096, file_fields={"recording"}, text_fields=set()))
    assert list(store.root.iterdir()) == []

def test_filename_is_metadata_only(tmp_path):
    store = LocalStorage(tmp_path / "media")
    request = StreamRequest([part(filename="../../sneaky.exe") + b"\r\n--boundary--\r\n"])
    uploaded = asyncio.run(receive_upload(request, store, max_bytes=4096, file_fields={"recording"}, text_fields=set()))
    assert uploaded.filenames["recording"] == "sneaky.exe"
    assert uploaded.files["recording"] == uploaded.prefix + "/recording"
    assert store.path(uploaded.files["recording"]).read_bytes() == b"audio"
    assert len(uploaded.prefix) == 32
    with pytest.raises(ValueError):
        store.delete("../outside")
    with pytest.raises(ValueError):
        store.delete(".")

@pytest.mark.parametrize("info", [
    {"streams": [{"codec_type": "video"}], "format": {"duration": "1"}},
    {"streams": [{"codec_type": "audio"}], "format": {}},
    {"streams": [{"codec_type": "audio"}], "format": {"duration": "nan"}},
    {"streams": [{"codec_type": "audio"}], "format": {"duration": "0"}},
])
def test_ffprobe_rejects_without_audio_and_duration(tmp_path, monkeypatch, info):
    monkeypatch.setattr(media.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(info)))
    with pytest.raises(media.MediaValidationError):
        media.probe_media(tmp_path / "lies.mp3")

def test_ffprobe_uses_contents_and_audio_stream_duration(tmp_path, monkeypatch):
    def probe(command, **kwargs):
        assert "-show_streams" in command and "-show_format" in command
        assert kwargs["check"] is True and kwargs["timeout"] > 0
        return SimpleNamespace(stdout=json.dumps({"streams": [{"codec_type": "audio", "duration": "2.5"}],
                                                  "format": {"format_name": "wav"}}))
    monkeypatch.setattr(media.subprocess, "run", probe)
    assert media.probe_media(tmp_path / "fake.txt") == {"duration": 2.5, "kind": "audio", "media_type": "audio/wav"}

def test_non_media_upload_rejected_and_removed(api_env, monkeypatch):
    api_env.client.headers["Authorization"] = "Bearer test-token"
    def reject(path):
        assert path.read_bytes() == b"not media"
        raise media.MediaValidationError("should never leak")
    monkeypatch.setattr(main, "probe_media", reject)
    response = api_env.client.post("/meetings", files={"recording": ("valid.wav", b"not media", "audio/wav")})
    assert response.status_code == 415
    assert "should never leak" not in response.text
    assert list(api_env.store.root.iterdir()) == []
    assert api_env.client.get("/meetings").json() == []

def test_api_size_limit_returns_413_and_cleans(api_client, api_env, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.0002")
    response = api_client.post("/meetings", files={"recording": ("clip.wav", b"x" * 1000)})
    assert response.status_code == 413
    assert list(api_env.store.root.iterdir()) == []

def test_invalid_sidecar_removed_without_rows(api_client, api_env):
    response = api_client.post("/meetings", files={"recording": ("clip.wav", b"audio"), "asr_json": ("asr.json", b"{}")})
    assert response.status_code == 422
    assert api_client.get("/meetings").json() == []
    assert list(api_env.store.root.iterdir()) == []

def test_ffprobe_ignores_unavailable_container_duration(tmp_path, monkeypatch):
    info = {"format": {"duration": "N/A"}, "streams": [{"codec_type": "audio", "duration": "3"}]}
    monkeypatch.setattr(media.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(info)))
    assert media.probe_media(tmp_path / "recording")["duration"] == 3.0

def test_unauthorized_upload_does_not_read_or_store_files(api_env, monkeypatch):
    def forbidden(path):
        pytest.fail("Unauthorized uploads must not be probed")
    monkeypatch.setattr(main, "probe_media", forbidden)
    response = api_env.client.post("/meetings", files={"recording": ("clip.wav", b"audio")})
    assert response.status_code == 401
    assert list(api_env.store.root.iterdir()) == []
