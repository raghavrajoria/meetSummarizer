"""Bounded multipart parsing directly from the ASGI request stream."""

from dataclasses import dataclass, field
from urllib.parse import parse_qs
import re
import uuid

from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header

from .storage import Storage


class UploadError(ValueError):
    pass


class UploadTooLarge(UploadError):
    pass


def safe_filename(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r'[\x00-\x1f\x7f<>:"/\\|?*]', "_", name).strip(" .")[:120] or "recording"


@dataclass
class Uploaded:
    prefix: str
    files: dict[str, str] = field(default_factory=dict)
    filenames: dict[str, str] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)


async def receive_upload(request, store: Storage, *, max_bytes: int, file_fields: set[str], text_fields: set[str]) -> Uploaded:
    """Limit the entire multipart body, including metadata and its envelope."""
    if max_bytes <= 0:
        raise UploadError("Upload limit must be positive")
    content_type, options = parse_options_header(request.headers.get("content-type", ""))
    if content_type == b"application/x-www-form-urlencoded" and text_fields == {"session_json"}:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_bytes:
                raise UploadTooLarge("Upload limit exceeded")
            body.extend(chunk)
        fields = parse_qs(body.decode("utf-8", errors="strict"), keep_blank_values=True, max_num_fields=4)
        if any(key not in text_fields or len(values) != 1 for key, values in fields.items()):
            raise UploadError("Unknown or repeated form field")
        uploaded = Uploaded(uuid.uuid4().hex, fields={key: values[0] for key, values in fields.items()})
        store.path(uploaded.prefix).mkdir(parents=True, exist_ok=False)
        return uploaded
    boundary = options.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary or len(boundary) > 200:
        raise UploadError("Expected multipart/form-data")
    uploaded = Uploaded(uuid.uuid4().hex)
    directory = store.path(uploaded.prefix)
    directory.mkdir(parents=True, exist_ok=False)
    current = {"handle": None}
    seen = set()
    finished = False

    def begin():
        current.update(headers={}, header_name=bytearray(), header_value=bytearray(), header_bytes=0,
                       name=None, text=bytearray(), handle=None)

    def header_field(data, start, end):
        current["header_name"].extend(data[start:end])
        check_headers(end - start)

    def header_value(data, start, end):
        current["header_value"].extend(data[start:end])
        check_headers(end - start)

    def check_headers(count):
        current["header_bytes"] += count
        if current["header_bytes"] > 8192:
            raise UploadError("Multipart headers too large")

    def header_end():
        current["headers"][bytes(current["header_name"]).lower()] = bytes(current["header_value"])
        current["header_name"].clear()
        current["header_value"].clear()

    def headers_finished():
        disposition, opts = parse_options_header(current["headers"].get(b"content-disposition", b""))
        name = opts.get(b"name", b"").decode("utf-8", errors="strict")
        if disposition != b"form-data" or name in seen or name not in file_fields | text_fields:
            raise UploadError("Unknown or repeated multipart field")
        seen.add(name)
        current["name"] = name
        if name in file_fields:
            if b"filename" not in opts:
                raise UploadError("Expected file field")
            key = f"{uploaded.prefix}/{name}"
            uploaded.files[name] = key
            uploaded.filenames[name] = safe_filename(opts[b"filename"].decode("utf-8", errors="replace"))
            current["handle"] = store.path(key).open("xb")
        elif b"filename" in opts:
            raise UploadError("Expected text field")

    def data_received(data, start, end):
        if current["handle"] is not None:
            current["handle"].write(data[start:end])
        else:
            current["text"].extend(data[start:end])
            if current["name"] != "session_json" and len(current["text"]) > 2048:
                raise UploadError("Text field too large")

    def end_part():
        if current["handle"] is not None:
            current["handle"].close()
            current["handle"] = None
        else:
            uploaded.fields[current["name"]] = current["text"].decode("utf-8", errors="strict")

    def end():
        nonlocal finished
        finished = True

    try:
        parser = MultipartParser(boundary, {
            "on_part_begin": begin, "on_header_field": header_field, "on_header_value": header_value,
            "on_header_end": header_end, "on_headers_finished": headers_finished,
            "on_part_data": data_received, "on_part_end": end_part, "on_end": end,
        })
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_bytes:
                raise UploadTooLarge("Upload limit exceeded")
            parser.write(chunk)
        parser.finalize()
        if not finished:
            raise UploadError("Incomplete multipart body")
        return uploaded
    except BaseException:
        if current.get("handle") is not None:
            current["handle"].close()
        store.delete(uploaded.prefix)
        raise
