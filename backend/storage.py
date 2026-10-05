"""Media storage interface and local filesystem implementation."""

from __future__ import annotations

import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Iterator, Protocol


class Storage(Protocol):
    def path(self, key: str) -> Path: ...
    def delete(self, key: str) -> None: ...
    def save(self, source: BinaryIO, filename: str) -> str: ...
    def size(self, key: str) -> int: ...
    def content_type(self, key: str) -> str: ...
    def iter_bytes(self, key: str, start: int, length: int) -> Iterator[bytes]: ...


class LocalStorage:
    """Store imported media under the ignored local data directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        if candidate == self.root or not candidate.is_relative_to(self.root):
            raise ValueError("Invalid storage key")
        return candidate

    def path(self, key: str) -> Path:
        return self._path(key)

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    def save(self, source: BinaryIO, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if not suffix or len(suffix) > 12 or not suffix[1:].isalnum():
            suffix = ".media"
        key = f"{uuid.uuid4().hex}{suffix}"
        destination = self._path(key)
        with destination.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        return key

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    def content_type(self, key: str) -> str:
        return mimetypes.guess_type(key)[0] or "application/octet-stream"

    def iter_bytes(self, key: str, start: int, length: int) -> Iterator[bytes]:
        path = self._path(key)

        def chunks() -> Iterator[bytes]:
            remaining = length
            with path.open("rb") as source:
                source.seek(start)
                while remaining > 0:
                    chunk = source.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return chunks()
