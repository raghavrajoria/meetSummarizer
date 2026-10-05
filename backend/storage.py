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


    def flush(self, prefix):
        pass  # Local writes are already durable on the shared volume.

    def ready(self):
        return self.root.is_dir()

class S3Storage(LocalStorage):
    """Local staging only; all durable objects live in S3/MinIO."""
    def __init__(self, root):
        super().__init__(root)
        import boto3, os
        from botocore.config import Config
        self.bucket=os.environ.get("S3_BUCKET","indicmeet")
        self.client=boto3.client("s3",endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,region_name=os.environ.get("S3_REGION","us-east-1"),aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),config=Config(signature_version="s3v4",s3={"addressing_style":"path"}))
    def path(self,key):
        path=self._path(key)
        if not path.exists():
            from botocore.exceptions import ClientError
            try:
                self.client.head_object(Bucket=self.bucket,Key=key)
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in {"404","NoSuchKey","NotFound"}: raise
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".part")
                self.client.download_file(self.bucket,key,str(temporary));temporary.replace(path)
        return path
    def flush(self,prefix):
        path=self._path(prefix)
        files=path.rglob("*") if path.is_dir() else [path]
        for file in files:
            if file.is_file(): self.client.upload_file(str(file),self.bucket,file.relative_to(self.root).as_posix())
    def size(self,key):
        from botocore.exceptions import ClientError
        try: return self.client.head_object(Bucket=self.bucket,Key=key)["ContentLength"]
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404","NoSuchKey","NotFound"}: raise FileNotFoundError(key) from None
            raise
    def iter_bytes(self,key,start,length):
        body=self.client.get_object(Bucket=self.bucket,Key=key,Range=f"bytes={start}-{start+length-1}")["Body"]
        try:
            while chunk:=body.read(64*1024): yield chunk
        finally: body.close()
    def delete(self,key):
        self._path(key)
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket,Prefix=key):
            objects=[{"Key":obj["Key"]} for obj in page.get("Contents",[]) if obj["Key"]==key or obj["Key"].startswith(key+"/")]
            if objects:self.client.delete_objects(Bucket=self.bucket,Delete={"Objects":objects})
        super().delete(key)
    def ready(self):
        self.client.head_bucket(Bucket=self.bucket);return True

def configured_storage(root):
    import os
    mode=os.environ.get("STORAGE_BACKEND","local")
    if mode not in {"local","s3"}:raise ValueError("STORAGE_BACKEND must be local or s3")
    return S3Storage(root) if mode=="s3" else LocalStorage(root)
