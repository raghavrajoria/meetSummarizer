"""FastAPI endpoints for importing and reviewing meeting sessions."""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import Base, DATA_DIR, engine, get_db
from .models import MeetingSession
from .storage import LocalStorage, Storage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
storage: Storage = LocalStorage(DATA_DIR / "media")
RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="meetSummerizer API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8001", "http://127.0.0.1:8001"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Range"],
    expose_headers=["Accept-Ranges", "Content-Range", "Content-Length"],
)


def get_storage() -> Storage:
    return storage


def frontend_payload(record: MeetingSession) -> dict[str, Any]:
    payload = dict(record.payload)
    payload["id"] = record.id
    payload["title"] = record.title
    payload["group"] = record.group_name
    payload.setdefault("date", record.date)
    payload.setdefault("dateLabel", record.date or "Date not recorded")
    payload.setdefault("time", "")
    payload.setdefault("status", "processed")
    payload.setdefault("summary", "")
    payload.setdefault("speakers", [])
    payload.setdefault("intelligence", {})
    payload["media"] = f"/sessions/{record.id}/media" if record.media_key else ""
    return payload


@app.get("/sessions")
def list_sessions(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    records = db.scalars(select(MeetingSession).order_by(MeetingSession.date.desc(), MeetingSession.title)).all()
    return [frontend_payload(record) for record in records]


@app.get("/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    record = db.get(MeetingSession, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return frontend_payload(record)


@app.get("/sessions/{session_id}/media")
def get_media(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
    media_store: Storage = Depends(get_storage),
):
    record = db.get(MeetingSession, session_id)
    if record is None or not record.media_key:
        raise HTTPException(status_code=404, detail="Session media not found")
    try:
        size = media_store.size(record.media_key)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Session media not found") from None

    range_header = request.headers.get("range")
    status_code = 200
    start, end = 0, size - 1
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(size)}

    if range_header:
        match = RANGE_PATTERN.fullmatch(range_header.strip())
        if not match or size == 0:
            raise HTTPException(status_code=416, detail="Invalid byte range", headers={"Content-Range": f"bytes */{size}"})
        first, last = match.groups()
        if first == "":
            if last == "":
                raise HTTPException(status_code=416, detail="Invalid byte range", headers={"Content-Range": f"bytes */{size}"})
            suffix_length = int(last)
            if suffix_length == 0:
                raise HTTPException(status_code=416, detail="Invalid byte range", headers={"Content-Range": f"bytes */{size}"})
            start = max(0, size - suffix_length)
        else:
            start = int(first)
        end = min(int(last), size - 1) if last and first else size - 1
        if start >= size or start > end:
            raise HTTPException(status_code=416, detail="Range not satisfiable", headers={"Content-Range": f"bytes */{size}"})
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(end - start + 1)

    return StreamingResponse(
        media_store.iter_bytes(record.media_key, start, end - start + 1),
        status_code=status_code,
        media_type=media_store.content_type(record.media_key),
        headers=headers,
    )


@app.post("/sessions/import", status_code=201)
async def import_session(
    session_json: str = Form(...),
    media: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    media_store: Storage = Depends(get_storage),
) -> dict[str, Any]:
    try:
        payload = json.loads(session_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"session_json is invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise HTTPException(status_code=422, detail="session_json must be an object with a segments array")
    session_id = str(payload.get("id") or "").strip()
    title = str(payload.get("title") or "").strip()
    if not session_id or len(session_id) > 160 or not title:
        raise HTTPException(status_code=422, detail="session_json requires a short id and title")
    if db.get(MeetingSession, session_id) is not None:
        raise HTTPException(status_code=409, detail="A session with this id already exists")

    media_key = None
    if media is not None and media.filename:
        media_key = media_store.save(media.file, media.filename)
    payload["id"] = session_id
    payload["title"] = title
    record = MeetingSession(
        id=session_id,
        title=title,
        group_name=str(payload.get("group") or ""),
        date=str(payload.get("date") or ""),
        payload=payload,
        media_key=media_key,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return frontend_payload(record)
