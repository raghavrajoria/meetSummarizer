"""FastAPI endpoints for importing and reviewing meeting sessions."""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from .database import MEDIA_DIR, get_db
from .models import MeetingSession, Job, SummaryEdit, AccessSession
from .jobs import DatabaseJobQueue, JobQueue, job_payload
from .logging_config import configure_logging, RequestLogMiddleware
from .security import AuthMiddleware
from .uploads import receive_upload, UploadError, UploadTooLarge
from .media import probe_media, MediaValidationError
from .processing import validate_sidecars
from .maintenance import delete_meeting, MeetingBusy
from .storage import LocalStorage, Storage
from indicmeet.settings import get_settings


storage: Storage = LocalStorage(MEDIA_DIR)
RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")


configure_logging()
app = FastAPI(title="meetSummerizer API", version="0.2.0")
app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Range", "Authorization", "X-Request-ID"],
    expose_headers=["Accept-Ranges", "Content-Range", "Content-Length", "X-Request-ID"],
)
app.add_middleware(RequestLogMiddleware)


def get_storage() -> Storage:
    return storage


def get_queue() -> JobQueue:
    return DatabaseJobQueue(lease_seconds=get_settings().worker_lease_seconds)


def frontend_payload(record: MeetingSession, job: Job | None = None, db=None) -> dict[str, Any]:
    payload = dict(record.payload)
    for private in ("source_files", "storage_prefix", "transcript_key"):
        payload.pop(private, None)
    payload["original_summary"] = payload.get("summary", "")
    edit = db.get(SummaryEdit, record.id) if db else None
    if edit:
        payload["summary"] = edit.text
    payload["summary_edited"] = edit is not None
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
    if job:
        payload["status"] = "processed" if job.status == "done" else job.status
        payload["job_id"] = job.id
    return payload


@app.get("/sessions")
@app.get("/meetings")
def list_sessions(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    records = db.scalars(select(MeetingSession).order_by(MeetingSession.date.desc(), MeetingSession.title)).all()
    return [frontend_payload(record, db.scalar(select(Job).where(Job.meeting_id == record.id)), db) for record in records]


@app.get("/sessions/{session_id}")
@app.get("/meetings/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    record = db.get(MeetingSession, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return frontend_payload(record, db.scalar(select(Job).where(Job.meeting_id == record.id)), db)


@app.get("/sessions/{session_id}/media")
@app.get("/meetings/{session_id}/media")
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
        media_type=record.payload.get("media_type") or media_store.content_type(record.media_key),
        headers=headers,
    )


@app.post("/sessions/import", status_code=201)
async def import_session(
    request: Request,
    db: Session = Depends(get_db),
    media_store: Storage = Depends(get_storage),
) -> dict[str, Any]:
    uploaded = await parse_upload(request, media_store, {"media"}, {"session_json"})
    try:
        try:
            payload = json.loads(uploaded.fields.get("session_json", ""))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="session_json is invalid JSON") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
            raise HTTPException(status_code=422, detail="session_json must be an object with a segments array")
        session_id = str(payload.get("id") or "").strip()
        title = str(payload.get("title") or "").strip()
        if not session_id or len(session_id) > 160 or not title or len(title) > 500:
            raise HTTPException(status_code=422, detail="session_json requires a short id and title")
        if db.get(MeetingSession, session_id) is not None:
            raise HTTPException(status_code=409, detail="A session with this id already exists")
        media_key = uploaded.files.get("media")
        info = await run_in_threadpool(checked_media, media_store.path(media_key)) if media_key else {}
        # These keys are server-owned; never accept arbitrary filesystem keys.
        for private in ("source_files", "storage_prefix", "transcript_key", "media_type"):
            payload.pop(private, None)
        payload.update(info, id=session_id, title=title, storage_prefix=uploaded.prefix)
        record = MeetingSession(id=session_id, title=title,
                                group_name=str(payload.get("group") or "")[:300],
                                date=str(payload.get("date") or "")[:32], payload=payload, media_key=media_key)
        db.add(record)
        db.flush()
        from datetime import datetime
        now = datetime.utcnow()
        job = Job(meeting_id=session_id, request_id=request.state.request_id, status="done", stage="done",
                  progress=100, completed_at=now, stage_timings={})
        db.add(job)
        db.commit()
        return frontend_payload(record, job)
    except BaseException:
        db.rollback()
        media_store.delete(uploaded.prefix)
        raise


async def parse_upload(request, store, file_fields, text_fields):
    try:
        return await receive_upload(request, store, max_bytes=int(get_settings().max_upload_mb * 1024 * 1024),
                                    file_fields=file_fields, text_fields=text_fields)
    except UploadTooLarge:
        raise HTTPException(status_code=413, detail="Upload limit exceeded") from None
    except (UploadError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid multipart upload") from None


def checked_media(path):
    try:
        return probe_media(path)
    except MediaValidationError:
        raise HTTPException(status_code=415, detail="Media must contain audio and a duration") from None
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Media validation unavailable") from None


@app.post("/meetings", status_code=202)
async def create_meeting(request: Request, db: Session = Depends(get_db),
                         media_store: Storage = Depends(get_storage), queue: JobQueue = Depends(get_queue)):
    uploaded = await parse_upload(request, media_store, {"recording", "asr_json", "diarization_csv", "attendees_json"},
                                  {"title", "group", "date"})
    try:
        if "recording" not in uploaded.files:
            raise HTTPException(status_code=422, detail="recording is required")
        info = await run_in_threadpool(checked_media, media_store.path(uploaded.files["recording"]))
        try:
            await run_in_threadpool(validate_sidecars, uploaded.files, media_store)
        except (ValueError, RuntimeError, TypeError, KeyError, UnicodeError):
            raise HTTPException(status_code=422, detail="Invalid meeting metadata") from None
        title = uploaded.fields.get("title", uploaded.filenames["recording"]).strip()
        if not title or len(title) > 500:
            raise HTTPException(status_code=422, detail="title must contain 1 to 500 characters")
        record = MeetingSession(id=uploaded.prefix, title=title,
                                group_name=uploaded.fields.get("group", "")[:300],
                                date=uploaded.fields.get("date", "")[:32],
                                media_key=uploaded.files["recording"], payload={
                                    **info, "status": "queued", "source_files": uploaded.files,
                                    "storage_prefix": uploaded.prefix,
                                    "original_filename": uploaded.filenames["recording"],
                                })
        db.add(record)
        db.flush()
        job = queue.enqueue(db, record.id, request.state.request_id)
        db.commit()
        return {**frontend_payload(record, job), "job": job_payload(job)}
    except BaseException:
        db.rollback()
        media_store.delete(uploaded.prefix)
        raise


@app.get("/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_payload(job)


@app.get("/meetings/{session_id}/job")
def meeting_job(session_id: str, db: Session = Depends(get_db)):
    job = db.scalar(select(Job).where(Job.meeting_id == session_id))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_payload(job)


@app.post("/jobs/{job_id}/retry", status_code=202)
def retry_job(job_id: str, request: Request, db: Session = Depends(get_db), queue: JobQueue = Depends(get_queue)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        job = queue.retry(db, job, request.state.request_id)
    except ValueError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Only failed jobs can be retried") from None
    db.commit()
    return job_payload(job)


@app.delete("/meetings/{session_id}", status_code=204)
@app.delete("/sessions/{session_id}", status_code=204)
def remove_meeting(session_id: str, db: Session = Depends(get_db), media_store: Storage = Depends(get_storage)):
    record = db.get(MeetingSession, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        delete_meeting(db, media_store, record)
    except MeetingBusy:
        db.rollback()
        raise HTTPException(status_code=409, detail="Meeting is running; retry deletion after completion") from None
    except OSError:
        db.rollback()
        raise HTTPException(status_code=503, detail="Media deletion unavailable") from None
    return Response(status_code=204)


@app.get("/meetings/{session_id}/transcript")
def download_transcript(session_id: str, db: Session = Depends(get_db)):
    record = db.get(MeetingSession, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    job = db.scalar(select(Job).where(Job.meeting_id == session_id))
    if job and job.status != "done":
        raise HTTPException(status_code=409, detail="Transcript is not ready")
    return JSONResponse(record.payload.get("transcript", record.payload.get("segments", [])),
                        headers={"Content-Disposition": 'attachment; filename="transcript.json"'})


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/readyz")
def ready(db: Session = Depends(get_db), media_store: Storage = Depends(get_storage)):
    import shutil
    import uuid
    temporary_key = uuid.uuid4().hex + ".ready"
    try:
        # Query mapped columns, so missing tables/columns require a migration.
        db.execute(select(MeetingSession).limit(1))
        db.execute(select(Job).limit(1))
        if not shutil.which(get_settings().ffprobe_binary):
            raise RuntimeError("ffprobe unavailable")
        with media_store.path(temporary_key).open("xb"):
            pass
        return {"status": "ready"}
    except Exception:
        db.rollback()
        return JSONResponse({"status": "not_ready"}, status_code=503)
    finally:
        try:
            media_store.delete(temporary_key)
        except OSError:
            pass


@app.get("/config")
def public_config():
    return {"demo_mode":get_settings().demo_mode,"max_upload_mb":get_settings().max_upload_mb,"import_mode":get_settings().asr_mode=="import"}

@app.post("/auth/login")
async def login(request: Request, db: Session=Depends(get_db)):
    import secrets, time
    from .security import users, password_matches, token_hash
    try:
        body=await request.json()
        username, password=body["username"],body["password"]
        if not isinstance(username,str) or not isinstance(password,str) or len(password)>1024:
            raise ValueError()
    except (ValueError,KeyError,TypeError):
        raise HTTPException(422,"username and password are required") from None
    user=users().get(username,{})
    if not password_matches(password,user.get("password_hash","")):
        raise HTTPException(401,"Invalid username or password")
    role=user.get("role","editor")
    if role not in {"viewer","editor","admin"}:
        raise HTTPException(503,"Account configuration invalid")
    token=secrets.token_urlsafe(32)
    lifetime=min(3600,max(60,int(__import__("os").environ.get("ACCESS_TOKEN_SECONDS","900"))))
    db.add(AccessSession(id=token_hash(token),username=username,role=role,expires=int(time.time())+lifetime))
    db.commit()
    return {"access_token":token,"token_type":"bearer","expires_in":lifetime,"username":username,"role":role}

@app.post("/auth/logout",status_code=204)
def logout(request:Request,db:Session=Depends(get_db)):
    session=request.state.identity.get("session")
    if session and session!="integration":
        row=db.get(AccessSession,session)
        if row:
            db.delete(row); db.commit()
    return Response(status_code=204)

@app.get("/meetings/{session_id}/media-url")
def media_url(session_id:str,request:Request,db:Session=Depends(get_db)):
    import time
    from .security import media_signature
    record=db.get(MeetingSession,session_id)
    if not record or not record.media_key:
        raise HTTPException(404,"Media not found")
    path=f"/meetings/{session_id}/media"
    expires=str(int(time.time())+120)
    session=request.state.identity.get("session","integration")
    try:
        signature=media_signature(path,expires,session)
    except ValueError:
        raise HTTPException(503,"Media signing unavailable") from None
    return {"url":f"{path}?expires={expires}&session={session}&signature={signature}","expires_in":120}

@app.put("/meetings/{session_id}/summary")
async def save_summary(session_id:str,request:Request,db:Session=Depends(get_db)):
    from datetime import datetime
    record=db.get(MeetingSession,session_id)
    if not record:
        raise HTTPException(404,"Meeting not found")
    body=await request.json()
    text=body.get("text") if isinstance(body,dict) else None
    if not isinstance(text,str) or len(text)>50000:
        raise HTTPException(422,"text must be a string up to 50000 characters")
    db.merge(SummaryEdit(meeting_id=session_id,text=text,updated_at=datetime.utcnow()))
    db.commit()
    return {"summary":text,"summary_edited":True}

@app.delete("/meetings/{session_id}/summary",status_code=204)
def revert_summary(session_id:str,db:Session=Depends(get_db)):
    if not db.get(MeetingSession,session_id):
        raise HTTPException(404,"Meeting not found")
    row=db.get(SummaryEdit,session_id)
    if row:
        db.delete(row); db.commit()
    return Response(status_code=204)
