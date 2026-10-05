"""Persistent job state and an interchangeable queue contract."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol
import uuid

from sqlalchemy import select, update

from .database import SessionLocal
from .models import Job, MeetingSession


@dataclass(frozen=True)
class Claim:
    id: str
    meeting_id: str
    request_id: str
    worker_token: str
    attempt: int = 0


class JobQueue(Protocol):
    def enqueue(self, db, meeting_id: str, request_id: str) -> Job: ...
    def claim_next(self) -> Claim | None: ...
    def advance(self, claim: Claim, stage: str, progress: int, timings: dict | None = None) -> None: ...
    def renew(self, claim: Claim) -> bool: ...
    def finish(self, claim: Claim, payload: dict) -> None: ...
    def fail(self, claim: Claim, error: str) -> None: ...
    def retry(self, db, job: Job, request_id: str) -> Job: ...


class LostClaim(RuntimeError):
    pass


def job_payload(job: Job) -> dict:
    def stamp(value):
        return value.isoformat(timespec="milliseconds") + "Z" if value else None
    return {
        "id": job.id, "meeting_id": job.meeting_id, "status": job.status,
        "stage": job.stage, "progress": job.progress, "error": job.error,
        "created_at": stamp(job.created_at), "updated_at": stamp(job.updated_at),
        "started_at": stamp(job.started_at), "completed_at": stamp(job.completed_at),
        "remote_asr_job_id": job.remote_asr_job_id,
        "attempts": job.attempts, "stage_timings": job.stage_timings,
    }


class DatabaseJobQueue:
    def __init__(self, session_factory=SessionLocal, *, lease_seconds: float = 180):
        if lease_seconds <= 0:
            raise ValueError("Worker lease must be positive")
        self.sessions = session_factory
        self.lease_seconds = lease_seconds

    def enqueue(self, db, meeting_id, request_id):
        job = Job(meeting_id=meeting_id, request_id=request_id)
        db.add(job)
        db.flush()
        return job

    def claim_next(self):
        now = datetime.utcnow()
        with self.sessions() as db:
            # An already-submitted remote job resumes polling under a new lease.
            db.execute(update(Job).where(Job.status == "running", Job.lease_expires_at < now,
                Job.remote_asr_job_id.is_not(None)).values(status="queued", worker_token=None,
                lease_expires_at=None, updated_at=now))
            # Jobs without a confirmed remote submission remain explicitly retryable.
            db.execute(update(Job).where(Job.status == "running", Job.lease_expires_at < now).values(
                status="failed", error="worker lease expired", completed_at=now, updated_at=now,
                worker_token=None, lease_expires_at=None))
            db.commit()
            candidates = db.scalars(select(Job.id).where(Job.status == "queued").order_by(Job.created_at).limit(20)).all()
            for job_id in candidates:
                token = uuid.uuid4().hex
                changed = db.execute(update(Job).where(Job.id == job_id, Job.status == "queued").values(
                    status="running", stage="starting", progress=0, started_at=now, updated_at=now,
                    attempts=Job.attempts + 1, worker_token=token,
                    lease_expires_at=now + timedelta(seconds=self.lease_seconds)))
                if changed.rowcount == 1:
                    job = db.get(Job, job_id)
                    claim = Claim(job.id, job.meeting_id, job.request_id, token, job.attempts)
                    db.commit()
                    return claim
                db.rollback()
        return None

    def _owned(self, claim):
        return (Job.id == claim.id, Job.status == "running", Job.worker_token == claim.worker_token,
                Job.lease_expires_at > datetime.utcnow())

    def advance(self, claim, stage, progress, timings=None):
        with self.sessions() as db:
            values = dict(stage=stage, progress=progress, updated_at=datetime.utcnow())
            if timings is not None:
                values["stage_timings"] = dict(timings)
            result = db.execute(update(Job).where(*self._owned(claim)).values(**values))
            if result.rowcount != 1:
                raise LostClaim("Job no longer owned")
            db.commit()

    def renew(self, claim):
        with self.sessions() as db:
            result = db.execute(update(Job).where(*self._owned(claim)).values(
                lease_expires_at=datetime.utcnow() + timedelta(seconds=self.lease_seconds)))
            db.commit()
            return result.rowcount == 1

    def finish(self, claim, payload):
        with self.sessions() as db:
            now = datetime.utcnow()
            result = db.execute(update(Job).where(*self._owned(claim)).values(
                status="done", stage="done", progress=100, error=None,
                completed_at=now, updated_at=now, worker_token=None, lease_expires_at=None))
            if result.rowcount != 1:
                raise LostClaim("Job no longer owned")
            record = db.get(MeetingSession, claim.meeting_id)
            record.payload = {**record.payload, **payload, "status": "processed"}
            db.commit()

    def fail(self, claim, error):
        with self.sessions() as db:
            now = datetime.utcnow()
            result = db.execute(update(Job).where(*self._owned(claim)).values(
                status="failed", error=error, completed_at=now, updated_at=now,
                worker_token=None, lease_expires_at=None))
            db.commit()
            return result.rowcount == 1

    def retry(self, db, job, request_id):
        retained={key:value for key,value in (job.asr_requests or {}).items() if value.get("status") not in {"failed","timeout","submitting","ambiguous"}}
        retained_id=next((value.get("job_id") for value in retained.values() if value.get("job_id")),None)
        changed = db.execute(update(Job).where(Job.id == job.id, Job.status == "failed").values(
            asr_requests=retained, remote_asr_job_id=retained_id,
            status="queued", stage="queued", progress=0, error=None, updated_at=datetime.utcnow(),
            started_at=None, completed_at=None, lease_expires_at=None, worker_token=None,
            request_id=request_id, stage_timings={}))
        if changed.rowcount != 1:
            raise ValueError("Only failed jobs can be retried")
        db.flush()
        db.refresh(job)
        return job

    def load_asr(self,claim,scope):
        with self.sessions() as db:
            job=db.scalar(select(Job).where(*self._owned(claim)))
            if job is None:raise LostClaim("ASR checkpoint claim lost")
            return (job.asr_requests or {}).get(scope)

    def save_asr(self,claim,scope,state):
        with self.sessions() as db:
            job=db.scalar(select(Job).where(*self._owned(claim)))
            if job is None:raise LostClaim("ASR checkpoint claim lost")
            values={**(job.asr_requests or {}),scope:dict(state)}
            job.asr_requests=values
            if state.get("job_id"):job.remote_asr_job_id=state["job_id"]
            db.commit()
