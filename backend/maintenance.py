"""Filesystem/row deletion and optional terminal-job retention."""

from datetime import datetime, timedelta

from sqlalchemy import delete, select, update

from .models import Job, MeetingSession, SummaryEdit


class MeetingBusy(ValueError):
    pass


def delete_meeting(db, store, record, *, expired_before=None):
    # This UPDATE locks the job on both SQLite and Postgres before touching files.
    # Claim/retry must wait, then find the row gone after this transaction commits.
    conditions = [Job.meeting_id == record.id, Job.status != "running"]
    if expired_before is not None:
        conditions.extend([Job.status.in_(["done", "failed"]), Job.completed_at < expired_before])
    locked = db.execute(update(Job).where(*conditions).values(updated_at=datetime.utcnow()))
    if locked.rowcount == 0 and db.scalar(select(Job.id).where(Job.meeting_id == record.id)):
        raise MeetingBusy("Meeting is running or no longer eligible for retention")
    prefix = record.payload.get("storage_prefix")
    if prefix:
        store.delete(prefix)
    elif record.media_key:
        store.delete(record.media_key)
    db.execute(delete(Job).where(Job.meeting_id == record.id))
    db.execute(delete(SummaryEdit).where(SummaryEdit.meeting_id == record.id))
    db.delete(record)
    db.commit()


def purge_expired(session_factory, store, retention_days):
    if retention_days <= 0:
        return 0
    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    removed = 0
    with session_factory() as db:
        ids = db.scalars(select(Job.meeting_id).where(Job.status.in_(["done", "failed"]), Job.completed_at < cutoff)).all()
        for meeting_id in ids:
            record = db.get(MeetingSession, meeting_id)
            if record is None:
                continue
            try:
                delete_meeting(db, store, record, expired_before=cutoff)
                removed += 1
            except MeetingBusy:
                db.rollback()
    return removed
