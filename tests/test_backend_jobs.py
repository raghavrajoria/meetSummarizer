"""Persistent lifecycle, retry, ownership, deletion and retention."""
from datetime import datetime, timedelta
import json
import pytest
from sqlalchemy import select
from backend.jobs import DatabaseJobQueue, LostClaim
from backend.models import Job, MeetingSession
from backend.worker import Worker
from backend.maintenance import purge_expired

def create(client, rows=None):
    files = {"recording": ("recording.wav", b"audio")}
    if rows is not None:
        files["asr_json"] = ("asr.json", json.dumps(rows).encode())
    response = client.post("/meetings", data={"title": "Meeting"}, files=files, headers={"X-Request-ID": "job-test"})
    assert response.status_code == 202, response.text
    return response.json()

def test_job_queued_running_done_and_transcript(api_client, api_env, asr_rows):
    meeting = create(api_client, asr_rows)
    job_id = meeting["job_id"]
    assert meeting["status"] == "queued"
    assert api_client.get(f"/jobs/{job_id}").json()["status"] == "queued"
    assert api_client.get(f"/meetings/{meeting['id']}/transcript").status_code == 409
    observed = []
    def processor(record, store, stage):
        observed.append(api_client.get(f"/jobs/{job_id}").json()["status"])
        stage("asr", 40, lambda: None)
        return {"transcript": asr_rows, "segments": [], "summary": "Summary"}
    worker = Worker(api_env.queue, api_env.sessions, api_env.store, processor)
    assert worker.run_once() is True
    assert observed == ["running"]
    job = api_client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "done" and job["progress"] == 100
    assert job["attempts"] == 1 and job["error"] is None
    assert job["started_at"] and job["completed_at"]
    assert job["stage_timings"]["asr"] >= 0
    assert api_client.get(f"/meetings/{meeting['id']}/job").json() == job
    assert api_client.get(f"/meetings/{meeting['id']}").json()["status"] == "processed"
    transcript = api_client.get(f"/meetings/{meeting['id']}/transcript")
    assert transcript.json() == asr_rows
    assert 'filename="transcript.json"' in transcript.headers["content-disposition"]
    assert worker.run_once() is False

def test_failure_is_private_and_retry_requeues(api_client, api_env, caplog):
    meeting = create(api_client)
    job_id = meeting["job_id"]
    assert api_client.post(f"/jobs/{job_id}/retry").status_code == 409
    def processor(record, store, stage):
        def fail():
            raise RuntimeError("Private transcript must never appear")
        stage("asr", 40, fail)
    worker = Worker(api_env.queue, api_env.sessions, api_env.store, processor)
    assert worker.run_once()
    failed = api_client.get(f"/jobs/{job_id}").json()
    assert failed["status"] == "failed" and failed["error"] == "asr failed"
    assert failed["stage_timings"]["asr"] >= 0
    assert "Private transcript" not in caplog.text
    assert "Private transcript" not in json.dumps(failed)
    response = api_client.post(f"/jobs/{job_id}/retry", headers={"X-Request-ID": "retry-request"})
    assert response.status_code == 202
    retried = response.json()
    assert retried["status"] == "queued" and retried["error"] is None
    assert retried["completed_at"] is None and retried["stage_timings"] == {}
    claim = api_env.queue.claim_next()
    assert claim.request_id == "retry-request"
    api_env.queue.finish(claim, {"transcript": []})
    assert api_client.get(f"/jobs/{job_id}").json()["attempts"] == 2

def test_delete_removes_generated_files_and_rows(api_client, api_env, asr_rows):
    meeting = create(api_client, asr_rows)
    prefix = meeting["id"]
    assert api_env.store.path(prefix).exists()
    assert api_client.delete(f"/meetings/{prefix}").status_code == 204
    assert not api_env.store.path(prefix).exists()
    with api_env.sessions() as db:
        assert db.get(MeetingSession, prefix) is None
        assert db.get(Job, meeting["job_id"]) is None
    assert api_client.get(f"/jobs/{meeting['job_id']}").status_code == 404
    assert api_env.queue.claim_next() is None

def test_delete_running_returns_conflict_then_deletes_terminal(api_client, api_env):
    meeting = create(api_client)
    claim = api_env.queue.claim_next()
    assert api_client.delete(f"/meetings/{meeting['id']}").status_code == 409
    assert api_env.store.path(meeting["id"]).exists()
    api_env.queue.fail(claim, "asr failed")
    assert api_client.delete(f"/meetings/{meeting['id']}").status_code == 204

def test_only_one_queue_can_claim_and_stale_owner_cannot_finish(api_client, api_env):
    meeting = create(api_client)
    claim = api_env.queue.claim_next()
    other = DatabaseJobQueue(api_env.sessions)
    assert other.claim_next() is None
    assert api_env.queue.renew(claim)
    api_env.queue.fail(claim, "asr failed")
    with api_env.sessions() as db:
        job = db.get(Job, meeting["job_id"])
        other.retry(db, job, "new-request")
        db.commit()
    new_claim = other.claim_next()
    assert new_claim.worker_token != claim.worker_token
    with pytest.raises(LostClaim):
        api_env.queue.finish(claim, {"transcript": ["stale"]})
    other.finish(new_claim, {"transcript": []})

def test_expired_lease_fails_without_reprocessing(api_client, api_env):
    meeting = create(api_client)
    claim = api_env.queue.claim_next()
    with api_env.sessions() as db:
        job = db.get(Job, claim.id)
        job.lease_expires_at = datetime.utcnow() - timedelta(seconds=1)
        db.commit()
    assert not api_env.queue.renew(claim)
    assert api_env.queue.claim_next() is None
    result = api_client.get(f"/jobs/{meeting['job_id']}").json()
    assert result["status"] == "failed" and result["error"] == "worker lease expired"

def test_retention_only_removes_expired_terminal_meetings(api_client, api_env):
    old = create(api_client)
    claim = api_env.queue.claim_next()
    api_env.queue.finish(claim, {"transcript": []})
    with api_env.sessions() as db:
        db.get(Job, claim.id).completed_at = datetime.utcnow() - timedelta(days=5)
        db.commit()
    queued = create(api_client)
    assert purge_expired(api_env.sessions, api_env.store, 0) == 0
    assert purge_expired(api_env.sessions, api_env.store, 2) == 1
    assert api_client.get(f"/meetings/{old['id']}").status_code == 404
    assert api_client.get(f"/meetings/{queued['id']}").status_code == 200

def test_retention_lock_preserves_retried_job(api_client, api_env):
    from backend.maintenance import delete_meeting, MeetingBusy
    meeting = create(api_client)
    claim = api_env.queue.claim_next()
    api_env.queue.fail(claim, "asr failed")
    cutoff = datetime.utcnow() - timedelta(days=2)
    with api_env.sessions() as db:
        job = db.get(Job, claim.id)
        job.completed_at = cutoff - timedelta(days=1)
        db.commit()
    assert api_client.post(f"/jobs/{claim.id}/retry").status_code == 202
    with api_env.sessions() as db:
        record = db.get(MeetingSession, meeting["id"])
        with pytest.raises(MeetingBusy):
            delete_meeting(db, api_env.store, record, expired_before=cutoff)
        db.rollback()
    assert api_env.store.path(meeting["id"]).exists()
    assert api_client.get(f"/jobs/{claim.id}").json()["status"] == "queued"
