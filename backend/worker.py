"""Run with root Python: python -m backend.worker [--once]."""

import argparse
import logging
import threading
import time

from indicmeet.settings import get_settings

from .database import SessionLocal, MEDIA_DIR
from .jobs import DatabaseJobQueue, LostClaim
from .logging_config import configure_logging, request_id
from .maintenance import purge_expired
from .models import MeetingSession
from .processing import process_meeting
from .storage import configured_storage

logger = logging.getLogger("indicmeet.worker")


class Worker:
    def __init__(self, queue, session_factory, store, processor=process_meeting):
        self.queue, self.sessions, self.store, self.processor = queue, session_factory, store, processor

    def run_once(self):
        claim = self.queue.claim_next()
        if claim is None:
            return False
        token = request_id.set(claim.request_id)
        stop = threading.Event()
        lease = get_settings().worker_lease_seconds

        def heartbeat():
            correlation = request_id.set(claim.request_id)
            try:
                while not stop.wait(max(0.1, lease / 3)):
                    try:
                        if not self.queue.renew(claim):
                            break
                    except Exception as exc:
                        logger.error("heartbeat_failed exception_type=%s", type(exc).__name__)
                        break
            finally:
                request_id.reset(correlation)

        thread = threading.Thread(target=heartbeat, daemon=True)
        timings, current_stage = {}, "starting"

        def stage(name, progress, action):
            nonlocal current_stage
            current_stage = name
            self.queue.advance(claim, name, progress)
            started = time.monotonic()
            settings = get_settings()
            logger.info("stage_started stage=%s input_id=%s model=%s device=%s progress=%s", name, claim.meeting_id, "fake" if settings.demo_mode else settings.asr_mode if name == "asr" else settings.groq_model if name in {"enrichment", "summary"} else "none", "remote" if name in {"asr","enrichment","summary"} else "cpu", progress)
            outcome = "failed"
            try:
                result = action()
                outcome = "ok"
                return result
            finally:
                timings[name] = round(time.monotonic() - started, 6)
                self.queue.advance(claim, name, progress, timings)
                logger.info("stage_finished stage=%s seconds=%.6f outcome=%s progress=%s", name, timings[name], outcome, progress)

        try:
            logger.info("job_running job_id=%s", claim.id)
            thread.start()
            with self.sessions() as db:
                record = db.get(MeetingSession, claim.meeting_id)
                if record is None:
                    raise LostClaim("Meeting deleted")
                db.expunge(record)
            payload = self.processor(record, self.store, stage)
            self.queue.finish(claim, payload)
            logger.info("job_done job_id=%s", claim.id)
        except LostClaim:
            logger.warning("job_claim_lost job_id=%s", claim.id)
        except Exception as exc:
            # Exception strings can contain transcript text, URLs, or credentials.
            self.queue.fail(claim, current_stage + " failed")
            logger.error("job_failed job_id=%s stage=%s exception_type=%s", claim.id, current_stage, type(exc).__name__)
        finally:
            stop.set()
            if thread.ident is not None:
                thread.join(timeout=5)
            request_id.reset(token)
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="IndicMeet database job worker")
    parser.add_argument("--once", action="store_true", help="Claim at most one job, then exit")
    args = parser.parse_args(argv)
    configure_logging()
    settings = get_settings()
    store = configured_storage(MEDIA_DIR)
    worker = Worker(DatabaseJobQueue(lease_seconds=settings.worker_lease_seconds), SessionLocal, store)
    from .worker_health import start_health, pulse
    if not args.once: start_health(store)
    last_retention = 0.0
    try:
        while True:
            pulse()
            worked = worker.run_once()
            if time.monotonic() - last_retention >= 3600:
                purge_expired(SessionLocal, store, settings.retention_days)
                last_retention = time.monotonic()
            if args.once:
                return 0
            if not worked:
                from .services import wait
                wait(settings.worker_poll_seconds)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.error("worker_stopped exception_type=%s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
