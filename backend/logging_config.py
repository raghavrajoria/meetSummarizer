"""JSON logging with request correlation and no input/exception payloads."""

import contextvars
import json
import logging
import time
import uuid

from starlette.responses import JSONResponse

request_id = contextvars.ContextVar("request_id", default="-")
logger = logging.getLogger("indicmeet.api")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "time": self.formatTime(record), "level": record.levelname,
            "request_id": getattr(record, "request_id", request_id.get()),
            "event": record.getMessage(),
        })


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.getLogger().handlers = [handler]
    logging.getLogger().setLevel(logging.INFO)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        child = logging.getLogger(name)
        child.handlers = []
        child.propagate = True
    # Multipart's debug messages include uploaded bytes; keep them disabled.
    logging.getLogger("python_multipart").setLevel(logging.WARNING)


class RequestLogMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"x-request-id", b"").decode("ascii", errors="ignore")
        rid = supplied if 0 < len(supplied) <= 64 and all(c.isalnum() or c in "-_" for c in supplied) else uuid.uuid4().hex
        token = request_id.set(rid)
        scope.setdefault("state", {})["request_id"] = rid
        started, status = time.monotonic(), 500
        response_started = False

        async def correlated_send(message):
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                message["headers"] = list(message.get("headers", [])) + [(b"x-request-id", rid.encode("ascii"))]
            await send(message)

        try:
            await self.app(scope, receive, correlated_send)
        except Exception as exc:
            logger.error("request_failed exception_type=%s", type(exc).__name__)
            if response_started:
                raise
            await JSONResponse({"detail": "Internal server error"}, status_code=500)(scope, receive, correlated_send)
        finally:
            # Do not log request URLs, query strings, headers, or response bodies.
            logger.info("request_completed method=%s status=%s seconds=%.3f", scope["method"], status, time.monotonic() - started)
            request_id.reset(token)
