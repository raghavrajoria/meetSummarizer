"""Environment bearer authentication for every non-health HTTP route."""

import hmac

from starlette.responses import JSONResponse
from indicmeet.settings import get_settings


class AuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in {"/healthz", "/readyz"}:
            return await self.app(scope, receive, send)
        header = dict(scope.get("headers", [])).get(b"authorization", b"")
        scheme, _, candidate = header.partition(b" ")
        valid = False
        for configured in get_settings().api_tokens:
            valid |= hmac.compare_digest(candidate, configured.encode("utf-8"))
        if scheme.lower() != b"bearer" or not valid:
            return await JSONResponse({"detail": "Unauthorized"}, status_code=401,
                                      headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
        await self.app(scope, receive, send)
