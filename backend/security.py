"""Hashed credentials, revocable short sessions, and scoped media signatures."""
import hashlib
import hmac
import json
import os
import secrets
import time
from urllib.parse import parse_qs
from starlette.responses import JSONResponse
from .database import SessionLocal
from .models import AccessSession
from indicmeet.settings import get_settings

def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310000).hex()
    return "pbkdf2_sha256$310000$" + salt + "$" + digest

def password_matches(password, stored):
    try:
        kind, rounds, salt, digest = stored.split("$")
        if kind != "pbkdf2_sha256" or rounds != "310000":
            return False
        return hmac.compare_digest(password_hash(password,salt),stored)
    except (ValueError, TypeError):
        return False

def users():
    configured = json.loads(os.environ.get("AUTH_USERS_JSON") or "{}")
    if get_settings().demo_mode and not configured:
        return {"demo": {"password_hash": password_hash("demo-password", "demo-only"), "role": "editor"}}
    return configured

def secret():
    value = os.environ.get("MEDIA_SIGNING_SECRET", "")
    if not value and get_settings().demo_mode:
        value = "local-demo-only-do-not-use-in-production"
    if len(value) < 32:
        raise ValueError("MEDIA_SIGNING_SECRET must contain at least 32 characters")
    return value.encode()

def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()

def media_signature(path, expires, session):
    return hmac.new(secret(), f"{path}|{expires}|{session}".encode(), hashlib.sha256).hexdigest()

def lookup(token):
    with SessionLocal() as db:
        record = db.get(AccessSession,token_hash(token))
        if record and record.expires > int(time.time()):
            return {"username": record.username, "role": record.role, "session": record.id}
    return None

def signed_identity(scope):
    query = parse_qs(scope.get("query_string",b"").decode())
    try:
        expires, session, signature = (query[k][0] for k in ("expires","session","signature"))
        if int(expires) <= int(time.time()) or int(expires)>int(time.time())+3600:
            return None
        if not hmac.compare_digest(signature,media_signature(scope["path"],expires,session)):
            return None
        if session == "integration":
            return {"role":"viewer"} if get_settings().api_tokens else None
        with SessionLocal() as db:
            row = db.get(AccessSession,session)
            if row and row.expires > int(time.time()):
                return {"username":row.username,"role":row.role,"session":row.id}
    except (KeyError, ValueError, UnicodeError):
        return None

class AuthMiddleware:
    def __init__(self, app):
        self.app=app
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in {"/healthz","/readyz","/auth/login","/config"}:
            return await self.app(scope,receive,send)
        scheme,_,candidate=dict(scope.get("headers",[])).get(b"authorization",b"").partition(b" ")
        identity=None
        if scheme.lower()==b"bearer":
            token=candidate.decode(errors="replace")
            valid = False
            for configured in get_settings().api_tokens:
                valid |= hmac.compare_digest(candidate, configured.encode())
            if valid:
                identity={"role":"editor","session":"integration"}
            elif token:
                identity=lookup(token)
        if not identity and scope["method"]=="GET" and scope["path"].endswith("/media"):
            identity=signed_identity(scope)
        if not identity:
            return await JSONResponse({"detail":"Unauthorized"},status_code=401,headers={"WWW-Authenticate":"Bearer"})(scope,receive,send)
        if scope["method"] not in {"GET","HEAD","OPTIONS"} and identity["role"]=="viewer" and scope["path"]!="/auth/logout":
            return await JSONResponse({"detail":"Read-only account"},status_code=403)(scope,receive,send)
        scope.setdefault("state",{})["identity"]=identity
        await self.app(scope,receive,send)
