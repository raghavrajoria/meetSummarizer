"""Worker HTTP health/readiness without credentials or meeting payloads."""
import json,os,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
last_pulse=0
def pulse():
    global last_pulse
    last_pulse=time.monotonic()
def start_health(store):
    from .database import SessionLocal
    from .models import Job
    from .services import dependencies_ready
    from sqlalchemy import select
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            ready=self.path=="/healthz"
            if self.path=="/readyz":
                try:
                    with SessionLocal() as db:db.execute(select(Job).limit(1))
                    ready=dependencies_ready(store) and last_pulse>0
                except Exception:ready=False
            code=200 if ready else 503
            self.send_response(code);self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(json.dumps({"status":"ready" if ready else "not_ready"}).encode())
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(("0.0.0.0",int(os.environ.get("WORKER_HEALTH_PORT","8081"))),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
