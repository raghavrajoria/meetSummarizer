# meetSummerizer

meetSummerizer is a local meeting intelligence application for importing recordings and speaker-labeled transcripts, reviewing evidence-linked summaries, and playing transcript segments alongside the meeting media.

## Folder map

- `backend/` — FastAPI service, SQLite persistence, and media storage adapter.
- `frontend/` — React meeting review app (Vite) plus its shared CSS design system.
- `indicmeet/` — current ASR and summary pipeline modules.
- `experiments/` — legacy scripts, notebooks, and model experiments.
- `fixtures/` — small sample session data and media for local development.
- `data/` — local datasets and generated artifacts; excluded from Git.
- `docs/` — project handoff and work log.
- `notebooks/` — project notebooks; credential-bearing notebooks stay excluded.

## Local API

Use the root `.venv` (Python 3.10+) and install the pinned backend additions with the commands below. Set `API_TOKENS` in `.env`, then migrate before starting the API:

```powershell
.venv\Scripts\python -m alembic upgrade head
.\scripts\run_local.ps1
```

In a second terminal, start the React development server:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (normally `http://localhost:5173`). Vite proxies `/api` requests to the local backend. To build the production UI, run `npm run build` from `frontend/`; the static bundle is written to `frontend/dist/`. For a deployed API, define `window.MEETINGS_API_BASE` before the React entry script loads.

Import the sample session and its matching three-minute media clip with:

```powershell
curl.exe -X POST http://127.0.0.1:8000/sessions/import -H "Authorization: Bearer YOUR_TOKEN" `
  -F "session_json=<fixtures/session.json" `
  -F "media=@fixtures/group_discussion_3min.mp4"
```

The API returns the UI's existing meeting object shape: `id`, `title`, `group`, `date`, `dateLabel`, `time`, `media`, `status`, `summary`, `speakers`, `intelligence`, and `segments`. `intelligence` retains `discussed`, `keyDiscussion`, `decisions`, `actionItems`, `followUps`, `questions`, and `concerns`. Speakers retain `id`, `name`, `role`, `confidence`, `join`, and `leave`; transcript segments retain `t`, `time`, `speaker`, `lang`, `tx`, and `en`. Optional fields such as `quality`, `verified`, `evidence`, `end`, and `roman` add review state, evidence links, estimated transcript intervals, or transliteration without replacing the UI's established fields.

The SQLite database and imported media are written under the Git-ignored `data/` directory. Media access supports byte ranges for seeking. Local files use the `LocalStorage` implementation behind the `Storage` protocol so a future S3 implementation can replace it.

## Tests and processing pipeline

Install `backend/requirements-dev.txt` into the root `.venv` to run the offline summary regression and API tests with `.venv\Scripts\python -m pytest`. The ASR and diarization stages require the ML dependencies and a suitable GPU environment. For local/offline runs, precompute those stages and provide their JSON/CSV outputs:

```powershell
python -m indicmeet.pipeline --media recording.mp4 --session-id demo `
  --diar-csv diarization.csv --asr-json asr.json --dry-summary summary.json
```

The pipeline caches stage results in `data/sessions/<id>/`; use `--force` to rerun stages, and authenticated `curl.exe` imports to send the assembled session and media to the API. The legacy CLI `--import-url` helper does not supply a bearer token.

## Milestone 2: authenticated API and database worker

All routes, including `/sessions`, `/docs`, `/openapi.json`, downloads and job endpoints, require `Authorization: Bearer <token>`. Configure one or more tokens using comma-separated `API_TOKENS`. Missing/wrong credentials produce the same 401 response. An empty token list denies every protected request. `/healthz` and `/readyz` are public. CORS preflight is handled by the CORS middleware; authenticated requests require an allowed `CORS_ORIGINS` origin. The frontend has not yet been changed to supply bearer tokens.

New dependencies are pinned in `backend/requirements.txt`: Alembic 1.16.5 (migrations), Mako 1.3.10 (Alembic templates), and psycopg[binary] 3.2.10 (Postgres). No Redis/Celery package is required. Install and run everything from the root environment:

```powershell
cd D:\crazibrain\indicmeet-ai
.venv\Scripts\python -m pip install -r backend\requirements-dev.txt
.venv\Scripts\python -m alembic upgrade head
.\scripts\run_local.ps1
```

The launcher migrates, starts the API and a separate polling worker, writes their logs under `data/logs/`, then stops both on Ctrl+C or if either process exits. Use `-Port 8001` to choose another API port. To run in separate terminals for direct console logs:

```powershell
.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --no-access-log
.venv\Scripts\python -m backend.worker
```

SQLite is the local default. For Postgres, set `DATABASE_URL=postgresql://user:password@host/database`; it selects the pinned psycopg 3 driver. Do not put the URL in `alembic.ini`. Migration `0001_meetings_jobs` creates the current meeting schema and jobs, or adopts an existing Milestone 1 meeting table after checking its column set. Existing meetings remain readable. Downgrading removes jobs but deliberately preserves meetings and media. Back up the database/media before migration; offline SQL assumes a fresh schema.

| Method | Route | Behavior |
| --- | --- | --- |
| POST | `/meetings` | Multipart `recording` plus optional `asr_json`, `diarization_csv`, `attendees_json` files; optional `title`, `group`, `date` text. Returns 202 with meeting and queued job. |
| GET | `/meetings`, `/meetings/{id}` | List/detail; legacy `/sessions` routes remain available. |
| GET | `/jobs/{id}`, `/meetings/{id}/job` | State, stage, percentage, safe failure code, UTC timestamps, attempt count and per-stage seconds. |
| POST | `/jobs/{id}/retry` | Requeue a failed job using retained source files; other states return 409. |
| DELETE | `/meetings/{id}` | Delete files and meeting/job rows. Running jobs return 409; retry after completion. |
| GET | `/meetings/{id}/transcript` | JSON attachment when done; 409 while not ready. |
| GET | `/meetings/{id}/media` | Authenticated media with byte-range support. |
| GET | `/healthz`, `/readyz` | Liveness; readiness checks migrated DB, writable media storage and ffprobe availability. |

`MAX_UPLOAD_MB` defaults to 512 MiB and caps the entire streamed request, including metadata and multipart overhead. Uploads stop at the limit, and failed uploads remove their generated directory. Filenames are sanitized metadata; paths use generated IDs. ffprobe must find an audio stream and a finite positive duration, regardless of extension or Content-Type. `FFPROBE_BINARY`, `FFMPEG_BINARY`, and `MEDIA_TIMEOUT_SECONDS` (default 300) configure media tools and subprocess timeouts. Keep ffprobe/ffmpeg installed on the host.

An uploaded ASR JSON uses the Milestone 1 import provider and skips audio extraction/GPU processing. Without it, the worker extracts mono 16 kHz WAV and calls `get_asr_provider(ASR_MODE)`. The M1 provider interface supports `remote` and `import`; choose `remote` and configure `ASR_SERVICE_URL`/optional `ASR_SERVICE_TOKEN` for recordings without ASR JSON. `import` without a file and the legacy CLI's `local` mode fail safely in the worker. Optional diarization CSV requires start/end columns and assigns speakers using maximum time overlap. Attendee JSON accepts a list of names or an object with `attendees`/`participants` and `aliases`. Summary generation uses existing Groq configuration and its existing fallback behavior.

The `JobQueue` interface separates endpoints from database scheduling. Workers atomically claim jobs, heartbeat while processing and refuse stale ownership. `WORKER_POLL_SECONDS` defaults to 2; `WORKER_LEASE_SECONDS` defaults to 180. Expired running jobs become failed and require explicit retry. Per-stage timings are recorded even on stage failure. JSON logs contain a request ID (startup uses `-`), static event names and safe exception class names; transcript text, request bodies and raw exception messages are not logged. Send a valid `X-Request-ID` to correlate upload and worker events.

`RETENTION_DAYS=0` disables retention. A positive value removes expired done/failed meetings and their files, checked hourly by the worker. Queued/running jobs are retained. Meetings predating the job table have no job row and are not automatically purged. Database and filesystem deletion cannot be one transaction: rows commit after files are removed; retry deletion after a storage/database failure.

Tests use temporary SQLite/storage plus scoped mocks for media/provider/summary calls. They do not require GPU, Groq, an ASR service or a running API/worker. Migration tests use temporary databases and emit PostgreSQL DDL; a live Postgres integration run remains a separate deployment check.

```powershell
.venv\Scripts\python -m pytest -q --tb=short tests\test_backend_api.py tests\test_backend_security.py tests\test_backend_uploads.py tests\test_backend_jobs.py tests\test_backend_worker.py tests\test_backend_migrations.py
.venv\Scripts\python -m pytest -q --tb=short
```
