# Deployment package
Browser -> nginx frontend (/api proxy) -> FastAPI -> PostgreSQL jobs/session metadata + S3 recordings/artifacts. Worker owns DB leases, reads staged S3 artifacts, invokes the separately hosted ASR and Groq, writes canonical transcript/cited intelligence. Redis wakes workers; DB is durable queue source of truth. No GPU models are inside application containers.

## Local demo
Docker Desktop must run Linux containers. From repo root:
```
docker compose up --build -d --wait
python scripts/smoke_test.py
cd frontend
npm ci
npx playwright install chromium
npm run test:e2e
```
Open http://127.0.0.1:8080; account demo / demo-password. No Groq calls or GPU downloads. MinIO official images/binary distribution were unavailable during this release, so deploy/minio/Dockerfile builds the pinned official release source with two build workers; initial build takes longer. The repo does not contain its binary.

## Production
Set DEMO_MODE=false, AUTH_USERS_JSON with password hashes, random MEDIA_SIGNING_SECRET, database/storage secrets, exact HTTPS origin, ASR_MODE=remote, ASR_SERVICE_URL/token, GROQ_API_KEY. Follow ENVIRONMENT.md. Compose local public passwords are demo defaults only. Provision supported external DB/S3/queue and TLS/domain/hosts; adapt Compose or your deployment platform to those endpoints. Restrict API/worker internal ports and storage access; nginx handles browser traffic. Do not expose the local MinIO demo publicly.

## Startup and migrations
DB/Redis/MinIO become healthy, storage-init creates bucket, API and worker start. Each app entrypoint runs Alembic under a Postgres advisory lock then execs the service. Repeated `alembic upgrade head` is safe. Manual migration: `docker compose run --rm api python -m alembic upgrade head` requires overriding image entrypoint: `docker compose run --rm --entrypoint python api -m alembic upgrade head`.

Readiness: API /readyz checks mapped schema, ffprobe, storage and Redis; worker :8081/readyz checks DB/storage/Redis and initialized worker loop. /healthz is liveness. `docker compose ps` shows health. Logs are JSON stage/job/request metadata without transcripts, credentials or signed URLs; nginx and uvicorn URL access logs are disabled. Use `docker compose logs --tail 50 api worker` for failures.

## Ports, volumes, scaling
Loopback frontend8080/API8000/workerhealth8081; DB5432,Redis6379,MinIO9000 internal only. Named db-data,queue-data,storage-data hold durable data; api-staging/worker-staging are reconstructible caches. For multiple workers remove the single published worker health port, then `docker compose up -d --scale worker=2`. Atomic DB claims/renewal prevent double execution. Do not share local-only filesystem storage across hosts; use S3.

## Backup and retention
Back up Postgres and S3 together with their key references; verify restore before rollout. Preserve original transcripts and original AI output; edits are separate rows. Retention defaults disabled; RETENTION_DAYS=N removes completed/failed meetings through the same guarded deletion path. Never run retention against irreplaceable ground-truth/output/Kaggle directories. Local staging caches and LLM cache need their own bounded disk/privacy policy on deployed hosts. Stop with `docker compose down`; named volumes persist. No -v is used by verification.

## Verification
`scripts/check.ps1 -Python <venv-python>` or `scripts/check.sh` runs lint, suite, fixture schema and clean build. `python scripts/smoke_test.py --url http://127.0.0.1:8000` exercises login/upload/worker/citations/signed Range/edit/revert/export/delete/logout. Playwright `npm run test:e2e` asserts actual playback/seek, persisted edit, downloaded JSON file and deletion. All proof runs require explicit demo mode; real ASR/diarization accuracy, Groq Indian-language output quality and real LiveKit data are separate integration acceptance.

Compose optionally loads the ignored .env for all application settings; explicit service wiring takes precedence. S3_ENDPOINT_URL, S3_BUCKET, S3_REGION and REDIS_URL are configurable. Successful S3 uploads and completed worker jobs evict reconstructible staging copies; local durable storage is never evicted. Configure a private writable INDICMEET_LLM_CACHE on production workers and its disk/retention policy.
