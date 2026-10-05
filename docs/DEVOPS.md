# Deployment / runtime acceptance

Browser -> nginx /api proxy -> FastAPI -> external PostgreSQL job leases/metadata, S3 recordings/artifacts and Redis wakeups. Worker submits one async ASR job per audio scope, persists ID, resumes polling, gates review, enriches/summarizes with Groq and stores canonical output. No GPU model is in app images. Reference GPU service lives separately in asr_service/.

## Explicit local demo
Prerequisites Docker Linux engine and Compose v2.24+; follow README exactly:
```sh
docker compose --env-file .env.demo.example up --build -d --wait
docker compose --env-file .env.demo.example ps
# browser localhost8080, demo/demo-password, fixtures/demo.mp4 or demo.webm
docker compose --env-file .env.demo.example down
```
Only selecting the public demo env enables fake behavior. Local services are DB/Redis/source-built MinIO; app ports bind loopback, data services are internal. MinIO official source is pinned; first build is slow. It is a local integration fixture, not a supported production storage recommendation. Never expose this stack publicly. Named volumes survive down; no -v in verification.

## Production: standalone external-service template

Copy .env.example to ignored .env and privately fill all required values in docs/ENVIRONMENT.md. Provision PostgreSQL, Redis, supported S3 bucket/endpoint and teammate HTTPS ASR v2. Set private users/signing/provider secrets and exact HTTPS domain. Use **only** this Compose file; do not overlay the demo file (that would retain demo services):
```sh
docker compose -f docker-compose.prod.example.yml --env-file .env config --quiet
docker compose -f docker-compose.prod.example.yml --env-file .env up --build -d --wait
docker compose -f docker-compose.prod.example.yml --env-file .env ps
```
No MinIO, fake ASR, fake LLM, DB or Redis is provisioned by this template. API and worker internal only; frontend bound127.0.0.1:8080; operator terminates HTTPS and proxies domain there. Production startup refuses unsafe/default/demo settings before migrations, so configuration errors cannot accidentally produce a public demo. Config validation is exercised with fakes; actual external infrastructure/production deployment NOT VERIFIED.

## Migrations, health and recovery

Entrypoint runs Alembic head under Postgres advisory lock; API and worker may start together safely. Migration3 adds remote_asr_job_id/asr_requests to jobs. It preserves existing meetings/jobs. Manual upgrade:
```sh
docker compose -f docker-compose.prod.example.yml run --rm --entrypoint python api -m alembic upgrade head
```
API /healthz liveness, /readyz DB/schema/ffprobe/S3/Redis; worker8081/readyz initialized loop plus dependencies. ASR /healthz is host liveness/version; actual upload completion tests model readiness. Production accepts external readiness failures as failures; it does not substitute fixtures. Worker lease expires -> confirmed remote ID resumes polling; uncertain submissions fail for explicit retry. Total ASR deadline is persisted across restarts (3h default). Failed/timeout explicit retry may submit new GPU work; uncertain response should be investigated at host first.

Logs: JSON metadata job/request IDs, stage/progress/model/device/elapsed/outcome, no transcript/audio filename/secret/request URL. nginx/uvicorn access logs are off. Avoid printing `.env` or `docker compose config` without --quiet (resolved secrets). S3 staging is a reconstructible cache; success evicts local copies. LLM cache needs private storage and bounded retention. Named staging volumes do not contain authoritative artifacts.

## Acceptance and operations

Follow docs/HANDOFF.md curl checks and scripts/smoke_test.py on an isolated demo/staging tenant; browser tests upload/edit/delete actual records, so never point them at real production meetings. Real-mode smoke requires representative approved audio and hosted ASR/Groq, not DEMO=true. Run rate/long-audio/error/restart acceptance on the ASR host; review model accuracy with humans.

Restrict network/API/storage access, configure DB/S3 consistent backups, test restore, set RETENTION_DAYS explicitly, and apply ASR host >=24h result retention. Never retention-clean original ground_truth/output/Kaggle paths. For multiple app workers remove single host health publication and scale; DB leases prevent duplicate claims. ASR reference service is single process/single GPU queue, no shared-state replicas. Original executed Kaggle equivalence, GPU requirements/image, real Groq/ASR/LiveKit accuracy/integration remain NOT VERIFIED.
