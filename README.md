# IndicMeet AI

React frontend, FastAPI API, database-backed worker and language-routed ASR service for recorded meetings. Original transcripts are preserved; summaries and action items cite segment IDs. PostgreSQL stores meeting payloads, Redis handles queue coordination, and S3-compatible storage holds recordings.

## Local demo

Start Docker Desktop (Linux containers), then run from the repository root:

```powershell
docker compose --env-file .env.demo.example up --build -d --wait
```

Open http://127.0.0.1:8080/. Local demo login: demo / demo-password. This profile serves precomputed recordings; live uploads are disabled. A fresh clone contains no real recordings or model output. To reproduce the owner's demo, obtain the approved local media/artifacts described in docs/DEMO.md, then run `python scripts/load_demo_meetings.py`. The loader refuses unaudited artifacts and non-demo profiles. No models or Groq calls are made by the loader.

Stopping with `docker compose --env-file .env.demo.example down` preserves named volumes. Never remove volumes containing recordings or transcripts accidentally.

## Development

Python3.10, Node20, FFmpeg and FFprobe are required for development/testing. Install isolated application/test dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt -r requirements-llm.txt
cd frontend
npm ci
npm run dev
```

Vite refreshes edits automatically and proxies `/api` to the backend on8000. Keep the Docker backend running. To deploy frontend changes locally:

```powershell
docker compose --env-file .env.demo.example build frontend
docker compose --env-file .env.demo.example up -d --no-deps --wait frontend
```

Run `scripts/check.ps1 -Python .venv\Scripts\python.exe` or `scripts/check.sh` for lint, Python tests, contract validation and frontend build. Required small fixtures are maintained for offline tests. Two real-artifact tests skip in fresh clones without owner-provided data. Browser fixture tests use intercepted APIs; upload smoke/workflow tests must use a separate disposable demo stack with DEMO_PRECOMPUTED_ONLY=false, never the presentation stack. Real-demo browser checks require E2E_REAL_DEMO=true and the approved local meetings.

## Production

Use docker-compose.prod.example.yml with external PostgreSQL, Redis and S3-compatible storage; a domain/HTTPS proxy; non-demo hashed accounts and private credentials. Production requires the authenticated HTTPS remote ASR v2 service and real Groq credentials. Read docs/DEVOPS.md, docs/ENVIRONMENT.md, docs/HANDOFF.md and docs/ASR_API_CONTRACT.md. The GPU ASR host is separate; see asr_service/README.md. Public hosting and complete live-model acceptance remain unverified.

## Repository scope

`main` contains application code, deployment files, migrations, dependencies, tests, small required fixtures and current documentation. Full development history, experiments, legacy UI scripts and historical reports remain on branch codex/archive-development-2026-10-07. See docs/REPOSITORY_LAYOUT.md. Secrets, recordings, model weights, caches and local environments remain ignored; archive branches do not include those private local files. No history was rewritten.
