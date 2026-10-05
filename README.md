# IndicMeet AI

Recorded meetings become speaker-attributed native/Roman/English transcripts and cited meeting intelligence. React frontend, FastAPI, DB-leased worker, canonical review gate, signed playback, separate saved summary edits and hosted asynchronous ASR contract v2. Original native text stays immutable. Real model accuracy is not proven by the demo.

## Quickstart: explicit offline demo

Prerequisites: Git, Docker Desktop/Engine running Linux containers with Compose v2.24+, available loopback ports8080/8000/8081, disk/network for the first image build. No Python/GPU/secret is needed for the browser demo. Run literally from the cloned repository:
```sh
git clone --branch release/complete https://github.com/raghavrajoria/meetSummerizer.git
cd meetSummerizer
docker compose --env-file .env.demo.example up --build -d --wait
docker compose --env-file .env.demo.example ps
```
The release branch must first be pushed by the owner; before push, clone the local release branch for verification. Select `.env.demo.example` explicitly: it sets DEMO=true and public local credentials. Without it, demo defaults are not enabled and missing configuration fails. Open http://127.0.0.1:8080, log in demo / demo-password, upload fixtures/demo.mp4 (Chrome/Edge) or fixtures/demo.webm (bundled test Chromium). Watch processing, view transcript/summary, play/seek citations, save an edit, reload, download transcript, then delete. Banner says stored ASR/fake LLM; Indic demo translation/romanization is explicitly unavailable. First build compiles pinned MinIO source because public images were unavailable; cached subsequent builds are faster.
```sh
docker compose --env-file .env.demo.example down
```
Named volumes persist; do not remove them accidentally. No large models or paid Groq calls occur.

## Local development and checks

Python3.10 and Node20 are app/test prerequisites. New venv: `pip install -r backend/requirements-dev.txt -r requirements-llm.txt`. Explicit demo environment DEMO=true APP_ENV=demo, isolated INDICMEET_DATA_DIR; `python -m alembic upgrade head`; API `python -m uvicorn backend.main:app --port 8000 --no-access-log`; worker `python -m backend.worker`. Frontend `npm ci`, `npm run dev` proxies /api to8000. Production is the default outside explicit demo; don't reuse local credentials.

`scripts/check.ps1 -Python <venv-python>` / scripts/check.sh runs lint, tests, schema and clean build. With running explicit demo: `python scripts/smoke_test.py`; frontend `npm ci`, `npx playwright install chromium`, `npm run test:e2e`. Tests require actual media playback/seek and downloaded JSON, not just HTTP status. `python scripts/release_verify.py` performs new clone/new backend+LLM venvs, installs, migrations, full suite including async/refusal/fake-service tests, clean build, explicit-demo Compose, smoke/e2e/ps/down. GPU requirements are separate; never load those models on this workstation.

Read [handoff](docs/HANDOFF.md), [deployment](docs/DEVOPS.md), [environment](docs/ENVIRONMENT.md), [ASR v2](docs/ASR_API_CONTRACT.md), [reference ASR host](asr_service/README.md), [manager release summary](docs/RELEASE_SUMMARY.md), [decisions](DECISIONS.md), [engineering requirements](AGENTS.md). Runtime output tails and exact tested code hash are in docs/FINISHING_EVIDENCE.md; earlier release evidence is historical.

Production uses docker-compose.prod.example.yml standalone, external PostgreSQL/Redis/S3 and HTTPS remote ASR; private hashed users/tokens/signing secret, exact HTTPS origin. It refuses demo/default/weak configuration. Hosting and real ASR/diarization, real Groq Indian-language quality and real LiveKit acceptance remain team work. Small approved fixtures are tracked; irreplaceable full references/media/output/Kaggle data remain ignored and unchanged.
