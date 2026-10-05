# IndicMeet AI
Recorded Indian-language meetings become speaker-attributed native/Roman/English transcripts and cited meeting intelligence. Original-language text remains immutable. React frontend, FastAPI API, durable database-job worker, signed media playback, separate summary edits, canonical segments and external GPU contracts.

## Quickstart: offline demo
From repo root with Docker Desktop running:
```
docker compose up --build -d --wait
```
Open http://127.0.0.1:8080, sign in demo / demo-password, upload fixtures/demo.mp4. Demo is visibly labeled and uses stored ASR/fake LLM only. It proves application plumbing, not model quality. First build compiles pinned MinIO official source because its public images were unavailable.

## Local Python/React
Use Python 3.10; create a new venv, install `pip install -r backend/requirements-dev.txt -r requirements-llm.txt`. Set DEMO_MODE=true, an isolated INDICMEET_DATA_DIR; `python -m alembic upgrade head`; launch `python -m uvicorn backend.main:app --port 8000 --no-access-log` and `python -m backend.worker`. In frontend: `npm ci && npm run dev`. Vite proxies /api to port8000. Keep GPU packages in a separate Python3.11 environment using requirements-ml.txt; never load large models on this workstation.

## Checks and deployment
`scripts/check.ps1 -Python <venv-python>` (Windows) or scripts/check.sh. Running stack: `python scripts/smoke_test.py`, then frontend `npm run test:e2e` (install Chromium once with `npx playwright install chromium`). Fresh-clone verification is recorded in docs/RELEASE_EVIDENCE.md after execution.

Real mode requires DEMO_MODE=false, hashed login users, signing secret, GROQ_API_KEY, ASR_MODE=remote and ASR_SERVICE_URL/token. See [environment](docs/ENVIRONMENT.md), [deployment](docs/DEVOPS.md), [ASR contract](docs/ASR_API_CONTRACT.md), [GPU procedure](docs/GPU_DIARIZATION.md), [decision log](DECISIONS.md), and [engineering requirements](AGENTS.md).

Small tracked fixtures are under fixtures/. Full irreplaceable reference data stays in the existing ignored data/ground_truth and output/Kaggle locations; do not overwrite them. Reference subsets are explicitly soft LLM-generated references, not human ground truth or accuracy proof.
