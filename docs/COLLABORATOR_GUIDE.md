# Collaborator guide

Start with the repository README.md. The frontend is React/Vite; API and worker are Python. Keep model-host ML dependencies separate from application dependencies.

1. Install backend/requirements-dev.txt and requirements-llm.txt in an isolated Python3.10 environment; run frontend npm ci.
2. Start the configured local Docker backend. Run frontend npm run dev for live UI development; /api proxies to8000.
3. Run scripts/check.ps1 or scripts/check.sh. Canonical transcripts and review gates are specified in AGENTS.md and docs/ASR_API_CONTRACT.md.
4. Production configuration is documented in docs/ENVIRONMENT.md and docs/DEVOPS.md. Store secrets in private environment configuration, never Git. Run migrations through the application entrypoint or Alembic.
5. Use scripts/load_demo_meetings.py only for the explicit precomputed demo and approved local data. Do not run fake-upload tests against its presentation database.
6. Branch organization and archived experiments: docs/REPOSITORY_LAYOUT.md. Historical notebooks/output are not production dependencies.
