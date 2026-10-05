FROM python:3.10.16-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 INDICMEET_DATA_DIR=/var/lib/indicmeet
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY constraints.txt requirements-llm.txt ./
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt -r requirements-llm.txt
COPY indicmeet indicmeet
COPY backend backend
COPY alembic.ini ./
COPY fixtures/demo_asr.json fixtures/demo_asr.json
RUN useradd --uid 10001 --create-home app && mkdir -p /var/lib/indicmeet && chown app:app /var/lib/indicmeet
USER app
ENTRYPOINT ["python", "-m", "backend.entrypoint"]
