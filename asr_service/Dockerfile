# NOT VERIFIED on real GPU/models. Build only on the teammate GPU host.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 ASR_STATE_DIR=/var/lib/asr ASR_DEVICE=cuda:0
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY requirements-ml.txt ./
COPY asr_service/requirements.txt asr_service/requirements.txt
RUN pip install --no-cache-dir -r asr_service/requirements.txt
COPY asr_service asr_service
COPY indicmeet indicmeet
COPY backend/media.py backend/media.py
COPY backend/__init__.py backend/__init__.py
COPY backend/logging_config.py backend/logging_config.py
RUN useradd --uid 10001 --create-home asr && mkdir -p /var/lib/asr && chown asr:asr /var/lib/asr
USER asr
EXPOSE 9002
CMD ["python","-m","asr_service"]
