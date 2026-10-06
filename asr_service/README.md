# Reference ASR host (v2)

**NOT VERIFIED on real GPU/models**: model installation/download, CUDA Docker build, model/API compatibility, accuracy and performance. All local tests use an injected fake Model; importing the service loads no model. This service wraps the reconstructed existing pipeline, not a claim of equivalence with the original executed Kaggle notebook.

Read ../docs/ASR_API_CONTRACT.md first. Single FastAPI process and one in-process GPU worker; disk-persisted jobs/results/idempotency keys survive restart. Do not run multiple uvicorn workers or replicas sharing ASR_STATE_DIR: process-local queues/locks are not a distributed GPU scheduler. Host restart requeues running work under the original ID. Completion/failed results retain >=24h; unfinished jobs are not expired. Secure durable state volume contains audio/transcripts and must have host-owned backup/retention/access policy. A worker blocked in a model call requires process restart; HTTP polling remains bounded on the app side.

## Estimated sizing, not measured

Starting estimate: one NVIDIA GPU with 24–32 GB VRAM (48 GB may be needed for concurrent loaded model allocations), 32–64 GB system RAM, 50–100 GB model/cache disk plus retained recordings/results, NVIDIA container runtime and CUDA12.8-compatible driver. Model downloads/revisions/license terms and actual peak VRAM are operator acceptance items. One GPU worker does not guarantee that all models fit. Load-test representative long audio before promising throughput.

## Environment

- ASR_SERVICE_TOKEN: required bearer, >=16 characters; private env/secret, not Docker build arg.
- ASR_MODEL_VERSION: required nonempty deployment/model revision identifier; appears in health and every segment.
- HF_TOKEN (or HUGGINGFACE_TOKEN): gated model token; host must accept model terms before use.
- WHISPER_MODEL=large-v3; WHISPER_COMPUTE_TYPE=float16; WHISPER_BEAM_SIZE=5.
- INDICCONFORMER_MODEL=ai4bharat/indic-conformer-600m-multilingual; MMS_LID_MODEL=facebook/mms-lid-256; ASR_INDIC_DECODER=ctc; ASR_MAX_INDIC_CHUNK_SECONDS=8.
- ASR_DEVICE=cuda:0; PYANNOTE_DEVICE=cuda; PYANNOTE_MODEL=pyannote/speaker-diarization-3.1 (choose authorized compatible revision).
- ASR_STATE_DIR=/var/lib/asr; ASR_RETENTION_SECONDS=86400 minimum; ASR_MAX_DURATION_SECONDS=7200; ASR_MAX_UPLOAD_BYTES=536870912; ASR_MAX_QUEUED_JOBS=32; ASR_SYNC_TIMEOUT_SECONDS=300.

## GPU host run (not executed locally)

Use separate Python3.11/3.12 environment; requirements-ml.txt exact model pins and asr_service/requirements.txt exact web pins. Ubuntu24.04 CUDA image uses system Python3.12. Transitive GPU requirements/model remote-code dependencies still need real host resolution; this is not a verified GPU lockfile. Never install this environment into app containers.
```sh
python -m venv .venv-asr
. .venv-asr/bin/activate
pip install -r asr_service/requirements.txt
# Export private token/model version/HF token/device before launch.
python -m asr_service
# Or build from repo root, only on the GPU host:
docker build -f asr_service/Dockerfile -t indicmeet-asr:v2 .
docker run --rm --gpus all --env-file /private/asr.env -p 127.0.0.1:9002:9002 -v asr-state:/var/lib/asr indicmeet-asr:v2
```
When turns are omitted, pyannote generates turns; Whisper LID is primary for English, MMS cross-checks Indic routes, IndicConformer uses CTC and bounded silence splits, quality gate retains review/rejected text for inspection. Model loading is lazy inside the sequential worker. Full model version must cover all participating models and routing revision.

## Verify

Health is liveness/version and can pass before first model load. Authenticated upload/job completion proves model readiness. Prepare the approved tiny fixture locally:
```sh
ffmpeg -nostdin -i fixtures/demo.mp4 -ar 16000 -ac 1 -c:a pcm_s16le /tmp/fixture.wav
export ASR_SERVICE_URL=http://127.0.0.1:9002
python scripts/asr_contract_smoke.py --url "$ASR_SERVICE_URL" --audio /tmp/fixture.wav
```
Also run docs/ASR_API_CONTRACT.md curl calls (401, async 202/poll, sync small, >=300s sync rejection), restart while a job is running, poll same ID, inspect metadata/version, retain completed result for24h, then evaluate human-checked native transcripts/routing/diarization separately. The same contract client is tested against scripts/fake_asr_server.py in pytest without downloads. Local `python -m pytest -q tests/test_asr_async.py tests/test_asr_service.py` is the fake-model contract suite. It is not a GPU accuracy benchmark.
# Explicit CPU public-content acceptance mode

ASR_DEVICE=cpu selects sequential CPU model passes in an isolated child process: faster-whisper int8 (WHISPER_MODEL default small), MMS-LID, IndicConformer. Models are released between passes; no pyannote on CPU and nonempty request turns required. ASR_RAM_HEADROOM_GB defaults3, ASR_CPU_THREADS defaults2, ASR_CPU_MAX_SECONDS defaults10800. A psutil guard checks before loads/segments, and the parent samples available host RAM every second and terminates the child process tree below threshold. Job metrics include sampled peak process-tree RSS, elapsed time and successful model-stage timings. RAM aborts fail with RAM_SAFETY_ABORT; they do not yield fake transcripts. CPU child diagnostics inherit the private service log. Local-real binds the native ASR service to127.0.0.1.

Native Windows acceptance uses the existing .venv first; its ML versions are not the pinned GPU requirements and are recorded in REAL_RUN_EVIDENCE.md. Do not infer GPU-image compatibility from CPU execution. See docs/HANDOFF.md for local-real infrastructure, memory limits, WSL2 notes and GPU probe. Exact model revisions still require operator pinning before a production model release.
