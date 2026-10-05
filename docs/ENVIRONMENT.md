# Environment configuration
Defaults are defined in indicmeet/settings.py; empty values use those defaults. `.env` is local only and ignored; exported variables take precedence. Docker uses Compose interpolation and explicit environment entries; never bake `.env` into images. Local compose is a demo, bound to loopback, with clearly public demo credentials. Production sets DEMO_MODE=false and replaces every demo credential.

| Variables | Default / required / missing behavior |
|---|---|
| DEMO_MODE, STRICT_REVIEW | false, true. Demo injects stored ASR/fake LLM and demo/demo-password only if no users configured. Strict excludes review and rejected before enrichment/summary. |
| AUTH_USERS_JSON | Production required JSON mapping username to `{password_hash,role}`; absent means no login accounts. Role viewer/editor/admin; absent role editor. Generate PBKDF2 using `backend.security.password_hash` in a private terminal; never put plaintext passwords in JSON. |
| ACCESS_TOKEN_SECONDS, MEDIA_SIGNING_SECRET | 900 seconds (bounded 60–3600); signing secret >=32 characters required outside demo, missing signing yields 503 media-url. Demo secret must be replaced before hosting. |
| API_TOKENS | Optional comma-separated integration bearers. Missing disables integration credentials; login still works. Never embed in browser builds. |
| INDICMEET_DATA_DIR, INDICMEET_MEDIA_DIR, DATABASE_URL | repo data/, data/media/, SQLite data/meetSummerizer.sqlite3. Docker uses /var/lib/indicmeet and Postgres. Unavailable DB prevents readiness/migrations. |
| CORS_ORIGINS | localhost/127.0.0.1 on 5173/8001; production use same-origin nginx or exact HTTPS origin. |
| FFPROBE_BINARY, FFMPEG_BINARY, MEDIA_TIMEOUT_SECONDS | ffprobe, ffmpeg, 300. Missing executables prevent media validation/processing; API readiness checks ffprobe. |
| MAX_UPLOAD_MB, IMPORT_REQUEST_TIMEOUT_SECONDS | 512 MB total multipart request; 120 seconds CLI import. nginx allows 513 MB; API applies actual 512 MB bound. Invalid/empty/non-audio media rejected. |
| ASR_MODE, ASR_SERVICE_URL, ASR_SERVICE_TOKEN, ASR_SERVICE_TIMEOUT_SECONDS | import; URL/token absent; 120 seconds. Import needs ASR JSON unless demo. Remote needs full URL; host may require bearer. Failures leave a retryable job. Local requires GPU and explicit turns. |
| GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL, GROQ_TEMPERATURE | Secret required for production LLM; official Groq chat completions URL; openai/gpt-oss-20b; 0.2. Demo ignores keys. |
| GROQ_REQUEST_TIMEOUT_SECONDS, GROQ_MAX_WAIT_SECONDS, INDICMEET_LLM_CACHE | 120,120,repo llm_cache/. Missing key fails production summary; bounded recovery records enrichment failures. Cache contains model responses: keep private, apply retention/backup policy. |
| HF_TOKEN, HUGGINGFACE_TOKEN | GPU host optional alias; gated pyannote/model access requires accepted terms and token. No local model downloads in demo/API. |
| PYANNOTE_MODEL, PYANNOTE_DEVICE | pyannote/speaker-diarization-3.1,auto (GPU procedure sets cuda). Use model version authorized by host; adapter handles Annotation and DiarizeOutput. |
| WHISPER_MODEL, WHISPER_COMPUTE_TYPE, WHISPER_BEAM_SIZE | large-v3,float16,5 on GPU host. |
| INDICCONFORMER_MODEL, MMS_LID_MODEL, ASR_DEVICE | ai4bharat/indic-conformer-600m-multilingual,facebook/mms-lid-256,cuda:0. CPU large ASR fails clearly. |
| ASR_INDIC_DECODER, ASR_MAX_INDIC_CHUNK_SECONDS | ctc,8. Silence splitting prefers energy gaps; equal fallback only when no silence found. |
| WORKER_POLL_SECONDS, WORKER_LEASE_SECONDS, RETENTION_DAYS | 2,180,0 (retention disabled). Lease heartbeat renews while processing; expired jobs become failed, explicit retry required. Retention deletes terminal jobs/media after N days. |
| STORAGE_BACKEND | local; s3 enables boto3 storage with separate staging volumes. Unknown mode fails startup. |
| S3_ENDPOINT_URL, S3_BUCKET, S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY | AWS default endpoint,indicmeet,us-east-1,keys required for MinIO or provider credential chain for AWS. S3 outage fails readiness/storage writes. Do not print credentials. |
| REDIS_URL | Optional; missing uses DB polling. Compose redis://queue:6379/0. Configured Redis failure fails readiness, wakeup falls back to DB polling. |
| WORKER_HEALTH_PORT, SERVICE_ROLE | 8081,api; entrypoint selects worker with SERVICE_ROLE=worker. |
| API_PORT, WORKER_PORT, FRONTEND_PORT | Compose loopback published ports 8000,8081,8080. |
| POSTGRES_PASSWORD | Compose local-demo-password; set private production value and matching DATABASE_URL. |
| E2E_BASE_URL | Test runner http://127.0.0.1:8080; points to demo frontend, never production meetings. |

Production infrastructure overrides are team-owned; use supported managed S3 or your chosen supported service. The archived MinIO source build here exists only to reproduce the requested local integration test.
