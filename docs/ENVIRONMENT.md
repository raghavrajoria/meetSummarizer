# Environment and safe startup

Production is the default. Empty values are absent. Exported environment overrides ignored local .env. Never bake .env or tokens into images. `.env.example` is a blank production template; `.env.demo.example` is public local-only configuration selected explicitly with Compose --env-file. Only DEMO=true permits demo users/stored ASR/fake LLM. DEMO_MODE is obsolete and rejected when true; it cannot enable a demo. APP_ENV=production refuses DEMO=true, import/local ASR, fake/debug flags, default/weak secrets, demo users and CORS wildcard. API lifespan, worker main and Docker entrypoint validate; entrypoint does so before DB connection/migrations. APP_ENV=development is deliberate local development, not a production bypass.

| Variables | Behavior |
|---|---|
| DEMO, APP_ENV | false/production default; explicit DEMO=true defaults demo only when APP_ENV not configured; demo Compose explicitly sets APP_ENV=demo |
| AUTH_USERS_JSON | Production required nonempty mapping username to PBKDF2 password_hash and viewer/editor/admin role; reject demo/default passwords, plaintext/malformed hashes; generate private hashes with backend.security.password_hash |
| MEDIA_SIGNING_SECRET, ACCESS_TOKEN_SECONDS | Production random >=32 characters, no default/demo/test/proof value; token900s bounded60–3600 |
| API_TOKENS | Optional private integration bearers; default/weak values rejected in production; never frontend-build secrets |
| DATABASE_URL, INDICMEET_DATA_DIR, INDICMEET_MEDIA_DIR | Production PostgreSQL with private password; demo local Compose Postgres; local Python can use isolated SQLite; data/media defaults remain local ignored paths |
| STORAGE_BACKEND, S3_ENDPOINT_URL, S3_BUCKET, S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY | Production s3 and private non-default keys, HTTPS custom endpoint (AWS SDK default HTTPS endpoint allowed), explicit bucket/region; stage copies evicted after upload/jobs; durable artifacts remain S3 |
| REDIS_URL | Production required external Redis (prefer private rediss endpoint); wakeups are advisory, DB owns durable jobs; no local Redis service in prod template |
| CORS_ORIGINS | Production required exact HTTPS origins; wildcard always rejected; demo/dev localhost5173/8001 defaults |
| ASR_MODE, ASR_SERVICE_URL, ASR_SERVICE_TOKEN | Production remote / HTTPS BASE URL / private token required; default ASR_MODE=remote; explicit demo can use import. Local GPU never belongs in app containers |
| ASR_POLL_INTERVAL_SECONDS, ASR_JOB_TIMEOUT_SECONDS, ASR_SERVICE_TIMEOUT_SECONDS | 2s,10800s total including restart,120s per HTTP request; positive; backoff capped30s; confirmed job ID checkpointed to DB |
| GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL, GROQ_TEMPERATURE | Production private key / official Groq chat endpoint / openai/gpt-oss-20b /0.2; fake client only explicit demo or injected tests |
| GROQ_REQUEST_TIMEOUT_SECONDS, GROQ_MAX_WAIT_SECONDS, INDICMEET_LLM_CACHE |120/120/ignored local llm_cache; response cache is sensitive, apply private directory/retention policy |
| STRICT_REVIEW |true; rejected always excluded, review excluded when strict; mul/und never accepted |
| WORKER_POLL_SECONDS, WORKER_LEASE_SECONDS, RETENTION_DAYS |2/180/0; heartbeat while remote polling; confirmed jobs resume after expired worker lease; others explicit retry; terminal meeting retention disabled at0 |
| MAX_UPLOAD_MB, IMPORT_REQUEST_TIMEOUT_SECONDS, MEDIA_TIMEOUT_SECONDS |512/120/300; bounded upload/CLI import/ffmpeg operations |
| FFPROBE_BINARY, FFMPEG_BINARY |ffprobe/ffmpeg required for uploaded media; installed in app image |
| SERVICE_ROLE, WORKER_HEALTH_PORT |api or worker/8081; health remains private |
| API_PORT, WORKER_PORT, FRONTEND_PORT |local demo8000/8081/8080, loopback; production publishes only loopback frontend for host TLS proxy |
| POSTGRES_PASSWORD |explicit local demo env-file only; no public fallback when unconfigured; production DB external |
| HF_TOKEN/HUGGINGFACE_TOKEN, PYANNOTE_MODEL/PYANNOTE_DEVICE, WHISPER_MODEL/WHISPER_COMPUTE_TYPE/WHISPER_BEAM_SIZE, INDICCONFORMER_MODEL, MMS_LID_MODEL, ASR_DEVICE/ASR_INDIC_DECODER/ASR_MAX_INDIC_CHUNK_SECONDS |separate GPU host only; exact defaults/sizing/required version in asr_service/README.md; GPU stack NOT VERIFIED |
| E2E_BASE_URL, E2E_BROWSER_CHANNEL, E2E_MEDIA_FIXTURE |test frontend URL; optional installed chrome/edge channel; WebM default/original MP4 optional; never test destructive workflow against production meetings |

Debug flags DEBUG/APP_DEBUG and FAKE_ASR/FAKE_LLM are not supported production features; setting them truthy fails startup. Strong secret checks reject known demo/default values, short or low-diversity values; they do not certify entropy. Use a secure secret generator. Infra connectivity/readiness and actual hosted model acceptance are still required after configuration passes.
