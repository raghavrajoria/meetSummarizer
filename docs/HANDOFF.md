# Handoff — release/complete

## Page 1 — DevOps checklist

Read README.md, docs/DEVOPS.md, docs/ENVIRONMENT.md, docker-compose.prod.example.yml, DECISIONS.md and docs/FINISHING_EVIDENCE.md. The application demo is verified; public hosting and real-model acceptance are not.

- [ ] Provide application/worker hosts, PostgreSQL DB, Redis queue, supported S3 storage/bucket, private secrets, domain/DNS and TLS.
- [ ] Obtain teammate ASR HTTPS base URL/token implementing docs/ASR_API_CONTRACT.md v2, verify /healthz version and a completed upload result; configure polling/deadline.
- [ ] Set APP_ENV=production, DEMO=false, ASR_MODE=remote, STORAGE_BACKEND=s3. Use non-demo hashed AUTH_USERS_JSON, random MEDIA_SIGNING_SECRET, private Groq/ASR/S3/DB credentials and exact HTTPS CORS origin. Startup must refuse demo/weak settings. Never expose `.env` or local MinIO.
- [ ] Start standalone production Compose (not overlay), migrate DB, inspect healthy services; configure frontend domain/TLS proxy, private API/worker/DB/queue/storage.
- [ ] Back up DB/S3 consistently and test restore; private ASR result/state volume and cache policy; choose meeting/cache retention, preserve original transcripts/AI claims, summary edits remain separate.
- [ ] Verify on isolated staging with approved data. Secrets already exported in private shell:
```sh
curl -fsS "$APP_BASE/healthz"
curl -fsS "$APP_BASE/readyz"
curl -i "$APP_BASE/meetings" # expect 401 without bearer
curl -fsS -H "Authorization: Bearer $APP_TOKEN" "$APP_BASE/meetings"
curl -fsS -H "Authorization: Bearer $APP_TOKEN" "$APP_BASE/meetings/MEETING_ID/media-url"
# Use returned signed relative URL (valid120s, revoked by logout); no bearer header required:
curl -i -H 'Range: bytes=0-31' "$APP_BASE/SIGNED_MEDIA_PATH" # expect206/Content-Range/32bytes
```
- [ ] Use actual browser login -> upload -> processing -> transcript/summary -> playback/seek -> edit/reload -> export. Run smoke/e2e only on isolated demo/staging records; those tests delete their records.

## Page 2 — ASR host checklist

Read docs/ASR_API_CONTRACT.md (normative v2), asr_service/README.md, asr_service/Dockerfile and requirements, requirements-ml.txt, AGENTS.md §5/8, DECISIONS.md. **Real GPU/model execution and image are NOT VERIFIED here.**

- [ ] Provision compatible NVIDIA GPU/driver/runtime, memory/disk/state volume; sizing in README is an estimate, measure peak before commitments.
- [ ] Install pinned separate web/ML environment, resolve actual remote-model code dependencies, accept gated model licenses, configure HF token. Do not add GPU stack to app images.
- [ ] Set private ASR_SERVICE_TOKEN, honest ASR_MODEL_VERSION, model names/revisions/device/compute type; secure retained audio/results. Run one process/GPU worker; disk persists queue/results/idempotency keys. Retention >=24h after done/failed; pending/running not expired.
- [ ] Serve HTTPS v2: async POST202 -> authenticated GET job; sync only <300s; IDs may be omitted, client assigns; every asr.model_version required. Validate limits/auth/turns/failed/overload/idempotency/unknown-ID semantics.
```sh
curl -fsS "$ASR_SERVICE_URL/healthz" #200/status ok/version
curl -i "$ASR_SERVICE_URL/jobs/unknown" #401 without bearer
curl -i -H "Authorization: Bearer $ASR_SERVICE_TOKEN" "$ASR_SERVICE_URL/jobs/unknown" #404
curl -i -H "Authorization: Bearer $ASR_SERVICE_TOKEN" -H 'Idempotency-Key: handoff-001' -F 'audio=@/tmp/fixture.wav;type=audio/wav' "$ASR_SERVICE_URL/transcribe" #202/job_id
curl -fsS -H "Authorization: Bearer $ASR_SERVICE_TOKEN" "$ASR_SERVICE_URL/jobs/JOB_ID" #same ID until done/result or failed/error
curl -fsS -H "Authorization: Bearer $ASR_SERVICE_TOKEN" -F 'audio=@/tmp/fixture.wav;type=audio/wav' "$ASR_SERVICE_URL/transcribe?sync=true" #200 for <300s
python scripts/asr_contract_smoke.py --url "$ASR_SERVICE_URL" --audio /tmp/fixture.wav
```
- [ ] Restart host during queued/running job and app worker during polling; preserve IDs/results, never silently resubmit. Verify failed result and completion remain24h, expired unknown404, 300s sync413 with async guidance, 401/413/415/422/429/503 errors.
- [ ] Evaluate human-checked ASR/diarization accuracy on real audio and model routing; real Groq quality on Indian languages; real LiveKit manifests/events/tracks. Soft references/fake clients do not prove these. Compare original executed Kaggle notebook when available.
