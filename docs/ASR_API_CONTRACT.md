# Hosted ASR contract v2 — owner approved 2026-10-05

This is an audio-to-canonical-segments service on the teammate GPU host. App images contain no GPU models. ASR_SERVICE_URL is now a **base URL**, such as https://asr.team.example, not the full /transcribe endpoint. No audio is sent to a different third party. Reference implementation: asr_service/; real GPU/model execution is NOT VERIFIED.

## Transport and authentication

POST /transcribe receives multipart/form-data with audio (16 kHz mono, 16-bit PCM WAV) and optional turns (JSON string array of {start:number,end:number,speaker:string}). Turns use recording-relative seconds, finite 0 <= start <= end <= audio duration, nonempty speaker, no extra fields. When omitted, the host diarizes. Empty/undecodable or wrong-format audio is 415. Default limits: 512 MiB audio and 7200 seconds; host/operator may lower these only after communicating the limit. Upload/duration overflow is 413. Invalid turns are 422. Authorization: Bearer ASR_SERVICE_TOKEN is required for POST and GET /jobs; missing/wrong token is 401 with WWW-Authenticate: Bearer. /healthz is public liveness/version metadata, not an accuracy or model-readiness guarantee.

Client sends an Idempotency-Key per local DB job attempt and audio scope. Host MUST persist it with the request fingerprint and return the existing job_id on an identical request; the same key with different audio/turns is 409. Keep this mapping while its result is retained. This closes duplicate-submit ambiguity when a response is lost. Reference host scopes keys to its single configured client credential. A future multitenant host must scope them by authenticated client.

## Async (default for all worker jobs)

POST /transcribe -> 202 {"job_id":"server-generated-opaque-id"}. IDs are 1–160 characters from A–Z, a–z, 0–9, hyphen or underscore; no URL/path characters. Submission response does not contain transcript text.

GET /jobs/{job_id} -> 200 object:
```json
{"status":"running","progress":40,"stage":"language_routed_asr"}
```
status is queued|running|done|failed; progress is a number 0–100; stage is a short stage identifier. result appears **only** for done, as the segment array. error appears **only** for failed, as {code:string,detail:string}. Error detail must not contain credentials, audio paths or transcript text. Unknown/expired job_id is 404. Completed AND failed results are retained at least 24 hours from completion; pending/running jobs must not be removed by age. The host persists jobs/results to disk; one GPU worker executes jobs sequentially. After host crash a pending/in-flight job is requeued under the same ID; it may recompute audio, but it is not a new client submission.

```json
{"status":"failed","progress":40,"stage":"failed","error":{"code":"MODEL_UNAVAILABLE","detail":"ASR processing failed; inspect private host diagnostics"}}
```
Queue overload returns 429 with Retry-After seconds; model unavailable returns 503. Error HTTP bodies use {detail:string}. Transient GET 429/503/network errors are polled with bounded backoff, never a new POST.

## Synchronous exception

POST /transcribe?sync=true returns 200 with the segment array **only for duration <300 seconds**. Exactly 300 seconds or longer returns 413 with detail "Synchronous audio must be under 5 minutes; use async POST /transcribe". The same single GPU queue executes sync and async jobs. Reference sync wait is bounded by ASR_SYNC_TIMEOUT_SECONDS (300 default); 503 timeout returns X-ASR-Job-ID so the accepted work remains observable. Application workers always use async.

## Health and segment result

GET /healthz -> 200 {"status":"ok","model_version":"team-deployment-2026-10-05"}. This identifier MUST match every returned segment's nonempty asr.model_version. The host must set an honest deployment identifier covering model revisions, decoder/routing code and configuration; it is not a fabricated accuracy assertion.

done example:
```json
{"status":"done","progress":100,"stage":"done","result":[{"start":0.0,"end":1.0,"speaker":"SPEAKER_01","speaker_name":null,"language":"en","text_native":"We agreed to review the report.","text_roman":null,"text_english":null,"quality":"accepted","reasons":[],"asr":{"method":"whisper","model_version":"team-deployment-2026-10-05","whisper_lang":"en","whisper_lang_conf":0.97,"mms_lang":null,"mms_lang_conf":null}}]}
```
All canonical fields in AGENTS.md §5 are required except host segment_id, which may be absent or arbitrary: **the client ignores it**. Native text is immutable. Language is ISO 639-1, or mul/und with quality=review only; original mixed is asr.source_language. Urdu routes to Hindi but original label remains metadata. Rejected segments stay in transcript but never reach an LLM; strict review also excludes review before enrichment and summary. Language and transcription confidence stay separate. asr.model_version is mandatory in v2 and cannot be inferred by the client. Legacy import adapters explicitly label missing provenance legacy-import-unversioned; this fallback does NOT apply to remote v2 results. App persists segment metadata and exposes unique asr_model_versions in meeting API/UI metadata.

## Exact CLIENT ID algorithm (Python 3.10 reference)

After validating/normalizing segments, use finite float start/end, exact speaker string and unchanged text_native. No Unicode normalization. Group by (round(float(start),3),speaker), using Python binary64 round-to-nearest/ties-to-even behavior. Primary ID hashes JSON encoded UTF-8, ensure_ascii=False, separators=(",",":"), of [rounded_start,speaker]. Prefix seg_ plus first 20 lower-case SHA256 hex digits. For a group with more than one row, append underscore plus first 12 SHA256 hex digits of default-separator UTF-8 JSON [full_start,full_end,text_native]. Full timestamps are floats, not integer tokens. Exact duplicate/colliding final IDs are rejected, never silently merged. Output order does not enter the hash. Adding a member to a formerly single-member collision group changes that group's IDs; rerunning the same content is stable.

```python
from collections import Counter
import hashlib, json
groups = Counter((round(float(s['start']), 3), s['speaker']) for s in segments)
for s in segments:
    start, end = float(s['start']), float(s['end'])
    group = (round(start, 3), s['speaker'])
    primary = json.dumps(list(group), ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    s['segment_id'] = 'seg_' + hashlib.sha256(primary).hexdigest()[:20]
    if groups[group] > 1:
        content = json.dumps([start, end, s['text_native']], ensure_ascii=False).encode('utf-8')
        s['segment_id'] += '_' + hashlib.sha256(content).hexdigest()[:12]
assert len({s['segment_id'] for s in segments}) == len(segments)
```
Speaker reassignment/per-track merging runs this same algorithm again. No unspecified discriminator or host ID is used.

## App polling/restart/retry

ASR_POLL_INTERVAL_SECONDS=2 default; multiply wait by 1.5, cap at 30 seconds, bound every wait/request by remaining ASR_JOB_TIMEOUT_SECONDS=10800 (3 hours). ASR_SERVICE_TIMEOUT_SECONDS=120 is individual HTTP request timeout, not total job deadline. Clock/deadline starts before submission and is persisted with base URL, idempotency key, job_id/status in DB jobs.asr_requests; jobs.remote_asr_job_id exposes latest confirmed host ID. Per-track scopes use deterministic temporary track filenames. Worker lease heartbeat continues during polling. Expired worker leases with confirmed host ID are reclaimed for polling; no new POST occurs. Endpoint changes or uncertain submissions fail closed into explicit retry; uncertain submission may already exist on host, so inspect host before retrying. Host failed/404/invalid result/total timeout becomes the existing failed DB job. Explicit retry clears failed/timeout/ambiguous states and uses a new attempt idempotency key; it retains confirmed unfinished states when recovering other worker errors. Restart does not reset the total deadline.

## Implementer smoke calls (shell; token is already in private environment)

```sh
curl -fsS "$ASR_SERVICE_URL/healthz"
curl -fsS -H "Authorization: Bearer $ASR_SERVICE_TOKEN" -H 'Idempotency-Key: acceptance-001' -F 'audio=@/tmp/fixture.wav;type=audio/wav' -F 'turns=[{"start":0,"end":1,"speaker":"SPEAKER_01"}]' "$ASR_SERVICE_URL/transcribe"
# Copy returned job_id; poll the same ID until done or failed:
curl -fsS -H "Authorization: Bearer $ASR_SERVICE_TOKEN" "$ASR_SERVICE_URL/jobs/JOB_ID"
curl -fsS -H "Authorization: Bearer $ASR_SERVICE_TOKEN" -F 'audio=@/tmp/fixture.wav;type=audio/wav' "$ASR_SERVICE_URL/transcribe?sync=true"
```
Contract client check: python scripts/asr_contract_smoke.py --url "$ASR_SERVICE_URL" --audio /tmp/fixture.wav. This uses the actual v2 RemoteAsrProvider. Default tests use local fake servers/models, not the GPU host. Host reverse proxy must also enforce upload limits, TLS and private access; never enable URL/header/transcript access logging.
# Additive speaker-name provenance (owner authorized 2026-10-06)

Canonical segments include speaker_name_source: attendee_list | inferred | none. Missing provenance on a non-null imported speaker_name is conservatively inferred; null names are none. An attendee list provides spelling/identity metadata, not proof that a diarization cluster belongs to that name. Action intelligence separately includes owner_name_source and owner_source_segment_ids; name-bearing transcript evidence joins task citations. Existing IDs/native text/quality rules are unchanged.
