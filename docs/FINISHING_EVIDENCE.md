# Finishing evidence — 2026-10-05

Tested branch: release/complete, commit `8571ee741647660cd6012afa137cd8084b52c5ef`. This report/PR body and the env-quoting clarification are a later documentation-only commit; application/config/test code is the fresh-clone code above. Repository `D:\crazibrain\indicmeet-ai`. No pushes, main changes or history rewrites. Owner approval for v2/default changes is recorded in DECISIONS.md and AGENTS.md.

## Verified result

Fresh clone + two new venvs + exact pinned app/LLM installs + migrations on empty and existing DB + lint + **126 full tests** + **27 dedicated finishing tests** + artifact/history scans + clean npm install/build + README explicit demo Compose build/up/ps + API smoke + **2 browser tests** + Compose down + clean git status all passed. Tests include a real worker crash after host accepted submission, restart under a new lease, same host ID completion, and **exactly one POST**. Reference GPU service runs with a fake model over an actual loopback HTTP client; no large model import/download occurs. These are transport/control-flow proofs, not model-accuracy proof.

| Component | Status | Evidence | Gap |
|---|---|---|---|
| Async v2 submit/poll, slow/failure/404/timeout/transient errors | VERIFIED-RUN | tests/test_asr_async.py and loopback fake server | Real endpoint not available |
| Persisted job ID, worker-crash recovery, retry generation | VERIFIED-RUN | Actual Worker crash/restart test; 1 POST/3 polls; DB checkpoint/lease tests | Real long-duration/load acceptance |
| Client IDs, submillisecond collisions, ignored host IDs | VERIFIED-RUN | Real stored-row and rounded-start regression | None for exercised algorithm |
| Required asr.model_version, meeting API/UI metadata | VERIFIED-RUN | Missing-version rejection, fake worker provenance, browser metadata paragraph | Real model revision/config must be supplied |
| Reference ASR host queue/disk/restart/retention/auth/sync/error semantics | VERIFIED-RUN with fake model |5 service tests including actual HTTP client; single execution worker; >=24h retention clock | GPU image/dependencies/models NOT VERIFIED |
| Production safe/default/demo refusal | VERIFIED-RUN |11 configuration tests and network-disabled built-container refusals | Real external production deployment NOT VERIFIED |
| Production template has external DB/Redis/S3/ASR only | VERIFIED-RUN for config | Compose config --quiet exit0; services exactly api/frontend/worker | Actual external resources not provisioned |
| Login/upload/progress/transcript/summary/playback/edit/export/delete | VERIFIED-RUN in explicit demo | API smoke and2 Playwright tests; model metadata and mixed review badges checked | Real ASR/Groq quality NOT VERIFIED |
| Real/fixture schema adaptation | VERIFIED-RUN |639 rows/13 artifacts,0 failed artifacts;32 fixture rows also in clean clone | Repeated copies/subsets are not639 distinct observations or an accuracy score |
| Git/ignore/handoff | VERIFIED-RUN for repository checks |30 reachable commits/302 blobs at tested tip;0 credential matches/large blobs/suspicious tracked files; check-ignore output below | Remote branch remains unpublished by instruction |

## Contract before -> v2 (10 lines for teammate)

1. URL: full POST endpoint -> ASR_SERVICE_URL base URL.
2. Submit: synchronous200 array -> default POST /transcribe returns202 job_id.
3. Fetch: result in POST -> authenticated GET /jobs/id, result only for done.
4. Sync: unrestricted long call -> sync=true only <300s; >=300s is413/use async.
5. Restart: no remote checkpoint -> DB-persisted IDs and original deadline, resume polling.
6. IDs: host IDs accepted -> client ignores them and uses documented SHA256/rounded collision algorithm.
7. Version: absent -> required asr.model_version in every segment; meeting API/UI expose it.
8. Health: unspecified -> GET /healthz200/status ok/model_version (liveness, not accuracy).
9. Retention: unspecified -> completed/failed results >=24h; unknown/expired ID404.
10. Retry: one request timeout -> bounded polling/backoff,3h default deadline, durable host idempotency keys and explicit failed/ambiguous retry.

## README walkthrough and production checks

Executed on a clean local release/complete clone because owner prohibited push and remote branch is not yet published. README's remote git clone line remains contingent on owner's push; no claim that cloning an unpublished remote branch succeeded. From that fresh checkout, the **literal** commands `docker compose --env-file .env.demo.example up --build -d --wait`, `... ps`, and `... down` ran on the README default ports8000/8080/8081. COMPOSE_PROJECT_NAME only isolated proof resources. Browser followed the documented demo steps; stores and fake summaries were visibly labeled. No `.env` was required for this demo proof. Only after the completed demo run was an ignored synthetic production .env created to parse the separate production template; its values are not credentials for any actual service and are not printed or committed.

Standalone production template config passed and lists only api/frontend/worker, no MinIO/DB/queue/fake service. Network-disabled containers with APP_ENV=production and DEMO=true or ASR_MODE=import/default signing secret both exited1 before migration/network connection with clear unsafe-configuration messages. Safe synthetic production config validation passed without network/model calls. NVIDIA base-tag manifest exists (`docker manifest inspect nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`, schemaVersion2/manifest list), but no GPU image build or layer/model download occurred; actual image/dependency/model runtime remains NOT VERIFIED.

## Git credential/ignore/directory proof

At tested tip, scripts/audit_git.py:302 reachable blobs,0 known credential-pattern matches,0 blobs>5MiB,0 suspicious tracked files. Additional scripts/audit_credentials.py scanned all reachable blobs (including document XML) and attributes matches to every commit/file if any: **30 commits,302 blobs,0 matching blobs,0 commit/file hits**. No hit exists to list. The complete quoted gsk_/hf_ pickaxe lines are included below; they are scanner regex/loops, model metadata variable names, report references and abbreviated placeholders such as `hf_xxx` or `hf_prGPdz...`, not a complete matching credential. Final documentation adds commits/blobs and is scanned again; no secret value is printed.

eval/score_wer.py is tracked evaluation source; eval/results and bytecode ignored. deploy/ is tracked deployment source. scripts/ is tracked executable check/launch/provider tooling. Entire colab/ is local notebook/output storage and ignored; its existing notebook is untouched. There are no untracked non-ignored directories at the tested clean tip. .env and env variants ignored except public .env.example/.env.demo.example; approved tiny fixtures remain tracked outside data. No protected original data/output/Kaggle files were overwritten.

```
.gitignore:1:.env	.env
.gitignore:2:.venv/	.venv/probe
.gitignore:3:models/	models/probe
.gitignore:4:NeMo/	NeMo/probe
.gitignore:13:llm_cache/	llm_cache/probe
.gitignore:5:data/	data/probe
.gitignore:38:colab/	colab/output.bin
.gitignore:16:node_modules/	frontend/node_modules/probe
.gitignore:17:dist/	frontend/dist/probe
.gitignore:14:__pycache__/	indicmeet/__pycache__/probe
.gitignore:15:.pytest_cache/	.pytest_cache/probe
.gitignore:35:eval/results/	eval/results/probe
```
Tracked directory sources:
```
deploy/minio/Dockerfile
eval/score_wer.py
scripts/asr_contract_smoke.py
scripts/audit_credentials.py
scripts/audit_git.py
scripts/check.ps1
scripts/check.sh
scripts/diarize.py
scripts/fake_asr_server.py
scripts/release_verify.py
scripts/run_local.ps1
scripts/smoke_test.py
scripts/validate_contract.py
```

## What the team reads/provides

DevOps reads README.md, docs/HANDOFF.md page1, docs/DEVOPS.md, docs/ENVIRONMENT.md, docker-compose.prod.example.yml and DECISIONS.md. Provides **hosts, DB, queue, storage, secrets, domain/TLS**, private access/backup/restore/retention/cache policy and production acceptance. No public demo credentials or MinIO in production template.

ASR host reads docs/HANDOFF.md page2, docs/ASR_API_CONTRACT.md, asr_service/README.md, asr_service/Dockerfile, asr_service/requirements.txt, requirements-ml.txt, AGENTS.md §5/8 and DECISIONS.md. Provides the **v2 service per docs/ASR_API_CONTRACT.md**, compatible GPU/runtime/model environment, authorized model access/token/version, private durable state/results and actual contract/accuracy/load/restart acceptance.

Manager reads docs/RELEASE_SUMMARY.md. The application demo works; the reference service/fake tests and safe production package exist. Real-model production readiness is not claimed.

## Still NOT VERIFIED

- GPU CUDA image build, ML transitive/remote-code dependency resolution, actual model loading and performance/VRAM/disk sizing. Sizing is explicitly an estimate. No local large model was loaded or downloaded.
- Real ASR/diarization accuracy, equivalence with original executed Kaggle notebook, real Groq output quality on Indian languages, real LiveKit recordings/manifests/events/tracks.
- External host/DB/Redis/S3/domain/TLS deployment, real service credentials, production load and backup/restore acceptance; publication/remote clone/PR not executed.

## Exact commands and output tails

From release repo: `C:\Users\raghav\AppData\Local\Temp\indicmeet-audit-20261005\working-venv\Scripts\python.exe scripts\release_verify.py`. Full stdout files: `C:\Users\raghav\.codex\visualizations\2026\10\05\01a10a7c-9e53-78a1-8b7a-1c67c8f68c15\indicmeet-finishing-evidence`. Original final proof: `C:\Users\raghav\AppData\Local\Temp\indicmeet-release-proof-j1bu25j9\evidence`.8 full-suite warnings are dependency deprecations (Starlette AnyIO alias and Alembic path separator); dedicated suite has1 warning. No skipped tests were counted as passes; lint/build passed, npm reported0 vulnerabilities. Named proof volumes remain, verification containers/network were stopped; no down -v.

### New backend/dev/LLM venv pinned install

```text
  Using cached sniffio-1.3.1-py3-none-any.whl (10 kB)
Collecting distro<2,>=1.7.0
  Using cached distro-1.9.0-py3-none-any.whl (20 kB)
Collecting python-dateutil<3.0.0,>=2.1
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Collecting six>=1.5
  Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Installing collected packages: urllib3, tzdata, typing-extensions, tomli, sniffio, six, ruff, python-multipart, pygments, psycopg-binary, pluggy, packaging, MarkupSafe, jmespath, iniconfig, idna, h11, greenlet, distro, colorama, click, charset_normalizer, certifi, async-timeout, annotated-types, annotated-doc, uvicorn, typing-inspection, SQLAlchemy, requests, redis, python-dateutil, pydantic-core, psycopg, opentelemetry-api, Mako, httpcore, exceptiongroup, pytest, pydantic, botocore, anyio, alembic, starlette, s3transfer, httpx, groq, fastapi, boto3
Successfully installed Mako-1.3.10 MarkupSafe-3.0.4 SQLAlchemy-2.0.54 alembic-1.16.5 annotated-doc-0.0.5 annotated-types-0.8.0 anyio-4.15.1 async-timeout-5.0.1 boto3-1.40.50 botocore-1.40.76 certifi-2026.7.22 charset_normalizer-3.5.2 click-8.5.0 colorama-0.4.6 distro-1.9.0 exceptiongroup-1.3.1 fastapi-0.142.2 greenlet-3.5.6 groq-1.7.0 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 idna-3.20 iniconfig-2.3.0 jmespath-1.1.0 opentelemetry-api-1.45.0 packaging-26.3 pluggy-1.6.0 psycopg-3.2.10 psycopg-binary-3.2.10 pydantic-2.11.7 pydantic-core-2.33.2 pygments-2.21.0 pytest-9.1.1 python-dateutil-2.9.0.post0 python-multipart-0.0.32 redis-6.4.0 requests-2.34.2 ruff-0.14.0 s3transfer-0.14.0 six-1.17.0 sniffio-1.3.1 starlette-0.52.1 tomli-2.4.1 typing-extensions-4.16.0 typing-inspection-0.4.4 tzdata-2026.5 urllib3-2.8.0 uvicorn-0.54.0

[notice] A new release of pip is available: 23.0.1 -> 26.2.1
[notice] To update, run: C:\Users\raghav\AppData\Local\Temp\indicmeet-release-proof-j1bu25j9\repo\.venv-backend\Scripts\python.exe -m pip install --upgrade pip
```

### New isolated LLM venv pinned install

```text
Collecting h11>=0.16
  Using cached h11-0.16.0-py3-none-any.whl (37 kB)
Installing collected packages: urllib3, typing-extensions, sniffio, idna, h11, distro, charset_normalizer, certifi, annotated-types, typing-inspection, requests, pydantic-core, httpcore, exceptiongroup, pydantic, anyio, httpx, groq
Successfully installed annotated-types-0.8.0 anyio-4.15.1 certifi-2026.7.22 charset_normalizer-3.5.2 distro-1.9.0 exceptiongroup-1.3.1 groq-1.7.0 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 idna-3.20 pydantic-2.11.7 pydantic-core-2.33.2 requests-2.34.2 sniffio-1.3.1 typing-extensions-4.16.0 typing-inspection-0.4.4 urllib3-2.8.0

[notice] A new release of pip is available: 23.0.1 -> 26.2.1
[notice] To update, run: C:\Users\raghav\AppData\Local\Temp\indicmeet-release-proof-j1bu25j9\repo\.venv-llm\Scripts\python.exe -m pip install --upgrade pip
```

### Isolated LLM fake cited summary

```text
ISOLATED LLM PASS cited claims= 2
```

### python -m alembic upgrade head (empty DB; exit0)

```text
(no stdout; exit0)
```

### python -m alembic upgrade head (existing DB; exit0)

```text
(no stdout; exit0)
```

### Read-only DB schema/version proof

```text
EMPTY/REPEAT MIGRATION RESULT revision=0003_async_asr remote_asr_job_id=True asr_requests=True
```

### python -m ruff check backend indicmeet asr_service tests scripts

```text
All checks passed!
```

### python -m pytest -q --tb=short

```text
........................................................................ [ 57%]
......................................................                   [100%]
============================== warnings summary ===============================
.venv-backend\lib\site-packages\starlette\testclient.py:45
  C:\Users\raghav\AppData\Local\Temp\indicmeet-release-proof-j1bu25j9\repo\.venv-backend\lib\site-packages\starlette\testclient.py:45: DeprecationWarning: The anyio.abc.BlockingPortal alias is deprecated, use anyio.from_thread.BlockingPortal instead.
    _PortalFactoryType = Callable[[], AbstractContextManager[anyio.abc.BlockingPortal]]

tests/test_backend_migrations.py::test_first_migration_and_downgrade_preserve_meetings[False]
tests/test_backend_migrations.py::test_first_migration_and_downgrade_preserve_meetings[False]
tests/test_backend_migrations.py::test_first_migration_and_downgrade_preserve_meetings[False]
tests/test_backend_migrations.py::test_first_migration_and_downgrade_preserve_meetings[True]
tests/test_backend_migrations.py::test_first_migration_and_downgrade_preserve_meetings[True]
tests/test_backend_migrations.py::test_first_migration_and_downgrade_preserve_meetings[True]
tests/test_backend_migrations.py::test_postgres_driver_and_offline_migration
  C:\Users\raghav\AppData\Local\Temp\indicmeet-release-proof-j1bu25j9\repo\.venv-backend\lib\site-packages\alembic\config.py:598: DeprecationWarning: No path_separator found in configuration; falling back to legacy splitting on spaces, commas, and colons for prepend_sys_path.  Consider adding path_separator=os to Alembic config.
    util.warn_deprecated(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
126 passed, 8 warnings in 31.02s
```

### python -m pytest -q tests/test_asr_async.py tests/test_asr_service.py tests/test_production_config.py --tb=short

```text
...........................                                              [100%]
============================== warnings summary ===============================
.venv-backend\lib\site-packages\starlette\testclient.py:45
  C:\Users\raghav\AppData\Local\Temp\indicmeet-release-proof-j1bu25j9\repo\.venv-backend\lib\site-packages\starlette\testclient.py:45: DeprecationWarning: The anyio.abc.BlockingPortal alias is deprecated, use anyio.from_thread.BlockingPortal instead.
    _PortalFactoryType = Callable[[], AbstractContextManager[anyio.abc.BlockingPortal]]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
27 passed, 1 warning in 17.85s
```

### python scripts/validate_contract.py fixtures

```text
fixtures\demo_asr.json: before=0/3 after=3/3
fixtures\ground_truth\agm_subset.json: before=0/3 after=3/3
fixtures\ground_truth\bengali_subset.json: before=0/3 after=3/3
fixtures\ground_truth\mixed_rows.json: before=0/2 after=2/2
fixtures\mixed_segments.json: before=0/2 after=2/2
fixtures\session.json: before=0/19 after=19/19
VALIDATED segments=32 failed_artifacts=0
```

### python scripts/audit_git.py

```text
HISTORY blobs=302 credential_pattern_hits=0 blobs_over_5MB=0
TRACKED suspicious_files=0
LARGEST TRACKED 473050 docs/moving-project-to-google-colab.md
LARGEST TRACKED 171157 fixtures/demo.webm
LARGEST TRACKED 107833 fixtures/demo.mp4
LARGEST TRACKED 58909 experiments/meetreader_sample_data.js
LARGEST TRACKED 36272 frontend/package-lock.json
LARGEST TRACKED 33685 frontend/css/style.css
LARGEST TRACKED 26271 frontend/js/app.js
LARGEST TRACKED 25980 docs/RELEASE_EVIDENCE.md
LARGEST TRACKED 25453 indicmeet/summary.py
LARGEST TRACKED 21966 indicmeet/pipeline.py
git log --all --oneline -S gsk_:
0a2c952 docs: add production handoff, explicit demo quickstart and credential audit
8171bd5 docs: record fresh clone runtime evidence and remaining team acceptance
7af6978 test: add browser workflow and fresh clone release verification
git log --all --oneline -S hf_:
0a2c952 docs: add production handoff, explicit demo quickstart and credential audit
8171bd5 docs: record fresh clone runtime evidence and remaining team acceptance
7af6978 test: add browser workflow and fresh clone release verification
2559768 milestone 1: cache keys, asr provider, settings, tests (includes pre-existing wip)
6f8680d feat: pipeline modules, backend, frontend, docs, fixtures
git log --all --oneline -S AKIA:
0a2c952 docs: add production handoff, explicit demo quickstart and credential audit
8171bd5 docs: record fresh clone runtime evidence and remaining team acceptance
7af6978 test: add browser workflow and fresh clone release verification
git log --all --oneline -S ASIA:
0a2c952 docs: add production handoff, explicit demo quickstart and credential audit
8171bd5 docs: record fresh clone runtime evidence and remaining team acceptance
7af6978 test: add browser workflow and fresh clone release verification
git log --all --oneline -S PRIVATE KEY:
0a2c952 docs: add production handoff, explicit demo quickstart and credential audit
8171bd5 docs: record fresh clone runtime evidence and remaining team acceptance
7af6978 test: add browser workflow and fresh clone release verification
```

### python scripts/audit_credentials.py (full attributed pickaxe output)

```text
CREDENTIAL AUDIT reachable_commits=30 blobs=302 matched_blobs=0 commit_file_hits=0
PICKAXE NON-CREDENTIAL LINE {"commit": "0a2c952", "file": "scripts/audit_credentials.py", "line": "PATTERNS=[rb\"gsk_[A-Za-z0-9]{20,}\",rb\"hf_[A-Za-z0-9]{20,}\",rb\"AKIA[0-9A-Z]{16}\",rb\"ASIA[0-9A-Z]{16}\",rb\"-----BEGIN [A-Z ]*PRIVATE KEY-----\"]"}
PICKAXE NON-CREDENTIAL LINE {"commit": "0a2c952", "file": "scripts/audit_credentials.py", "line": "    for needle in ('gsk_','hf_'):"}
PICKAXE NON-CREDENTIAL LINE {"commit": "8171bd5", "file": "docs/RELEASE_EVIDENCE.md", "line": "git log --all --oneline -S gsk_:"}
PICKAXE NON-CREDENTIAL LINE {"commit": "7af6978", "file": "scripts/audit_git.py", "line": "    patterns=[rb\"gsk_[A-Za-z0-9]{20,}\",rb\"hf_[A-Za-z0-9]{20,}\",rb\"(?:AKIA|ASIA)[A-Z0-9]{16}\",rb\"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\"]"}
PICKAXE NON-CREDENTIAL LINE {"commit": "7af6978", "file": "scripts/audit_git.py", "line": "    for needle in (\"gsk_\",\"hf_\",\"AKIA\",\"ASIA\",\"PRIVATE KEY\"):"}
PICKAXE NON-CREDENTIAL LINE {"commit": "8171bd5", "file": "docs/RELEASE_EVIDENCE.md", "line": "git log --all --oneline -S hf_:"}
PICKAXE NON-CREDENTIAL LINE {"commit": "2559768", "file": "indicmeet/pipeline.py", "line": "    token = app_settings.hf_token"}
PICKAXE NON-CREDENTIAL LINE {"commit": "2559768", "file": "indicmeet/settings.py", "line": "    hf_token: str | None"}
PICKAXE NON-CREDENTIAL LINE {"commit": "2559768", "file": "indicmeet/settings.py", "line": "        hf_token=_value(\"HF_TOKEN\", \"\") or _value(\"HUGGINGFACE_TOKEN\", \"\") or None,"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/INDICMEET_HANDOFF.md", "line": "hf_token = UserSecretsClient().get_secret(\"tts\")"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/INDICMEET_HANDOFF.md", "line": "login(token=hf_token)"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/INDICMEET_HANDOFF.md", "line": "Never type the abbreviated token shown in the Kaggle UI, such as `hf_...AkWw`; the abbreviation is display-only. `get_secret(\"tts\")` returns the full value."}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/INDICMEET_HANDOFF.md", "line": "    token=hf_token,"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/INDICMEET_HANDOFF.md", "line": "    hf_token=hf_token,"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/INDICMEET_WORKLOG_2026-09-28.md", "line": "hf_token = UserSecretsClient().get_secret(\"tts\")"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/INDICMEET_WORKLOG_2026-09-28.md", "line": "login(token=hf_token)"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/moving-project-to-google-colab.md", "line": "--> 438             hf_hub_download("}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/moving-project-to-google-colab.md", "line": "2. Add a secret named exactly `HF_TOKEN`, paste your token (`hf_prGPdz...`) as its value."}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/moving-project-to-google-colab.md", "line": "    def __init__(self, device=\"cuda\", whisper_size=\"small\", hf_token=None):"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/moving-project-to-google-colab.md", "line": "        if hf_token:"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/moving-project-to-google-colab.md", "line": "            login(token=hf_token)"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/moving-project-to-google-colab.md", "line": "# asr = IndicMeetASR(hf_token=\"hf_xxx\")"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/moving-project-to-google-colab.md", "line": "2. **`hf_token` handling** \u2014 don't hardcode it; pull from an environment variable or secrets manager in production, not a function argument passed around in code."}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/moving-project-to-google-colab.md", "line": "hf_token = userdata.get('HF_TOKEN')"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/moving-project-to-google-colab.md", "line": "asr = IndicMeetASR(hf_token=hf_token)"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "docs/moving-project-to-google-colab.md", "line": "That's it \u2014 the class already accepts `hf_token` as a parameter and calls `login(token=hf_token)` internally, so nothing else changes."}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "experiments/indicmeet_asr_colab.py", "line": "    def __init__(self, device=\"cuda\", whisper_size=\"small\", hf_token=None):"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "experiments/indicmeet_asr_colab.py", "line": "        if hf_token:"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "experiments/indicmeet_asr_colab.py", "line": "            login(token=hf_token)"}
PICKAXE NON-CREDENTIAL LINE {"commit": "6f8680d", "file": "experiments/indicmeet_asr_colab.py", "line": "# asr = IndicMeetASR(hf_token=\"hf_xxx\")"}
NO REAL CREDENTIAL PATTERN MATCHES; quoted pickaxe lines are regex patterns, metadata identifiers, or report references
```

### npm ci

```text

added 20 packages, and audited 21 packages in 3s

5 packages are looking for funding
  run `npm fund` for details

found 0 vulnerabilities
```

### npm run build

```text

> indicmeet-frontend@0.1.0 build
> vite build

vite v6.4.3 building for production...
transforming...
✓ 30 modules transformed.
rendering chunks...
computing gzip size...
dist/upload.html                 0.35 kB │ gzip:  0.26 kB
dist/processing.html             0.36 kB │ gzip:  0.26 kB
dist/meeting.html                0.73 kB │ gzip:  0.40 kB
dist/dashboard.html              0.73 kB │ gzip:  0.40 kB
dist/meetings.html               0.73 kB │ gzip:  0.40 kB
dist/index.html                  0.79 kB │ gzip:  0.43 kB
dist/assets/main-pXq__elY.css   27.89 kB │ gzip:  6.03 kB
dist/assets/main-BzVxm1Xg.js   237.50 kB │ gzip: 73.94 kB
✓ built in 1.76s
```

### docker compose --env-file .env.demo.example up --build -d --wait

```text
 Container indicmeet-fresh-j1bu25j9-frontend-1  Started
 Container indicmeet-fresh-j1bu25j9-frontend-1  Waiting
 Container indicmeet-fresh-j1bu25j9-db-1  Waiting
 Container indicmeet-fresh-j1bu25j9-queue-1  Waiting
 Container indicmeet-fresh-j1bu25j9-minio-1  Waiting
 Container indicmeet-fresh-j1bu25j9-storage-init-1  Waiting
 Container indicmeet-fresh-j1bu25j9-api-1  Waiting
 Container indicmeet-fresh-j1bu25j9-worker-1  Waiting
 Container indicmeet-fresh-j1bu25j9-db-1  Healthy
 Container indicmeet-fresh-j1bu25j9-queue-1  Healthy
 Container indicmeet-fresh-j1bu25j9-worker-1  Healthy
 Container indicmeet-fresh-j1bu25j9-storage-init-1  Exited
 Container indicmeet-fresh-j1bu25j9-minio-1  Healthy
 Container indicmeet-fresh-j1bu25j9-api-1  Healthy
 Container indicmeet-fresh-j1bu25j9-frontend-1  Healthy
```

### docker compose --env-file .env.demo.example ps

```text
NAME                                  IMAGE                               COMMAND                  SERVICE    CREATED          STATUS                    PORTS
indicmeet-fresh-j1bu25j9-api-1        indicmeet-fresh-j1bu25j9-api        "python -m backend.e…"   api        21 seconds ago   Up 12 seconds (healthy)   127.0.0.1:8000->8000/tcp
indicmeet-fresh-j1bu25j9-db-1         postgres:16.6-bookworm              "docker-entrypoint.s…"   db         21 seconds ago   Up 19 seconds (healthy)   5432/tcp
indicmeet-fresh-j1bu25j9-frontend-1   indicmeet-fresh-j1bu25j9-frontend   "/docker-entrypoint.…"   frontend   20 seconds ago   Up 6 seconds (healthy)    127.0.0.1:8080->80/tcp
indicmeet-fresh-j1bu25j9-minio-1      indicmeet-fresh-j1bu25j9-minio      "minio server /data …"   minio      21 seconds ago   Up 19 seconds (healthy)   
indicmeet-fresh-j1bu25j9-queue-1      redis:7.4.2-alpine                  "docker-entrypoint.s…"   queue      21 seconds ago   Up 19 seconds (healthy)   6379/tcp
indicmeet-fresh-j1bu25j9-worker-1     indicmeet-fresh-j1bu25j9-worker     "python -m backend.e…"   worker     21 seconds ago   Up 12 seconds (healthy)   127.0.0.1:8081->8081/tcp
```

### python scripts/smoke_test.py --url http://127.0.0.1:8000

```text
job stage=starting progress=0 status=running
job stage=done progress=100 status=done
transcript segments=3 cited_overview_claims=2
SMOKE PASS login upload worker citations signed-range edit revert export delete logout
```

### npm run test:e2e

```text

> indicmeet-frontend@0.1.0 test:e2e
> playwright test


Running 2 tests using 1 worker

  ok 1 e2e\workflow.spec.js:4:1 › demo login upload progress playback edit reload download and delete (9.9s)
  ok 2 e2e\workflow.spec.js:56:1 › real mixed transcript rows render mul und review badges without hiding text (1.4s)

  2 passed (13.4s)
```

### docker compose --env-file .env.demo.example ps (after browser)

```text
NAME                                  IMAGE                               COMMAND                  SERVICE    CREATED          STATUS                    PORTS
indicmeet-fresh-j1bu25j9-api-1        indicmeet-fresh-j1bu25j9-api        "python -m backend.e…"   api        58 seconds ago   Up 49 seconds (healthy)   127.0.0.1:8000->8000/tcp
indicmeet-fresh-j1bu25j9-db-1         postgres:16.6-bookworm              "docker-entrypoint.s…"   db         58 seconds ago   Up 56 seconds (healthy)   5432/tcp
indicmeet-fresh-j1bu25j9-frontend-1   indicmeet-fresh-j1bu25j9-frontend   "/docker-entrypoint.…"   frontend   57 seconds ago   Up 43 seconds (healthy)   127.0.0.1:8080->80/tcp
indicmeet-fresh-j1bu25j9-minio-1      indicmeet-fresh-j1bu25j9-minio      "minio server /data …"   minio      58 seconds ago   Up 56 seconds (healthy)   
indicmeet-fresh-j1bu25j9-queue-1      redis:7.4.2-alpine                  "docker-entrypoint.s…"   queue      58 seconds ago   Up 56 seconds (healthy)   6379/tcp
indicmeet-fresh-j1bu25j9-worker-1     indicmeet-fresh-j1bu25j9-worker     "python -m backend.e…"   worker     58 seconds ago   Up 49 seconds (healthy)   127.0.0.1:8081->8081/tcp
```

### docker compose --env-file .env.demo.example down

```text
 Container indicmeet-fresh-j1bu25j9-queue-1  Removing
 Container indicmeet-fresh-j1bu25j9-queue-1  Removed
 Container indicmeet-fresh-j1bu25j9-minio-1  Stopped
 Container indicmeet-fresh-j1bu25j9-minio-1  Removing
 Container indicmeet-fresh-j1bu25j9-minio-1  Removed
 Container indicmeet-fresh-j1bu25j9-db-1  Stopped
 Container indicmeet-fresh-j1bu25j9-db-1  Removing
 Container indicmeet-fresh-j1bu25j9-db-1  Removed
 Network indicmeet-fresh-j1bu25j9_default  Removing
 Network indicmeet-fresh-j1bu25j9_default  Removed
```

### git status --short (fresh clone, exit0)

```text
(no stdout; exit0)
```

### Synthetic safe production configuration validation

```text
SAFE PRODUCTION CONFIG PASS (no network/model calls)

EXIT=0
```

### docker compose -f docker-compose.prod.example.yml --env-file .env config --quiet

```text

EXIT=0
```

### docker compose -f docker-compose.prod.example.yml --env-file .env config --services

```text
api
frontend
worker

EXIT=0
```

### Built container APP_ENV=production DEMO=true --network none (expected exit1)

```text
ValueError: Unsafe application configuration: Production refuses DEMO=true; Production requires ASR_MODE=remote; Production requires HTTPS ASR_SERVICE_URL base URL; Production requires private ASR_SERVICE_TOKEN; Production requires private GROQ_API_KEY; Production requires external S3 storage; Production requires PostgreSQL DATABASE_URL; DATABASE_URL password is missing/default/weak; S3_ACCESS_KEY is missing/default/weak; S3_SECRET_KEY is missing/default/weak; Production requires REDIS_URL; Production requires exact HTTPS CORS_ORIGINS; MEDIA_SIGNING_SECRET is missing/default/weak; AUTH_USERS_JSON requires hashed non-demo users

EXIT=1
```

### Built container import/default secret --network none (expected exit1)

```text
ValueError: Unsafe application configuration: Production requires ASR_MODE=remote; Production requires HTTPS ASR_SERVICE_URL base URL; Production requires private ASR_SERVICE_TOKEN; Production requires private GROQ_API_KEY; Production requires external S3 storage; Production requires PostgreSQL DATABASE_URL; DATABASE_URL password is missing/default/weak; S3_ACCESS_KEY is missing/default/weak; S3_SECRET_KEY is missing/default/weak; Production requires REDIS_URL; Production requires exact HTTPS CORS_ORIGINS; MEDIA_SIGNING_SECRET is missing/default/weak; AUTH_USERS_JSON requires hashed non-demo users

EXIT=1
```

### Import reference service without any heavy model modules

```text
NO HEAVY MODEL IMPORTS PASS

EXIT=0
```

## Owner-only publish/PR commands (not executed)

Run from the release repository after reviewing this evidence:
```powershell
git push -u origin release/complete
gh pr create --base main --head release/complete --title "Complete meeting workflow with async ASR v2 and safe deployment defaults" --body-file docs/PR_BODY.md
```
The PR body is prepared for review. These publish externally; owner runs them. No main switch/merge/push, force push or squash occurred.

DONE (with evidence): Final fresh clone passed pinned installs, empty/repeat migrations,126 full tests,27 finishing tests, lint/build,6 healthy services, API smoke,2 browser tests and clean shutdown; actual worker crash recovered with1 host submission; real artifacts639/0fail; production template/refusal proofs and credential/ignore scans passed.
NOT VERIFIED: Real GPU/model environment/image/accuracy/performance, original Kaggle equivalence, real Groq Indian-language quality, real LiveKit integration, actual production infrastructure/load/restore and remote publication/PR.
DECISIONS: Owner-approved async v2 and client collision IDs; persisted deadline/idempotency/checkpoints; versioned host/explicit unversioned imports; separate fake-tested single-GPU reference; fail-closed production/explicit demo; source-vs-local-artifact ignore policy.
NEEDS OWNER: None pending; provided approvals were applied.
NEXT: Owner reviews and runs the exact push/PR commands; DevOps and ASR host follow HANDOFF checklists and run real integration acceptance before production release.
