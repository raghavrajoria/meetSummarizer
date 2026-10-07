# Engineering decisions

## 2026-10-05 Preserve current work and separate runtimes
Context: The working backend was uncommitted and its dependencies differed from the GPU experiments.
Options: Fold old work into the release / preserve it as separate commits first.
Chosen: Preserve the original API, worker, migrations, tests and launcher in five commits; exact backend/LLM pins and a separate ML environment.
Why: Reviewable provenance and reproducible small-runtime tests without loading local large models.
Reversible: yes

## 2026-10-05 Small versioned fixtures
Context: Full reference datasets and media are ignored and irreplaceable.
Options: Publish full datasets / keep originals and track approved small subsets.
Chosen: Track a ten-second clip, stored demo ASR and trimmed reference subsets, explicitly labeled as soft references.
Why: Reproducible tests with no changes to original artifacts and no media over 5 MB.
Reversible: yes

## 2026-10-05 Canonical imports and citations
Context: Existing artifacts include missing confidence and two distinct texts at the same speaker/start/end.
Options: Reject legacy artifacts / explicit canonical adapter with collision discriminator.
Chosen: Preserve source text; primary ID hashes start and speaker, collisions add a deterministic end/text hash; exact duplicates fail. Unknown languages stay review. Extraction uses a private numeric adapter, then publishes validated canonical citations only. Strict review is default.
Why: Real artifacts import without weakening the quality gate or linking evidence by timestamp.
Reversible: yes

## 2026-10-05 Login, media playback and separate edits
Context: Browser media elements cannot attach bearer headers; fake login and local edits break the workflow.
Options: Cookie auth / revocable opaque bearer sessions plus signed media URLs.
Chosen: Environment JSON users with PBKDF2 hashes, hashed tokens in DB, 15-minute sessions, viewer/editor roles; media signatures last 120 seconds and check logout revocation. Summary edits live in a separate table.
Why: Supports the existing bearer integrations and browser playback without exposing a long-lived credential in a URL.
Reversible: yes

## 2026-10-05 External model and LiveKit boundaries
Context: GPU endpoint and real LiveKit exports are unavailable; no local large-model loading is permitted.
Options: Implicit local fallback / explicit remote contract and tested fakes.
Chosen: Remote multipart canonical API; GPU CSV script with both pyannote output adapters; LiveKit composite naming by overlap with ties left unnamed. Explicit demo mode injects a fake client; production has no fixture fallback. Enrichment preserves native text and records failures.
Why: Local workflows can be tested without transmitting recordings or claiming model accuracy.
Reversible: yes

## 2026-10-05 Durable deployment and unavailable MinIO images
Context: The requested local S3 stack cannot pull MinIO community images (Docker Hub denied, Quay 401, binary download 410).
Options: Abandon stack test / build the requested MinIO from the pinned official release source.
Chosen: Build MinIO locally from the official annotated release tag object f19c534b9f457773dcd043d977433e1a71525c3b. S3 holds durable objects; API/worker staging volumes are independent. DB leases own jobs; Redis is an optional wakeup accelerator. Postgres advisory lock serializes migrations.
Why: Exercises actual S3/DB/queue integration without a paid service or substitute third party receiving recordings. MinIO is local demo infrastructure; production requires team-provisioned supported storage.
Reversible: yes

## 2026-10-05 Owner-approved mixed/unknown language markers
Context: Semantic validation found 23 real rows labeled mixed, which cannot be ISO 639-1.
Options: Reject unresolved artifacts / permit mul and und only under review.
Chosen: Owner explicitly approved mul/und review markers; preserve original labels in asr.source_language. Validate other values against the frozen official ISO table; never guess an accepted language from script.
Why: Preserves stored evidence while enforcing the language gate; no silent routing guess.
Reversible: yes

Owner clarification: every mul/und row is review; original rejected status is retained in asr.source_quality and blocked even in relaxed mode. Real mixed reference rows are tracked and exercise LLM exclusion and browser visibility. The frontend displays the language code and review badge directly.

## 2026-10-05 Final import and deployment edge cases
Context: The shape adapter accepted sparse real artifacts, but upload pre-validation still required obsolete fields; speaker reassignment could discard collision suffixes.
Options: Keep inconsistent import paths / use the adapter for file imports and reidentify all aligned rows together.
Chosen: Canonical import validation, collision-safe reidentification after alignment, optional private env file for all container settings, configurable service endpoints and S3 staging eviction after durable writes/processing. Added idempotent bucket regression.
Why: Fresh clones must exercise the same contract as real imports and separate worker/API storage. Local durable files are never evicted.
Reversible: yes

## 2026-10-05 Portable browser playback fixture
Context: Fresh-clone Playwright reached signed media HTTP 206 but bundled Windows Chromium canPlayType returned empty for H.264/AAC and probably for VP8/Opus.
Options: Ignore playback / require a proprietary browser / encode the approved short clip in WebM and retain optional installed-browser MP4 verification.
Chosen: Add small VP8/Opus fixture; require actual playback and seek in the portable test; expose optional browser and fixture selectors.
Why: The regression must verify decoded media without assuming the test browser ships licensed codecs.
Reversible: yes

## 2026-10-05 Owner-approved async ASR contract v2
Context: Owner approved async hosted jobs, client-owned IDs, required model version and production refusal defaults in the finishing task.
Options: Long synchronous calls / durable async submit-and-poll.
Chosen: Persist submission intent, host IDs, deadline, endpoint and per-track checkpoints in DB jobs; bound polling; resume confirmed jobs; uncertain submission requires explicit retry. Host supports durable idempotency keys; explicit retry gets a new attempt key.
Why: Worker restart or response loss must not silently submit expensive GPU work twice.
Reversible: yes

## 2026-10-05 Exact collision IDs and model provenance
Context: Host IDs are not authoritative; millisecond rounding can collide even when exact starts differ.
Options: Trust host IDs / documented client hashing with rounded grouping and full timestamp/text suffix.
Chosen: SHA256 of compact UTF8 JSON [round(start,3),speaker],20hex; collision groups append12hex of default JSON [full_start,full_end,native_text]. Duplicate IDs fail. Require host asr.model_version; label historical imports legacy-import-unversioned; expose meeting model versions.
Why: Reproducible stable reruns and honest provenance without inventing legacy model versions.
Reversible: yes

## 2026-10-05 Separate reference GPU service
Context: Teammate needs implementable v2 host, but no GPU/model loading is allowed here.
Options: Wait for teammate / ship an injected-model reference with disk state.
Chosen: Separate asr_service images/environment, one in-process worker, atomic disk statuses/results and >=24h terminal retention, fake-model tests. CUDA/model build/runtime remains NOT VERIFIED.
Why: Validate transport/queue/restart semantics without pretending to verify model accuracy or GPU compatibility.
Reversible: yes

## 2026-10-05 Explicit demo and fail-closed production
Context: Earlier Compose enabled demo by default and included public local credentials.
Options: Warn only / refuse unsafe startup and require explicit demo selection.
Chosen: Default production/remote, validate startup before migrations; DEMO=true via selected .env.demo.example only; production template standalone with external DB/Redis/S3/ASR and no demo services.
Why: Unconfigured deployment must not silently publish fake processing or demo credentials.
Reversible: yes

## 2026-10-05 Repository directory ownership
Context: Owner requested eval/colab/deploy/scripts and ignore policy review.
Options: Ignore all tooling / track source and exclude generated output.
Chosen: eval/score_wer.py, deploy/ and scripts/ are source and remain tracked. eval/results and caches ignored; the entire colab/ directory is local notebook/generated artifact storage and remains ignored; approved tiny fixtures remain tracked. Add explicit colab output rules and ASR state ignore; .env.demo.example is an intentional public configuration exception.
Why: A clone needs evaluation/deployment/check tooling, not retained media, credentials, model weights or notebook output.
Reversible: yes
# 2026-10-06 Public CPU run and explicit local-real deployment
Context: Owner explicitly requested real CPU inference despite older workstation guidance, with public AGM content only and 3 GiB free RAM headroom. Existing production policy required externally hosted HTTPS services.
Options: weaken production/defaults / explicit narrowly scoped local-real profile.
Chosen: APP_ENV=production, DEMO=false, DEPLOY_PROFILE=local-real; HTTP allowed only for exact 127.0.0.1, localhost, host.docker.internal hosts for ASR, CORS and the compose-provided MinIO endpoint. All other secret/auth/storage/database/fake/demo checks retained. Dedicated compose infrastructure has private random credentials and loopback ports.
Why: The owner approved the explicit local profile; MinIO must also use the same exact-host exception for local S3 to work. Never reuse this profile on public deployment.
Reversible: yes

# 2026-10-06 Sequential isolated CPU passes and memory watchdog
Context: i5-1334U, 15.7 GB RAM, no GPU; model loading previously crashed. CPU request explicitly supersedes AGENTS.md section 2 restriction for this public test.
Options: eager resident models / sequential whole-clip model passes in killable subprocess.
Chosen: Native Windows first; Whisper small int8, MMS-LID CPU, IndicConformer CPU, models released between passes; check available RAM before loads and segments, independent parent samples RAM each second and terminates the model process tree below 3 GiB. Pyannote never runs in CPU mode; request turns required. Two CPU threads default.
Why: Sequential whole-clip passes avoid reloading models per segment and retain the existing routing logic. Native compatibility and actual memory remain acceptance conditions. Linux fallback only if IndicConformer fails on Windows, not as an excuse to lower the headroom.
Reversible: yes

# 2026-10-06 Speaker-name provenance and owner-name evidence
Context: Owner requested demo-fixture name audit before real models. Imported names lacked provenance; inferred action-owner basis was lost in UI and reply evidence was missing from citations.
Options: leave display ambiguous / additive provenance and owner evidence.
Chosen: Owner-authorized speaker_name_source=attendee_list|inferred|none; unproven imports inferred, null names none. Preserve owner_name_source and owner_source_segment_ids, include name-bearing segments in action citations, display inferred in transcript/actions.
Why: An attendee spelling match alone does not verify identity. Demo has no names/actions; focused synthetic tests cover that missing behavior separately.
Reversible: yes

## 2026-10-06 Precomputed real-recording demo
Context: Owner requests existing real outputs only, with honest source attribution and no new model calls.
Options: Fixture simulation / import audited stored results.
Chosen: Separate indicmeet-precomputed Compose project; demo-only idempotent loader; refuse live uploads; cached Scrum Groq only; missing summaries explicit.
Why: Excludes fake/reference data, avoids old demo-volume contamination and preserves production behavior. Owner-stated Kaggle provenance explicitly distinguished from artifact proof. Full226-row group replaces exactly matching19-row excerpt; estimated ends marked, speakers anonymous. Scrum staged action items identified as simulated, Whisper methods do not prove IndicConformer execution.
Reversible: yes

## 2026-10-06 Restore owner HTML layout with connected React workflows
Context: Owner supplied original UI screenshots, requested only meeting count/time statistics and readable speaker names.
Options: Keep simplified React scaffolding / restore original design-system markup and screenshot layout.
Chosen: Original meetReaderCB shell, account footer, login, Overview/library rows and separate meeting Overview/Transcript panels. All content still fetched via existing authenticated API; selected view persisted in URL.
Why: Restores the requested presentation without substituting screenshot fixture content. Audited Scrum names use transcript references and inferred labels; unknown speakers get deterministic demo aliases, explicitly marked. Canonical IDs, original transcript and production identity behavior unchanged. PostgreSQL query verified payload transcript counts4/156/226; media stays in MinIO.
Reversible: yes

## 2026-10-07 Simplify demo transcript and align recording filters
Context: Owner requests removal of Both/Original/Roman/English controls and unused top search icon.
Options: Keep multi-view controls / show original transcript once.
Chosen: Original text only in transcript UI, with search/copy/download retained; removed top search icon; meeting search and date controls share an aligned responsive toolbar.
Why: Removes duplicate English lines and aligns recording controls without changing stored transcript/enrichment or API behavior.
Reversible: yes

## 2026-10-07 Essential main tree and preserved development archive
Context: Owner explicitly authorizes a branch preserving extras and a cleaned main branch.
Options: Delete development history / archive the full snapshot and remove unused files in a normal descendant commit.
Chosen: codex/archive-development-2026-10-07 at e19b954 preserves all tracked extras; cleanup prepared on codex/clean-main and main fast-forwards only after verification. Retain runtime/pipeline/model-host/deployment source, tests and required small fixtures. Move browser test payload into tests/fixtures; archive experiments, legacy frontend files, obsolete contract and generated/historical reports.
Why: A clean application tree must still build/test; removing all fixtures would break verified workflows. Private local recordings/models/secrets remain ignored, untouched and excluded from branches. Fresh-clone real-artifact tests explicitly skip without owner data; real browser tests opt in.
Reversible: yes
