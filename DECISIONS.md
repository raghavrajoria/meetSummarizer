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
