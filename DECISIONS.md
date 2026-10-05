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
