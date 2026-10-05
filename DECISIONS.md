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
