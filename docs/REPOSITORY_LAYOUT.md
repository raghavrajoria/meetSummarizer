# Repository layout

The complete tracked development snapshot is preserved on `codex/archive-development-2026-10-07` at e19b954a58c9a64d16a52e24a80c726f428a4f55. `main` advances with ordinary commits; no history is erased. Older files therefore remain recoverable from Git history as well as the archive branch. Removing files from the main tree does not shrink historical Git objects.

## Essential source retained

frontend/ (React/Vite), backend/ (API/worker/migrations), indicmeet/ (canonical pipeline), asr_service/ (model-host service), deploy/, eval/score_wer.py (tested metric tool), scripts/ (launch/check/import/evaluation tools), tests/, required small fixtures/, dependency and lock files, Docker/Compose configuration, docs/, README.md, AGENTS.md and DECISIONS.md.

Fixtures remain only where needed for pipeline tests, schema checks, fake-service tests or isolated browser playback. Browser speaker-name session fixture moved from a generated docs report into tests/fixtures/speaker_name_session.json. No fixture or ground-truth reference is presented as real model output. Irreplaceable full recordings, references, model weights and notebook outputs remain ignored and untouched on local disk.

## Archived extras

These paths are absent from the main tree; their original content is preserved on the archive branch:

- experiments/auto_lang_asr.py
- experiments/build_group_discussion_fixture.mjs
- experiments/indicconformer_multiwindow.py
- experiments/indicconformer_test.py
- experiments/indicmeet_asr_colab.py
- experiments/indicmeet_llm_legacy.py
- experiments/meetreader_sample_data.js
- experiments/qwen_test.py
- experiments/requirements_legacy.txt
- experiments/transcribe_test.py
- experiments/vad_lang_asr.py
- frontend/js/app.js
- frontend/js/player.js
- notebooks/.gitkeep
- frontend/serve.py
- frontend/DEPLOY.txt
- frontend/README.txt
- fixtures/dry_summary.json
- fixtures/group_discussion_diarization.csv
- docs/ASR_SERVICE_CONTRACT.md
- docs/DEMO_BROWSER_EVIDENCE.json
- docs/DEMO_INVENTORY.md
- docs/FINISHING_EVIDENCE.md
- docs/INDICMEET_HANDOFF.md
- docs/INDICMEET_WORKLOG_2026-09-28.md
- docs/IndicMeet_ASR_Progress_Report.md
- docs/PR_BODY.md
- docs/REAL_RUN_EVIDENCE.md
- docs/RELEASE_EVIDENCE.md
- docs/RELEASE_SUMMARY.md
- docs/SPEAKER_NAME_EVIDENCE.md
- docs/moving-project-to-google-colab.md
- docs/SPEAKER_NAME_EVIDENCE.json

To inspect an archived file without changing the checkout:

```powershell
git show codex/archive-development-2026-10-07:experiments/qwen_test.py
```

To work on old experiments, create a separate checkout/worktree from the archive branch. Do not copy secrets or large local artifacts into it. Unrelated untracked shell-output files were left untouched.
