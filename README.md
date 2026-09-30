# meetSummerizer

meetSummerizer is a local meeting intelligence application for importing recordings and speaker-labeled transcripts, reviewing evidence-linked summaries, and playing transcript segments alongside the meeting media.

## Folder map

- `backend/` — FastAPI service, SQLite persistence, and media storage adapter.
- `frontend/` — static HTML, CSS, and JavaScript meeting review interface.
- `indicmeet/` — current ASR and summary pipeline modules.
- `experiments/` — legacy scripts, notebooks, and model experiments.
- `fixtures/` — small sample session data and media for local development.
- `data/` — local datasets and generated artifacts; excluded from Git.
- `docs/` — project handoff and work log.
- `notebooks/` — project notebooks; credential-bearing notebooks stay excluded.

## Local API

Install the packages in `backend/requirements.txt` in a fresh Python 3.10+ environment, then start the API with `python -m uvicorn backend.main:app --reload`. Start the static frontend with `python frontend/serve.py`; it requests sessions from `http://127.0.0.1:8000` and continues to use the existing HTML, CSS, and JavaScript.

Import the sample session and its matching three-minute media clip with:

```powershell
curl.exe -X POST http://127.0.0.1:8000/sessions/import `
  -F "session_json=<fixtures/session.json" `
  -F "media=@fixtures/group_discussion_3min.mp4"
```

The API returns the UI's existing meeting object shape: `id`, `title`, `group`, `date`, `dateLabel`, `time`, `media`, `status`, `summary`, `speakers`, `intelligence`, and `segments`. `intelligence` retains `discussed`, `keyDiscussion`, `decisions`, `actionItems`, `followUps`, `questions`, and `concerns`. Speakers retain `id`, `name`, `role`, `confidence`, `join`, and `leave`; transcript segments retain `t`, `time`, `speaker`, `lang`, `tx`, and `en`. Optional fields such as `quality`, `verified`, `evidence`, `end`, and `roman` add review state, evidence links, estimated transcript intervals, or transliteration without replacing the UI's established fields.

The SQLite database and imported media are written under the Git-ignored `data/` directory. Media access supports byte ranges for seeking. Local files use the `LocalStorage` implementation behind the `Storage` protocol so a future S3 implementation can replace it.

## Tests and processing pipeline

Install `backend/requirements-dev.txt` into the backend environment to run the offline summary regression and API tests with `python -m pytest`. The ASR and diarization stages require the ML dependencies and a suitable GPU environment. For local/offline runs, precompute those stages and provide their JSON/CSV outputs:

```powershell
python -m indicmeet.pipeline --media recording.mp4 --session-id demo `
  --diar-csv diarization.csv --asr-json asr.json --dry-summary summary.json
```

The pipeline caches stage results in `data/sessions/<id>/`; use `--force` to rerun stages, and `--import-url http://127.0.0.1:8000/sessions/import` to send the assembled session and media to the local API.
