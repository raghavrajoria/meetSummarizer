# IndicMeet collaborator guide

This guide covers a local Windows checkout of meetSummerizer. The app currently uses a static HTML/CSS/JavaScript frontend with a FastAPI backend; it is not a React app.

## 1. Get set up

Clone the repository and open PowerShell in the project root. Python 3.10 or newer and FFmpeg (`ffmpeg` and `ffprobe` on `PATH`) are needed for the pipeline. Node.js is only needed for the fixture-building helper.

```powershell
git clone https://github.com/raghavrajoria/meetSummerizer.git
Set-Location .\meetSummerizer
python --version
ffmpeg -version
ffprobe -version
```

Create the backend environment and install its runtime and test dependencies:

```powershell
python -m venv backend\.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
```

For real Groq summaries, create a local `.env` from the example and add your own credentials:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `GROQ_API_KEY` for local summary generation. `HF_TOKEN` is used only when the pyannote diarization stage needs authenticated model access. Never commit `.env`, tokens, credential-bearing notebooks, or private media. The `.env`, virtual environments, model/data directories, `colab/*.ipynb`, and common media extensions are Git-ignored; still check staged files before committing.

## 2. Run the API and UI

Start the backend in one terminal from the project root:

```powershell
.\backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal, start the static frontend:

```powershell
Set-Location .\frontend
..\backend\.venv\Scripts\python.exe serve.py
```

Open <http://localhost:5173/>. The UI requests session data from the API at <http://127.0.0.1:8000>.

The API currently provides:

- `GET /sessions` and `GET /sessions/{id}`
- `GET /sessions/{id}/media` with byte-range support for seeking
- `POST /sessions/import` with `session_json` and optional media form fields

Import the sample session from a project-root terminal:

```powershell
curl.exe -X POST http://127.0.0.1:8000/sessions/import `
  -F "session_json=<fixtures/session.json" `
  -F "media=@fixtures/group_discussion_3min.mp4"
```

The sample video is local fixture media and is excluded from Git, so a fresh clone may not contain `fixtures/group_discussion_3min.mp4`. Obtain it from the project owner through the team's approved file-sharing method before using that import command. Do not commit it. Imported media and the SQLite database are stored under ignored `data/`.

## 3. Run the pipeline

The CLI is `python -m indicmeet.pipeline`. A precomputed/offline run looks like this:

```powershell
python -m indicmeet.pipeline `
  --media <path-to-recording> `
  --session-id <unique-id> `
  --title "Meeting title" `
  --attendees fixtures/attendees_scrum.json `
  --diar-csv <path-to-diarization.csv> `
  --asr-json <path-to-asr.json> `
  --out data/sessions/<unique-id>
```

Use `--dry-summary fixtures/dry_summary.json` for a plumbing smoke test; it loads a canned summary, so it does **not** test summary quality. For a real summary, omit `--dry-summary` and provide `GROQ_API_KEY` in the environment or root `.env`. To have the pipeline post its result to the API, add:

```text
--import-url http://127.0.0.1:8000/sessions/import
```

The attendees file format is:

```json
{
  "attendees": ["Shashank", "Neha", "Deepika", "KT", "Manoj", "Sindhu", "Sanam"],
  "aliases": {
    "KT": ["Kitty", "Kirti", "Keith", "Keithy"],
    "Sindhu": ["Sendu"]
  }
}
```

Each stage reports its input, output, and elapsed time. JSON cache files and the extracted 16 kHz mono WAV are stored in the session output folder; existing stage results are reused. Pass `--force` to rerun cached stages. The normal diarization/ASR stages require their ML dependencies and an appropriate GPU environment; if they are unavailable, run those stages in the team's approved Kaggle notebook and pass the resulting CSV/JSON files to the local pipeline.

Do **not** use `data/outputs/meeting_diarization_exclusive.csv` for `data/recordings/scrumMeetingDemo.mp4`: the CSV ends at about 2,252 seconds, while that scrum video is about 1,286 seconds. The durations show they are not a match. Run diarization for the scrum recording instead.

## 4. Tests and contribution workflow

Run the project's offline summary and backend/API tests from the project root:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest -q
```

The suite covers the offline summary regression, attendee/alias handoff, session listing/detail, media ranges, and session import. It does not call Groq or require a GPU.

Before sharing changes, review `git status`, `git diff`, and the staged-file list. Keep generated outputs, recordings, model files, environments, tokens, and private notebooks out of commits. Use a feature branch for collaborator changes and include test results in the pull request. Ask the project owner before changing the frontend framework or the established session data shape.

## Repository map

- `backend/` — FastAPI, SQLite persistence, local media storage, and runtime/dev requirements.
- `frontend/` — static HTML, CSS, JavaScript, and the local range-capable web server.
- `indicmeet/` — summary, ASR, and pipeline modules.
- `fixtures/` — small session/attendee/test data; media may be provided separately.
- `tests/` — offline summary, pipeline contract, and API tests.
- `experiments/` — legacy/model experiments; not the current application entry point.
- `docs/` — collaborator and project handoff documentation.
- `data/` — ignored local recordings, caches, imported media, and generated outputs.
