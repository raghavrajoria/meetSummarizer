# Precomputed real-recording demo

Open http://127.0.0.1:8080 and sign in as `demo`. The demo-only login details appear on the login page and are defined in backend/security.py; no password is reproduced here. Configuration: .env.demo.example. Compose project: indicmeet-precomputed. Stack remains running.

Click Sign in, Meetings, Open meeting. Play the recording, click a transcript timestamp to seek, then switch Native / Roman / English. Missing translations are explicitly unavailable. Scrum has cached Groq claims and evidence links; AGM and group show exactly "summary not generated" with no empty editor or invented extraction placeholders.

| Meeting | Media duration | Transcript | Talking point / provenance |
|---|---:|---:|---|
| Daily Scrum | 1286.300 s | 156 rows | Real pipeline output (Kaggle T4, late Sep 2026, owner-stated). Source: staged scrum demo; action items are simulated. Owner identifies 30 Sep re-run. Artifact methods:150 whisper_english,6 whisper_latin_unresolved. No model/date fields; no claim IndicConformer handled these segments. Cached Groq summary linked by saved overview and cached task matches; exact Groq model/date unknown. |
| Full group discussion | 2251.680 s | 226 rows | Real pipeline output (Kaggle T4, late Sep 2026, owner-stated). Hindi and English discussion; speakers anonymous. Owner identifies reconstructed indicmeet_asr_v2, Whisper large-v3 + IndicConformer routing. 166 accepted/56 review/4 rejected; en136/hi84/mr2/ru1/zh1/ko1/ur1. Stored labels retained in source metadata; canonical routing/gates apply. Native source preserved,84 stored Roman fields retained with generator/date unknown. End times estimated from next exported start; final2251.680s. Summary not generated. |
| AGM 30-second sample | 30.000 s | 4 rows | 30-second sample, single speaker label (no diarization). Measured real CPU execution2026-10-06: small-int8 Whisper + MMS-LID256 + IndicConformer600M CTC. Summary not generated. |

## Pairing and exclusions

Full group transcript exports data/transcripts/transcript.txt and transcript (1).txt are byte-identical,226 rows,last start2245s. The19 excerpt rows match first180s exactly in start,speaker,language,quality,native text; their ends were estimated. Three-minute source and session media have identical SHA256 d43ac42fadbb9a8b044f4954979b6aba3ebfed92b749d29c74f0ff2b305e9f8e;180.040s. Excerpt removed because full meeting covers it.

The375-turn CSV data/outputs/meeting_diarization_exclusive.csv ends2251.915s,0.235s beyond full group media2251.680s. Duration supports group pairing, not AGM1200s; source identity and diarization word alignment are not independently proven. The CSV is not substituted for estimated export ends.

fixtures/demo.mp4 is a10.000s low-resolution extract of the group-discussion fixture; demo.webm is its10.008s VP8/Opus transcode. They contain third-party group-discussion video, not generated synthetic media; original creator/URL unknown. Both excluded, and demo_asr.json/dry summary excluded. Tracked media: demo.mp4 107833bytes; demo.webm171157bytes. No new audio/video added to Git. Owner will check GitHub visibility.

Search under data/, llm_cache and historical data/outputs/prior_output/ found no paired group summary or separate hindi_enriched artifact.13 raw caches lack recording identity; unpaired caches are not imported. Ground-truth files are LLM-generated references and never meetings. Ambiguous Bengali/Qwen/full_transcript legacy outputs are skipped without reliable media/run bindings. See DEMO_INVENTORY.md for every inventoried asset and verdict.

Not shown: live processing, GPU speed, CUDA deployment accuracy, large-v3 evaluation, real-participant speaker names, or verified diarization accuracy. No ASR models or Groq calls run during this demo task. The demo banner appears only in demo mode. Live upload is disabled for the precomputed profile.

## Verification

Loader: three imports; second run three UNCHANGED, no extra meetings. Browser command: E2E_BROWSER_CHANNEL=msedge npm --prefix frontend run test:e2e -- precomputed-demo.spec.js;2 passed in12.1s. Every loaded recording decoded and played, timestamp seeking and three transcript tabs passed. Production-config banner absence passed. Measured playback evidence: data/demo_browser_evidence.json.

Full Python suite: .venv\Scripts\python.exe -m pytest -q;167 passed,8 warnings in44.07s (including upload-guard regression). Ruff: all checks passed.
