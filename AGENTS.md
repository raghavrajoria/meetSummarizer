# AGENTS.md — IndicMeet AI

You are the primary engineer on this repo. Make decisions yourself, verify your own work, and report evidence. The owner (Raghav) should not need to re-check your work line by line. He reviews your decision log and your evidence, not your diffs.

## 1. Product in one paragraph

IndicMeet AI turns a recorded Indian-language meeting (English, Hindi primary; other Indian languages secondary) into: speaker-attributed transcript in native script, Roman transliteration, English translation, and structured meeting intelligence (summary, decisions, action items, questions, risks) with source references back to transcript segments.

Pipeline:

```
recording -> ffmpeg (16 kHz mono wav) -> speaker turns -> language-routed ASR
 -> aligned transcript (stable segment ids) -> review gate
 -> cleanup / transliteration / translation -> LLM intelligence (Groq)
```

The original-language transcript is the source of truth. Never overwrite it with cleaned, transliterated or translated text. Enrichment lives in separate fields.

## 2. Environment facts (verify, do not assume)

- Windows. Repo root `D:\crazibrain\indicmeet-ai`. Python 3.10.11 in `.venv`. If `python` is not on PATH in your shell, use `.venv\Scripts\python.exe`. Do not reinstall Python.
- No NVIDIA GPU locally. GPU stages (Whisper large-v3, IndicConformer, diarization) run on Kaggle. Locally you can run: ffmpeg, the LLM stage, alignment, validation, tests, the API.
- 15.7 GB RAM. Loading large models locally has previously crashed (`mkl_malloc`). Do not try to load Whisper large-v3, Qwen 1.7B or IndicConformer on this machine.
- Models are hosted by a teammate, not by Raghav. Never hardcode a model location or assume local GPU. ASR must sit behind a provider interface (`transcribe(audio_path, turns) -> list[Segment]`) with at least a `LocalGPU` implementation (current code) and a `Remote` implementation (HTTP, configurable base URL via env var). Until the teammate's endpoint exists, write the Remote client against a documented request/response contract and test it against a fake server.
- Secrets (Groq key, HF token) come from environment variables or `.env` (gitignored). Never commit them, never print them, never put them in logs.

## 3. Decision authority

### Decide and proceed (do not ask)
- Anything local, reversible, and inside this repo: code structure, naming, refactors needed for the task, tests, fixtures, schemas, CLI flags, logging format, error handling, retry policy, batch sizes.
- Choosing between two reasonable implementations. Pick one, write down why in `DECISIONS.md`.
- Fixing bugs you find in adjacent code if the fix is small and tested. Note it in the decision log.
- Creating branches and small commits.

### Stop and ask (one focused question, with your recommended default)
- Spending money or adding a paid service/API.
- Changing the public data contract (section 5) after other components depend on it.
- Deleting or overwriting anything in `ground_truth/`, `output/`, or any Kaggle artifact. Those are irreplaceable.
- Force-push, history rewrite, or touching `main` directly.
- Installing heavy or exotic dependencies (vLLM, flash-attn, quantization libs, full NeMo). They already failed or were ruled out here.
- Anything that would send meeting audio or transcripts to a third-party service not already in the design.
- Choices that change what the product promises to users (e.g. dropping a language, auto-accepting rejected segments).

If blocked on a stop-and-ask item, do not idle. Finish everything that does not depend on the answer, then ask.

## 4. Verification is your job

You may not say "done", "fixed" or "working" unless you ran something that proves it in this session. Evidence means command output, not reasoning.

Definition of done for any change:
1. Code runs: show the exact command and its tail output.
2. A test covers the behavior. Bug fixes get a regression test that fails before the fix and passes after.
3. `pytest` passes in full (`tests/` is currently empty; creating a real suite is part of your job).
4. For data-shape changes, validate real artifacts against the schema, not just toy dicts.
5. Logging for any stage over ~10 s shows: stage, input, model, device, elapsed, progress, warnings, errors, completion.
6. `DECISIONS.md` updated if you made a non-trivial choice.

Build this test harness early, then use it for everything:
- `tests/test_contract.py`: schema validation for ASR output, alignment output, and the LLM input/output.
- `tests/test_gate.py`: quality gate (empty text, repetition, unexpected language, rejected segments never reach the LLM, `--skip-review` excludes review segments from the summary too).
- `tests/test_alignment.py`: overlapping speech, turns longer than 25 s, chunk boundaries, stable segment ids across reruns.
- `tests/test_llm_stage.py`: fake Groq client. Covers omitted segment ids, bad JSON, rate limits (bounded retries), batch shrink on failure, source references that point to real segment ids.
- `tests/fixtures/`: small hand-checked segment JSONs, including the validated AGM window (870 to 990 s) and a Bengali sample, with expected outputs.
- A `make check` (or `scripts/check.ps1`) that runs lint, tests and a schema check on the sample artifacts in one command.

Use fixtures and fakes for anything that needs a GPU or network. A test that needs Kaggle is not a test you can run; label it `integration` and keep it out of the default run.

Be honest about what you could not verify. "Not run, needs GPU" is an acceptable report. A claimed pass you did not run is not.

## 5. Canonical data contract

One segment shape everywhere. If current code disagrees, the code changes, not the contract, unless you log a decision.

```json
{
  "segment_id": "seg_000123",
  "start": 12.34,
  "end": 18.90,
  "speaker": "SPEAKER_01",
  "speaker_name": null,
  "language": "hi",
  "text_native": "मैं कल रिपोर्ट भेज दूंगा",
  "text_roman": null,
  "text_english": null,
  "quality": "accepted",
  "reasons": [],
  "asr": {"method": "indicconformer_hi", "whisper_lang": "hi", "whisper_lang_conf": 0.97,
          "mms_lang": "hi", "mms_lang_conf": 0.99}
}
```

- `language`: ISO 639-1 (`en`, `hi`, `bn`, ...), plus owner-approved `mul` (mixed) and `und` (unknown) for review/rejected segments only (2026-10-05). Preserve the original label in `asr.source_language`; never auto-accept these markers. `ur` is normalized to `hi` for routing but keep the raw value in `asr`.
- `quality`: exactly `accepted | review | rejected`. `rejected` text never goes to the LLM. `review` is excluded from the summary when review mode is strict.
- `segment_id` is deterministic (derived from start time and speaker), stable across reruns.
- Language confidence and transcription quality are different things. Do not merge them.

## 6. Known issues to confirm and fix first

These were reported by a previous review. Confirm each yourself before acting; some may be stale.

1. Local `indicmeet_asr.py` is the old v1 (Whisper `small`, RNNT, `device="cuda"` default). The newer v2 (diarization-turn segmentation, Whisper large-v3, CTC decoding, MMS-LID cross-check, quality gate) lives in `indicmeet_asr_v2.py`, whose docstring says "reconstructed". It may differ from what actually ran on Kaggle. Decide which is canonical, diff against the Kaggle notebook if available, and record the decision.
2. Field mismatch between ASR and LLM stages: ASR emits `lang` and `quality`; LLM reads `language` and other names (check `quality_flag` vs `quality`, `reject` vs `rejected`). Fix with the contract in section 5 plus an adapter test with real artifacts.
3. `--skip-review` skips enrichment of review segments but still lets them into the summary. Fix and test.
4. Diarization code is not in the repo. v2 only reads a CSV with `start,end,speaker`. Add the diarization step (pyannote, run on Kaggle) as a script that produces that CSV, so runs are reproducible.
5. `indicmeet_asr_v2.py` hardcodes `compute_type="float16"` and CUDA. Make device and compute type configurable, with a clear error on CPU rather than a crash.
6. Equal-length splitting of long turns (25 s) and IndicConformer sub-chunks (8 s) can cut mid-word. Prefer splitting at VAD silences; fall back to equal splits only when no silence is found.
7. Groq stage: add smaller-batch recovery, retry for omitted segment ids, bounded rate-limit backoff, and an input-size strategy for long meetings (map-reduce summary).
8. Git has few or no commits of the real work. Commit in small, reviewable steps on a branch.

## 7. Speaker source: diarization vs LiveKit logs

The meeting app records via LiveKit. A requirements doc was sent to that developer asking for `session.json` and `events.jsonl` (join/leave, mic on/off, `active_speakers`) with `t_ms` relative to recording start, plus optionally per-participant audio tracks.

Design for both sources behind one interface:
- `SpeakerSource.turns(session) -> list[Turn]` with implementations `Diarization` (current) and `LiveKitEvents` (when logs exist).
- If per-participant tracks exist, transcribe each track separately and merge by time. That removes diarization and overlap errors.
- If only the composite recording plus `active_speakers` exists, use events to name diarization clusters (map cluster to identity by overlap in time) rather than replacing diarization.
- Build the `LiveKitEvents` reader against the documented format with fixtures. Do not wait for real data.

## 8. Language routing rules (hard-won, do not regress)

- Wrong language ID turns a good ASR into confident garbage. LID is the highest-risk stage.
- Whisper's built-in LID beats MMS-LID on Indian-accented English. MMS-LID is a cross-check for Indic languages, not the primary signal for English.
- Latin-script output alone does not mean English. Require Whisper to also say `en`.
- Hindi and Urdu labels are acoustically the same spoken language here. Route both to `hi`.
- IndicConformer RNNT silently returns empty text on segments over about 10 to 13 s. Keep sub-chunks at 8 s or less.
- Languages outside the allowlist are flagged for review, not silently accepted.

## 9. Evaluation rules

- Report numbers with their reference. The current "42/42 routing" and "about 10.5% WER" were measured against LLM-generated transcripts, which are not human ground truth. Label them soft.
- The 85.6% speaker-match score on the two-minute AGM window is a custom metric, not DER. Do not present it as diarization accuracy.
- Add real metrics when you can: WER/CER with a human-checked subset, language-ID accuracy, DER where reference RTTM exists.
- Never tune thresholds on the same sample you report results on.

## 10. Code and logging rules

- One component at a time. No unrelated refactors.
- Long commands print stage, input, output, model, device, elapsed, progress, warnings, failures. Keep that format consistent via a small shared logger.
- Never log full transcripts or audio paths containing personal names at INFO level. Log ids, counts, durations.
- Handle per-segment failures without aborting the run; record the exception in `reasons` and continue. Checkpoint long runs.
- Pin dependencies in requirements files. `qwen-asr` pins `transformers==4.57.6` and conflicted with the ASR stack before; keep the LLM and ASR environments separate or verify they coexist.
- Python 3.10 syntax only.

## 11. Order of work

Unless the owner redirects you:
1. Confirm section 6 items, create `DECISIONS.md`, set up pytest and `make check`.
2. Fix the data contract and the review-gate bug, with tests.
3. Implement the alignment module with stable ids and tests.
4. Provider interface for ASR, plus the Remote client against a fake server.
5. Make the Groq stage resilient, with fake-client tests.
6. One reviewed end-to-end run on the validated 870 to 990 s AGM window using stored ASR output: transcript, speaker attribution, translation, extracted claims, each claim traced to segment ids.
7. Only then: FastAPI wrapper, job queue, storage, UI. The React port of the demo UI waits until the pipeline works end to end.

## 12. How to report

Every report ends with this block:

```
DONE (with evidence): <what, plus command + result>
NOT VERIFIED: <what you could not run and why>
DECISIONS: <entries added to DECISIONS.md, one line each>
NEEDS OWNER: <only stop-and-ask items, each with your recommended default>
NEXT: <what you will do next>
```

`DECISIONS.md` entry format:

```
## YYYY-MM-DD <short title>
Context: <one or two lines>
Options: <A / B>
Chosen: <X>
Why: <reason>
Reversible: yes/no
```

No filler, no praise. If the owner's request is wrong or risky, say so plainly and propose the better option.