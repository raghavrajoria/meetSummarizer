# IndicMeet Work Log — 28 September 2026

## Scope

This document summarizes the IndicMeet meeting-intelligence work completed on 28 September 2026. It covers pyannote setup and diarization, validation against a human-reviewed audio window, Gemini transcript experiments, handoff documentation, and the first Groq-based transcript-enrichment implementation.

## Outcome at the end of the day

By the end of the session:

- The pyannote speaker-diarization pipeline loaded successfully in Kaggle and ran on GPU.
- The full approximately 20-minute Hinglish AGM was diarized and exported to CSV.
- The run produced 269 speaker turns across 9 anonymous speaker clusters, covering the recording through approximately 19:53.
- A difficult two-minute, five-speaker window was manually checked and used as a small diarization benchmark.
- Pyannote kept the five speakers consistent and achieved an 85.6% custom speaker-match score on the coarse human reference.
- Gemini was tested as a transcript/timestamp source and rejected as ground truth because its outputs were incomplete and its timestamps drifted substantially.
- A detailed project handoff was created.
- A Groq integration was added for transcript cleanup, transliteration, English translation, and evidence-linked meeting intelligence.
- Kaggle testing exposed two Groq reliability issues: omitted segment IDs and token-per-minute rate limiting. Safer retry strategies were designed, but the complete end-to-end LLM output was not yet verified.

## 1. Pyannote and Hugging Face setup

### Problems diagnosed

The initial failure was not caused by missing Hugging Face access. Model files had downloaded, but pyannote telemetry failed because of an OpenTelemetry compatibility issue:

```text
AttributeError: type object 'TraceFlags' has no attribute 'RANDOM_TRACE_ID'
```

The following separate authentication and access issues were then identified:

- `401 Unauthorized`: an invalid or placeholder Hugging Face token was being used.
- Kaggle secret lookup failure: the account-level secret named `tts` had not been enabled for the current notebook.
- `403 Forbidden`: access had not yet been accepted for `pyannote/speaker-diarization-community-1`, which was required by the pipeline configuration.

### Fixes applied

- Disabled pyannote telemetry before importing the library:

```python
import os
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
```

- Loaded the full Hugging Face token securely from Kaggle Secrets instead of copying the abbreviated UI value:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

hf_token = UserSecretsClient().get_secret("tts")
login(token=hf_token)
```

- Enabled the `tts` secret for the notebook.
- Accepted access for all required gated repositories:
  - `pyannote/segmentation-3.0`
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/speaker-diarization-community-1`
- Confirmed successful downloads of the PLDA files and speaker model.
- Loaded the pipeline and moved it to CUDA.

### API/runtime corrections

- The installed pyannote version returns a `DiarizeOutput` object.
- Speaker turns must be read from `diarization.speaker_diarization`, not by calling `itertracks` directly on `diarization`.
- `diarization.exclusive_speaker_diarization` should be used when downstream ASR alignment needs exactly one speaker at each instant.
- The PyTorch `std(): degrees of freedom is <= 0` message was identified as a warning caused by very short sequences, not necessarily a failed run.
- The first full run was manually interrupted, so `diarization` was never assigned. The full run succeeded after using CUDA and allowing processing to finish.

## 2. Full AGM diarization

The source recording was:

```text
/kaggle/input/datasets/raghavrajoria/indicmeetdata/hinglish_AGM_call.wav
```

The completed run produced:

- 269 diarization turns
- 9 anonymous speaker clusters
- Coverage through approximately 19:53
- 37 turns shorter than 0.5 seconds
- 79 turns shorter than 1 second
- Some overlapping turns

The result was saved as:

```text
/kaggle/working/agm_diarization.csv
```

Speaker totals:

| Speaker | Turns | Detected speech seconds |
|---|---:|---:|
| `SPEAKER_00` | 20 | 226.7 |
| `SPEAKER_08` | 57 | 205.1 |
| `SPEAKER_07` | 22 | 122.5 |
| `SPEAKER_05` | 21 | 118.9 |
| `SPEAKER_03` | 43 | 93.2 |
| `SPEAKER_01` | 42 | 76.1 |
| `SPEAKER_02` | 27 | 53.3 |
| `SPEAKER_04` | 25 | 36.0 |
| `SPEAKER_06` | 12 | 33.3 |

Short turns were not automatically treated as genuine new speakers because some are caused by pauses, overlap, or boundary artifacts.

## 3. Gemini transcript experiments

Gemini was tested on the full AGM and on a shorter two-minute clip.

### Full-recording results

The first Gemini output:

- Had 46 large segments.
- Included only start timestamps.
- Stopped around 15:49 even though the recording continued to about 19:53.
- Included unusable citation placeholders.
- Used speaker labels that did not consistently match pyannote.

The second Gemini output:

- Had 133 segments with start and end timestamps.
- Stopped at 16:48.040 and ended mid-sentence.
- Contained transcription mistakes, including calling the AGM an EGM.
- Still required manual verification of speakers and timing.

An approximate alignment file was created:

```text
/kaggle/working/agm_aligned_transcript.csv
```

It remains a draft and must not be treated as reference truth.

### Two-minute test

The clip from 14:30 to 16:30 of the full recording was sent to Gemini. Timestamp drift remained severe:

| Event | Human timing | Gemini timing |
|---|---:|---:|
| Saraph finishes | about 00:32 | 00:23 |
| Sivaraman starts his main question | about 00:49 | about 00:38 |
| Sivaraman finishes | about 01:26 | about 01:08 |

Decision: Gemini may help recover draft wording, but its speaker labels and timestamps are not suitable as diarization ground truth.

## 4. Human-checked diarization benchmark

The reviewed window covered full-recording seconds 870–990, or 14:30–16:30.

Manual listening identified five speakers and two overlapping responses. The cough and overlapping intervals were excluded from single-speaker scoring.

Pyannote clusters mapped to people as follows:

| Pyannote cluster | Human identity |
|---|---|
| `SPEAKER_00` | Saraph |
| `SPEAKER_03` | Main meeting speaker |
| `SPEAKER_01` | Moderator |
| `SPEAKER_06` | Sivaraman |
| `SPEAKER_05` | Rajendra |

Benchmark result:

```text
Human-labeled speech: 107.0 seconds
Correctly matched:      91.6 seconds
Speaker match score:    85.6%
```

Important interpretation:

- This is a custom overlap-based speaker-match score, not formal diarization error rate.
- The human boundaries were approximate and included natural pauses, while pyannote marks detected speech.
- The score therefore understates the visible speaker-attribution quality.
- Most observed boundaries were within roughly one second of the human notes.
- Sivaraman and Rajendra kept consistent cluster identities across their turns.
- Tiny label changes around overlaps lasted only hundredths of a second and were treated as artifacts.

Conclusion: pyannote was validated on this one difficult five-speaker sample, but more windows are still needed before claiming full-meeting accuracy.

## 5. Existing ASR work captured in the handoff

The day's handoff preserved the current ASR architecture and earlier fixes:

- Faster Whisper large-v3 for initial transcription and language identification.
- IndicConformer 600M for supported Indic-language ASR.
- MMS-LID as an audio-level language cross-check.
- Silero VAD for speech segmentation.
- CTC decoding for IndicConformer.
- Quality states: accepted, review, and rejected.

Known limitation retained for follow-up: numeric or address-heavy Hindi may be transliterated into Latin text and incorrectly classified as English, including the `hi_152` case.

## 6. Groq LLM integration

### Local implementation created

`indicmeet_llm.py` was added with two passes:

1. Segment enrichment
   - Conservative transcript cleanup
   - Roman transliteration
   - English translation
   - Confidence classification
   - Original transcript text, timestamps, speaker IDs, and quality fields preserved

2. Meeting intelligence
   - Summary
   - Decisions
   - Action items
   - Open questions
   - Risks
   - Source segment IDs required for extracted claims

Safety and data-integrity behavior:

- Transcript text is treated as untrusted data, not as instructions.
- Rejected segments are excluded from the meeting-intelligence pass.
- Review segments are included by default but can be skipped.
- Unknown or malformed model output IDs raise an error instead of silently attaching text to the wrong segment.
- The source transcript is not overwritten.

The default model was set to `openai/gpt-oss-20b`, with configuration for `GROQ_MODEL`, batch size, optional enrichment, optional summary, and review-segment handling.

### Kaggle setup prepared

Instructions were provided to:

- Install the `groq` package.
- Read `GROQ_API_KEY` from Kaggle Secrets.
- Create `/kaggle/working/indicmeet_llm.py` with a `%%writefile` cell.
- Run the script against `/kaggle/working/agm_full_aligned.json`.

### Issues found during live testing

1. The initial batch size of 12 allowed the model to omit segment IDs `57–60`.
2. The script correctly stopped rather than silently misaligning the output.
3. The retry strategy was changed to use smaller batches and retry omitted segments individually.
4. A later request hit Groq's 8,000-token-per-minute rate limit.
5. A bounded retry loop was proposed to parse Groq's suggested wait time, sleep, and retry up to eight times.

End-of-day status: the integration existed and the failure modes were understood, but a complete, reviewed `agm_full_aligned_llm.json` had not yet been produced.

## 7. Files created or updated

Local project files:

| File | Work performed |
|---|---|
| `INDICMEET_HANDOFF.md` | Created a detailed handoff covering architecture, fixes, benchmark evidence, limitations, and next steps; later extended with Groq usage. |
| `indicmeet_llm.py` | Added Groq-based enrichment and meeting-intelligence processing. |
| `requirements-llm.txt` | Added the optional Groq dependency. |

Important Kaggle artifacts:

| Artifact | Purpose |
|---|---|
| `agm_diarization.csv` | Full AGM pyannote turns. |
| `agm_aligned_transcript.csv` | Approximate Gemini alignment; draft only. |
| `indicmeet_asr_v2.py` | Reusable multilingual ASR module in Kaggle. |
| `gemini_transcript.json` | Gemini transcript experiment. |

The local Git repository still had no commits. The project files were untracked at the time of review.

## 8. What was completed correctly

- Hugging Face authentication and gated-model access were fixed without exposing the token.
- The telemetry problem was separated from authentication and repository-permission problems.
- Pyannote was loaded on CUDA and the full AGM was processed.
- The current pyannote output API was handled correctly.
- Full diarization was exported for downstream work.
- A challenging test window was selected and checked by ear.
- Anonymous speaker clusters were mapped to known speakers with temporal overlap.
- Non-speech and overlap were excluded from simple single-speaker scoring.
- Gemini's timing errors were caught instead of being accepted as truth.
- Groq outputs were validated by segment ID to prevent silent corruption.
- LLM-derived text was kept separate from the original transcript.
- Meeting-intelligence claims were designed to retain source-segment evidence.

## 9. Remaining work

### Immediate

1. Export the validated 870–990 second audio window.
2. Run `IndicMeetASR` on that clip.
3. Add 870 seconds to ASR-relative timestamps.
4. Align ASR segments to `exclusive_speaker_diarization` using maximum temporal overlap.
5. Apply the verified five-speaker mapping.
6. Review rejected/review segments and a small random sample of accepted segments.

### Next validation stage

- Save a single JSON containing `start`, `end`, `speaker`, `language`, `text`, `method`, and `quality`.
- Review the known Sivaraman and Rajendra passages for word accuracy.
- Add two or three more human-checked windows:
  - a quiet section,
  - an overlap-heavy section,
  - a Hindi/English code-switching section.
- Measure formal DER/JER once RTTM reference annotations exist.
- Measure ASR WER/CER once normalized human reference text exists.

### Groq completion

- Confirm that the Kaggle version uses small batches.
- Add bounded rate-limit retries to the actual script.
- Retry omitted segments individually.
- Produce and manually inspect the final enriched transcript.
- Verify native-script cleanup, transliteration, translation, and every cited meeting claim.
- Compare the summary structure with the intended static webpage once that page or its URL is available.

### Later engineering

- Resolve numeric/address-heavy Indic speech routing.
- Add real-name speaker enrollment or collaborator metadata.
- Run the full speaker-aligned transcript only after sample validation.
- Add backend/frontend infrastructure: FastAPI, workers, database, object storage, and UI.

## Final status

The diarization stage moved from blocked setup to a completed full-meeting run with one human-validated sample. The next critical path is ASR-to-speaker alignment and transcript review. The Groq layer was started and designed defensively, but still requires rate-limit handling and a successful reviewed end-to-end Kaggle run.
