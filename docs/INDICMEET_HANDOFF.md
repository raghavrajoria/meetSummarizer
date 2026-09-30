# IndicMeet AI — Session Handoff

_Last updated: 28 September 2026_

## 1. Current objective

Build and validate a multilingual Indian meeting-intelligence pipeline:

```text
Meeting audio/video
  -> audio preparation and VAD
  -> speaker diarization
  -> multilingual ASR and language routing
  -> speaker/transcript alignment
  -> native transcript and optional transliteration/translation
  -> meeting summary, decisions, action items, requirements, and risks
```

The original-language transcript is the source of truth. Translation and meeting intelligence should consume a reviewed transcript.

## 2. Files and data used

### Kaggle audio

```text
/kaggle/input/datasets/raghavrajoria/indicmeetdata/hinglish_AGM_call.wav
/kaggle/input/datasets/raghavrajoria/indicmeetdata/bengali_meeting_audio.wav
```

The AGM recording is approximately 20 minutes long. A local copy also exists at:

```text
C:\Users\raghav\Downloads\hinglish_AGM_call.wav
```

### Kaggle benchmark clips

```text
/kaggle/working/indicmeet_benchmark/clips/
```

This directory contains Hindi and Bengali clips such as `hi_144.wav`, `hi_152.wav`, and `bn_019.wav`.

### Generated artifacts

```text
/kaggle/working/indicmeet_asr_v2.py
/kaggle/working/gemini_transcript.json
/kaggle/working/agm_diarization.csv
/kaggle/working/agm_aligned_transcript.csv
```

Files under `/kaggle/working` survive while the current Kaggle session is active and can be downloaded or saved as notebook output. Files under `/kaggle/input` remain attached to the notebook.

## 3. ASR work completed before diarization

The reusable `indicmeet_asr_v2.py` module was created in Kaggle. Its main architecture is:

- Faster Whisper large-v3 for initial transcription and language identification.
- IndicConformer 600M for supported Indic-language ASR.
- MMS-LID as an audio-level language cross-check.
- Silero VAD for speech segmentation.
- CTC decoding for IndicConformer.

Important fixes already incorporated in v2:

1. IndicConformer was switched from RNNT to CTC because RNNT silently returned empty text on some short clips.
2. Low-confidence Hindi/Urdu routing now uses MMS-LID confirmation rather than routing unconditionally.
3. Latin-script Whisper output is no longer automatically considered reliable English in every case.
4. Results include quality states such as accepted, review, and rejected.

Known ASR limitation:

- Numeric or address-heavy Hindi can be transliterated into Latin text and confidently misidentified as English. The `hi_152` case remains an open edge case.

## 4. Loading pyannote diarization

### Hugging Face access

The following gated repositories were involved:

- `pyannote/segmentation-3.0`
- `pyannote/speaker-diarization-3.1`
- `pyannote/speaker-diarization-community-1`

Access was accepted for the required repositories. The Kaggle secret containing the Hugging Face token is named `tts`. It is attached to the notebook and can be loaded without printing the token:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

hf_token = UserSecretsClient().get_secret("tts")
login(token=hf_token)
```

Never type the abbreviated token shown in the Kaggle UI, such as `hf_...AkWw`; the abbreviation is display-only. `get_secret("tts")` returns the full value.

### Telemetry/OpenTelemetry error

The first pipeline load failed with:

```text
AttributeError: type object 'TraceFlags' has no attribute 'RANDOM_TRACE_ID'
```

This was an OpenTelemetry API/SDK mismatch inside pyannote's optional telemetry, not a Hugging Face permission problem. Telemetry was disabled before importing pyannote:

```python
import os
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

import torch
from pyannote.audio import Pipeline
```

OpenTelemetry 1.45.0 was already installed. The OmegaConf dependency parsing warning was unrelated.

### Authentication and access errors resolved

Errors encountered and their causes:

- `401 Unauthorized` from `whoami-v2`: a placeholder or invalid token value was passed.
- `No user secrets exist ... label tts`: the Kaggle secret existed at account level but was not enabled for the current notebook.
- `403 Forbidden` for `speaker-diarization-community-1`: the account had not yet accepted that repository's gated conditions.

After attaching the `tts` secret and accepting access, the PLDA files and model downloaded successfully:

```text
plda/xvec_transform.npz
plda/plda.npz
pytorch_model.bin
```

### Working pipeline load

```python
import os
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

import torch
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from pyannote.audio import Pipeline

hf_token = UserSecretsClient().get_secret("tts")
login(token=hf_token)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token,
)

pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("Pipeline loaded:", pipeline is not None)
```

The installed pyannote version returns a `DiarizeOutput`, so speaker turns are accessed through `speaker_diarization`:

```python
diarization = pipeline(
    "/kaggle/input/datasets/raghavrajoria/indicmeetdata/hinglish_AGM_call.wav"
)

for turn, _, speaker in diarization.speaker_diarization.itertracks(
    yield_label=True
):
    print(f"{turn.start:.2f}s–{turn.end:.2f}s  {speaker}")
```

Calling `diarization.itertracks(...)` directly fails because `DiarizeOutput` itself does not expose that method.

### Runtime observations

- A `std(): degrees of freedom is <= 0` warning may appear for extremely short sequences. It is a warning, not necessarily a failed run.
- The first full AGM run was manually interrupted during speaker embedding, so `diarization` was never assigned and a later reference caused `NameError`.
- Moving the pipeline to CUDA and allowing it to finish resolved this.
- If the Kaggle kernel restarts, the Python variables and loaded models must be recreated. This does not damage model repositories or input data.

## 5. Full AGM diarization result

The AGM run completed and produced:

- 269 diarization turns.
- 9 anonymous speaker clusters.
- Recording coverage through approximately 19:53.

The CSV was saved as:

```text
/kaggle/working/agm_diarization.csv
```

Speaker totals from the CSV:

| Speaker | Turns | Detected speech seconds |
|---|---:|---:|
| SPEAKER_00 | 20 | 226.7 |
| SPEAKER_08 | 57 | 205.1 |
| SPEAKER_07 | 22 | 122.5 |
| SPEAKER_05 | 21 | 118.9 |
| SPEAKER_03 | 43 | 93.2 |
| SPEAKER_01 | 42 | 76.1 |
| SPEAKER_02 | 27 | 53.3 |
| SPEAKER_04 | 25 | 36.0 |
| SPEAKER_06 | 12 | 33.3 |

The file contains 37 turns shorter than 0.5 seconds, 79 turns shorter than one second, and some overlapping turns. Very short turns should not automatically be treated as distinct real speakers.

CSV creation code:

```python
import pandas as pd

annotation = diarization.speaker_diarization

rows = []
for turn, _, speaker in annotation.itertracks(yield_label=True):
    rows.append({
        "start": turn.start,
        "end": turn.end,
        "duration": turn.end - turn.start,
        "speaker": speaker,
    })

turns_df = pd.DataFrame(rows)
turns_df.to_csv("/kaggle/working/agm_diarization.csv", index=False)
```

For downstream ASR alignment where one speaker is needed at each instant, use:

```python
annotation = diarization.exclusive_speaker_diarization
```

Keep the regular `speaker_diarization` result when evaluating real overlapping speech.

## 6. Gemini transcript experiments

Gemini was asked to transcribe the full AGM with speakers and timestamps. Two outputs were tried.

### First Gemini output

- 46 large segments.
- Start timestamps only.
- Ended at about 15:49 even though the audio continues to about 19:53.
- Included unusable `[cite: 1]` placeholders.
- Speaker labels did not consistently correspond with pyannote labels.

An approximate alignment CSV was made by using the next Gemini segment's start as the previous segment's estimated end:

```text
/kaggle/working/agm_aligned_transcript.csv
```

This is only a draft. It must not be used as ground truth.

### Second Gemini output

- 133 segments.
- Included start and end timestamps.
- Stopped at 16:48.040 and ended mid-sentence: `I would like to`.
- Contained transcription mistakes such as identifying the AGM as an EGM.
- Speaker labels and timing still needed verification.

### Two-minute Gemini experiment

A two-minute clip from 870 to 990 seconds was sent to Gemini to reduce context length. It still showed severe timestamp drift:

- Human listening: Saraph ended at approximately 00:32 relative to the clip.
- Gemini: Saraph ended at 00:23.
- Human listening: Sivaraman began the main question at approximately 00:49.
- Gemini: the question began around 00:38.
- Human listening: Sivaraman finished around 01:26.
- Gemini: he finished around 01:08.

Conclusion: Gemini transcripts may help recover draft wording, but Gemini-generated timestamps and speaker labels must not be used as benchmark ground truth.

## 7. Human-checked two-minute diarization benchmark

The tested clip covers full-recording time:

```text
870s–990s, or 00:14:30–00:16:30
```

It was selected because pyannote reported multiple speakers and many changes. Manual listening identified these approximate clip-relative events:

| Clip time | Human event |
|---|---|
| 00:00–00:32 | Saraph finishes his speech and says his closing lines |
| 00:32–00:35 | Main meeting speaker says thank you twice |
| 00:36 | Cough/non-speech |
| 00:37–00:42 | Moderator invites Sivaraman Hariharan |
| 00:43–00:46 | Sivaraman asks whether he is audible |
| Around 00:48 | Main speaker and moderator respond, with overlap |
| 00:49–01:26 | Sivaraman asks his questions |
| 01:27–01:29 | Main speaker thanks Sivaraman |
| 01:30–01:36 | Moderator invites Rajendra Jamnadas |
| Around 01:38 | Rajendra says hello |
| Around 01:42 | Main speaker and moderator respond with overlap |
| 01:44 onward | Rajendra begins his speech |

The human reference used for single-speaker scoring excluded the cough, uncertain gaps, and overlapping responses:

```python
reference = [
    (0, 32, "SARAPH"),
    (32, 35, "MAIN"),
    (37, 42, "MODERATOR"),
    (43, 46, "SIVARAMAN"),
    (49, 86, "SIVARAMAN"),
    (87, 89, "MAIN"),
    (90, 96, "MODERATOR"),
    (98, 101, "RAJENDRA"),
    (104, 120, "RAJENDRA"),
]
```

### Automatic cluster mapping

The anonymous pyannote clusters mapped to the human voices as follows:

| Pyannote cluster | Human identity |
|---|---|
| SPEAKER_00 | SARAPH |
| SPEAKER_03 | MAIN |
| SPEAKER_01 | MODERATOR |
| SPEAKER_06 | SIVARAMAN |
| SPEAKER_05 | RAJENDRA |

### Result

```text
Human-labeled speech: 107.0 seconds
Correctly matched:     91.6 seconds
Speaker match score:   85.6%
```

This is not formal diarization error rate. The reference intervals were deliberately coarse and contain natural pauses, while pyannote labels speech activity only. Therefore, 85.6% understates the visible speaker attribution quality.

Observed boundary quality was strong:

- Saraph: predicted 00:00–00:31.32 versus human end near 00:32.
- Main: predicted start 00:32.49 versus human start near 00:32.
- Moderator: predicted turns from 00:36.30 through 00:42.01 versus human 00:37–00:42.
- Sivaraman: predicted start 00:43.19 and retained the same cluster across pauses.
- Main returned at 01:27.35 versus human start near 01:27.
- Moderator returned at 01:30.20 versus human start near 01:30.
- Rajendra began at 01:38.13 versus human start near 01:38.

There were tiny label changes lasting hundredths of a second around overlaps. These should not be interpreted as meaningful speaker changes.

### Current diarization conclusion

Pyannote diarization is validated on this small, human-checked five-speaker sample. It correctly separated and consistently clustered the five voices, with most boundaries within roughly one second of manual observations.

This does not yet prove full-meeting accuracy. More samples should be reviewed before reporting a general diarization metric.

## 8. Code used to locate useful test windows

```python
import torchaudio
from IPython.display import Audio, display

audio_path = "/kaggle/input/datasets/raghavrajoria/indicmeetdata/hinglish_AGM_call.wav"

annotation = diarization.speaker_diarization
turns = [
    (turn.start, turn.end, speaker)
    for turn, _, speaker in annotation.itertracks(yield_label=True)
]

waveform, sample_rate = torchaudio.load(audio_path)
audio_duration = waveform.shape[1] / sample_rate

window_seconds = 120
step_seconds = 30
candidates = []

for start in range(
    0,
    max(1, int(audio_duration - window_seconds) + 1),
    step_seconds,
):
    end = start + window_seconds
    active_turns = [
        (max(start, turn_start), min(end, turn_end), speaker)
        for turn_start, turn_end, speaker in turns
        if turn_start < end and turn_end > start
    ]

    speech_seconds = sum(
        turn_end - turn_start
        for turn_start, turn_end, _ in active_turns
    )

    ordered = sorted(active_turns, key=lambda item: item[0])
    speakers = {speaker for _, _, speaker in ordered}
    switches = sum(
        ordered[index][2] != ordered[index - 1][2]
        for index in range(1, len(ordered))
    )

    score = (
        speech_seconds
        + 8 * switches
        + 5 * max(0, len(speakers) - 1)
    )

    candidates.append({
        "start": start,
        "end": end,
        "speech_seconds": round(speech_seconds, 1),
        "speakers": len(speakers),
        "switches": switches,
        "score": score,
    })

candidates.sort(key=lambda item: item["score"], reverse=True)

for candidate in candidates[:5]:
    print(candidate)

best = candidates[0]
start_sample = int(best["start"] * sample_rate)
end_sample = int(best["end"] * sample_rate)
clip = waveform[:, start_sample:end_sample].mean(dim=0)

print(f"Playing {best['start']}s–{best['end']}s")
display(Audio(clip.numpy(), rate=sample_rate))
```

Top candidates from this run were:

```text
870–990 seconds
900–1020 seconds
810–930 seconds
840–960 seconds
1020–1140 seconds
```

## 9. What was done correctly

- Hugging Face access and Kaggle secret handling were fixed without exposing the token.
- The OpenTelemetry error was correctly separated from the gated-model access errors.
- The pyannote pipeline loaded and ran on CUDA.
- The current pyannote API's `DiarizeOutput` structure was handled correctly.
- Full AGM diarization was completed and exported to CSV.
- A complex two-minute section was selected programmatically and checked by ear.
- Arbitrary pyannote cluster labels were mapped to human identities using temporal overlap.
- Overlapping speech and non-speech were excluded from the simple single-speaker score.
- Gemini timing errors were detected rather than accepted as reference data.

## 10. What remains uncertain

- The 85.6% value is a custom speaker-match score, not DER.
- The manually labeled intervals are approximate to about one second.
- Only one two-minute section has been human-checked carefully.
- Overlap detection has not yet been scored.
- Nine clusters in the full meeting have not all been verified as nine distinct people.
- Full-meeting ASR text accuracy and speaker/transcript alignment remain unfinished.
- Gemini outputs are incomplete and timestamp-unstable, so they cannot be used as reference truth.

## 11. Recommended next steps

### Immediate next step: test ASR on the validated two-minute clip

1. Export the 870–990 second clip to `/kaggle/working/agm_test_870_990.wav`.
2. Run `IndicMeetASR` on that clip.
3. Convert ASR-relative timestamps back to full-recording timestamps by adding 870 seconds.
4. Assign speakers using maximum temporal overlap with `diarization.exclusive_speaker_diarization`.
5. Use the verified mapping in this handoff to display real names for this sample.
6. Listen only to segments flagged for review or rejected, plus a small random sample of accepted segments.

Clip export code:

```python
import torchaudio

audio_path = "/kaggle/input/datasets/raghavrajoria/indicmeetdata/hinglish_AGM_call.wav"
clip_path = "/kaggle/working/agm_test_870_990.wav"

waveform, sample_rate = torchaudio.load(audio_path)
clip = waveform[
    :,
    int(870 * sample_rate):int(990 * sample_rate),
]

torchaudio.save(clip_path, clip, sample_rate)
print(clip_path)
```

Run ASR:

```python
from indicmeet_asr_v2 import IndicMeetASR

asr = IndicMeetASR(
    device="cuda",
    whisper_size="large-v3",
    whisper_compute_type="float16",
    hf_token=hf_token,
)

asr_results = asr.transcribe_file(clip_path)
```

### After the ASR sample works

- Align ASR segments to pyannote turns by maximum overlap.
- Save one JSON containing `start`, `end`, `speaker`, `language`, `text`, `method`, and `quality`.
- Review ASR wording for the known Sivaraman and Rajendra portions.
- Add two or three more short human-checked windows from other parts of the AGM.
- Include a quieter section, an overlap-heavy section, and a Hindi/English code-switching section.
- Only after these checks, run the complete speaker-aligned transcript over the full AGM.

### Later engineering work

- Speaker-to-real-name enrollment or collaborator metadata.
- Better handling of numeric/address-heavy Indic speech.
- Formal ASR WER/CER using normalized human reference text.
- Formal diarization evaluation using RTTM reference annotations and DER/JER.
- Transliteration and English translation after the native transcript is accepted.
- LLM meeting intelligence after transcript verification.
- Backend/frontend integration: FastAPI, workers, database, object storage, and UI.

## 13. Groq LLM integration (new)

`indicmeet_llm.py` implements two transcript-consuming passes:

1. Segment enrichment: conservative native-script cleanup, Roman transliteration,
   and English translation. It batches segments, preserves each original `text`
   and all alignment/quality fields, and writes results under `segment.llm`.
2. Meeting intelligence: summary, decisions, action items, open questions, and
   risks. Each extracted item must cite source segment IDs. Rejected segments are
   excluded; review segments are included by default and can be omitted.

Install `groq` from `requirements-llm.txt`, set `GROQ_API_KEY` as an environment
secret, then run:

```powershell
$env:GROQ_API_KEY = "..."  # Prefer a secret manager in persistent environments.
python .\indicmeet_llm.py .\agm_full_aligned.json
```

The default output is `agm_full_aligned_llm.json`. The source transcript is not
overwritten. `GROQ_MODEL` can override the default `openai/gpt-oss-20b` model;
`--batch-size`, `--skip-review`, `--no-enrich`, and `--no-summary` are available.
Run on a copy of the reviewed transcript and inspect Indic-script cleanup,
transliteration, and cited meeting claims before treating them as verified.

## 12. Suggested opening message for the next chat

```text
Continue the IndicMeet AI work using INDICMEET_HANDOFF.md in the project folder.
The pyannote diarization pipeline is working and has been human-validated on the
870–990 second AGM window. The next task is to run indicmeet_asr_v2.py on that
two-minute clip, align its timestamped ASR output with the existing pyannote
speaker turns, apply the verified speaker mapping, and save a reviewable JSON.
Preserve original-language text and quality flags. Do not use Gemini timestamps
as ground truth.
```
