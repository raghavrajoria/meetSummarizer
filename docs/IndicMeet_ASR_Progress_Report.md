# IndicMeet AI — ASR Pipeline: Progress Report

## 1. Objective of This Phase

Move ASR experimentation from a local, GPU-less Windows machine (blocked by RAM limits, e.g. Qwen 1.7B failing to load) to Google Colab with a Tesla T4 GPU, and answer the core open question from the original roadmap: **can we build a reliable multilingual ASR pipeline for real Indian meeting audio, with English and Hindi as primary languages and other Indian languages as secondary/safety coverage?**

Answer, based on evidence gathered: **yes**, with a specific architecture validated below.

---

## 2. Key Diagnostic Finding

Every earlier "bad ASR" result (Whisper small/large-v3, Qwen3-ASR 0.6B/1.7B, all producing garbled/hallucinated output on the original HDFC concall recording) was traced to **language identification failure**, not audio quality or model capacity.

Proof: when the correct language code was manually forced (Bengali, via `IndicConformer(wav, "bn", ...)`), the same audio that had produced nonsense with every prior model/config suddenly transcribed near-perfectly against a human-verified ground truth transcript.

**Conclusion: language ID is the single highest-risk stage in the entire pipeline** — more critical than which ASR model is chosen, since a wrong language code turns even a good ASR model into confident-looking garbage (phonetic transliteration into the wrong script).

---

## 3. Models Validated

| Component | Model | Result |
|---|---|---|
| ASR (Indian languages) | `ai4bharat/indic-conformer-600m-multilingual` | Strong accuracy on Bengali and Hindi once given correct language code. Gated HF model — requires access request + token. RNNT decoder silently returns empty output on segments >~10-13s; fixed by sub-chunking at 8s. |
| ASR (English) | `faster-whisper` (small) | Excellent on Indian-accented English, including code-switched phrases. |
| Language ID (audio) | `facebook/mms-lid-256` | Reliable on Bengali/Hindi (95-99%+ confidence). Systematically unreliable on Indian-accented English, especially short formulaic phrases (5-30% confidence even when correct) — not fixable by threshold tuning alone. |
| Language ID (final, working) | **Whisper's own built-in LID** (`language=None`) | Dramatically outperformed MMS-LID on English (60-99% confidence vs. 5-30%). Became the primary routing signal. |
| VAD | Silero VAD | Accurate speech-segment boundaries, confirmed against ground-truth speaker-turn timestamps. |

### Note on IndicLID
AI4Bharat's "IndicLID" was initially considered but is a **text**-based language identifier (script classification), not audio-based — not usable for this pipeline stage. Corrected mid-investigation.

---

## 4. Final Validated Routing Architecture

```
Audio
  → VAD (Silero, min_speech_duration_ms=500)
  → Merge close segments (max_gap=1.0s, min_duration=1.5s)
  → Whisper transcription + LID in one pass (language=None)
  → If Whisper's output is Latin-script → trust as English, done
  → If Whisper LID reports "hi" or "ur" → re-run through IndicConformer (hi)
       (Hindi/Urdu are acoustically the same spoken language; Whisper
        sometimes labels Hindi speech "ur" — this is expected, not an error)
  → If Whisper LID reports another Indic language → cross-check against
       MMS-LID; route to IndicConformer only if both agree or MMS-LID
       confidence > 85%
```

This two-stage design (audio-level LID + script-level verification) was necessary because no single model was reliable alone.

---

## 5. Test Results

### Test 1 — HDFC Bank Concall (Bengali/Hindi/English, secondary-language coverage)
- Real, messy audio: shouting, cross-talk, rapid interruptions, secondary languages (Assamese, Telugu) briefly detected but not present — traced to LID confusion during clapping/cross-talk, not genuine.
- Once routing was corrected: strong match against ground truth on Bengali segments (e.g. "৮ তারিখে ১০টা সেভিংস খুলেছে..." matched near word-for-word).

### Test 2 — Axis Bank AGM Q&A (English/Hindi, primary target market) — 20 minutes, 42 segments
- **Language routing accuracy: 42/42 correct (100%)**, verified against a full human/LLM-generated ground-truth transcript.
- **Word Error Rate: ~10.5%** on sampled English segments (formal measurement across a larger sample still pending — see Open Items).
- Successfully handled short formulaic phrases ("please go ahead," "thank you, sir") that had broken every earlier routing approach.
- Correctly resolved Hindi/Urdu label confusion by treating them as equivalent for routing purposes.

---

## 6. Other Findings Along the Way

- **`hinglish_test_4to12min.wav`** (an early test file) was found to be permanently clipped at the recording stage (`min`/`max` at exactly ±1.0) — unrecoverable even with `ffmpeg`'s `adeclip` filter. Dropped as a test asset; not a pipeline defect.
- **Speaker naming** (attaching real names, not just `SPEAKER_01` labels, to diarized voices) was scoped as a separate problem from diarization itself. It requires either (a) voice enrollment against a reference sample per person, or (b) correlating diarized segments with meeting-platform metadata (attendee list / per-participant mic-active signal from the collaborator's recording app) — option (b) is almost certainly the practical path given the product's architecture, and should be confirmed with the collaborator.
- Ground truth for both test recordings was generated via an LLM transcription tool, producing segment-level JSON (timestamp, speaker, language, native-script text, English translation) — this format is reusable as an evaluation reference for future model comparisons.

---

## 7. Open Items / Next Steps

1. **Formal WER measurement** across a larger paired sample (8-10+ segments per language) to get a defensible average accuracy number.
2. **Wrap the routing logic into a reusable ASR module/class** so it can be dropped into the FastAPI worker described in the original architecture roadmap.
3. **Proceed to diarization** (e.g. pyannote) as the next roadmap stage, now that ASR has a validated, evidence-backed foundation under it.
