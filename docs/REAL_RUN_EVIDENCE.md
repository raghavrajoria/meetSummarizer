# CPU acceptance evidence — 2026-10-06

Native Windows30-second ASR-only preflight completed. Command: `.venv/Scripts/python.exe -u scripts/run_real_local.py --preflight data/local-real/agm-30.wav`. No Groq calls occurred. Five-minute API upload failed before model execution with HTTP422 because the CSV reader supplied a private idx field; remote-client payload filtering was fixed with a regression test. The owner then redirected work to the pre-computed demo; no five-minute ASR retry was run. No GPU performance, CUDA image, large-v3 accuracy or speaker diarization is proven.

First preflight with Docker running aborted at2.93GiB available after3.094s. Retried with Docker stopped. Native installed ML stack: torch2.14.0+cpu, torchaudio2.11.0+cpu, transformers4.57.6, faster-whisper1.2.1, onnxruntime1.20.1, psutil7.2.2, soundfile0.14.0. These differ from the GPU pins.

## Completed preflight (measured)

```json
{
  "status": "done",
  "rows": [
    {
      "segment_id": "seg_5de2e3e88e2442049fa3",
      "start": 0.0,
      "end": 1.1,
      "speaker": "VAD_SINGLE_SPEAKER",
      "speaker_name": null,
      "speaker_name_source": "none",
      "language": "en",
      "text_native": "That's the king is it",
      "text_roman": null,
      "text_english": null,
      "quality": "review",
      "reasons": [
        "very_short_clip"
      ],
      "asr": {
        "model_version": "cpu-small-int8-indic600m-mms256-20261006",
        "method": "whisper_english",
        "whisper_lang": "en",
        "whisper_lang_conf": 0.654,
        "mms_lang": "bod",
        "mms_lang_conf": 0.659,
        "confidence": null
      }
    },
    {
      "segment_id": "seg_534a5cb459bb03b89413",
      "start": 1.3,
      "end": 7.3,
      "speaker": "VAD_SINGLE_SPEAKER",
      "speaker_name": null,
      "speaker_name_source": "none",
      "language": "hi",
      "text_native": "आ तो भाई हमें मौका तो ये अपनी मीटिंग में जरूर विचार करिएगा और आशा करता हैो आप जरूर कदम उठाएंगे",
      "text_roman": null,
      "text_english": null,
      "quality": "accepted",
      "reasons": [],
      "asr": {
        "model_version": "cpu-small-int8-indic600m-mms256-20261006",
        "method": "indicconformer_hi",
        "whisper_lang": "hi",
        "whisper_lang_conf": 0.832,
        "mms_lang": "hi",
        "mms_lang_conf": 0.889,
        "confidence": null
      }
    },
    {
      "segment_id": "seg_b1dfc4d054f1ec75d874",
      "start": 7.4,
      "end": 23.7,
      "speaker": "VAD_SINGLE_SPEAKER",
      "speaker_name": null,
      "speaker_name_source": "none",
      "language": "hi",
      "text_native": "जय जय भारत समय देने के लिए एक बार से आप सभी के जितने भी क डायरेक्टर जितने भी अधिकारी के न जितने भी हमारे कर्मचारी काम कर रहे हैं उनके उनके परिवार के लिए भग से प्रार्थना करता हूं भगवान दो हजार बाईस त का जो फाइनेंशियल सबके लिए वेलथी हेल्थीस्पे और सेफ्टी के साथ बीती तो",
      "text_roman": null,
      "text_english": null,
      "quality": "accepted",
      "reasons": [],
      "asr": {
        "model_version": "cpu-small-int8-indic600m-mms256-20261006",
        "method": "indicconformer_hi",
        "whisper_lang": "hi",
        "whisper_lang_conf": 0.967,
        "mms_lang": "hi",
        "mms_lang_conf": 0.976,
        "confidence": null
      }
    },
    {
      "segment_id": "seg_7902b581375beaab3626",
      "start": 23.8,
      "end": 30.0,
      "speaker": "VAD_SINGLE_SPEAKER",
      "speaker_name": null,
      "speaker_name_source": "none",
      "language": "hi",
      "text_native": "जए उज़ा बरज़् मखिज़़ समजदने की लिए, सब बवाईज़ करश्ट ले दोना हाजोर की निद, नमसकर कता अजो दोना हाजोर की लिए.",
      "text_roman": null,
      "text_english": null,
      "quality": "review",
      "reasons": [
        "lid_disagreement whisper=hi mms=npi"
      ],
      "asr": {
        "model_version": "cpu-small-int8-indic600m-mms256-20261006",
        "method": "unrouted_whisper_hi",
        "whisper_lang": "hi",
        "whisper_lang_conf": 0.471,
        "mms_lang": "npi",
        "mms_lang_conf": 0.989,
        "confidence": null
      }
    }
  ],
  "metrics": {
    "peak_process_tree_rss_bytes": 4558311424,
    "elapsed_seconds": 457.64099999999985,
    "exit_code": 0,
    "stage_timings": {
      "whisper": 183.17100000000005,
      "mms_lid": 244.82899999999995,
      "indicconformer_routing": 24.233999999999924
    }
  },
  "soft_evaluation": {
    "label": "soft: LLM-generated reference",
    "WER": "NOT VERIFIED: no complete overlapping reference spans",
    "CER": "NOT VERIFIED"
  },
  "real_time_factor": 15.254699999999994,
  "peak_process_tree_ram_gib": 4.245258331298828
}
```

## Five-minute attempt

Probe0.062s; metadata0s; audio extraction0.141s; failed ASR submission0.344s; wall time6.265s. Complete-pipeline real-time factor, final transcript, summary, quality/language counts, Groq quality and WER/CER are NOT VERIFIED. An empty failed result is not a100% WER benchmark.
