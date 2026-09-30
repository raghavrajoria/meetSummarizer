"""
indicmeet_asr.py — Language-routed ASR pipeline for IndicMeet AI.

Validated pipeline: VAD -> Whisper (transcription + LID) -> script-verification
-> route to IndicConformer for Indic languages, keep Whisper output for English.

Dependencies (install once, e.g. in your FastAPI worker's requirements):
    pip install faster-whisper transformers torch torchaudio onnxruntime-gpu
    pip install git+https://github.com/snakers4/silero-vad  (or torch.hub as below)
"""

import re
import time
import torch
import torchaudio
import numpy as np
from faster_whisper import WhisperModel
from transformers import AutoModel, Wav2Vec2ForSequenceClassification, AutoFeatureExtractor


class IndicMeetASR:
    def __init__(self, device="cuda", whisper_size="small", hf_token=None):
        self.device = device

        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)

        print("Loading Whisper...")
        self.whisper = WhisperModel(whisper_size, device=device, compute_type="float16")

        print("Loading IndicConformer...")
        self.indic_model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True
        ).to(device)

        print("Loading MMS-LID...")
        self.lid_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-256")
        self.lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/mms-lid-256"
        ).to(device)

        print("Loading Silero VAD...")
        self.vad_model, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False
        )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = vad_utils

        self.LID_TO_INDICCONFORMER = {
                "bn": "bn",
                "hi": "hi",
                "ur": "hi",
                "gu": "gu",
                "mr": "mr",
                "ta": "ta",
                "te": "te",
                "kn": "kn",
                "ml": "ml",
                "pa": "pa",
                "or": "or",
                "as": "as"
        }
        self.SECONDARY_LANGS = set(self.LID_TO_INDICCONFORMER.keys())
        self.SECONDARY_MIN_CONF = 0.90
        print("IndicMeetASR ready.")

    # ---------- VAD ----------

    def get_segments(self, wav_path, min_speech_ms=500, max_gap_s=1.0, min_duration_s=1.5):
        wav = self.read_audio(wav_path, sampling_rate=16000)
        raw = self.get_speech_timestamps(
            wav, self.vad_model, sampling_rate=16000,
            min_speech_duration_ms=min_speech_ms, return_seconds=True
        )
        return self._merge_segments(raw, max_gap_s, min_duration_s)

    @staticmethod
    def _merge_segments(segments, max_gap, min_duration):
        if not segments:
            return []
        merged, current = [], segments[0].copy()
        for seg in segments[1:]:
            if seg['start'] - current['end'] <= max_gap:
                current['end'] = seg['end']
            else:
                merged.append(current)
                current = seg.copy()
        merged.append(current)
        return [s for s in merged if (s['end'] - s['start']) >= min_duration]

    # ---------- LID helpers ----------

    @staticmethod
    def _is_latin_script(text, threshold=0.7):
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        latin = sum(1 for c in letters if ord(c) < 0x250)
        return (latin / len(letters)) >= threshold

    def _mms_lid(self, wav_chunk):
        inputs = self.lid_processor(wav_chunk.numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.lid_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = torch.argmax(probs).item()
        lang = self.lid_model.config.id2label[idx]
        return lang, probs[idx].item()

    def _transcribe_indic_chunked(self, wav_chunk, lang_code, max_chunk_s=8.0, sr=16000):
        duration = wav_chunk.shape[0] / sr
        if duration <= max_chunk_s:
            return self.indic_model(wav_chunk.unsqueeze(0), lang_code, "rnnt")
        n_chunks = int(duration // max_chunk_s) + 1
        chunk_len = int(len(wav_chunk) / n_chunks)
        texts = []
        for i in range(n_chunks):
            sub = wav_chunk[i * chunk_len: (i + 1) * chunk_len]
            if len(sub) < sr * 0.5:
                continue
            texts.append(self.indic_model(sub.unsqueeze(0), lang_code, "rnnt"))
        return " ".join(texts)

    # ---------- Main routing ----------

    def transcribe_chunk(self, wav_chunk):
        """Returns dict: {text, lang, confidence, method}"""
        audio_np = wav_chunk.numpy().astype(np.float32)
        w_segments, w_info = self.whisper.transcribe(
            audio_np, language=None, beam_size=5, vad_filter=False
        )
        whisper_text = " ".join(seg.text for seg in w_segments)
        w_lang, w_conf = w_info.language, w_info.language_probability

        if self._is_latin_script(whisper_text):
            return {"text": whisper_text, "lang": "eng", "confidence": w_conf,
                    "method": "whisper_script_verified"}

        if w_lang in ("hi", "ur"):
            text = self._transcribe_indic_chunked(wav_chunk.to(self.device), "hi")
            return {"text": text, "lang": "hin", "confidence": w_conf,
                    "method": "indicconformer_via_whisper_lid"}

        if w_lang in self.LID_TO_INDICCONFORMER:
            mms_lang, mms_conf = self._mms_lid(wav_chunk)
            if mms_lang == w_lang or mms_conf > self.SECONDARY_MIN_CONF:
                indic_lang = self.LID_TO_INDICCONFORMER[w_lang]
                text = self._transcribe_indic_chunked(wav_chunk.to(self.device), indic_lang)
                return {"text": text, "lang": w_lang, "confidence": w_conf,
                        "method": f"indicconformer_{w_lang}"}

        return {"text": whisper_text, "lang": w_lang, "confidence": w_conf,
                "method": f"unrouted_whisper_guess_{w_lang}"}

    # ---------- Full pipeline ----------

    def transcribe_file(self, wav_path, verbose=True):
        full_wav, sr = torchaudio.load(wav_path)
        full_wav = torch.mean(full_wav, dim=0)
        if sr != 16000:
            full_wav = torchaudio.transforms.Resample(sr, 16000)(full_wav)

        segments = self.get_segments(wav_path)
        results = []
        t0 = time.time()

        for i, seg in enumerate(segments):
            chunk = full_wav[int(seg['start'] * 16000): int(seg['end'] * 16000)]
            result = self.transcribe_chunk(chunk)
            result.update({"idx": i, "start": seg['start'], "end": seg['end']})
            results.append(result)
            if verbose:
                print(f"[{i}] {seg['start']:.1f}-{seg['end']:.1f}s | "
                      f"{result['lang']} ({result['confidence']:.1%}) | {result['method']}")
                print(f"     -> {result['text']}\n")

        if verbose:
            print(f"Total: {time.time()-t0:.1f}s for {len(segments)} segments")
        return results


# ---------- Usage ----------
# asr = IndicMeetASR(hf_token="hf_xxx")
# results = asr.transcribe_file("/content/meeting_audio.wav")