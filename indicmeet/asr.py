"""indicmeet_asr_v2.py - reconstructed. Diarization-turn segmentation -> Whisper (LID) -> route to IndicConformer (CTC)."""
import os, re, time, json, math, csv
from collections import Counter
from .settings import get_settings


class IndicMeetASR:
    SR = 16000
    EXPECTED_LANGUAGES = {"en", "hi", "ur", "bn", "gu", "mr", "ta", "te", "kn", "ml", "pa", "or", "as"}
    WHISPER_TO_INDIC = {"bn": "bn", "hi": "hi", "ur": "hi", "gu": "gu", "mr": "mr", "ta": "ta",
                        "te": "te", "kn": "kn", "ml": "ml", "pa": "pa", "or": "or", "as": "as"}
    MMS_TO_ISO1 = {"eng": "en", "hin": "hi", "urd": "ur", "ben": "bn", "guj": "gu", "mar": "mr",
                   "tam": "ta", "tel": "te", "kan": "kn", "mal": "ml", "pan": "pa", "ory": "or",
                   "ori": "or", "asm": "as"}

    def __init__(self, device=None, whisper_size=None, indic_decoder=None, max_indic_chunk_s=None):
        settings = get_settings()
        if not (device or settings.asr_device).startswith("cuda"):
            raise RuntimeError("Large ASR requires the teammate GPU host; use ASR_MODE=remote or import")
        import numpy as np
        import pandas as pd
        import torch
        import torchaudio

        settings = get_settings()
        device = device or settings.asr_device
        whisper_size = whisper_size or settings.whisper_model
        indic_decoder = indic_decoder or settings.asr_indic_decoder
        max_indic_chunk_s = max_indic_chunk_s or settings.asr_max_indic_chunk_seconds
        self._numpy, self._torch, self._torchaudio = np, torch, torchaudio
        self.settings = settings
        from faster_whisper import WhisperModel
        from transformers import AutoModel, Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
        self.device = device
        self.indic_decoder = indic_decoder
        self.max_indic_chunk_s = max_indic_chunk_s
        t0 = time.time()

        print(f"[1/3] Loading Whisper {whisper_size} ...")
        dev_type, _, dev_idx = device.partition(":")
        self.whisper = WhisperModel(whisper_size, device=dev_type, device_index=int(dev_idx or 0),
                                    compute_type=settings.whisper_compute_type)
        print(f"      done ({time.time()-t0:.0f}s elapsed)")

        print("[2/3] Loading IndicConformer 600M ...")
        self.indic_model = AutoModel.from_pretrained(
            settings.indicconformer_model, trust_remote_code=True).to(device)
        print(f"      done ({time.time()-t0:.0f}s elapsed)")

        print("[3/3] Loading MMS-LID ...")
        self.lid_processor = AutoFeatureExtractor.from_pretrained(settings.mms_lid_model)
        self.lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(settings.mms_lid_model).to(device)
        self.lid_model.eval()
        print(f"IndicMeetASR ready in {time.time()-t0:.0f}s")

    # ---------- helpers ----------
    @staticmethod
    def _is_latin_script(text, threshold=0.7):
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        return sum(1 for c in letters if ord(c) < 0x250) / len(letters) >= threshold

    @staticmethod
    def _norm(lang):
        return "hi" if lang == "ur" else lang

    def _mms_lid(self, wav_chunk):
        x = wav_chunk.detach().cpu().numpy().astype(self._numpy.float32)[: 30 * self.SR]
        if len(x) < self.SR:
            x = self._numpy.pad(x, (0, self.SR - len(x)))
        inputs = self.lid_processor(x, sampling_rate=self.SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self.lid_model(**inputs).logits
        probs = self._torch.softmax(logits, dim=-1)[0]
        idx = int(self._torch.argmax(probs))
        raw = self.lid_model.config.id2label[idx]
        return self.MMS_TO_ISO1.get(raw, raw), float(probs[idx])

    def _indic_decode(self, wav, lang_code):
        with self._torch.no_grad():
            out = self.indic_model(wav.unsqueeze(0).to(self.device), lang_code, self.indic_decoder)
        if isinstance(out, (list, tuple)):
            out = " ".join(str(o) for o in out)
        return str(out).strip()

    def _indic_transcribe(self, wav_chunk, lang_code):
        wav_chunk = wav_chunk.detach().cpu().float()
        dur = wav_chunk.shape[0] / self.SR
        if dur <= self.max_indic_chunk_s:
            return self._indic_decode(wav_chunk, lang_code)
        from .splitting import split_samples
        spans = split_samples(wav_chunk.tolist(), self.SR, self.max_indic_chunk_s)
        texts = []
        for start, end in spans:
            sub = wav_chunk[start:end]
            if sub.shape[0] < int(0.3 * self.SR):
                continue
            t = self._indic_decode(sub, lang_code)
            if t:
                texts.append(t)
        return " ".join(texts)

    # ---------- quality gate (applied to every return path) ----------
    def _finalize(self, text, lang, method, dur, reasons, **extra):
        reasons = list(reasons)
        reject = False
        text = (text or "").strip()
        if not re.sub(r"[\W_]+", "", text):
            reasons.append("empty_or_punctuation_only")
            reject = True
        toks = [t.strip(".,!?;:।\"'()-") for t in text.lower().split()]
        toks = [t for t in toks if t]
        if len(toks) >= 6:
            top = Counter(toks).most_common(1)[0][1]
            if top / len(toks) > 0.5 or (len(toks) >= 8 and len(set(toks)) / len(toks) < 0.35):
                reasons.append("repetitive_output")
                reject = True
        if lang not in self.EXPECTED_LANGUAGES:
            reasons.append(f"language_outside_allowlist:{lang}")
        quality = "rejected" if reject else ("review" if reasons else "accepted")
        return {"text": text, "lang": lang, "method": method, "quality": quality,
                "reasons": reasons, "duration": round(dur, 2), "confidence": None, **extra}

    # ---------- main routing ----------
    def transcribe_chunk(self, wav_chunk):
        dur = wav_chunk.shape[0] / self.SR
        audio_np = wav_chunk.detach().cpu().numpy().astype(self._numpy.float32)
        segs, info = self.whisper.transcribe(audio_np, language=None, beam_size=self.settings.whisper_beam_size,
                                             vad_filter=False, condition_on_previous_text=False)
        whisper_text = " ".join(s.text.strip() for s in segs).strip()
        w_lang, w_conf = info.language, float(info.language_probability)
        mms_lang, mms_conf = self._mms_lid(wav_chunk)
        agree = self._norm(mms_lang) == self._norm(w_lang)
        latin = self._is_latin_script(whisper_text)
        base = dict(whisper_lang=w_lang, whisper_lang_conf=round(w_conf, 3),
                    mms_lang=mms_lang, mms_lang_conf=round(mms_conf, 3))

        # 1) Latin output + Whisper says English -> English (flag if weak/short)
        if latin and w_lang == "en":
            r = []
            if w_conf < 0.5: r.append("low_whisper_lid_conf")
            if dur < 1.5: r.append("very_short_clip")
            return self._finalize(whisper_text, "en", "whisper_english", dur, r, **base)

        # 2) Indic language per Whisper: route only if MMS-LID confirms (or is unsure and Whisper is confident)
        indic_code = self.WHISPER_TO_INDIC.get(w_lang)
        if indic_code:
            routable = agree or (mms_conf < 0.5 and w_conf >= 0.7 and not latin)
            if routable:
                text = self._indic_transcribe(wav_chunk, indic_code)
                r = []
                if not agree: r.append("mms_not_confirming")
                if w_conf < 0.5: r.append("low_whisper_lid_conf")
                m = f"indicconformer_{indic_code}" + ("_latin_override" if latin else "")
                return self._finalize(text, indic_code, m, dur, r, **base)

        # 3) Latin output but language claim unresolved -> keep Whisper text, force review
        if latin:
            lang = w_lang if w_lang in self.EXPECTED_LANGUAGES else "en"
            return self._finalize(whisper_text, lang, "whisper_latin_unresolved", dur,
                                  ["latin_script_language_mismatch"], **base)

        # 4) Everything else: Whisper text, unrouted
        r = [f"lid_disagreement whisper={w_lang} mms={mms_lang}"] if indic_code else []
        return self._finalize(whisper_text, w_lang, f"unrouted_whisper_{w_lang}", dur, r, **base)

    # ---------- turns from diarization CSV ----------
    @staticmethod
    def build_turns(csv_path, min_dur=0.3, max_gap=0.7, max_len=25.0, audio_duration=None):
        with open(csv_path, newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"start", "end", "speaker"}.issubset(reader.fieldnames):
                raise ValueError("Diarization CSV must contain start, end, and speaker columns")
            rows = [{"start": float(row["start"]), "end": float(row["end"]),
                     "speaker": row["speaker"]} for row in reader]
        rows.sort(key=lambda row: row["start"])
        merged = []
        for row in rows:
            if merged and merged[-1]["speaker"] == row["speaker"] and row["start"] - merged[-1]["end"] <= max_gap:
                merged[-1]["end"] = max(merged[-1]["end"], row["end"])
            else:
                merged.append(row)
        turns = []
        for t in merged:
            start = max(0.0, t["start"])
            end = t["end"]
            if audio_duration is not None:
                end = min(end, max(0.0, float(audio_duration)))
                start = min(start, max(0.0, float(audio_duration)))
            d = end - start
            if d < min_dur:
                continue
            n = 1
            step = d / n
            for i in range(n):
                turn_start = round(start + i * step, 3)
                turn_end = round(start + (i + 1) * step, 3)
                if audio_duration is not None:
                    turn_start = min(turn_start, float(audio_duration))
                    turn_end = min(turn_end, float(audio_duration))
                turns.append({"start": turn_start,
                              "end": turn_end,
                              "speaker": t["speaker"]})
        return turns

    # ---------- full run with progress + checkpoints ----------
    def transcribe_turns(self, wav_path, turns, out_json=None, checkpoint_every=25, verbose=True):
        wav, sr = self._torchaudio.load(wav_path)
        wav = wav.mean(dim=0)
        if sr != self.SR:
            wav = self._torchaudio.transforms.Resample(sr, self.SR)(wav)
        audio_duration = wav.shape[0] / self.SR
        bounded_turns = []
        for turn in turns:
            start = min(max(0.0, float(turn["start"])), audio_duration)
            end = min(max(0.0, float(turn["end"])), audio_duration)
            if end > start:
                bounded_turns.append({**turn, "start": start, "end": end})
        from .splitting import split_samples
        turns = []
        for turn in bounded_turns:
            offset = int(turn["start"] * self.SR)
            samples = wav[offset:int(turn["end"] * self.SR)].tolist()
            for first, last in split_samples(samples, self.SR, 25):
                turns.append({**turn, "start": (offset + first) / self.SR, "end": (offset + last) / self.SR})
        total_audio = sum(t["end"] - t["start"] for t in turns)
        print(f"Turns: {len(turns)} | speech to process: {total_audio/60:.1f} min | device: {self.device}")
        results, done_audio, t0 = [], 0.0, time.time()
        for i, t in enumerate(turns):
            chunk = wav[int(t["start"] * self.SR): int(t["end"] * self.SR)]
            try:
                res = self.transcribe_chunk(chunk)
            except Exception as e:
                res = {"text": "", "lang": "unknown", "method": "error", "quality": "rejected",
                       "reasons": [f"exception:{type(e).__name__}:{str(e)[:120]}"],
                       "duration": round(chunk.shape[0] / self.SR, 2), "confidence": None}
            res.update({"idx": i, "start": t["start"], "end": t["end"], "speaker": t["speaker"]})
            results.append(res)
            done_audio += t["end"] - t["start"]
            if verbose:
                print(f"[{i+1}/{len(turns)}] {t['start']:.1f}-{t['end']:.1f}s {t['speaker']} | "
                      f"{res['lang']} | {res['quality']} | {res['method']} | {time.time()-t0:.0f}s elapsed")
                print(f"     -> {res['text'][:200]}")
            if out_json and (i + 1) % checkpoint_every == 0:
                json.dump(results, open(out_json, "w"), ensure_ascii=False, indent=1)
        if out_json:
            json.dump(results, open(out_json, "w"), ensure_ascii=False, indent=1)
        print("\nDONE in %.0fs" % (time.time() - t0))
        print("quality:", dict(Counter(r["quality"] for r in results)))
        print("lang   :", dict(Counter(r["lang"] for r in results)))
        return results
