"""indicmeet_asr_v2.py - reconstructed. Diarization-turn segmentation -> Whisper (LID) -> route to IndicConformer (CTC)."""
import os, re, time, json, math
import numpy as np
import pandas as pd
import torch
import torchaudio
from collections import Counter


class IndicMeetASR:
    SR = 16000
    EXPECTED_LANGUAGES = {"en", "hi", "ur", "bn", "gu", "mr", "ta", "te", "kn", "ml", "pa", "or", "as"}
    WHISPER_TO_INDIC = {"bn": "bn", "hi": "hi", "ur": "hi", "gu": "gu", "mr": "mr", "ta": "ta",
                        "te": "te", "kn": "kn", "ml": "ml", "pa": "pa", "or": "or", "as": "as"}
    MMS_TO_ISO1 = {"eng": "en", "hin": "hi", "urd": "ur", "ben": "bn", "guj": "gu", "mar": "mr",
                   "tam": "ta", "tel": "te", "kan": "kn", "mal": "ml", "pan": "pa", "ory": "or",
                   "ori": "or", "asm": "as"}

    def __init__(self, device="cuda:0", whisper_size="large-v3", indic_decoder="ctc", max_indic_chunk_s=8.0):
        from faster_whisper import WhisperModel
        from transformers import AutoModel, Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
        self.device = device
        self.indic_decoder = indic_decoder
        self.max_indic_chunk_s = max_indic_chunk_s
        t0 = time.time()

        print(f"[1/3] Loading Whisper {whisper_size} ...")
        dev_type, _, dev_idx = device.partition(":")
        self.whisper = WhisperModel(whisper_size, device=dev_type, device_index=int(dev_idx or 0),
                                    compute_type="float16")
        print(f"      done ({time.time()-t0:.0f}s elapsed)")

        print("[2/3] Loading IndicConformer 600M ...")
        self.indic_model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True).to(device)
        print(f"      done ({time.time()-t0:.0f}s elapsed)")

        print("[3/3] Loading MMS-LID ...")
        self.lid_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-256")
        self.lid_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-256").to(device)
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
        x = wav_chunk.detach().cpu().numpy().astype(np.float32)[: 30 * self.SR]
        if len(x) < self.SR:
            x = np.pad(x, (0, self.SR - len(x)))
        inputs = self.lid_processor(x, sampling_rate=self.SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.lid_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs))
        raw = self.lid_model.config.id2label[idx]
        return self.MMS_TO_ISO1.get(raw, raw), float(probs[idx])

    def _indic_decode(self, wav, lang_code):
        with torch.no_grad():
            out = self.indic_model(wav.unsqueeze(0).to(self.device), lang_code, self.indic_decoder)
        if isinstance(out, (list, tuple)):
            out = " ".join(str(o) for o in out)
        return str(out).strip()

    def _indic_transcribe(self, wav_chunk, lang_code):
        wav_chunk = wav_chunk.detach().cpu().float()
        dur = wav_chunk.shape[0] / self.SR
        if dur <= self.max_indic_chunk_s:
            return self._indic_decode(wav_chunk, lang_code)
        n = math.ceil(dur / self.max_indic_chunk_s)
        step = math.ceil(wav_chunk.shape[0] / n)
        texts = []
        for i in range(n):
            sub = wav_chunk[i * step:(i + 1) * step]
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
                "reasons": reasons, "duration": round(dur, 2), **extra}

    # ---------- main routing ----------
    def transcribe_chunk(self, wav_chunk):
        dur = wav_chunk.shape[0] / self.SR
        audio_np = wav_chunk.detach().cpu().numpy().astype(np.float32)
        segs, info = self.whisper.transcribe(audio_np, language=None, beam_size=5,
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
    def build_turns(csv_path, min_dur=0.3, max_gap=0.7, max_len=25.0):
        df = pd.read_csv(csv_path).sort_values("start").reset_index(drop=True)
        merged = []
        for _, r in df.iterrows():
            if merged and merged[-1]["speaker"] == r.speaker and r.start - merged[-1]["end"] <= max_gap:
                merged[-1]["end"] = float(r.end)
            else:
                merged.append({"start": float(r.start), "end": float(r.end), "speaker": r.speaker})
        turns = []
        for t in merged:
            d = t["end"] - t["start"]
            if d < min_dur:
                continue
            n = max(1, math.ceil(d / max_len))
            step = d / n
            for i in range(n):
                turns.append({"start": round(t["start"] + i * step, 3),
                              "end": round(t["start"] + (i + 1) * step, 3),
                              "speaker": t["speaker"]})
        return turns

    # ---------- full run with progress + checkpoints ----------
    def transcribe_turns(self, wav_path, turns, out_json=None, checkpoint_every=25, verbose=True):
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0)
        if sr != self.SR:
            wav = torchaudio.transforms.Resample(sr, self.SR)(wav)
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
                       "duration": round(chunk.shape[0] / self.SR, 2)}
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
