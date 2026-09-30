import importlib.util
import sys
import os
import ctypes
from pathlib import Path

import torch

# --- Native DLL init (must happen after torch import, before torchaudio import) ---
FFMPEG_BIN = r"C:\ffmpeg\ffmpeg-9.0.2-full_build-shared\bin"
TORCHCODEC_DIR = str(Path(sys.prefix) / "Lib" / "site-packages" / "torchcodec")
TORCHCODEC_CORE_DLL = str(Path(TORCHCODEC_DIR) / "libtorchcodec_core9.dll")

os.add_dll_directory(FFMPEG_BIN)
os.add_dll_directory(TORCHCODEC_DIR)
ctypes.WinDLL(TORCHCODEC_CORE_DLL)
print(f"[init] libtorchcodec_core9.dll pre-loaded OK (torch {torch.__version__})")
# --- end native DLL init ---

import torchaudio
import soundfile as sf
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel
from indic_transliteration import sanscript


MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
FULL_AUDIO_FILE = Path("output/hinglish_test_4to12min.wav")
MODEL_DIR = Path("models/indic-conformer-600m-multilingual")
SAMPLE_RATE = 16000

# Set to a fixed list of (start_seconds, duration_seconds) to spot-check
# specific windows, OR leave as None to auto-chunk the ENTIRE file below.
WINDOWS = None

# Used only when WINDOWS is None: chunk the whole file into consecutive
# windows of this length, covering 100% of the audio (no gaps).
AUTO_CHUNK_SECONDS = 10


def build_full_file_windows(total_duration, chunk_seconds):
    windows = []
    t = 0.0
    while t < total_duration:
        windows.append((t, min(chunk_seconds, total_duration - t)))
        t += chunk_seconds
    return windows

# Language ID is unreliable on very short clips. We detect language using a
# WIDER window centered on each segment (more context = better LID), then
# run ASR only on the original, shorter segment.
LID_CONTEXT_SECONDS = 30

# Whisper's detected language code -> whether IndicConformer likely supports it.
# IndicConformer covers the 22 scheduled Indian languages; we just pass the
# code straight through and let it fail loudly if unsupported.
UNSUPPORTED_FOR_ASR = {"en"}  # no English CTC head, confirmed earlier

# Whisper language code -> indic_transliteration script name, for
# deterministic (non-ML, non-guessing) native-script -> Roman conversion.
# This is a fixed character mapping, not a model, so it won't introduce
# new guesses on top of what the ASR already produced.
SCRIPT_MAP = {
    "bn": sanscript.BENGALI,
    "hi": sanscript.DEVANAGARI,
    "mr": sanscript.DEVANAGARI,
    "ne": sanscript.DEVANAGARI,
    "sa": sanscript.DEVANAGARI,
    "gu": sanscript.GUJARATI,
    "pa": sanscript.GURMUKHI,
    "ta": sanscript.TAMIL,
    "te": sanscript.TELUGU,
    "kn": sanscript.KANNADA,
    "ml": sanscript.MALAYALAM,
    "or": sanscript.ORIYA,
}


def to_roman(text, lang_code):
    script = SCRIPT_MAP.get(lang_code)
    if script is None:
        return "(no transliteration mapping for this language)"
    try:
        return sanscript.transliterate(text, script, sanscript.ITRANS)
    except Exception as e:
        return f"(transliteration failed: {e})"


def load_indic_model_class():
    print("[setup] Loading IndicConformer repository...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    repo_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    model_file = Path(repo_path) / "model_onnx.py"
    spec = importlib.util.spec_from_file_location("indic_conformer_model", model_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["indic_conformer_model"] = module
    spec.loader.exec_module(module)
    return module.IndicASRModel


def load_full_audio():
    print(f"[setup] Loading full audio: {FULL_AUDIO_FILE}")
    if not FULL_AUDIO_FILE.exists():
        raise FileNotFoundError(f"Audio file not found: {FULL_AUDIO_FILE}")

    waveform, sample_rate = torchaudio.load(str(FULL_AUDIO_FILE))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0).float()
    total_duration = waveform.numel() / SAMPLE_RATE
    print(f"Full audio duration: {total_duration:.2f}s")
    return waveform, total_duration


def slice_window(waveform, total_duration, start_sec, duration_sec):
    if start_sec >= total_duration:
        return None
    end_sec = min(start_sec + duration_sec, total_duration)
    start_idx = int(start_sec * SAMPLE_RATE)
    end_idx = int(end_sec * SAMPLE_RATE)
    return waveform[start_idx:end_idx]


def detect_language(whisper_model, clip_waveform, tmp_path):
    # faster-whisper needs a file path; write the clip out to a scratch wav
    sf.write(tmp_path, clip_waveform.numpy(), SAMPLE_RATE)
    segments, info = whisper_model.transcribe(
        tmp_path,
        language=None,          # force auto-detect, no hardcoding
        beam_size=1,            # we only want the LID result, not a good transcript
        vad_filter=False,
        condition_on_previous_text=False,
    )
    # Consume the generator minimally to make sure detection actually ran
    list(segments)
    return info.language, info.language_probability


def main():
    import gc

    print("=" * 70)
    print("AUTOMATIC LANGUAGE DETECTION + INDICCONFORMER ASR")
    print("=" * 70)

    full_waveform, total_duration = load_full_audio()
    print()

    active_windows = WINDOWS if WINDOWS is not None else build_full_file_windows(
        total_duration, AUTO_CHUNK_SECONDS
    )
    print(f"Processing {len(active_windows)} windows "
          f"({'manual list' if WINDOWS is not None else f'auto {AUTO_CHUNK_SECONDS}s chunks, full file'})\n")

    tmp_clip_path = "output/_tmp_lang_id_clip.wav"

    # ---- PHASE 1: language detection only. Use 'tiny', not 'small' -----
    # We only need the language guess, not transcript quality, so the
    # smallest Whisper model is enough and cuts memory pressure a lot.
    print("[phase 1] Loading faster-whisper 'small' (language ID only)...")
    whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

    detections = []  # (label, clip, lang_code, lang_prob)

    for start_sec, duration_sec in active_windows:
        clip = slice_window(full_waveform, total_duration, start_sec, duration_sec)
        if clip is None:
            print(f"[skip] {start_sec:.0f}s beyond audio length")
            continue

        # Build a wider context clip for language detection only, centered
        # on this segment where possible.
        pad = (LID_CONTEXT_SECONDS - duration_sec) / 2
        lid_start = max(0, start_sec - pad)
        lid_clip = slice_window(full_waveform, total_duration, lid_start, LID_CONTEXT_SECONDS)
        if lid_clip is None or lid_clip.numel() == 0:
            lid_clip = clip  # fallback if near the end of the file

        label = f"{start_sec:06.1f}s-{start_sec + duration_sec:06.1f}s"
        lang_code, lang_prob = detect_language(whisper_model, lid_clip, tmp_clip_path)
        print(f"[{label}] detected language: {lang_code}  (confidence: {lang_prob:.2f}, "
              f"using {LID_CONTEXT_SECONDS}s context)")
        detections.append((label, clip, lang_code, lang_prob))

    if os.path.exists(tmp_clip_path):
        os.remove(tmp_clip_path)

    # Fully release Whisper before loading IndicConformer. Do not keep
    # both models resident at once on a 16GB machine.
    del whisper_model
    gc.collect()
    print("\n[phase 1 done] Whisper unloaded from memory.\n")

    # ---- PHASE 2: ASR only, using each window's detected language ------
    IndicASRModel = load_indic_model_class()
    print("[phase 2] Loading IndicConformer model...")
    asr_model = IndicASRModel.from_pretrained(MODEL_ID)
    print("Model loaded.\n")

    results = []

    for label, clip, lang_code, lang_prob in detections:
        print("-" * 70)
        print(f"WINDOW {label}  (lang={lang_code}, p={lang_prob:.2f})")

        if lang_code in UNSUPPORTED_FOR_ASR:
            print(f"  [skip ASR] '{lang_code}' has no IndicConformer CTC head")
            results.append((label, lang_code, lang_prob, None))
            continue

        try:
            with torch.no_grad():
                text = asr_model(clip.unsqueeze(0), lang_code, decoding="ctc")
            roman = to_roman(text, lang_code)
            print(f"  Native : {text}")
            print(f"  Roman  : {roman}")
            results.append((label, lang_code, lang_prob, text, roman))
        except Exception as e:
            print(f"  [ASR ERROR for '{lang_code}']: {e}")
            results.append((label, lang_code, lang_prob, f"ERROR: {e}", ""))

        print()

    out_path = Path("output/full_transcript.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in results:
            if len(item) == 5:
                label, lang_code, lang_prob, text, roman = item
            else:
                label, lang_code, lang_prob, text = item
                roman = ""
            f.write(f"[{label}] lang={lang_code} (p={lang_prob:.2f})\n")
            f.write(f"  Native : {text}\n")
            if roman:
                f.write(f"  Roman  : {roman}\n")
            f.write("\n")
    print(f"\nFull transcript written to: {out_path}\n")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_processed = sum(min(d, total_duration - s) for s, d in active_windows if s < total_duration)
    print(f"Full recording length : {total_duration:.1f}s")
    print(f"Total processed here  : {total_processed:.1f}s "
          f"({100 * total_processed / total_duration:.1f}% of file, non-contiguous)")
    print()
    for item in results:
        if len(item) == 5:
            label, lang_code, lang_prob, text, roman = item
        else:
            label, lang_code, lang_prob, text = item
            roman = ""
        print(f"[{label}] lang={lang_code} (p={lang_prob:.2f})")
        print(f"  Native : {text}")
        if roman:
            print(f"  Roman  : {roman}")
    print("=" * 70)


if __name__ == "__main__":
    main()