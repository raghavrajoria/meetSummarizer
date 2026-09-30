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
from huggingface_hub import snapshot_download


MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
FULL_AUDIO_FILE = Path("output/meeting_audio.wav")
MODEL_DIR = Path("models/indic-conformer-600m-multilingual")

# Edit these to control what gets tested.
# Each entry: (start_seconds, duration_seconds)
WINDOWS = [
    (50, 10),
    (110, 10),
    (170, 10),
    (230, 10),
    (290, 10),
    (350, 10),
]

LANGUAGES = [
    ("hi", "Hindi"),
    ("mr", "Marathi"),
    ("bn", "Bengali"),
]

SAMPLE_RATE = 16000


def load_indic_model_class():
    print("[1/3] Downloading/loading IndicConformer repository...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    repo_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )

    model_file = Path(repo_path) / "model_onnx.py"
    print(f"Model code: {model_file}")

    spec = importlib.util.spec_from_file_location(
        "indic_conformer_model",
        model_file,
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules["indic_conformer_model"] = module
    spec.loader.exec_module(module)

    return module.IndicASRModel


def load_full_audio():
    print(f"[2/3] Loading full audio: {FULL_AUDIO_FILE}")

    if not FULL_AUDIO_FILE.exists():
        raise FileNotFoundError(f"Audio file not found: {FULL_AUDIO_FILE}")

    waveform, sample_rate = torchaudio.load(str(FULL_AUDIO_FILE))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != SAMPLE_RATE:
        print(f"Resampling {sample_rate} Hz -> {SAMPLE_RATE} Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=SAMPLE_RATE,
        )
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


def main():
    print("=" * 70)
    print("INDICCONFORMER 600M - MULTI-WINDOW TEST")
    print("=" * 70)

    IndicASRModel = load_indic_model_class()

    print("[3/3] Loading IndicConformer model...")
    model = IndicASRModel.from_pretrained(MODEL_ID)
    print("Model loaded successfully.")
    print()

    full_waveform, total_duration = load_full_audio()
    print()

    all_results = []

    for start_sec, duration_sec in WINDOWS:
        clip = slice_window(full_waveform, total_duration, start_sec, duration_sec)

        if clip is None:
            print(f"[skip] Window {start_sec}s-{start_sec + duration_sec}s "
                  f"is beyond audio length ({total_duration:.1f}s)")
            continue

        clip_duration = clip.numel() / SAMPLE_RATE
        label = f"{start_sec:03d}s-{int(start_sec + clip_duration):03d}s"

        print("-" * 70)
        print(f"WINDOW {label} (actual clip length: {clip_duration:.2f}s)")
        print("-" * 70)

        window_result = {"window": label}

        for lang_code, lang_name in LANGUAGES:
            try:
                with torch.no_grad():
                    text = model(
                        clip.unsqueeze(0),
                        lang_code,
                        decoding="ctc",
                    )
                window_result[lang_code] = text
                print(f"  [{lang_code}] {text}")
            except Exception as e:
                window_result[lang_code] = f"ERROR: {e}"
                print(f"  [{lang_code}] ERROR: {e}")

        print()
        all_results.append(window_result)

    print("=" * 70)
    print("SUMMARY - ALL WINDOWS")
    print("=" * 70)

    for r in all_results:
        print(f"\n[{r['window']}]")
        for lang_code, _ in LANGUAGES:
            print(f"  {lang_code}: {r.get(lang_code, 'N/A')}")

    print("=" * 70)


if __name__ == "__main__":
    main()