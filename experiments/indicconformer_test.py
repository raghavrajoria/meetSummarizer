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
AUDIO_FILE = Path("output/speech_test_60s.wav")
MODEL_DIR = Path("models/indic-conformer-600m-multilingual")


def load_indic_model_class():
    print("[1/4] Downloading/loading IndicConformer repository...")

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


def load_audio():
    print(f"[2/4] Loading audio: {AUDIO_FILE}")

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_FILE}"
        )

    waveform, sample_rate = torchaudio.load(str(AUDIO_FILE))

    print(f"Original sample rate: {sample_rate}")
    print(f"Original shape: {tuple(waveform.shape)}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        print(f"Resampling {sample_rate} Hz -> 16000 Hz")

        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000,
        )

        waveform = resampler(waveform)

    waveform = waveform.squeeze(0).float()

    print(f"Final shape: {tuple(waveform.shape)}")
    print(f"Duration: {waveform.numel() / 16000:.2f}s")

    return waveform


def main():
    print("=" * 70)
    print("INDICCONFORMER 600M TEST")
    print("=" * 70)

    IndicASRModel = load_indic_model_class()

    print("[3/4] Loading IndicConformer model...")

    model = IndicASRModel.from_pretrained(MODEL_ID)

    print("Model loaded successfully.")

    waveform = load_audio()

    print("[4/4] Running CTC decoding...")
    print()

    languages = [
        ("hi", "Hindi"),
        ("mr", "Marathi"),
        ("bn", "Bengali"),
    ]

    results = {}

    for lang_code, lang_name in languages:
        print(f"--- {lang_name} ({lang_code}) ---")

        try:
            with torch.no_grad():
                text = model(
                    waveform.unsqueeze(0),
                    lang_code,
                    decoding="ctc",
                )

            results[lang_code] = text
            print(text)
            print()

        except Exception as e:
            results[lang_code] = f"ERROR: {e}"
            print(f"ERROR: {e}")
            print()

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for lang_code, text in results.items():
        print(f"{lang_code}: {text}")

    print("=" * 70)


if __name__ == "__main__":
    main()