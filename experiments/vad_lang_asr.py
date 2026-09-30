import importlib.util
import sys
import os
import ctypes
import gc
from pathlib import Path

import torch

# ============================================================
# Native DLL init
# Must happen after torch import, before torchaudio import
# ============================================================

FFMPEG_BIN = r"C:\ffmpeg\ffmpeg-9.0.2-full_build-shared\bin"
TORCHCODEC_DIR = str(Path(sys.prefix) / "Lib" / "site-packages" / "torchcodec")
TORCHCODEC_CORE_DLL = str(Path(TORCHCODEC_DIR) / "libtorchcodec_core9.dll")

os.add_dll_directory(FFMPEG_BIN)
os.add_dll_directory(TORCHCODEC_DIR)

ctypes.WinDLL(TORCHCODEC_CORE_DLL)

print(
    f"[init] libtorchcodec_core9.dll pre-loaded OK "
    f"(torch {torch.__version__})"
)

# ============================================================
# Imports
# ============================================================

import torchaudio
import soundfile as sf

from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel
from indic_transliteration import sanscript
from silero_vad import load_silero_vad, get_speech_timestamps


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

FULL_AUDIO_FILE = Path("output/meeting_audio.wav")
MODEL_DIR = Path("models/indic-conformer-600m-multilingual")

SAMPLE_RATE = 16000

# ============================================================
# VAD tuning
# ============================================================

# Maximum length of one speech segment.
# Long uninterrupted speech will be split around available silences.
VAD_MAX_SEGMENT_SECONDS = 15.0

# A silence this long is allowed to end the current speech segment.
# Shorter pauses remain inside the same segment.
VAD_MIN_SILENCE_MS = 300

# Ignore extremely short speech/noise events.
VAD_MIN_SPEECH_MS = 300

# Keep a small amount of audio around each VAD boundary.
# This helps avoid clipping the first/last phoneme of a word.
VAD_SPEECH_PAD_MS = 30

# ============================================================
# Language ID context
# ============================================================

# Whisper uses a wider context window only for language detection.
# Actual ASR still runs on the VAD segment itself.
LID_CONTEXT_SECONDS = 30

# IndicConformer does not have an English CTC head.
UNSUPPORTED_FOR_ASR = {"en"}

# ============================================================
# Transliteration mapping
# ============================================================

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


# ============================================================
# Transliteration helper
# ============================================================

def to_roman(text, lang_code):
    script = SCRIPT_MAP.get(lang_code)

    if script is None:
        return "(no transliteration mapping for this language)"

    try:
        return sanscript.transliterate(
            text,
            script,
            sanscript.ITRANS,
        )
    except Exception as e:
        return f"(transliteration failed: {e})"


# ============================================================
# Load IndicConformer repository
# ============================================================

def load_indic_model_class():
    print("[setup] Loading IndicConformer repository...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    repo_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )

    model_file = Path(repo_path) / "model_onnx.py"

    spec = importlib.util.spec_from_file_location(
        "indic_conformer_model",
        model_file,
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules["indic_conformer_model"] = module

    spec.loader.exec_module(module)

    return module.IndicASRModel


# ============================================================
# Load full audio
# ============================================================

def load_full_audio():
    print(f"[setup] Loading full audio: {FULL_AUDIO_FILE}")

    if not FULL_AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found: {FULL_AUDIO_FILE}"
        )

    waveform, sample_rate = torchaudio.load(
        str(FULL_AUDIO_FILE)
    )

    # Convert stereo/multi-channel audio to mono.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    # Resample if needed.
    if sample_rate != SAMPLE_RATE:
        print(
            f"[audio] Resampling "
            f"{sample_rate} Hz -> {SAMPLE_RATE} Hz"
        )

        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=SAMPLE_RATE,
        )

        waveform = resampler(waveform)

    waveform = waveform.squeeze(0).float()

    total_duration = waveform.numel() / SAMPLE_RATE

    print(f"Full audio duration: {total_duration:.2f}s")

    return waveform, total_duration


# ============================================================
# Run Silero VAD
# ============================================================

def run_vad(waveform):
    print("[setup] Loading Silero VAD (ONNX backend)...")

    vad_model = load_silero_vad(onnx=True)

    print("[VAD] Detecting speech segments...")

    raw_segments = get_speech_timestamps(
        waveform,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        max_speech_duration_s=VAD_MAX_SEGMENT_SECONDS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    )

    segments = [
        (
            seg["start"],
            seg["end"] - seg["start"],
        )
        for seg in raw_segments
    ]

    print(
        f"[VAD] Found {len(segments)} speech segments "
        f"(out of {waveform.numel() / SAMPLE_RATE:.1f}s total audio)"
    )

    total_speech = sum(
        duration for _, duration in segments
    )

    total_audio = waveform.numel() / SAMPLE_RATE

    print(
        f"[VAD] Total speech time: {total_speech:.1f}s "
        f"({100 * total_speech / total_audio:.1f}% of file)"
    )

    # Release VAD model before loading Whisper.
    del vad_model
    gc.collect()

    return segments


# ============================================================
# Slice waveform
# ============================================================

def slice_window(
    waveform,
    total_duration,
    start_sec,
    duration_sec,
):
    if start_sec >= total_duration:
        return None

    end_sec = min(
        start_sec + duration_sec,
        total_duration,
    )

    start_idx = int(start_sec * SAMPLE_RATE)
    end_idx = int(end_sec * SAMPLE_RATE)

    return waveform[start_idx:end_idx]


# ============================================================
# Automatic language detection using Whisper
# ============================================================

def detect_language(
    whisper_model,
    clip_waveform,
    tmp_path,
):
    # faster-whisper requires a file path.
    sf.write(
        tmp_path,
        clip_waveform.numpy(),
        SAMPLE_RATE,
    )

    segments, info = whisper_model.transcribe(
        tmp_path,
        language=None,                  # IMPORTANT: auto detect
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
    )

    # Consume generator so detection actually executes.
    list(segments)

    return (
        info.language,
        info.language_probability,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VAD-SEGMENTED LANGUAGE DETECTION + INDICCONFORMER ASR")
    print("=" * 70)

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    full_waveform, total_duration = load_full_audio()

    print()

    # --------------------------------------------------------
    # VAD
    # --------------------------------------------------------

    vad_segments = run_vad(full_waveform)

    print()

    if not vad_segments:
        print(
            "[abort] VAD found no speech segments. "
            "Check audio content."
        )
        return

    # --------------------------------------------------------
    # Temporary file used by Whisper LID
    # --------------------------------------------------------

    tmp_clip_path = "output/_tmp_lang_id_clip.wav"

    # --------------------------------------------------------
    # PHASE 1
    # Automatic language detection
    # --------------------------------------------------------

    print(
        "[phase 1] Loading faster-whisper "
        "'small' (language ID only)..."
    )

    whisper_model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )

    detections = []

    for segment_index, (
        start_sec,
        duration_sec,
    ) in enumerate(vad_segments):

        clip = slice_window(
            full_waveform,
            total_duration,
            start_sec,
            duration_sec,
        )

        if clip is None or clip.numel() == 0:
            continue

        # ----------------------------------------------------
        # Wider context for language detection
        # ----------------------------------------------------

        pad = (
            LID_CONTEXT_SECONDS - duration_sec
        ) / 2

        lid_start = max(
            0,
            start_sec - pad,
        )

        lid_clip = slice_window(
            full_waveform,
            total_duration,
            lid_start,
            LID_CONTEXT_SECONDS,
        )

        if (
            lid_clip is None
            or lid_clip.numel() == 0
        ):
            lid_clip = clip

        label = (
            f"{start_sec:07.2f}s-"
            f"{start_sec + duration_sec:07.2f}s"
        )

        lang_code, lang_prob = detect_language(
            whisper_model,
            lid_clip,
            tmp_clip_path,
        )

        print(
            f"[VAD {segment_index:04d}] "
            f"[{label}] "
            f"({duration_sec:.2f}s) "
            f"detected language: {lang_code} "
            f"(confidence: {lang_prob:.2f})"
        )

        detections.append(
            (
                label,
                clip,
                lang_code,
                lang_prob,
                start_sec,
                duration_sec,
            )
        )

    # --------------------------------------------------------
    # Cleanup Whisper
    # --------------------------------------------------------

    if os.path.exists(tmp_clip_path):
        os.remove(tmp_clip_path)

    del whisper_model
    gc.collect()

    print(
        "\n[phase 1 done] "
        "Whisper unloaded from memory.\n"
    )

    # --------------------------------------------------------
    # PHASE 2
    # IndicConformer ASR
    # --------------------------------------------------------

    IndicASRModel = load_indic_model_class()

    print(
        "[phase 2] Loading IndicConformer model..."
    )

    asr_model = IndicASRModel.from_pretrained(
        MODEL_ID
    )

    print("Model loaded.\n")

    results = []

    for (
        label,
        clip,
        lang_code,
        lang_prob,
        start_sec,
        duration_sec,
    ) in detections:

        print("-" * 70)

        print(
            f"SEGMENT {label} "
            f"(lang={lang_code}, "
            f"p={lang_prob:.2f}, "
            f"dur={duration_sec:.2f}s)"
        )

        # ----------------------------------------------------
        # Unsupported language
        # ----------------------------------------------------

        if lang_code in UNSUPPORTED_FOR_ASR:

            print(
                f"  [skip ASR] "
                f"'{lang_code}' has no "
                f"IndicConformer CTC head"
            )

            results.append(
                (
                    label,
                    lang_code,
                    lang_prob,
                    None,
                    "",
                    start_sec,
                    duration_sec,
                )
            )

            print()

            continue

        # ----------------------------------------------------
        # ASR
        # ----------------------------------------------------

        try:

            with torch.no_grad():

                text = asr_model(
                    clip.unsqueeze(0),
                    lang_code,
                    decoding="ctc",
                )

            roman = to_roman(
                text,
                lang_code,
            )

            word_count = len(
                text.split()
            )

            flag = ""

            if (
                word_count < 3
                and duration_sec > 3
            ):
                flag = " [SHORT/SUSPECT]"

            print(
                f"  Native : "
                f"{text}{flag}"
            )

            print(
                f"  Roman  : "
                f"{roman}"
            )

            results.append(
                (
                    label,
                    lang_code,
                    lang_prob,
                    text,
                    roman,
                    start_sec,
                    duration_sec,
                )
            )

        except Exception as e:

            print(
                f"  [ASR ERROR for "
                f"'{lang_code}']: {e}"
            )

            results.append(
                (
                    label,
                    lang_code,
                    lang_prob,
                    f"ERROR: {e}",
                    "",
                    start_sec,
                    duration_sec,
                )
            )

        print()

    # --------------------------------------------------------
    # Write transcript
    # --------------------------------------------------------

    out_path = Path(
        "output/full_transcript_vad.txt"
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:

        for (
            label,
            lang_code,
            lang_prob,
            text,
            roman,
            start_sec,
            duration_sec,
        ) in results:

            f.write(
                f"[{label}] "
                f"lang={lang_code} "
                f"(p={lang_prob:.2f}, "
                f"dur={duration_sec:.2f}s)\n"
            )

            f.write(
                f"  Native : {text}\n"
            )

            if roman:
                f.write(
                    f"  Roman  : {roman}\n"
                )

            f.write("\n")

    print(
        f"\nFull transcript written to: "
        f"{out_path}\n"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_speech = sum(
        item[6]
        for item in results
    )

    speech_segment_count = len(results)

    speech_percentage = (
        100 * total_speech / total_duration
        if total_duration > 0
        else 0
    )

    print(
        f"Full recording length : "
        f"{total_duration:.1f}s"
    )

    print(
        f"Speech segments found : "
        f"{speech_segment_count}"
    )

    print(
        f"Total speech duration : "
        f"{total_speech:.1f}s "
        f"({speech_percentage:.1f}% of file)"
    )

    print(
        f"Output file            : "
        f"{out_path}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()