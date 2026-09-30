import time
import wave
from pathlib import Path

from faster_whisper import WhisperModel


# ============================================================
# CONFIGURATION
# ============================================================
audio_path = Path("output/meeting_audio.wav")
output_path = Path("output/transcript_whisper_small_auto.txt")

model_name = "small"
device = "cpu"
compute_type = "int8"
language = None


# ============================================================
# HELPERS
# ============================================================

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}m {seconds:02d}s"


# ============================================================
# AUDIO INFO
# ============================================================

with wave.open(str(audio_path), "rb") as wav:
    total_seconds = wav.getnframes() / wav.getframerate()


print()
print("=" * 60)
print("INDICMEET AI - ASR TEST")
print("=" * 60)
print(f"Audio       : {audio_path}")
print(f"Duration    : {format_time(total_seconds)}")
print(f"Model       : faster-whisper / {model_name}")
print(f"Device      : {device}")
print(f"Compute     : {compute_type}")
print(f"Language    : {language}")
print("=" * 60)
print()


# ============================================================
# LOAD MODEL
# ============================================================

print("[1/3] Loading Whisper model...")
model_start = time.perf_counter()

model = WhisperModel(
    model_name,
    device=device,
    compute_type=compute_type,
)

model_elapsed = time.perf_counter() - model_start

print(f"[DONE] Model loaded in {format_time(model_elapsed)}")
print()


# ============================================================
# START TRANSCRIPTION
# ============================================================

print("[2/3] Starting transcription...")
print("      VAD filtering       : ENABLED")
print("      Previous text       : DISABLED")
print("      Beam size           : 5")
print()
print("Waiting for transcription segments...")
print("-" * 60)

transcription_start = time.perf_counter()

segments, info = model.transcribe(
    str(audio_path),
    language=None,
    beam_size=5,
    vad_filter=True,
    condition_on_previous_text=False,
)

print()
print(f"Detected language       : {info.language}")
print(f"Language probability    : {info.language_probability:.2f}")
print()
print("Transcription output:")
print("-" * 60)


# ============================================================
# PROCESS SEGMENTS
# ============================================================

transcript_lines = []
segment_count = 0
last_progress = -1

for segment in segments:

    text = segment.text.strip()

    if not text:
        continue

    segment_count += 1

    transcript_line = (
        f"[{segment.start:.2f}s -> {segment.end:.2f}s] "
        f"{text}"
    )

    transcript_lines.append(transcript_line)

    processed_seconds = min(segment.end, total_seconds)
    progress = (processed_seconds / total_seconds) * 100

    elapsed = time.perf_counter() - transcription_start

    print(
        f"[{format_time(processed_seconds)} / "
        f"{format_time(total_seconds)}] "
        f"{progress:6.1f}% | "
        f"Elapsed: {format_time(elapsed)}"
    )

    print(f"  {text}")
    print()


# ============================================================
# SAVE TRANSCRIPT
# ============================================================

print("-" * 60)
print("[3/3] Saving transcript...")

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Detected language: {info.language}\n")
    f.write(
        f"Language probability: "
        f"{info.language_probability:.2f}\n\n"
    )

    for line in transcript_lines:
        f.write(line + "\n")


# ============================================================
# COMPLETE
# ============================================================

total_elapsed = time.perf_counter() - transcription_start

print()
print("=" * 60)
print("ASR COMPLETE")
print("=" * 60)
print(f"Audio duration    : {format_time(total_seconds)}")
print(f"Segments          : {segment_count}")
print(f"Transcription time: {format_time(total_elapsed)}")
print(f"Output file       : {output_path}")
print("=" * 60)