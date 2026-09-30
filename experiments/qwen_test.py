import time
import wave
from pathlib import Path

from qwen_asr import Qwen3ASRModel


audio_path = Path("output/speech_test_10s.wav")
output_path = Path("output/transcript_qwen3_1.7b.txt")

model_name = "Qwen/Qwen3-ASR-1.7B"
device = "cpu"


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}m {seconds:02d}s"


with wave.open(str(audio_path), "rb") as wav:
    total_seconds = wav.getnframes() / wav.getframerate()


print()
print("=" * 65)
print("INDICMEET AI - QWEN3 ASR 1.7B TEST")
print("=" * 65)
print(f"Audio       : {audio_path}")
print(f"Duration    : {format_time(total_seconds)}")
print(f"Model       : {model_name}")
print(f"Device      : {device}")
print("=" * 65)
print()

print("[1/3] Loading Qwen3-ASR 1.7B...")
print("      Model size is approximately 4.7 GB.")
print("      CPU inference will be slow.")
print("      Do not start another Python/model process.")
print()

model_start = time.perf_counter()

model = Qwen3ASRModel.from_pretrained(
    model_name,
    dtype="float16",
    device_map="cpu",
)

model_elapsed = time.perf_counter() - model_start

print()
print(f"[DONE] Model loaded in {format_time(model_elapsed)}")
print()

print("[2/3] Starting transcription...")
print("      Language : AUTO")
print()

transcription_start = time.perf_counter()

results = model.transcribe(
    audio=str(audio_path),
    language=None,
)

transcription_elapsed = time.perf_counter() - transcription_start

print()
print("-" * 65)
print(f"Transcription finished in {format_time(transcription_elapsed)}")
print()

print("Transcription output:")
print("-" * 65)

transcript_lines = []

for result in results:
    text = result.text.strip()

    if not text:
        continue

    transcript_lines.append(text)

    print(text)

    if hasattr(result, "language"):
        print(f"Language: {result.language}")

    print()


print("[3/3] Saving transcript...")

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for line in transcript_lines:
        f.write(line + "\n")


print()
print("=" * 65)
print("QWEN3 ASR 1.7B COMPLETE")
print("=" * 65)
print(f"Audio duration    : {format_time(total_seconds)}")
print(f"Transcription time: {format_time(transcription_elapsed)}")
print(f"Output file       : {output_path}")
print("=" * 65)