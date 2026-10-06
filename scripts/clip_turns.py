"""Cut verified CSV turns or use explicitly labeled VAD single-speaker turns."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def crop_turns(rows, start, duration):
    result = []
    for row in rows:
        a, b = max(start, float(row['start'])), min(start+duration, float(row['end']))
        if b > a: result.append({'start': round(a-start, 6), 'end': round(b-start, 6), 'speaker': row['speaker']})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--duration', type=float, required=True)
    args = parser.parse_args()
    import wave
    with wave.open(str(args.audio)) as wav: duration = wav.getnframes()/wav.getframerate()
    if abs(duration-args.duration) > .1: raise ValueError('Source did not contain requested duration')
    if args.csv and args.csv.is_file():
        with args.csv.open(encoding='utf-8-sig', newline='') as stream: turns = crop_turns(csv.DictReader(stream), args.start, duration)
        source = 'stored_diarization_csv'
    else:
        print('AGM diarization CSV unavailable/unverified: using VAD-based single-speaker turns; not speaker diarization', flush=True)
        from asr_service.memory import check_memory
        check_memory()
        import torch
        import soundfile
        from silero_vad import load_silero_vad, get_speech_timestamps
        torch.set_num_threads(2)
        audio, sr = soundfile.read(args.audio, dtype='float32')
        model = load_silero_vad(onnx=True)
        spans = get_speech_timestamps(torch.from_numpy(audio), model, sampling_rate=sr, return_seconds=True)
        turns = [{'start': s['start'], 'end': min(duration, s['end']), 'speaker': 'VAD_SINGLE_SPEAKER'} for s in spans]
        source = 'vad_single_speaker_NOT_DIARIZATION'
    if not turns: raise ValueError('No speech turns')
    output = args.audio.with_suffix('.turns.csv')
    if output.exists(): raise FileExistsError(output)
    with output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['start','end','speaker']); writer.writeheader(); writer.writerows(turns)
    args.audio.with_suffix('.turns.json').write_text(json.dumps(turns), encoding='utf-8')
    args.audio.with_suffix('.clip.json').write_text(json.dumps({'source_start_seconds':args.start,'duration_seconds':duration,'speaker_source':source,'turns':len(turns)},indent=2),encoding='utf-8')
    print(f'clip_seconds={duration} turns={len(turns)} speaker_source={source}', flush=True)


if __name__ == '__main__': main()
