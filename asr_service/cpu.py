"""Sequential CPU passes using the canonical routing and quality rules."""
import gc
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from indicmeet.asr import IndicMeetASR
from indicmeet.settings import get_settings
from .memory import check_memory, MemorySafetyError

logger = logging.getLogger('asr_service')


def chunk_key(chunk):
    return hashlib.sha256(chunk.numpy().tobytes()).hexdigest()


class CpuASR(IndicMeetASR):
    def __init__(self):
        check_memory()
        import numpy
        import torch
        torch.set_num_threads(int(os.environ.get('ASR_CPU_THREADS', '2')))
        self._numpy, self._torch = numpy, torch
        self.settings = get_settings()
        self.device = 'cpu'
        self.indic_decoder = self.settings.asr_indic_decoder
        self.max_indic_chunk_s = self.settings.asr_max_indic_chunk_seconds
        self.whisper_cache, self.mms_cache = {}, {}
        self.whisper = self
        self.indic_model = None
        self.stage_timings = {}

    def transcribe(self, audio, **kwargs):
        text, lang, conf = self.whisper_cache[hashlib.sha256(audio.tobytes()).hexdigest()]
        return [SimpleNamespace(text=text)], SimpleNamespace(language=lang, language_probability=conf)

    def _mms_lid(self, chunk):
        return self.mms_cache[chunk_key(chunk)]

    def transcribe_turns(self, wav_path, turns, **kwargs):
        import soundfile
        from indicmeet.splitting import split_samples
        audio, sr = soundfile.read(wav_path, dtype='float32')
        if sr != 16000 or audio.ndim != 1: raise ValueError('CPU input must be 16 kHz mono WAV')
        wav = self._torch.from_numpy(audio)
        chunks = []
        for turn in turns:
            first, last = max(0,int(turn['start']*sr)), min(len(wav), int(turn['end']*sr))
            for a, b in split_samples(wav[first:last].tolist(), sr, 25):
                chunks.append(({**turn, 'start': (first+a)/sr, 'end': (first+b)/sr}, wav[first+a:first+b]))
        def report(stage, index, started):
            check_memory()
            logger.info('stage=%s device=cpu segment=%d/%d elapsed=%.3f', stage, index, len(chunks), time.monotonic()-started)
        started = time.monotonic()
        check_memory()
        from faster_whisper import WhisperModel
        logger.info('stage=model_loading model=%s device=cpu', os.environ.get('WHISPER_MODEL','small'))
        model = WhisperModel(os.environ.get('WHISPER_MODEL', 'small'), device='cpu', compute_type='int8', cpu_threads=int(os.environ.get('ASR_CPU_THREADS','2')), num_workers=1)
        for i, (_, chunk) in enumerate(chunks):
            report('whisper', i+1, started)
            segs, info = model.transcribe(chunk.numpy(), language=None, beam_size=self.settings.whisper_beam_size, vad_filter=False, condition_on_previous_text=False)
            self.whisper_cache[chunk_key(chunk)] = (' '.join(s.text.strip() for s in segs), info.language, info.language_probability)
        del model
        gc.collect()
        self.stage_timings['whisper'] = time.monotonic()-started
        started = time.monotonic()
        check_memory()
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
        logger.info('stage=model_loading model=%s device=cpu', self.settings.mms_lid_model)
        self.lid_processor = AutoFeatureExtractor.from_pretrained(self.settings.mms_lid_model)
        check_memory()
        self.lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(self.settings.mms_lid_model).to('cpu').eval()
        for i, (_, chunk) in enumerate(chunks):
            report('mms_lid', i+1, started)
            self.mms_cache[chunk_key(chunk)] = IndicMeetASR._mms_lid(self, chunk)
        del self.lid_model, self.lid_processor
        gc.collect()
        self.stage_timings['mms_lid'] = time.monotonic()-started
        started = time.monotonic()
        check_memory()
        from transformers import AutoModel
        logger.info('stage=model_loading model=%s device=cpu', self.settings.indicconformer_model)
        self.indic_model = AutoModel.from_pretrained(self.settings.indicconformer_model, trust_remote_code=True).to('cpu').eval()
        rows = []
        try:
            for i, (turn, chunk) in enumerate(chunks):
                report('routing_indicconformer', i+1, started)
                try: row = self.transcribe_chunk(chunk)
                except MemorySafetyError: raise
                except Exception as exc: row = self._finalize('', 'und', 'error', len(chunk)/sr, [f'exception:{type(exc).__name__}'])
                rows.append({**row, **turn, 'idx': i})
        finally:
            self.indic_model = None
            gc.collect()
        self.stage_timings['indicconformer_routing'] = time.monotonic()-started
        return rows


def main():
    import argparse
    from backend.logging_config import configure_logging
    configure_logging()
    parser = argparse.ArgumentParser()
    for name in ('audio','turns','output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    engine = CpuASR()
    try:
        rows = engine.transcribe_turns(str(args.audio), json.loads(args.turns.read_text(encoding='utf-8')))
    except Exception:
        args.output.write_text(json.dumps({'stage_timings':engine.stage_timings,'status':'failed'},ensure_ascii=False),encoding='utf-8')
        raise
    args.output.write_text(json.dumps({'rows': rows, 'stage_timings': engine.stage_timings}, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__': main()
