"""Lazy model boundary. Importing the web service never imports torch/models."""
from pathlib import Path
import os,tempfile
from typing import Protocol

class Model(Protocol):
    model_version: str
    def transcribe(self,audio:Path,turns:list[dict]|None,progress) -> list[dict]: ...

class GpuModel:
    def __init__(self):
        self.model_version=os.environ.get("ASR_MODEL_VERSION","").strip()
        if not self.model_version:raise ValueError("ASR_MODEL_VERSION deployment identifier is required")
        self.engine=None

    def transcribe(self,audio,turns,progress):
        if os.environ.get('ASR_DEVICE') == 'cpu':
            return self._cpu_transcribe(audio, turns, progress)
        # NOT VERIFIED on real GPU/models. Only reached by the single host worker.
        if self.engine is None:
            progress(5,"model_loading")
            from indicmeet.asr import IndicMeetASR
            self.engine=IndicMeetASR()
        with tempfile.TemporaryDirectory(prefix="asr-gpu-") as temp:
            from backend.media import extract_audio
            wav=extract_audio(audio,Path(temp)/"audio.wav")
            if turns is None:
                progress(15,"diarization")
                from indicmeet.pipeline import _gpu_diarize
                turns=_gpu_diarize(wav,Path(temp)/"turns.csv")
            progress(30,"language_routed_asr")
            rows=self.engine.transcribe_turns(str(wav),turns,verbose=False)
            from indicmeet.contract import canonicalize
            rows=canonicalize(rows)
            for row in rows:row["asr"]["model_version"]=self.model_version
            return rows

    def _cpu_transcribe(self, audio, turns, progress):
        import json, subprocess, sys
        from .memory import check_memory, supervise
        if not turns: raise ValueError('CPU ASR requires request turns; pyannote is never run on CPU')
        check_memory()
        with tempfile.TemporaryDirectory(prefix='asr-cpu-') as temp:
            folder = Path(temp)
            request, output = folder/'turns.json', folder/'result.json'
            request.write_text(json.dumps(turns), encoding='utf-8')
            command = [os.environ.get('ASR_CPU_PYTHON', sys.executable), '-u', '-m', 'asr_service.cpu', '--audio', str(audio), '--turns', str(request), '--output', str(output)]
            process = subprocess.Popen(command)
            try:
                self.last_metrics = supervise(process, progress)
            except Exception as exc:
                self.last_metrics = getattr(exc,'metrics',{})
                raise
            if self.last_metrics['exit_code']: raise RuntimeError(f"CPU process failed exit={self.last_metrics['exit_code']}; inspect host log")
            body = json.loads(output.read_text(encoding='utf-8'))
            self.last_metrics.update(stage_timings=body['stage_timings'])
            from indicmeet.contract import canonicalize
            rows = canonicalize(body['rows'])
            for row in rows: row['asr']['model_version'] = self.model_version
            return rows
