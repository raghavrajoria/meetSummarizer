"""GPU host measurement. Torch counters exclude CTranslate2/ONNX allocations."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import wave
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import psutil
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,nargs=3,required=True,help='5,30,60-minute 16kHz mono WAVs')
    parser.add_argument('--turns',type=Path,nargs=3,help='Optional stored turns; omit to benchmark real GPU diarization too')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('NOT VERIFIED: CUDA GPU required')
    if args.output.exists(): raise FileExistsError('Refusing to overwrite sizing evidence')
    if not os.environ.get('ASR_DEVICE','cuda:0').startswith('cuda'): raise ValueError('GPU probe requires CUDA ASR_DEVICE')
    from asr_service.model import GpuModel
    model=GpuModel();results=[]
    for audio,turn_file,expected in zip(args.inputs,args.turns or [None]*3,(300,1800,3600)):
        with wave.open(str(audio)) as wav: duration=wav.getnframes()/wav.getframerate()
        if abs(duration-expected)>1: raise ValueError(f'Expected {expected}s input')
        turns=None
        if turn_file:
            with turn_file.open(encoding='utf-8',newline='') as stream: turns=[{'start':float(r['start']),'end':float(r['end']),'speaker':r['speaker']} for r in csv.DictReader(stream)]
        stop=threading.Event();peak={'rss_bytes':0,'gpu_memory_used_mib':0,'nvidia_smi':'NOT VERIFIED'}
        def sample():
            while not stop.is_set():
                peak['rss_bytes']=max(peak['rss_bytes'],psutil.Process().memory_info().rss)
                try:
                    out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],timeout=5,text=True)
                    peak['gpu_memory_used_mib']=max(peak['gpu_memory_used_mib'],max(int(x) for x in out.splitlines()));peak['nvidia_smi']='sampled device-wide memory, includes other processes'
                except (OSError,subprocess.SubprocessError,ValueError): pass
                stop.wait(.5)
        torch.cuda.reset_peak_memory_stats()
        sampler=threading.Thread(target=sample,daemon=True);sampler.start();started=time.monotonic()
        try: rows=model.transcribe(audio,turns,lambda p,s:print(f'duration={duration} stage={s} progress={p}',flush=True))
        finally: stop.set();sampler.join(timeout=6)
        elapsed=time.monotonic()-started
        results.append({'duration_seconds':duration,'elapsed_seconds':elapsed,'real_time_factor':elapsed/duration,'segments':len(rows),'diarization':'supplied turns' if turn_file else 'real GPU pyannote','torch_peak_vram_bytes':torch.cuda.max_memory_allocated(),**peak,'cold_model_load':len(results)==0,'counter_limit':'Torch VRAM excludes CTranslate2 and ONNX; use sampled device-wide counter too.'})
        args.output.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('GPU sizing probe complete')


if __name__=='__main__': main()
