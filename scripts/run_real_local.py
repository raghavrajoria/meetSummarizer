"""Run local real services or ASR-only preflight with bounded memory supervision."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.generate_local_real_config import read_env


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('.env.local-real'))
    parser.add_argument('--preflight',type=Path,help='30-second WAV, ASR only; no Groq calls')
    args=parser.parse_args()
    environment=dict(os.environ);environment.update(read_env(args.config));environment['PYTHONUNBUFFERED']='1'
    environment.setdefault('HF_HUB_DISABLE_XET','1')
    os.environ.update(environment)
    from backend.config_guard import validate_app_configuration
    validate_app_configuration()
    root=Path('data/local-real');root.mkdir(parents=True,exist_ok=True)
    if args.preflight:
        from asr_service.model import GpuModel
        turns=json.loads(args.preflight.with_suffix('.turns.json').read_text(encoding='utf-8'))
        model=GpuModel();started=time.monotonic()
        def progress(p,s): print(f'preflight progress={p} stage={s}',flush=True)
        try:
            rows=model.transcribe(args.preflight,turns,progress)
            result={'status':'done','rows':rows,'metrics':model.last_metrics}
        except Exception as exc:
            result={'status':'failed','error':str(exc),'metrics':getattr(model,'last_metrics',{}),'elapsed_seconds':time.monotonic()-started}
        (root/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
        if result['status']!='done': raise SystemExit(1)
        return
    from asr_service.memory import check_memory
    check_memory()
    for command in ([sys.executable,'-m','backend.init_storage'],[sys.executable,'-m','alembic','upgrade','head']): subprocess.run(command,env=environment,check=True)
    children=[];handles=[]
    try:
        for role,command in [('asr',[sys.executable,'-u','-m','asr_service']),('api',[sys.executable,'-u','-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8000','--no-access-log']),('worker',[sys.executable,'-u','-m','backend.worker'])]:
            handle=(root/(role+'.log')).open('a',encoding='utf-8');handles.append(handle)
            children.append(subprocess.Popen(command,env=environment,stdout=handle,stderr=subprocess.STDOUT))
        print('Local real services running; progress logs in data/local-real. Ctrl+C stops only these services.',flush=True)
        while all(p.poll() is None for p in children):
            check_memory();time.sleep(1)
        raise RuntimeError('Local service exited; inspect private diagnostics')
    finally:
        import psutil
        for p in children:
            if p.poll() is None:
                try:
                    process=psutil.Process(p.pid)
                    for child in process.children(recursive=True):child.kill()
                    process.kill()
                except psutil.NoSuchProcess: pass
                p.wait(timeout=10)
        for handle in handles:handle.close()


if __name__=='__main__':main()
