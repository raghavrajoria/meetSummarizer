"""Fresh local clone/new environments/full Compose smoke and browser proof."""
import os,pathlib,subprocess,sys,tempfile,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    folder=pathlib.Path(tempfile.mkdtemp(prefix="indicmeet-release-proof-"));clone=folder/"repo"
    logdir=folder/"evidence";logdir.mkdir()
    environment=dict(os.environ)
    environment.update(DEMO_PRECOMPUTED_ONLY="false",DEMO="true",APP_ENV="demo",DEMO_MODE="false",ASR_MODE="import",GROQ_API_KEY="",HF_TOKEN="",HUGGINGFACE_TOKEN="",API_TOKENS="",AUTH_USERS_JSON="",MEDIA_SIGNING_SECRET="local-proof-signing-secret-at-least-32-characters",API_PORT="8000",WORKER_PORT="8081",FRONTEND_PORT="8080",E2E_BASE_URL="http://127.0.0.1:8080")
    for name in ("DATABASE_URL","REDIS_URL","S3_ENDPOINT_URL","S3_ACCESS_KEY","S3_SECRET_KEY","STORAGE_BACKEND"):
        environment.pop(name,None)
    environment["INDICMEET_DATA_DIR"]=str(folder/"runtime")
    environment["INDICMEET_LLM_CACHE"]=str(folder/"llm-cache")
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if not branch: raise RuntimeError('Select a branch before fresh-clone verification')
    number=0
    def run(args,cwd,env=None):
        nonlocal number
        number+=1;log=logdir/f"{number:02d}.log";started=time.monotonic()
        print("COMMAND",subprocess.list2cmdline([str(a) for a in args]),flush=True)
        with log.open("w",encoding="utf-8") as output:
            result=subprocess.run([str(a) for a in args],cwd=cwd,env=env or environment,stdout=output,stderr=subprocess.STDOUT,text=True)
        print("\n".join(log.read_text(encoding="utf-8",errors="replace").splitlines()[-40:]),flush=True)
        print(f"RESULT exit={result.returncode} seconds={time.monotonic()-started:.1f} log={log}",flush=True)
        if result.returncode:raise RuntimeError(f"Verification failed at command {number}: {log}")
    print(f"FRESH_PROOF_FOLDER={folder}",flush=True)
    run(["git","clone","--no-local","--branch",branch,ROOT,clone],ROOT)
    run([sys.executable,"-m","venv",clone/".venv-backend"],clone)
    python=clone/".venv-backend"/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    run([python,"-m","pip","install","-r","backend/requirements-dev.txt","-r","requirements-llm.txt"],clone)
    run([sys.executable,"-m","venv",clone/".venv-llm"],clone)
    llm=clone/".venv-llm"/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    run([llm,"-m","pip","install","-r","requirements-llm.txt"],clone)
    run([llm,"-c","import json; from pathlib import Path; from indicmeet.summary import summarize; from indicmeet.demo import FakeGroq; rows=json.loads(Path('fixtures/demo_asr.json').read_text(encoding='utf-8')); result=summarize(rows,api_key='demo',client=FakeGroq(),log=lambda *a:None); assert result['overview_claims']; print('ISOLATED LLM PASS cited claims=',len(result['overview_claims']))"],clone)
    run([python,"-m","alembic","upgrade","head"],clone)
    run([python,"-m","alembic","upgrade","head"],clone)
    run([python,"-m","ruff","check","backend","indicmeet","asr_service","tests","scripts"],clone)
    run([python,"-m","pytest","-q","--tb=short"],clone)
    run([python,"-m","pytest","-q","tests/test_asr_async.py","tests/test_asr_service.py","tests/test_production_config.py","--tb=short"],clone)
    run([python,"scripts/validate_contract.py","fixtures"],clone)
    run([python,"scripts/audit_git.py"],clone)
    run([python,"scripts/audit_credentials.py"],clone)
    npm="npm.cmd" if os.name=="nt" else "npm"
    run([npm,"ci"],clone/"frontend")
    run([npm,"run","build"],clone/"frontend")
    project="indicmeet-fresh-"+folder.name.rsplit("-",1)[-1]
    environment["COMPOSE_PROJECT_NAME"]=project
    compose=["docker","compose","--env-file",".env.demo.example"]
    try:
        run(compose+["up","--build","-d","--wait"],clone)
        run(compose+["ps"],clone)
        run([python,"scripts/smoke_test.py","--url","http://127.0.0.1:8000"],clone)
        run([npm,"run","test:e2e"],clone/"frontend")
        run(compose+["ps"],clone)
        run(compose+["logs","--tail","35","api","worker"],clone)
    finally:
        run(compose+["down"],clone)
    run(["git","status","--short"],clone)
    print(f"FRESH CLONE PROOF PASS evidence={logdir}",flush=True)
if __name__=="__main__":main()
