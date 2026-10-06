"""Reference ASR v2 HTTP service. Run one uvicorn process per GPU."""
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio,hashlib,hmac,json,math,os,tempfile,time,wave
from fastapi import FastAPI,UploadFile,File,Form,Request,HTTPException
from .store import JobStore

def probe_wav(path):
    try:
        with wave.open(str(path),'rb') as audio:
            if audio.getnchannels()!=1 or audio.getframerate()!=16000 or audio.getsampwidth()!=2 or audio.getcomptype()!='NONE':raise ValueError()
            return audio.getnframes()/16000
    except (wave.Error,EOFError,ValueError):raise HTTPException(415,"Audio must be decodable 16kHz mono 16-bit PCM WAV") from None

def create_app(*,model=None,root=None,token=None,probe=probe_wav,max_duration=None,max_bytes=None,retention=None,clock=time.time):
    from .model import GpuModel
    model=model or GpuModel()
    token=token or os.environ.get("ASR_SERVICE_TOKEN","")
    if len(token)<16:raise ValueError("ASR_SERVICE_TOKEN must be configured (at least 16 characters)")
    maximum=max_duration or float(os.environ.get("ASR_MAX_DURATION_SECONDS","7200"))
    limit=max_bytes or int(os.environ.get("ASR_MAX_UPLOAD_BYTES",str(512*1024*1024)))
    store=JobStore(root or os.environ.get("ASR_STATE_DIR","/var/lib/asr"),model,retention or int(os.environ.get("ASR_RETENTION_SECONDS","86400")),clock)
    @asynccontextmanager
    async def lifespan(app):
        store.start()
        try:yield
        finally:store.close()
    app=FastAPI(title="IndicMeet ASR contract v2",lifespan=lifespan,docs_url=None,redoc_url=None)
    app.state.jobs=store
    @app.middleware("http")
    async def authenticate_before_parsing(request,call_next):
        if request.url.path=="/transcribe" or request.url.path.startswith("/jobs/"):
            scheme,_,candidate=request.headers.get('authorization','').partition(' ')
            if scheme.lower()!='bearer' or not hmac.compare_digest(candidate.encode(),token.encode()):
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail":"Unauthorized"},status_code=401,headers={"WWW-Authenticate":"Bearer"})
        return await call_next(request)
    def authorize(request):
        scheme,_,candidate=request.headers.get('authorization','').partition(' ')
        if scheme.lower()!='bearer' or not hmac.compare_digest(candidate.encode(),token.encode()):raise HTTPException(401,"Unauthorized",headers={"WWW-Authenticate":"Bearer"})
    @app.get('/healthz')
    def health():return {"status":"ok","model_version":model.model_version}
    @app.get('/jobs/{job_id}')
    def get_job(job_id:str,request:Request):
        authorize(request);body=store.get(job_id)
        if body is None:raise HTTPException(404,"Unknown or expired job_id")
        return body
    @app.post('/transcribe')
    async def transcribe(request:Request,audio:UploadFile=File(...),turns:str|None=Form(None),sync:bool=False):
        authorize(request)
        key=request.headers.get('Idempotency-Key','')
        existing=bool(key and key in store.keys)
        if store.unavailable and not existing:raise HTTPException(503,"ASR model unavailable")
        if not existing and store.queue.qsize()>=int(os.environ.get('ASR_MAX_QUEUED_JOBS','32')):raise HTTPException(429,"ASR queue full",headers={"Retry-After":"5"})
        if len(key)>160:raise HTTPException(422,"Idempotency-Key too long")
        with tempfile.TemporaryDirectory(prefix='asr-upload-',dir=store.root) as tmp:
            file=Path(tmp)/'audio.wav';size=0;digest=hashlib.sha256()
            with file.open('wb') as out:
                while chunk:=await audio.read(1024*1024):
                    size+=len(chunk)
                    if size>limit:raise HTTPException(413,"Audio exceeds host byte limit")
                    digest.update(chunk);out.write(chunk)
            duration=probe(file)
            if not math.isfinite(duration) or duration<=0:raise HTTPException(415,"Audio is empty or undecodable")
            if duration>maximum:raise HTTPException(413,"Audio exceeds host duration limit")
            if sync and duration>=300:raise HTTPException(413,"Synchronous audio must be under 5 minutes; use async POST /transcribe")
            parsed=None
            if turns is not None:
                try:
                    parsed=json.loads(turns)
                    if not isinstance(parsed,list):raise ValueError()
                    for turn in parsed:
                        if set(turn)!={'start','end','speaker'} or not isinstance(turn['start'],(int,float)) or not isinstance(turn['end'],(int,float)) or isinstance(turn['start'],bool) or isinstance(turn['end'],bool) or not isinstance(turn['speaker'],str) or not turn['speaker'] or not 0<=float(turn['start'])<=float(turn['end'])<=duration:raise ValueError()
                        if not math.isfinite(float(turn['start'])) or not math.isfinite(float(turn['end'])):raise ValueError()
                except (ValueError,TypeError,KeyError):raise HTTPException(422,"Invalid recording-relative turns") from None
            if os.environ.get('ASR_DEVICE')=='cpu' and not parsed:
                raise HTTPException(422,'CPU ASR requires nonempty request turns; pyannote is never run on CPU')
            digest.update(json.dumps(parsed,sort_keys=True,separators=(',',':')).encode())
            try:jid=store.submit(file,parsed,key,digest.hexdigest())
            except ValueError:raise HTTPException(409,"Idempotency-Key request conflict") from None
        if not sync:
            from fastapi.responses import JSONResponse
            return JSONResponse({"job_id":jid},status_code=202)
        deadline=time.monotonic()+float(os.environ.get('ASR_SYNC_TIMEOUT_SECONDS','300'))
        while time.monotonic()<deadline:
            body=store.get(jid)
            if body['status']=='done':return body['result']
            if body['status']=='failed':raise HTTPException(503,"ASR processing failed")
            await asyncio.sleep(.05)
        # Queue result still exists and may be fetched; never kill GPU work silently.
        raise HTTPException(503,"Synchronous wait timed out; use async",headers={"X-ASR-Job-ID":jid})
    return app
