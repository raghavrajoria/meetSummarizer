"""One sequential GPU worker; atomic disk checkpoints survive host restart."""
from pathlib import Path
import json,queue,threading,time,uuid,shutil,os,logging
from indicmeet.remote_asr import client_segments
logger=logging.getLogger("asr_service")

class JobStore:
    def __init__(self,root,model,retention=86400,clock=time.time):
        if retention<86400:raise ValueError("ASR retention must be at least 86400 seconds")
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.model,self.retention,self.clock=model,retention,clock
        self.lock=threading.RLock();self.queue=queue.Queue();self.stop=threading.Event()
        self.jobs={};self.keys={};self.thread=None;self.unavailable=False
        for file in self.root.glob('*/job.json'):
            job=json.loads(file.read_text(encoding="utf-8"));self.jobs[job["job_id"]]=job
            if job.get("key"):self.keys[job["key"]]=job["job_id"]
            if job["status"] in {"queued","running"}:
                job.update(status="queued",stage="restart_resume",progress=0);self._persist(job);self.queue.put(job["job_id"])
        self.cleanup()

    def _persist(self,job):
        folder=self.root/job["job_id"];folder.mkdir(exist_ok=True,mode=0o700)
        temp=folder/"job.json.tmp"
        with temp.open('w',encoding='utf-8') as f:
            json.dump(job,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
        os.chmod(temp,0o600);temp.replace(folder/"job.json")

    def submit(self,source,turns,key,fingerprint):
        with self.lock:
            if key in self.keys:
                old=self.jobs[self.keys[key]]
                if old["fingerprint"]!=fingerprint:raise ValueError("Idempotency-Key conflicts with another request")
                return old["job_id"]
            jid=uuid.uuid4().hex;folder=self.root/jid;folder.mkdir(mode=0o700)
            shutil.copyfile(source,folder/"audio.wav");os.chmod(folder/"audio.wav",0o600)
            job={"job_id":jid,"status":"queued","progress":0,"stage":"queued","created":self.clock(),"completed":None,"turns":turns,"key":key,"fingerprint":fingerprint}
            self.jobs[jid]=job
            if key:self.keys[key]=jid
            self._persist(job);self.queue.put(jid);return jid

    def get(self,jid):
        with self.lock:
            self.cleanup();job=self.jobs.get(jid)
            if not job:return None
            body={k:job[k] for k in ("status","progress","stage")}
            if job["status"]=="done":body["result"]=job["result"]
            if 'metrics' in job:body['metrics']=job['metrics']
            if job["status"]=="failed":body["error"]=job["error"]
            return body

    def cleanup(self):
        with self.lock:
            for jid,job in list(self.jobs.items()):
                if job.get("completed") is not None and self.clock()-job["completed"]>=self.retention:
                    # jid is a server-created UUID, never user path input.
                    if len(jid)!=32 or any(c not in '0123456789abcdef' for c in jid):raise ValueError("Unsafe persisted job id")
                    folder=self.root/jid
                    if folder.resolve().parent!=self.root.resolve():raise ValueError("Unsafe retention target")
                    shutil.rmtree(folder);self.jobs.pop(jid);self.keys.pop(job.get("key"),None)

    def start(self):
        if self.thread:raise RuntimeError("Only one GPU worker per service process")
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=5)

    def _run(self):
        while not self.stop.is_set():
            try:jid=self.queue.get(timeout=.2)
            except queue.Empty:self.cleanup();continue
            started=time.monotonic()
            def progress(value,stage):
                with self.lock:
                    job=self.jobs[jid];job.update(status="running",progress=max(0,min(99,int(value))),stage=stage);self._persist(job)
                logger.info("asr_progress job_id=%s progress=%s stage=%s model=%s device=%s",jid,value,stage,self.model.model_version,os.environ.get("ASR_DEVICE","fake"))
            try:
                progress(1,"starting")
                result=client_segments(self.model.transcribe(self.root/jid/"audio.wav",self.jobs[jid]["turns"],progress))
                if any(row["asr"]["model_version"]!=self.model.model_version for row in result):raise ValueError("Model version mismatch")
                with self.lock:self.jobs[jid].update(status="done",progress=100,stage="done",result=result,metrics=getattr(self.model,'last_metrics',{}),completed=self.clock());self._persist(self.jobs[jid])
                logger.info("asr_done job_id=%s elapsed=%.3f model=%s",jid,time.monotonic()-started,self.model.model_version)
            except Exception as exc:
                logger.exception('ASR host diagnostics job_id=%s', jid)
                from .memory import MemorySafetyError
                self.unavailable=isinstance(exc,(ImportError,RuntimeError)) and not isinstance(exc,MemorySafetyError)
                detail=str(exc) if isinstance(exc,MemorySafetyError) else "ASR processing failed; inspect private host diagnostics"
                with self.lock:self.jobs[jid].update(status="failed",stage="failed",metrics=getattr(self.model,'last_metrics',{}),error={"code":"RAM_SAFETY_ABORT" if isinstance(exc,MemorySafetyError) else "MODEL_UNAVAILABLE" if self.unavailable else "PROCESSING_FAILED","detail":detail},completed=self.clock());self._persist(self.jobs[jid])
                logger.error("asr_failed job_id=%s elapsed=%.3f exception_type=%s",jid,time.monotonic()-started,type(exc).__name__)
            finally:self.queue.task_done()
