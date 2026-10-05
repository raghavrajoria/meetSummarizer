"""ASR v2: one durable submission, restartable bounded polling, client IDs."""
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,time
import requests
from .contract import canonicalize,reidentify
from .settings import get_settings

remote_context=ContextVar("remote_asr_context",default=None)

@dataclass
class RemoteContext:
    key: str
    load: object
    save: object
    progress: object

class AsrHostError(RuntimeError):
    pass

def client_segments(value):
    if not isinstance(value,list):raise AsrHostError("ASR result must be an array")
    rows=[]
    required={"start","end","speaker","speaker_name","language","text_native","text_roman","text_english","quality","reasons","asr"}
    for index,row in enumerate(value):
        if not isinstance(row,dict) or not required.issubset(row) or not isinstance(row.get("asr"),dict):
            raise AsrHostError("ASR v2 requires canonical segment fields")
        if not isinstance(row,dict) or not isinstance(row.get("asr",{}).get("model_version"),str) or not row["asr"]["model_version"].strip():
            raise AsrHostError("ASR v2 requires asr.model_version")
        copied=dict(row);copied["segment_id"]=f"client-pending-{index}"
        rows.append(copied)
    return reidentify(canonicalize(rows))

class RemoteAsrProvider:
    def __init__(self,service_url=None,token=None,timeout=None,*,poll_interval=None,job_timeout=None,sleep=time.sleep,clock=time.time):
        settings=get_settings()
        self.service_url=(service_url or settings.asr_service_url or "").rstrip("/")
        self.token=settings.asr_service_token if token is None else token
        self.timeout=settings.asr_service_timeout_seconds if timeout is None else timeout
        self.interval=settings.asr_poll_interval_seconds if poll_interval is None else poll_interval
        self.job_timeout=settings.asr_job_timeout_seconds if job_timeout is None else job_timeout
        self.sleep,self.clock=sleep,clock
        if not self.service_url:raise ValueError("ASR_SERVICE_URL base URL is required")
        if min(self.timeout,self.interval,self.job_timeout)<=0:raise ValueError("ASR timeouts/poll interval must be positive")

    def health(self):
        response=requests.get(self.service_url+"/healthz",timeout=self.timeout)
        response.raise_for_status();body=response.json()
        if body.get("status")!="ok" or not body.get("model_version"):raise AsrHostError("Invalid ASR health response")
        return body

    def transcribe(self,audio_path:Path,*,asr_json_path=None,turns=None,sync=False):
        del asr_json_path
        context=remote_context.get();scope=audio_path.name
        state=context.load(scope) if context else None
        def save(value):
            nonlocal state
            state=dict(value)
            if context:context.save(scope,state)
        def fail(code):
            save({**(state or {}),"status":"failed","error":code})
            raise AsrHostError("ASR host "+code)
        headers={"Authorization":"Bearer "+self.token} if self.token else {}
        if sync and context:raise AsrHostError("Worker ASR requests must be asynchronous")
        if state:
            if state.get("base_url")!=self.service_url:raise AsrHostError("ASR endpoint changed; explicit retry required")
            if state.get("status") in {"submitting","ambiguous"}:raise AsrHostError("ASR submission outcome unknown; explicit retry required")
            if state.get("status") in {"failed","timeout"}:raise AsrHostError("ASR terminal request; explicit retry required")
        if not state:
            key=hashlib.sha256((context.key+":"+scope).encode()).hexdigest() if context else __import__('uuid').uuid4().hex
            save({"status":"submitting","started":self.clock(),"base_url":self.service_url,"idempotency_key":key})
            headers["Idempotency-Key"]=key
            try:
                with audio_path.open("rb") as audio:
                    response=requests.post(self.service_url+"/transcribe",params={"sync":"true"} if sync else None,files={"audio":(audio_path.name,audio,"audio/wav")},data={"turns":json.dumps(turns)} if turns is not None else {},headers=headers,timeout=self.timeout)
            except (requests.RequestException,OSError) as exc:
                save({**state,"status":"ambiguous"})
                raise AsrHostError("ASR submission outcome unknown; explicit retry required") from exc
            if sync:
                if response.status_code!=200:fail("HTTP_"+str(response.status_code))
                return client_segments(response.json())
            if response.status_code!=202:fail("HTTP_"+str(response.status_code))
            try:job_id=response.json()["job_id"]
            except (KeyError,ValueError,TypeError):fail("invalid_submission_response")
            if not isinstance(job_id,str) or not job_id or len(job_id)>160 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in job_id):fail("invalid_job_id")
            save({**state,"status":"submitted","job_id":job_id})
        delay=self.interval
        while True:
            remaining=self.job_timeout-(self.clock()-state["started"])
            if remaining<=0:
                save({**state,"status":"timeout"})
                raise AsrHostError("ASR job timeout; explicit retry required")
            try:
                response=requests.get(self.service_url+"/jobs/"+state["job_id"],headers=headers,timeout=min(self.timeout,remaining))
                if response.status_code in {429,503}:
                    retry=response.headers.get("Retry-After","")
                    pause=float(retry) if retry.isdigit() else delay
                    self.sleep(min(max(self.interval,pause),30,remaining));delay=min(delay*1.5,30);continue
                if response.status_code!=200:fail("HTTP_"+str(response.status_code))
                body=response.json()
            except requests.RequestException:
                self.sleep(min(delay,remaining));delay=min(delay*1.5,30);continue
            except ValueError:fail("invalid_job_response")
            if not isinstance(body,dict) or body.get("status") not in {"queued","running","done","failed"}:fail("invalid_job_status")
            progress=body.get("progress")
            if isinstance(progress,bool) or not isinstance(progress,(int,float)) or not 0<=progress<=100:fail("invalid_progress")
            if context:context.progress(int(progress))
            if body["status"]=="done":
                try:result=client_segments(body["result"])
                except (KeyError,ValueError,AsrHostError):fail("invalid_result")
                save({**state,"status":"done"});return result
            if body["status"]=="failed":fail("job_failed")
            self.sleep(min(delay,remaining));delay=min(delay*1.5,30)
