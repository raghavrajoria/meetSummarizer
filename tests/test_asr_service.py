"""Reference host fake-model tests; never import torch or download models."""
import io,time,wave
import pytest
from fastapi.testclient import TestClient
from asr_service.app import create_app
from asr_service.store import JobStore
from indicmeet.contract import canonicalize

TOKEN='offline-host-test-auth-7B92'
HEADERS={'Authorization':'Bearer '+TOKEN}
def wav(seconds=1):
    output=io.BytesIO()
    with wave.open(output,'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0\0'*int(seconds*16000))
    return output.getvalue()
class FakeModel:
    model_version='fake-reference-v2'
    def __init__(self,fail=False):self.fail=fail;self.calls=0;self.active=0;self.maximum=0
    def transcribe(self,audio,turns,progress):
        self.calls+=1;self.active+=1;self.maximum=max(self.maximum,self.active)
        try:
            progress(40,'asr');time.sleep(.015)
            if self.fail:raise ValueError('private transcript must not leak')
            row=canonicalize([{'start':0.,'end':.5,'speaker':'S','text':'native source','lang':'en','quality':'accepted'}])[0]
            row['asr']['model_version']=self.model_version;return [row]
        finally:self.active-=1
def complete(client,jid):
    for _ in range(100):
        body=client.get('/jobs/'+jid,headers=HEADERS).json()
        if body['status'] in {'done','failed'}:return body
        time.sleep(.01)
    raise AssertionError('Fake model job did not finish')
def test_service_health_auth_async_idempotency_and_single_worker(tmp_path):
    model=FakeModel()
    with TestClient(create_app(model=model,root=tmp_path,token=TOKEN)) as client:
        assert client.get('/healthz').json()=={'status':'ok','model_version':model.model_version}
        assert client.get('/jobs/unknown').status_code==401
        assert client.post('/transcribe').status_code==401
        assert client.get('/jobs/unknown',headers=HEADERS).status_code==404
        headers={**HEADERS,'Idempotency-Key':'same-request'}
        first=client.post('/transcribe',headers=headers,files={'audio':('clip.wav',wav(),'audio/wav')});assert first.status_code==202
        jid=first.json()['job_id']
        assert client.post('/transcribe',headers=headers,files={'audio':('clip.wav',wav(),'audio/wav')}).json()['job_id']==jid
        other=client.post('/transcribe',headers=HEADERS,files={'audio':('clip.wav',wav(),'audio/wav')}).json()['job_id']
        result=complete(client,jid);assert result['status']=='done' and result['result'][0]['asr']['model_version']==model.model_version and 'error' not in result
        assert complete(client,other)['status']=='done' and model.calls==2 and model.maximum==1
        client.app.state.jobs.unavailable=True
        assert client.post('/transcribe',headers=headers,files={'audio':('clip.wav',wav(),'audio/wav')}).json()['job_id']==jid
        assert client.post('/transcribe',headers=headers,files={'audio':('clip.wav',wav(2),'audio/wav')}).status_code==409
    restarted=JobStore(tmp_path,FakeModel());assert restarted.get(jid)['result']==result['result']

def test_sync_duration_errors_turns_and_model_failure(tmp_path):
    with TestClient(create_app(model=FakeModel(),root=tmp_path/'sync',token=TOKEN,probe=lambda path:1)) as client:
        response=client.post('/transcribe?sync=true',headers=HEADERS,files={'audio':('clip.wav',wav(),'audio/wav')});assert response.status_code==200 and response.json()[0]['text_native']=='native source'
        assert client.post('/transcribe',headers=HEADERS,files={'audio':('clip.wav',wav(),'audio/wav')},data={'turns':'[{"start": 0, "end": 9, "speaker":"S"}]'}).status_code==422
    with TestClient(create_app(model=FakeModel(),root=tmp_path/'long',token=TOKEN,probe=lambda path:300)) as client:
        response=client.post('/transcribe?sync=true',headers=HEADERS,files={'audio':('clip.wav',wav(),'audio/wav')});assert response.status_code==413 and 'async' in response.json()['detail']
    with TestClient(create_app(model=FakeModel(),root=tmp_path/'bad',token=TOKEN,max_bytes=100)) as client:
        assert client.post('/transcribe',headers=HEADERS,files={'audio':('clip.wav',wav(),'audio/wav')}).status_code==413
        assert client.post('/transcribe',headers=HEADERS,files={'audio':('clip.wav',b'bad','audio/wav')}).status_code==415
    with TestClient(create_app(model=FakeModel(fail=True),root=tmp_path/'fail',token=TOKEN)) as client:
        jid=client.post('/transcribe',headers=HEADERS,files={'audio':('clip.wav',wav(),'audio/wav')}).json()['job_id']
        body=complete(client,jid);assert body['status']=='failed' and set(body['error'])=={'code','detail'} and 'result' not in body and 'private transcript' not in str(body)

def test_restart_resumes_running_job_and_retention(tmp_path):
    model=FakeModel();clock=[100.];store=JobStore(tmp_path,model,clock=lambda:clock[0])
    source=tmp_path/'input.wav';source.write_bytes(wav());jid=store.submit(source,None,'key','hash')
    store.jobs[jid]['status']='running';store._persist(store.jobs[jid])
    restored=JobStore(tmp_path,model,clock=lambda:clock[0]);restored.start()
    try:
        for _ in range(100):
            if restored.get(jid)['status']=='done':break
            time.sleep(.01)
        assert restored.get(jid)['status']=='done' and model.calls==1
        clock[0]+=86399;assert restored.get(jid)
        clock[0]+=2;assert restored.get(jid) is None and not (tmp_path/jid).exists()
    finally:restored.close()
    with pytest.raises(ValueError,match='at least'):JobStore(tmp_path,model,retention=10)

def test_host_rate_limit_and_unavailable_semantics(tmp_path,monkeypatch):
    with TestClient(create_app(model=FakeModel(),root=tmp_path,token=TOKEN)) as client:
        monkeypatch.setenv('ASR_MAX_QUEUED_JOBS','0')
        response=client.post('/transcribe',headers=HEADERS,files={'audio':('clip.wav',wav(),'audio/wav')})
        assert response.status_code==429 and response.headers['retry-after']=='5'
        client.app.state.jobs.unavailable=True
        assert client.post('/transcribe',headers=HEADERS,files={'audio':('clip.wav',wav(),'audio/wav')}).status_code==503

def test_reference_host_fake_model_with_actual_remote_http_client(tmp_path):
    import socket,threading,uvicorn
    from indicmeet.remote_asr import RemoteAsrProvider
    with socket.socket() as reserve:reserve.bind(('127.0.0.1',0));port=reserve.getsockname()[1]
    model=FakeModel();app=create_app(model=model,root=tmp_path/'http-state',token=TOKEN)
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error',access_log=False))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    try:
        for _ in range(100):
            if server.started:break
            time.sleep(.02)
        assert server.started
        file=tmp_path/'fixture.wav';file.write_bytes(wav())
        provider=RemoteAsrProvider(f'http://127.0.0.1:{port}',token=TOKEN,poll_interval=.005,job_timeout=2)
        assert provider.health()['model_version']==model.model_version
        result=provider.transcribe(file)
        assert result[0]['asr']['model_version']==model.model_version and model.calls==1
    finally:server.should_exit=True;thread.join(5)
