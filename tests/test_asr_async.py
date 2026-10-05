"""Real loopback HTTP + durable DB tests; no model downloads."""
from datetime import datetime,timedelta
from pathlib import Path
import json
import pytest
from indicmeet.remote_asr import RemoteAsrProvider,RemoteContext,remote_context,AsrHostError,client_segments
from indicmeet.contract import canonicalize
from scripts.fake_asr_server import fake_server
from backend.models import Job,MeetingSession

def provider(url):return RemoteAsrProvider(url,token='test-token',poll_interval=.005,job_timeout=1,timeout=1)

def test_async_slow_host_client_ids_model_version_and_turns(tmp_path):
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture')
    with fake_server() as (url,host):
        result=provider(url).transcribe(audio,turns=[{'start':0,'end':1,'speaker':'S'}])
        assert host.posts==1 and host.polls==3 and b'name="turns"' in host.request_body
        assert result[0]['segment_id']!='host-id-is-ignored'
        assert result[0]['asr']['model_version']=='fake-asr-v2'
        assert provider(url).health()['model_version']=='fake-asr-v2'

@pytest.mark.parametrize('mode,code',[('failed','job_failed'),('unknown','HTTP_404'),('reject','HTTP_413')])
def test_host_failed_unknown_and_rejected_are_explicit(tmp_path,mode,code):
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture')
    with fake_server(mode=mode) as (url,host):
        with pytest.raises(AsrHostError,match=code):provider(url).transcribe(audio)
        assert host.posts==1

def test_bounded_poll_timeout_and_transient_backoff(tmp_path):
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture')
    with fake_server(mode='stall') as (url,host):
        clock=[0.];pauses=[]
        def sleep(seconds):pauses.append(seconds);clock[0]+=seconds
        p=RemoteAsrProvider(url,token='test-token',poll_interval=.1,job_timeout=.3,clock=lambda:clock[0],sleep=sleep)
        with pytest.raises(AsrHostError,match='timeout'):p.transcribe(audio)
        assert host.posts==1 and sum(pauses)==pytest.approx(.3)
    with fake_server(mode='transient') as (url,host):
        assert provider(url).transcribe(audio)
        assert host.posts==1 and host.polls==3

def test_worker_restart_resumes_persisted_host_id_without_post(api_env,tmp_path):
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture')
    with api_env.sessions() as db:
        db.add(MeetingSession(id='remote-resume',title='Resume',payload={}));db.flush()
        job=api_env.queue.enqueue(db,'remote-resume','request');db.commit();jid=job.id
    claim=api_env.queue.claim_next()
    with fake_server() as (url,host):
        import time
        api_env.queue.save_asr(claim,audio.name,{'status':'submitted','job_id':'fake-job-1','started':time.time(),'base_url':url,'idempotency_key':'persisted'})
        with api_env.sessions() as db:
            job=db.get(Job,jid);job.lease_expires_at=datetime.utcnow()-timedelta(seconds=1);db.commit()
        resumed=api_env.queue.claim_next();assert resumed and resumed.id==jid and resumed.worker_token!=claim.worker_token
        context=RemoteContext(jid,lambda scope:api_env.queue.load_asr(resumed,scope),lambda scope,state:api_env.queue.save_asr(resumed,scope,state),lambda progress:None)
        tok=remote_context.set(context)
        try:assert provider(url).transcribe(audio)
        finally:remote_context.reset(tok)
        assert host.posts==0 and host.polls==3
        with api_env.sessions() as db:assert db.get(Job,jid).remote_asr_job_id=='fake-job-1'

def test_uncertain_submission_never_auto_resubmits(tmp_path):
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture');state={'status':'submitting','base_url':'http://127.0.0.1:1'}
    context=RemoteContext('id',lambda scope:state,lambda *args:None,lambda *args:None);tok=remote_context.set(context)
    try:
        with pytest.raises(AsrHostError,match='unknown'):provider(state['base_url']).transcribe(audio)
    finally:remote_context.reset(tok)

def test_host_ids_ignored_collision_algorithm_uses_real_rows():
    raw=json.loads((Path(__file__).parents[1]/'fixtures/session.json').read_text(encoding='utf-8'))['segments'];rows=canonicalize(raw)
    for row in rows:row['asr']['model_version']='fake-host-v2';row['segment_id']='duplicate-host-id'
    result=client_segments(rows);assert len({r['segment_id'] for r in result})==len(rows)
    assert client_segments(rows)==result
    rows[0]['asr'].pop('model_version')
    with pytest.raises(AsrHostError,match='model_version'):client_segments(rows)

def test_rounded_start_collision_keeps_distinct_submillisecond_rows():
    rows=[{'start':start,'end':1.,'speaker':'S','text':'same native','lang':'en','quality':'accepted'} for start in (.0001,.0002)]
    out=canonicalize(rows);assert len({s['segment_id'] for s in out})==2
    assert canonicalize(rows)==out

def test_explicit_retry_clears_terminal_remote_checkpoint(api_env):
    with api_env.sessions() as db:
        db.add(MeetingSession(id='retry-asr',title='retry',payload={}));db.flush();job=api_env.queue.enqueue(db,'retry-asr','request');db.commit();jid=job.id
    claim=api_env.queue.claim_next()
    api_env.queue.save_asr(claim,'audio.wav',{'job_id':'old-host-job','status':'failed','base_url':'http://host'})
    api_env.queue.fail(claim,'ASR host failed')
    with api_env.sessions() as db:
        api_env.queue.retry(db,db.get(Job,jid),'retry');db.commit()
    next_claim=api_env.queue.claim_next()
    assert next_claim.attempt>claim.attempt
    assert api_env.queue.load_asr(next_claim,'audio.wav') is None

def test_actual_worker_crash_then_restart_polls_same_host_job(api_env,tmp_path):
    from backend.worker import Worker
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture')
    with api_env.sessions() as db:
        db.add(MeetingSession(id='worker-crash-asr',title='Resume real worker',payload={}));db.flush()
        job=api_env.queue.enqueue(db,'worker-crash-asr','request');db.commit();jid=job.id
    with fake_server() as (url,host):
        def crash(seconds):raise SystemExit('simulated process death after accepted host submission')
        first=RemoteAsrProvider(url,token='test-token',poll_interval=.005,job_timeout=5,sleep=crash)
        def processor(provider):
            return lambda record,store,stage:{'transcript':stage('asr',40,lambda:provider.transcribe(audio))}
        with pytest.raises(SystemExit):Worker(api_env.queue,api_env.sessions,api_env.store,processor=processor(first)).run_once()
        with api_env.sessions() as db:
            job=db.get(Job,jid)
            assert job.status=='running' and job.remote_asr_job_id=='fake-job-1'
            job.lease_expires_at=datetime.utcnow()-timedelta(seconds=1);db.commit()
        second=RemoteAsrProvider(url,token='test-token',poll_interval=.005,job_timeout=5)
        assert Worker(api_env.queue,api_env.sessions,api_env.store,processor=processor(second)).run_once()
        with api_env.sessions() as db:
            job=db.get(Job,jid);record=db.get(MeetingSession,'worker-crash-asr')
            assert job.status=='done' and job.progress==100 and job.attempts==2
            assert job.asr_requests[audio.name]['status']=='done'
            assert record.payload['transcript'][0]['asr']['model_version']=='fake-asr-v2'
        assert host.posts==1 and host.polls==3
