"""Authenticated real API upload/poll and honest public-content evidence report."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time
import wave
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.generate_local_real_config import read_env


def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1): current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def soft_scores(rows, reference, start, duration):
    if not rows: return {'label':'soft: LLM-generated reference','WER':'NOT VERIFIED: no completed transcript','CER':'NOT VERIFIED'}
    from indicmeet.contract import seconds
    selected=[]
    segments=reference.get('segments', reference) if isinstance(reference,dict) else reference
    for i,row in enumerate(segments):
        a=seconds(row.get('start',row.get('timestamp',0)))
        b=seconds(row.get('end',segments[i+1].get('start',segments[i+1].get('timestamp',a+1)))) if i+1<len(segments) else a+1
        # Only complete reference spans; cropped words cannot be meaningfully scored.
        if start<=a<b<=start+duration: selected.append((a,b,row.get('text_native',row.get('text',''))))
    if not selected: return {'label':'soft: LLM-generated reference','WER':'NOT VERIFIED: no complete overlapping reference spans','CER':'NOT VERIFIED'}
    lo,hi=selected[0][0]-start,selected[-1][1]-start
    ref=' '.join(s[2] for s in selected).lower()
    hyp=' '.join(s['text_native'] for s in rows if lo<=s['start'] and s['end']<=hi).lower()
    import re
    normalize=lambda t: re.sub(r'[^\w\s]','',t)
    ref,hyp=normalize(ref),normalize(hyp)
    return {'label':'soft: LLM-generated reference','WER':distance(ref.split(),hyp.split())/max(1,len(ref.split())),'CER':distance(list(ref),list(hyp))/max(1,len(ref)),'reference_words':len(ref.split()),'reference_spans':len(selected),'scored_clip_interval':[lo,hi],'alignment_limitation':'Reference timestamps are machine-generated; only complete hypothesis segments inside reference envelope used.'}


def report(session, job, duration, elapsed, metrics, scores):
    rows=session.get('transcript',[])
    valid={s['segment_id'] for s in rows}
    items=[item for value in session.get('intelligence',{}).values() if isinstance(value,list) for item in value if isinstance(item,dict)]
    citations=[ref for item in items for ref in item.get('source_segment_ids',[])]
    missing=[ref for ref in citations if ref not in valid]
    completed=job['status'] in {'done','completed'}
    return {'status':job['status'],'duration_seconds':duration,'elapsed_seconds':elapsed,'real_time_factor':elapsed/duration if completed else 'NOT VERIFIED: failed job','stage_timings':job.get('stage_timings',{}),'host_metrics':metrics or 'NOT VERIFIED','languages':dict(Counter(s['language'] for s in rows)) if completed else 'NOT VERIFIED','quality':dict(Counter(s['quality'] for s in rows)) if completed else 'NOT VERIFIED','review_segments':sum(s['quality']=='review' for s in rows) if completed else 'NOT VERIFIED','citation_count':len(citations) if completed else 'NOT VERIFIED','unresolved_citations':missing,'soft_evaluation':scores,'transcript':rows,'summary':session.get('summaryData',session.get('intelligence',{})),'limitations':['Does not prove GPU speed, CUDA image, large-v3 accuracy or real-audio diarization.']}


def main():
    import requests
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',required=True)
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--turns',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('.env.local-real'))
    parser.add_argument('--reference',type=Path,default=Path('data/ground_truth/agm_ground_truth.json'))
    parser.add_argument('--source-start',type=float,default=840)
    parser.add_argument('--timeout',type=float,default=10800)
    parser.add_argument('--evidence',type=Path,default=Path('docs/REAL_RUN_EVIDENCE.md'))
    args=parser.parse_args()
    cfg=read_env(args.config)
    if cfg.get('DEMO')!='false' or cfg.get('APP_ENV')!='production': raise ValueError('Real run requires validated non-demo production configuration')
    import os
    os.environ.update(cfg)
    from backend.config_guard import validate_app_configuration
    validate_app_configuration()
    api=requests.Session()
    def call(method,path,**kwargs):
        response=api.request(method,args.url.rstrip('/')+path,timeout=120,**kwargs); response.raise_for_status(); return response.json()
    login=call('POST','/auth/login',json={'username':cfg['LOCAL_REAL_USERNAME'],'password':cfg['LOCAL_REAL_PASSWORD']})
    api.headers['Authorization']='Bearer '+login['access_token']
    config=call('GET','/config')
    if config['demo_mode'] or config['import_mode']: raise ValueError('Refusing demo/import endpoint')
    with wave.open(str(args.audio)) as audio: duration=audio.getnframes()/audio.getframerate()
    started=time.monotonic()
    with args.audio.open('rb') as audio,args.turns.open('rb') as turns:
        uploaded=call('POST','/meetings',files={'recording':(args.audio.name,audio,'audio/wav'),'diarization_csv':(args.turns.name,turns,'text/csv')},data={'title':f'Public AGM CPU real run {duration:.0f}s'})
    meeting_id=uploaded.get('meeting_id',uploaded.get('id'))
    if not meeting_id: raise ValueError('Upload did not return meeting id')
    last_login=time.monotonic()
    while time.monotonic()-started<args.timeout:
        if time.monotonic()-last_login>600:
            login=call('POST','/auth/login',json={'username':cfg['LOCAL_REAL_USERNAME'],'password':cfg['LOCAL_REAL_PASSWORD']});api.headers['Authorization']='Bearer '+login['access_token'];last_login=time.monotonic()
        job=call('GET',f'/meetings/{meeting_id}/job')
        print(f"stage={job['stage']} progress={job['progress']} elapsed={time.monotonic()-started:.1f}s",flush=True)
        if job['status'] in ('completed','done','failed'): break
        time.sleep(5)
    else: raise TimeoutError('Real pipeline polling deadline reached; host run is not silently cancelled')
    session=call('GET',f'/meetings/{meeting_id}')
    metrics=None
    if job.get('remote_asr_job_id'):
        response=requests.get(cfg['ASR_SERVICE_URL'].rstrip('/')+'/jobs/'+job['remote_asr_job_id'],headers={'Authorization':'Bearer '+cfg['ASR_SERVICE_TOKEN']},timeout=30);response.raise_for_status();metrics=response.json().get('metrics')
    reference=json.loads(args.reference.read_text(encoding='utf-8')) if args.reference.exists() else {'segments':[]}
    evidence=report(session,job,duration,time.monotonic()-started,metrics,soft_scores(session.get('transcript',[]),reference,args.source_start,duration))
    args.evidence.parent.mkdir(parents=True,exist_ok=True)
    with args.evidence.open('a',encoding='utf-8') as stream: stream.write('\n## Public AGM API real run\n\n```json\n'+json.dumps(evidence,ensure_ascii=False,indent=2)+'\n```\n')
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('transcript','summary')},indent=2))
    if job['status']=='failed' or evidence['unresolved_citations']: raise SystemExit(1)


if __name__=='__main__': main()
