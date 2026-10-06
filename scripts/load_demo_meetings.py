"""Idempotent, explicit-demo-only import of audited pre-computed real outputs."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.inventory_demo_assets import sha, ROOT


def require_demo(config, environment=None):
    environment=environment or os.environ
    if environment.get('DEMO','').lower()!='true' or environment.get('APP_ENV')!='demo' or environment.get('DEPLOY_PROFILE',''):
        raise ValueError('Loader requires explicit DEMO=true APP_ENV=demo and no other deployment profile')
    if not config.get('demo_mode') or config.get('app_env','demo')!='demo' or config.get('deploy_profile',''): raise ValueError('Target API is not in the explicit demo profile')


def read_group_export(path):
    import re
    source=path.read_text(encoding='utf-8')
    pattern=r'^\[(\d\d):(\d\d):(\d\d)\]\s+(\S+)\s+\[([^]]+)\]\s+\((accepted|review|rejected)\)\s*\n(.*?)(?=^\[\d\d:|\Z)'
    rows=[]
    for i,match in enumerate(re.finditer(pattern,source,re.M|re.S)):
        h,m,se,speaker,language,quality,body=match.groups()
        def field(label):
            found=re.search(r'^\s*'+label+r'\s*:\s*(.*?)(?=^\s*(?:Native|Roman|English)\s*:|\Z)',body,re.M|re.S)
            return ' '.join(found.group(1).split()) if found else None
        rows.append(dict(idx=i,start=int(h)*3600+int(m)*60+int(se),speaker=speaker,lang=language,quality=quality,text=field('Native') or '',text_roman=field('Roman'),method='legacy_owner_stated_kaggle'))
    if len(rows)!=226: raise ValueError('Full group export must have 226 rows')
    for i,row in enumerate(rows):
        row.update(end=rows[i+1]['start'] if i+1<len(rows) else 2251.68,endEstimated=True)
    return rows


def admitted_artifact(path):
    relative=path.resolve().relative_to(ROOT.resolve()).as_posix()
    if relative=='data/transcripts/transcript.txt': return read_group_export(path)
    allowed={'data/scrum_asr.json','data/local-real/preflight.json','data/sessions/group-discussion-pipeline-check/asr.json','data/demo_group_full.json'}
    if relative not in allowed:raise ValueError('Artifact has no approved real-model provenance')
    body=json.loads(path.read_text(encoding='utf-8'))
    rows=body if isinstance(body,list) else body.get('rows',[])
    if not rows or (relative.endswith('preflight.json') and (body.get('status')!='done' or body.get('metrics',{}).get('exit_code')!=0)):
        raise ValueError('Artifact is not completed real output')
    for row in rows:
        method=row.get('asr',{}).get('method',row.get('method',''))
        if relative!='data/sessions/group-discussion-pipeline-check/asr.json' and (method.startswith(('stored','fake','import')) or not method):raise ValueError('Fixture/import method is not real ASR provenance')
    return rows


def cached_scrum_summary(root=ROOT):
    path=root/'data/sessions/scrum-demo/summary.json'
    if not path.exists():return None
    summary=json.loads(path.read_text(encoding='utf-8'))
    cache=[]
    for item in (root/'llm_cache').glob('*.json'):
        cache.append(json.loads(item.read_text(encoding='utf-8')))
    if not summary.get('stats',{}).get('api_calls') or not any(c.get('overview')==summary.get('overview') and summary.get('overview') for c in cache):return None
    # Every saved task must be found in a cached actual model response.
    if not all(any(any(a.get('task')==item.get('task') for a in c.get('action_items',[])) for c in cache) for item in summary.get('action_items',[])):return None
    return copy.deepcopy(summary)


def build_payload(spec):
    from indicmeet.contract import canonicalize
    from indicmeet.pipeline import _session_payload
    rows=admitted_artifact(ROOT/spec['transcript'])
    canonical=canonicalize(rows)
    summary=cached_scrum_summary() if spec['id']=='real-scrum' else None
    if summary:
        from indicmeet.summary import finish_action,make_canon
        segs=[{'idx':r.get('idx',i),'spk':r['speaker'],'text':r.get('text',r.get('text_native',''))} for i,r in enumerate(rows)]
        segs=sorted(segs,key=lambda s:next((r['start'] for r in rows if r.get('idx')==s['idx']),s['idx']))
        pos={s['idx']:i for i,s in enumerate(segs)}
        ids={r.get('idx',i):c['segment_id'] for i,(r,c) in enumerate(zip(rows,canonical))}
        valid={r.get('idx',i) for i,r in enumerate(rows) if canonical[i]['quality']=='accepted'}
        for key in ('key_discussion','decisions','action_items','follow_ups','questions','concerns'):
            kept=[]
            for item in summary.get(key,[]):
                refs=[e for e in item.get('ev',[]) if e in valid]
                if not refs:continue
                item['ev']=refs
                if key=='action_items':
                    finish_action(item,segs,pos,make_canon(None,None),False,lambda *a:None)
                    item['owner_source_segment_ids']=[ids[e] for e in item.get('owner_name_ev',[]) if e in ids]
                item['source_segment_ids']=list(dict.fromkeys([ids[e] for e in refs]+item.get('owner_source_segment_ids',[])))
                kept.append(item)
            summary[key]=kept
        summary['overview_claims']=[{'text':x['point'],'source_segment_ids':x['source_segment_ids']} for x in summary.get('key_discussion',[]) if not x.get('unverified')][:6]
    else:
        summary={'overview':'','overview_claims':[],**{k:[] for k in ('key_discussion','decisions','action_items','follow_ups','questions','concerns')}}
    media=ROOT/spec['media']
    payload=_session_payload(spec['id'],spec['title'],'audio' if media.suffix.lower() in {'.wav','.mp3','.flac','.ogg','.m4a'} else 'video',media.name,canonical,summary,None)
    payload.update(provenance={'verdict':spec.get('verdict','REAL MODEL OUTPUT'),'model':spec['model'],'model_version':spec['version'],'run_date':spec['date'],'evidence':spec['evidence'],'summary_provenance':'Cached Groq results; exact model/version/run date unknown' if summary.get('overview') else 'summary not generated'},summary_status='cached_real_groq' if summary.get('overview') else 'not_generated',demo_asset_fingerprint=sha(media)+':'+sha(ROOT/spec['transcript']))
    return payload


SPECS=[
 {'id':'real-scrum','title':'Daily Scrum — staged demonstration','media':'data/recordings/scrumMeetingDemo.mp4','transcript':'data/scrum_asr.json','model':'Whisper large-v3 + MMS-LID; recorded methods use Whisper (owner-stated stack)','version':'unknown','date':'late Sep 2026; 30 Sep re-run (owner-stated)','evidence':'Real pipeline output (Kaggle T4, late Sep 2026, owner-stated). Source: staged scrum demo; action items are simulated.'},
 {'id':'real-agm-cpu-30','title':'Public AGM — completed 30-second CPU sample','media':'data/local-real/agm-30.wav','transcript':'data/local-real/preflight.json','model':'Whisper small int8 + MMS-LID256 + IndicConformer600M CTC','version':'cpu-small-int8-indic600m-mms256-20261006','date':'2026-10-06','evidence':'30-second sample, single speaker label (no diarization)'},
 {'id':'real-group-discussion-full','title':'Group discussion - full 37.5-minute recording','media':'data/recordings/GroupDiscussionAuio.mp4','transcript':'data/transcripts/transcript.txt','model':'Whisper large-v3 + IndicConformer routing (owner-stated)','version':'indicmeet_asr_v2 reconstructed; exact revision unknown','date':'late Sep 2026 (owner-stated)','evidence':'Real pipeline output (Kaggle T4, late Sep 2026, owner-stated). Hindi and English discussion; speakers anonymous. Timestamp ends estimated from the next exported start; summary not generated.'},
]


def main():
    import requests
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    args=parser.parse_args()
    from scripts.generate_local_real_config import read_env
    environment=read_env(ROOT/'.env.demo.example')
    # Caller must choose this file explicitly via loader invocation; never import .env keys.
    require_demo({'demo_mode':True},environment)
    if os.environ.get('DEPLOY_PROFILE') or os.environ.get('APP_ENV','demo')!='demo' or os.environ.get('DEMO','true').lower()!='true':raise ValueError('Refusing non-demo caller profile')
    client=requests.Session();base=args.url.rstrip('/')
    config=client.get(base+'/config',timeout=10);config.raise_for_status();require_demo(config.json(),environment)
    login=client.post(base+'/auth/login',json={'username':'demo','password':'demo-password'},timeout=10);login.raise_for_status()
    client.headers['Authorization']='Bearer '+login.json()['access_token']
    loaded=[]
    for spec in SPECS:
        if not (ROOT/spec['media']).is_file() or not (ROOT/spec['transcript']).is_file():
            print(f"SKIPPED {spec['id']}: missing real transcript/media");continue
        payload=build_payload(spec)
        existing=client.get(base+'/meetings/'+spec['id'],timeout=10)
        if existing.status_code==200:
            if existing.json().get('demo_asset_fingerprint')!=payload['demo_asset_fingerprint']:raise ValueError('Existing ID differs from audited artifacts; refusing overwrite')
            print('UNCHANGED',spec['id']);loaded.append(spec['id']);continue
        if existing.status_code!=404:existing.raise_for_status()
        with (ROOT/spec['media']).open('rb') as stream:
            response=client.post(base+'/sessions/import',files={'media':((ROOT/spec['media']).name,stream)},data={'session_json':json.dumps(payload,ensure_ascii=False)},timeout=180)
        response.raise_for_status();loaded.append(spec['id']);print('LOADED',spec['id'])
    listed=client.get(base+'/meetings',timeout=20);listed.raise_for_status()
    body=listed.json();meetings=body if isinstance(body,list) else body.get('meetings',body.get('items',body.get('sessions',[])))
    # Do not delete unrelated user meetings. A dirty old demo volume is a refusal, not permission to erase it.
    unexpected=[m.get('id') for m in meetings if m.get('id') not in loaded]
    if unexpected:raise ValueError('Demo contains unaudited meetings; use a fresh isolated demo Compose project')
    (ROOT/'data/demo_loaded.json').write_text(json.dumps({'loaded':loaded,'specs':[s for s in SPECS if s['id'] in loaded]},indent=2),encoding='utf-8')
    print('DEMO REAL MEETINGS',len(loaded))


if __name__=='__main__':main()
