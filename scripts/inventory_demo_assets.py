"""Read-only recording/transcript inventory; never imports an ASR model."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
MEDIA={'.wav','.mp3','.mp4','.webm','.m4a','.ogg','.flac','.mov','.mkv'}
PRUNE={'.git','node_modules','models','NeMo','__pycache__','.pytest_cache','.ruff_cache','dist','test-results','playwright-report'}


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def duration(path):
    try:
        return round(float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)],text=True,timeout=30).strip()),3)
    except (OSError,subprocess.SubprocessError,ValueError):return 'unknown'


def verdict(path):
    name=path.as_posix().lower()
    if 'ground_truth' in name:return 'LLM-GENERATED REFERENCE','Ground-truth provenance documented in archive branch codex/archive-development-2026-10-07:docs/INDICMEET_HANDOFF.md; not human ground truth or ASR output.'
    if name in {'data/transcripts/transcript.txt','data/transcripts/transcript (1).txt','data/demo_group_full.json'}:
        return 'REAL MODEL OUTPUT (OWNER-STATED)','Owner-stated Kaggle T4 clean-room run, late Sep2026, indicmeet_asr_v2 reconstructed; Whisper large-v3 + IndicConformer routing. 226 rows:166 accepted/56 review/4 rejected; en136/hi84/other6. Full media2251.680s; diarization end2251.915s. End times reconstructed; exact versions unknown.'
    if name in {'data/sessions/group-discussion-pipeline-check/asr.json','fixtures/group_discussion_3min.mp4','data/sessions/group-discussion-pipeline-check/media.mp4'}:
        return 'REAL MODEL OUTPUT (OWNER-STATED EXCERPT)','19 native rows exactly match first180s of full owner-stated export. Source media SHA256 equals session media. Superseded by full meeting; fixture dry summary excluded.'
    if name.startswith('fixtures/') or 'group-discussion-pipeline-check' in name or name.startswith('frontend/media/'):
        return 'FAKE/FIXTURE','Fixture/demo artifact; group session derives from fixtures/session.json and dry_summary, not a proven model run.'
    if name in {'data/scrum_asr.json','data/sessions/scrum-demo/asr.json','data/sessions/scrum-demo/session.json'}:
        return 'REAL MODEL OUTPUT','156 stored segments: 150 whisper_english, 6 whisper_latin_unresolved; Whisper/MMS confidence metadata, matching 1286.3s scrum media and media.json source link. Owner-stated: indicmeet_asr_v2.py, Kaggle Tesla T4, Whisper large-v3 + IndicConformer CTC + MMS-LID, 30 Sep 2026 re-run. Fields consistent; model/date absent in artifact. Staged YouTube Scrum; actions simulated.'
    if name=='data/local-real/preflight.json':
        return 'REAL MODEL OUTPUT','2026-10-06 supervised actual CPU execution, preflight.log: small int8 Whisper, facebook/mms-lid-256 and IndicConformer CTC; exit=0, measured stage timings. Version cpu-small-int8-indic600m-mms256-20261006.'
    if name=='data/outputs/prior_output/transcript_qwen3_1.7b.txt':
        return 'UNESTABLISHED','Generator experiments/qwen_test.py names Qwen3-ASR-1.7B and speech_test_10s.wav, but no execution log/hash binds this text file to that run; conservatively skipped.'
    if name=='data/outputs/prior_output/full_transcript.txt':
        return 'UNESTABLISHED','Format matches experiments/auto_lang_asr.py IndicConformer CTC output; transcript ends513.3s, current script input is480s hinglish_test_4to12min.wav. Exact media/run binding not established.'
    return 'UNESTABLISHED','No reliable model-output/run/media association established; not loaded. File name or mtime alone is not proof.'


def main():
    entries=[]
    for folder,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in PRUNE and not d.startswith('.venv')]
        for name in files:
            path=Path(folder)/name;relative=path.relative_to(ROOT)
            if path.suffix.lower() in MEDIA:
                entries.append({'path':relative.as_posix(),'kind':'media','duration_seconds':duration(path),'bytes':path.stat().st_size})
            elif path.suffix.lower() in {'.json','.txt','.srt','.vtt'} and (relative.parts[0] in {'data','fixtures','llm_cache','output','ground_truth'}):
                rows=[]
                if path.suffix=='.json':
                    try:
                        body=json.loads(path.read_text(encoding='utf-8'))
                        rows=body if isinstance(body,list) else body.get('rows',body.get('transcript',body.get('segments',[]))) if isinstance(body,dict) else []
                    except (ValueError,UnicodeError):continue
                if (rows and isinstance(rows[0],dict) and any(k in rows[0] for k in ('text','text_native','tx'))) or path.suffix.lower() in {'.txt','.srt','.vtt'}:
                    v,e=verdict(relative);entries.append({'path':relative.as_posix(),'kind':'transcript','duration_seconds':'not media','verdict':v,'evidence':e,'segments':len(rows) or 'plain text'})
                elif relative.parts[0]=='llm_cache':
                    entries.append({'path':relative.as_posix(),'kind':'summary cache','duration_seconds':'not media','verdict':'CACHED GROQ CANDIDATE','evidence':'Raw cached JSON lacks input/model/date metadata; reuse only a summary with matching recording and cached overview plus matching extracted items.'})
    associations={
      'data/recordings/scrumMeetingDemo.mp4':['data/scrum_asr.json','data/sessions/scrum-demo/asr.json','data/sessions/scrum-demo/session.json'],
      'data/sessions/scrum-demo/media.mp4':['data/scrum_asr.json'],
      'data/sessions/scrum-demo/audio_16k_mono.wav':['data/scrum_asr.json'],
      'data/local-real/agm-30.wav':['data/local-real/preflight.json'],
      'fixtures/group_discussion_3min.mp4':['data/sessions/group-discussion-pipeline-check/asr.json (0–180s)'],
      'data/sessions/group-discussion-pipeline-check/media.mp4':['data/sessions/group-discussion-pipeline-check/asr.json'],
      'data/recordings/hinglish_AGM_call.wav':['data/ground_truth/agm_ground_truth.json','data/local-real/preflight.json (870–900s excerpt only)'],
      'data/recordings/GroupDiscussionAuio.mp4':['data/transcripts/transcript.txt','data/transcripts/transcript (1).txt'],
      'data/outputs/prior_output/speech_test_10s.wav':['data/outputs/prior_output/transcript_qwen3_1.7b.txt','data/outputs/prior_output/transcript_qwen3_0.6b.txt','data/outputs/prior_output/transcript_whisper_small_bn.txt (pairing unverified)'],
      'data/outputs/prior_output/bengali_meeting_audio.wav':['data/ground_truth/bengali_ground_truth.json','data/outputs/prior_output/full_transcript.txt (pairing unverified)'],
    }
    for entry in entries:
        if entry['kind']=='media':
            refs=associations.get(entry['path'],[])
            entry['transcripts']=refs
            if entry['path'] in {'data/recordings/scrumMeetingDemo.mp4','data/sessions/scrum-demo/media.mp4','data/sessions/scrum-demo/audio_16k_mono.wav','data/local-real/agm-30.wav','data/recordings/GroupDiscussionAuio.mp4'}:
                entry['verdict']='REAL MODEL OUTPUT';entry['evidence']='Playable media with associated proven stored output; duplicate representations loaded only once.'
            else:
                entry['verdict'],entry['evidence']=verdict(Path(entry['path']))
                if not refs:entry['evidence']+=' No associated real transcript located.'
    tracked=[]
    for name in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines():
        if Path(name).suffix.lower() in MEDIA:tracked.append({'path':name,'bytes':(ROOT/name).stat().st_size})
    result={'entries':sorted(entries,key=lambda e:e['path']),'tracked_media':tracked,'missing_directories':[p for p in ('output','data/archive') if not (ROOT/p).exists()],'visibility':'Not queried; owner will check GitHub visibility.'}
    output=ROOT/'data/demo_inventory.json';output.parent.mkdir(exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Demo provenance inventory — 2026-10-06','','No ASR models or Groq calls were run for this inventory. Unknown dates remain unknown; filesystem mtimes are not treated as run dates. UNESTABLISHED is deliberately outside the three requested positive verdicts: evidence is insufficient to honestly classify these artifacts.','','| File | Duration (s) | Transcript(s) / type | Provenance verdict | Evidence |','|---|---:|---|---|---|']
    for entry in result['entries']:
        detail=', '.join(entry.get('transcripts',[])) or entry['kind']
        lines.append(f"| {entry['path']} | {entry['duration_seconds']} | {detail} | {entry['verdict']} | {entry['evidence']} |")
    lines+=['','## Tracked media (git ls-files)','','| File | Bytes |','|---|---:|']+[f"| {t['path']} | {t['bytes']} |" for t in tracked]
    lines+=['','GitHub visibility was not queried; owner will check it. Media remain on local disk; no new media added to Git.','Root output/ does not exist; historical artifacts are under data/outputs/prior_output/.']
    (ROOT/'docs/DEMO_INVENTORY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(entries),'tracked_media':tracked},indent=2))


if __name__=='__main__':main()
