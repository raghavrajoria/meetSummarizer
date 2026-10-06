import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from asr_service import memory
from asr_service.model import GpuModel
from scripts.clip_turns import crop_turns
from scripts.real_pipeline_check import report, soft_scores


def test_memory_guard(monkeypatch):
    monkeypatch.setattr(memory.psutil,'virtual_memory',lambda:SimpleNamespace(available=2*2**30))
    with pytest.raises(memory.MemorySafetyError,match='RAM safety abort'):memory.check_memory(3)
    assert memory.check_memory(1)==2*2**30
    with pytest.raises(ValueError):memory.check_memory(-1)


def test_cpu_refuses_missing_turns(monkeypatch,tmp_path):
    monkeypatch.setenv('ASR_DEVICE','cpu');monkeypatch.setenv('ASR_MODEL_VERSION','cpu-test')
    with pytest.raises(ValueError,match='requires request turns'):GpuModel().transcribe(tmp_path/'audio.wav',None,lambda *a:None)


def test_cpu_isolated_fake_model(monkeypatch,tmp_path):
    monkeypatch.setenv('ASR_DEVICE','cpu');monkeypatch.setenv('ASR_MODEL_VERSION','cpu-test')
    monkeypatch.setattr(memory,'check_memory',lambda:10*2**30)
    def spawn(command):
        output=Path(command[command.index('--output')+1])
        output.write_text(json.dumps({'rows':[{'start':0,'end':1,'speaker':'S0','text':'hello','lang':'en','quality':'accepted'}],'stage_timings':{'whisper':1}}),encoding='utf-8')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('subprocess.Popen',spawn)
    monkeypatch.setattr(memory,'supervise',lambda *a:{'exit_code':0,'peak_process_tree_rss_bytes':123})
    model=GpuModel();rows=model.transcribe(tmp_path/'audio.wav',[{'start':0,'end':1,'speaker':'S0'}],lambda *a:None)
    assert rows[0]['asr']['model_version']=='cpu-test'
    assert model.last_metrics['stage_timings']=={'whisper':1}


def test_clip_relative_turns():
    assert crop_turns([{'start':860,'end':880,'speaker':'S0'},{'start':895,'end':910,'speaker':'S1'}],870,30)==[{'start':0,'end':10,'speaker':'S0'},{'start':25,'end':30,'speaker':'S1'}]


def test_soft_evaluation_no_overlap_is_not_verified():
    assert soft_scores([],{'segments':[{'timestamp':'00:04','text':'hello'}]},870,30)['WER'].startswith('NOT VERIFIED')


def test_remote_turn_payload_strips_private_csv_index(monkeypatch,tmp_path):
    from indicmeet.remote_asr import RemoteAsrProvider
    from indicmeet.contract import canonicalize
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fake')
    def post(url,**kwargs):
        assert json.loads(kwargs['data']['turns'])==[{'start':0,'end':1,'speaker':'S0'}]
        return SimpleNamespace(status_code=202,json=lambda:{'job_id':'job1'})
    rows=canonicalize([{'start':0,'end':1,'speaker':'S0','text':'hello','lang':'en','quality':'accepted'}])
    monkeypatch.setattr('indicmeet.remote_asr.requests.post',post)
    monkeypatch.setattr('indicmeet.remote_asr.requests.get',lambda *a,**k:SimpleNamespace(status_code=200,json=lambda:{'status':'done','progress':100,'result':rows}))
    assert RemoteAsrProvider('http://localhost:9002',token='fake-test').transcribe(audio,turns=[{'start':0,'end':1,'speaker':'S0','idx':99}])


def test_report_citation_validation():
    evidence=report({'transcript':[],'intelligence':{'actions':[{'source_segment_ids':['missing']}]}},{'status':'done'},30,60,None,{})
    assert evidence['real_time_factor']==2
    assert evidence['unresolved_citations']==['missing']


def test_watchdog_kills_only_its_child_on_low_ram(monkeypatch):
    killed=[]
    class Process:
        pid=123
        returncode=None
        def poll(self):return None
        def wait(self,timeout):self.returncode=-1
    monkeypatch.setattr(memory,'check_memory',lambda: (_ for _ in ()).throw(memory.MemorySafetyError('low RAM')))
    class Root:
        def children(self,recursive):return [SimpleNamespace(kill=lambda:killed.append('child'))]
        def kill(self):killed.append('root')
    monkeypatch.setattr(memory.psutil,'Process',lambda pid:Root())
    with pytest.raises(memory.MemorySafetyError) as error: memory.supervise(Process(),lambda *a:None)
    assert killed==['child','root']
    assert error.value.metrics['memory_guard_aborted']


def test_groq_cap_stops_before_http(monkeypatch):
    from indicmeet import summary
    monkeypatch.setenv('GROQ_MAX_LIVE_CALLS','1')
    monkeypatch.setattr(summary,'_live_attempts',1)
    monkeypatch.setattr(summary.requests,'post',lambda *a,**k:pytest.fail('HTTP request must not happen'))
    with pytest.raises(RuntimeError,match='cap reached'):summary._call_llm_raw('system','user',1,'private',lambda *a:None)


def test_cpu_passes_load_sequentially_and_preserve_routing(monkeypatch):
    import sys
    from array import array
    from asr_service.cpu import CpuASR
    from indicmeet.asr import IndicMeetASR
    events=[]
    class Chunk:
        def __init__(self,values):self.values=values;self.ndim=1;self.shape=(len(values),)
        def __len__(self):return len(self.values)
        def __getitem__(self,key):return Chunk(self.values[key])
        def numpy(self):return self
        def tobytes(self):return array('f',self.values).tobytes()
        def tolist(self):return self.values
        def detach(self):return self
        def cpu(self):return self
        def astype(self,dtype):return self
    class Whisper:
        def __init__(self,*a,**kwargs):
            events.append('load_whisper');assert kwargs['device']=='cpu' and kwargs['compute_type']=='int8'
        def transcribe(self,*a,**k):return [SimpleNamespace(text='hello')],SimpleNamespace(language='en',language_probability=.99)
        def __del__(self):events.append('unload_whisper')
    class Loaded:
        def __init__(self,name):self.name=name;events.append('load_'+name)
        def to(self,device):assert device=='cpu';return self
        def eval(self):return self
        def __del__(self):events.append('unload_'+self.name)
    class Factory:
        @staticmethod
        def from_pretrained(name,**k):return Loaded(name)
    monkeypatch.setitem(sys.modules,'soundfile',SimpleNamespace(read=lambda *a,**k:(Chunk([0.1]*16000),16000)))
    monkeypatch.setitem(sys.modules,'faster_whisper',SimpleNamespace(WhisperModel=Whisper))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoFeatureExtractor=Factory,Wav2Vec2ForSequenceClassification=Factory,AutoModel=Factory))
    monkeypatch.setattr(memory,'check_memory',lambda *a:10*2**30)
    monkeypatch.setattr('asr_service.cpu.check_memory',lambda *a:10*2**30)
    monkeypatch.setattr(IndicMeetASR,'_mms_lid',lambda *a:('en',.99))
    engine=CpuASR.__new__(CpuASR)
    engine.settings=SimpleNamespace(mms_lid_model='mms',indicconformer_model='indic',whisper_beam_size=5)
    engine._torch=SimpleNamespace(from_numpy=lambda a:a)
    engine._numpy=SimpleNamespace(float32='float32')
    engine.device='cpu';engine.whisper=engine;engine.whisper_cache={};engine.mms_cache={};engine.stage_timings={}
    rows=engine.transcribe_turns('fixture.wav',[{'start':0,'end':1,'speaker':'S0'}])
    assert rows[0]['text']=='hello' and rows[0]['method']=='whisper_english'
    assert events.index('unload_whisper')<events.index('load_mms')
    assert events.index('unload_mms')<events.index('load_indic')
