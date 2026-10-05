from pathlib import Path
from types import SimpleNamespace
from indicmeet.speakers import LiveKitEvents,Turn,name_clusters,pyannote_turns
ROOT=Path(__file__).parents[1]
def test_pyannote_both_output_apis():
    class Annotation:
        def itertracks(self,yield_label):
            assert yield_label
            return iter([(SimpleNamespace(start=1,end=2),None,"SPEAKER_01")])
    expected=[Turn(1.,2.,"SPEAKER_01")]
    assert pyannote_turns(Annotation())==expected
    assert pyannote_turns(SimpleNamespace(speaker_diarization=Annotation()))==expected
def test_livekit_mic_is_not_speech_and_overlap_names_clusters():
    source=LiveKitEvents(ROOT/"fixtures/livekit/session.json",ROOT/"fixtures/livekit/events.jsonl")
    assert source.turns()==[Turn(0.,5.,"alice"),Turn(5.,9.,"bob")]
    named=name_clusters([{"start":0.,"end":4.,"speaker":"SPEAKER_00"},{"start":6.,"end":8.,"speaker":"SPEAKER_01"}],source)
    assert [s["speaker_name"] for s in named]==["Alice","Bob"]
def test_livekit_completion_queues_once(api_env,api_client):
    prefix="meetings/fixture-livekit"
    root=api_env.store.path(prefix);root.mkdir(parents=True)
    for name in ("session.json","events.jsonl"):
        (root/name).write_bytes((ROOT/"fixtures/livekit"/name).read_bytes())
    (root/"recording.mp4").write_bytes(b"fixture")
    body={"session_id":"fixture-livekit","bucket":"indicmeet","prefix":prefix+"/","recording_key":prefix+"/recording.mp4","recording_type":"video","has_per_participant_tracks":False}
    first=api_client.post("/livekit/completed",json=body)
    assert first.status_code==202 and first.json()["job"]["status"]=="queued"
    assert api_client.post("/livekit/completed",json=body).json()["job"]["id"]==first.json()["job"]["id"]


def test_separate_tracks_merge_with_identity_and_common_clock(tmp_path):
    from indicmeet.speakers import transcribe_tracks
    calls=[]
    class Provider:
        def transcribe(self,path,turns):
            calls.append(turns[0]["speaker"])
            return [{"idx":0,"start":2.,"end":3.,"speaker":"cluster","text":"source","lang":"en","quality":"accepted"}]
    def extract(source,destination):return destination
    rows=transcribe_tracks({"bob":tmp_path/"bob.ogg","alice":tmp_path/"alice.ogg"},Provider(),extract,tmp_path,10,{"alice":"Alice","bob":"Bob"})
    assert calls==["alice","bob"]
    assert [r["speaker_name"] for r in rows]==["Alice","Bob"]
    assert len({r["segment_id"] for r in rows})==2
    assert all(r["start"]==2 and r["text_native"]=="source" for r in rows)
