import json
from pathlib import Path
from indicmeet.contract import canonicalize
from indicmeet.demo import FakeGroq
from indicmeet.enrichment import enrich
from indicmeet.summary import summarize
def test_rejected_and_strict_review_never_sent_to_fake_groq():
    rows=canonicalize(json.loads((Path(__file__).parents[1]/"fixtures/demo_asr.json").read_text(encoding="utf-8")))
    rows[1]["quality"]="review";rows[2]["quality"]="rejected"
    client=FakeGroq(); enriched=enrich(rows,client=client,strict=True)
    summary=summarize(enriched,api_key="test",strict_review=True,client=client,log=lambda *a:None)
    prompts="\n".join(p for _,p in client.prompts)
    assert rows[1]["text_native"] not in prompts and rows[2]["text_native"] not in prompts
    assert summary["overview_claims"]
    assert all(c["source_segment_ids"]==[rows[0]["segment_id"]] for c in summary["overview_claims"])
    assert [s["text_native"] for s in enriched]==[s["text_native"] for s in rows]


def test_skip_review_cli_sets_strict_before_pipeline(monkeypatch):
    from indicmeet import pipeline
    from indicmeet.settings import get_settings
    monkeypatch.setenv("STRICT_REVIEW","false")
    exercised=[]
    monkeypatch.setattr(pipeline,"run",lambda args:exercised.append(get_settings().strict_review))
    assert pipeline.main(["--media","unused.wav","--session-id","fixture","--skip-review"])==0
    assert exercised==[True]


def test_enrichment_recovers_omitted_ids_without_accepting_invented_ids():
    rows=canonicalize(json.loads((Path(__file__).parents[1]/"fixtures/demo_asr.json").read_text(encoding="utf-8")))
    calls=[]
    def client(system,user,*args):
        batch=json.loads(user);calls.append(len(batch))
        if len(calls)==1:return {"segments":[{"segment_id":"invented","text_roman":"bad","text_english":"bad"}]},"ok"
        return {"segments":[{"segment_id":s["segment_id"],"text_roman":"roman","text_english":"english"} for s in batch]},"ok"
    result=enrich(rows,client=client)
    assert calls==[3,1,1,1]
    assert all(s["text_english"]=="english" and s["text_native"]==rows[i]["text_native"] for i,s in enumerate(result))


def test_empty_and_repetition_quality_gate_without_loading_models():
    from indicmeet.asr import IndicMeetASR
    engine=IndicMeetASR.__new__(IndicMeetASR)
    empty=engine._finalize("", "en", "whisper", 1., [])
    repeated=engine._finalize("yes "*50, "en", "whisper", 2., [])
    assert empty["quality"]=="rejected"
    assert repeated["quality"] in {"review","rejected"}
