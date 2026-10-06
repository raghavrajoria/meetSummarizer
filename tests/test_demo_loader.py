import pytest
from scripts.load_demo_meetings import require_demo, admitted_artifact, build_payload, SPECS
from scripts.inventory_demo_assets import ROOT

@pytest.mark.parametrize("env",[{"DEMO":"false","APP_ENV":"demo"},{"DEMO":"true","APP_ENV":"production"},{"DEMO":"true","APP_ENV":"demo","DEPLOY_PROFILE":"local-real"}])
def test_refuses_other_profiles(env):
    with pytest.raises(ValueError): require_demo({"demo_mode":True},env)

def test_refuses_fixture_and_reference():
    for name in ["fixtures/demo_asr.json","data/ground_truth/agm_ground_truth.json"]:
        with pytest.raises(ValueError): admitted_artifact(ROOT/name)

def test_real_payloads_have_honest_provenance_and_no_fake_summaries():
    payloads={s['id']:build_payload(s) for s in SPECS}
    assert len(payloads['real-scrum']['transcript'])==156
    assert 'owner-stated' in payloads['real-scrum']['provenance']['evidence']
    for key in ['real-agm-cpu-30','real-group-discussion-full']:
        assert payloads[key]['summary_status']=='not_generated'
        assert not payloads[key]['summary']
        assert not payloads[key]['intelligence']['actionItems']
    assert payloads['real-agm-cpu-30']['provenance']['evidence']=='30-second sample, single speaker label (no diarization)'
    assert len(payloads['real-group-discussion-full']['transcript'])==226
    assert build_payload(SPECS[0])['demo_asset_fingerprint']==build_payload(SPECS[0])['demo_asset_fingerprint']


def test_group_export_matches_owner_counts_and_excerpt():
    from collections import Counter
    import json
    from scripts.load_demo_meetings import read_group_export
    rows=read_group_export(ROOT/'data/transcripts/transcript.txt')
    assert Counter(r['quality'] for r in rows)=={'accepted':166,'review':56,'rejected':4}
    assert Counter(r['lang'] for r in rows)=={'en':136,'hi':84,'mr':2,'ru':1,'zh':1,'ko':1,'ur':1}
    excerpt=json.loads((ROOT/'data/sessions/group-discussion-pipeline-check/asr.json').read_text(encoding='utf-8'))
    assert len([r for r in rows if r['start']<180])==len(excerpt)==19
    for a,b in zip(rows,excerpt):
        assert all(a[k]==b[k] for k in ['start','speaker','lang','quality','text'])


def test_precomputed_profile_blocks_live_upload(api_env, monkeypatch):
    monkeypatch.setenv('DEMO','true')
    monkeypatch.setenv('DEMO_PRECOMPUTED_ONLY','true')
    response=api_env.client.post('/meetings',headers={'Authorization':'Bearer test-token'})
    assert response.status_code==409
    assert api_env.client.get('/config').json()['precomputed_only'] is True
    monkeypatch.setenv('DEMO','false')
    assert api_env.client.get('/config').json()['precomputed_only'] is False
