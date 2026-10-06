import json
from pathlib import Path
from indicmeet.contract import canonicalize, Segment
from indicmeet.summary import summarize, finish_action, make_canon
from indicmeet.demo import FakeGroq
from indicmeet.pipeline import _session_payload


def test_demo_without_attendees_does_not_invent_names():
    rows = canonicalize(json.loads(Path('fixtures/demo_asr.json').read_text(encoding='utf-8')))
    result = summarize(rows, client=FakeGroq(), api_key='demo', log=lambda *a: None)
    session = _session_payload('audit', 'Audit', 'audio', 'demo.wav', rows, result, None)
    assert all(s['speaker_name'] is None and s['speaker_name_source'] == 'none' for s in session['transcript'])
    assert not result['speaker_hints'] and not result['action_items']
    assert all(s['name'] == s['id'] for s in session['speakers'])


def test_unproven_imported_name_is_marked_inferred():
    row = {'start': 0, 'end': 1, 'speaker': 'SPEAKER_00', 'speaker_name': 'Ravi', 'text': 'Hello', 'lang': 'en', 'quality': 'accepted'}
    segment = canonicalize([row])[0]
    assert segment['speaker_name_source'] == 'inferred'
    segment['speaker_name_source'] = 'attendee_list'
    assert Segment.model_validate(segment).speaker_name_source == 'attendee_list'


def test_owner_name_reply_has_its_own_evidence():
    segs = [{'idx': 0, 'spk': 'S0', 'text': 'I will send the report tomorrow.'}, {'idx': 1, 'spk': 'S1', 'text': 'Thanks, Ravi.'}]
    action = {'task': 'Send the report tomorrow', 'owner': 'Ravi', 'due': None, 'ev': [0]}
    finish_action(action, segs, {0: 0, 1: 1}, make_canon(None, None), False, lambda *a: None)
    assert action['owner'] == 'Ravi'
    assert action['owner_name_source'] == 'inferred'
    assert action['owner_name_ev'] == [1]


def test_owner_evidence_and_provenance_reach_ui_payload():
    rows = canonicalize([{'start': 0, 'end': 1, 'speaker': 'S0', 'text': 'I will send the report.', 'lang': 'en', 'quality': 'accepted'}, {'start': 1, 'end': 2, 'speaker': 'S1', 'text': 'Thanks, Ravi.', 'lang': 'en', 'quality': 'accepted'}])
    def client(system, user, *args):
        if system.startswith('Find every explicit'):
            return {'action_items': [{'task': 'Send the report tomorrow', 'owner': 'Ravi', 'due': None, 'ev': [0]}]}, 'ok'
        return {}, 'ok'
    result = summarize(rows, client=client, api_key='demo', log=lambda *a: None)
    action = result['action_items'][0]
    assert action['owner_source_segment_ids'] == [rows[1]['segment_id']]
    assert rows[1]['segment_id'] in action['source_segment_ids']
    payload = _session_payload('audit', 'Audit', 'audio', 'demo.wav', rows, result, None)
    assert payload['intelligence']['actionItems'][0]['owner_name_source'] == 'inferred'
