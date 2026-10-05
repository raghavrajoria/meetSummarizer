import json
from pathlib import Path
import pytest
from indicmeet.contract import canonicalize, Segment, review_gate
from indicmeet.pipeline import _session_payload

ROOT = Path(__file__).resolve().parents[1]

def test_stored_real_subset_and_optional_confidence():
    raw = json.loads((ROOT / "fixtures/demo_asr.json").read_text(encoding="utf-8"))
    segments = canonicalize(raw)
    assert all(Segment.model_validate(s) for s in segments)
    assert segments[0]["text_native"] == raw[0]["text"]
    assert segments[0]["asr"]["confidence"] is None

def test_ids_survive_insertion_and_reject_duplicate():
    raw = json.loads((ROOT / "fixtures/demo_asr.json").read_text(encoding="utf-8"))
    before = canonicalize(raw)
    added = {**raw[0], "speaker": "SPEAKER_NEW"}
    after = canonicalize([added] + raw)
    assert before[0]["segment_id"] == after[1]["segment_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        canonicalize([raw[0], raw[0]])

def test_exact_idx_time_collision_and_unknown_claim():
    raw = [{"idx": 100, "start": 0., "end": 1., "speaker": "SPEAKER_00", "text": "earlier", "lang": "en", "quality": "accepted"},
           {"idx": 0, "start": 10., "end": 11., "speaker": "SPEAKER_01", "text": "intended", "lang": "en", "quality": "accepted"}]
    canonical = canonicalize(raw)
    summary = {"key_discussion": [{"point": "intended", "source_segment_ids": [canonical[1]["segment_id"]]},
                                   {"point": "invented", "source_segment_ids": ["missing"]}]}
    payload = _session_payload("test", "test", "audio", "clip.wav", raw, summary, None)
    items = payload["intelligence"]["keyDiscussion"]
    assert len(items) == 1
    assert items[0]["evidence"][0]["segmentId"] == canonical[1]["segment_id"]
    assert items[0]["evidence"][0]["t"] == 10.

def test_strict_gate_and_rejected_exclusion():
    raw = json.loads((ROOT / "fixtures/demo_asr.json").read_text(encoding="utf-8"))
    segments = canonicalize(raw)
    segments[0]["quality"] = "review"
    segments[1]["quality"] = "rejected"
    assert segments[0] not in review_gate(segments, True)
    assert segments[1] not in review_gate(segments, False)
    assert segments[0] in review_gate(segments, False)


def test_real_fixture_collision_ids_survive_insertion():
    raw=json.loads((ROOT/"fixtures/session.json").read_text(encoding="utf-8"))["segments"]
    before=canonicalize(raw)
    after=canonicalize([{**raw[0],"speaker":"INSERTED"}]+raw)
    assert [s["segment_id"] for s in before]==[s["segment_id"] for s in after[1:]]
    assert len({s["segment_id"] for s in before})==len(raw)


def test_approved_mixed_unknown_markers_preserve_labels_and_force_review():
    raw=[{"start":0.,"end":1.,"speaker":"A","text":"mixed text","language":"mixed","quality":"accepted"}, {"start":2.,"end":3.,"speaker":"B","text":"unknown text","quality":"accepted"}]
    rows=canonicalize(raw)
    assert [s["language"] for s in rows]==["mul","und"]
    assert all(s["quality"]=="review" for s in rows)
    assert rows[0]["asr"]["source_language"]=="mixed"
    assert raw[0]["language"]=="mixed" and raw[0]["quality"]=="accepted"
    with pytest.raises(ValueError):Segment.model_validate({**rows[0],"quality":"accepted"})
    with pytest.raises(ValueError):Segment.model_validate({**rows[0],"language":"not-an-iso-code"})


def test_real_same_start_rows_keep_unique_ids_after_speaker_alignment():
    from indicmeet.contract import reidentify
    raw=json.loads((ROOT/"fixtures/session.json").read_text(encoding="utf-8"))["segments"]
    rows=canonicalize(raw)
    for row in rows:row["speaker"]="ALIGNED"
    aligned=reidentify(rows)
    assert len({r["segment_id"] for r in aligned})==len(rows)
    assert [r["text_native"] for r in aligned]==[r["text_native"] for r in rows]
    assert reidentify(aligned)==aligned

def test_sparse_legacy_rows_import_through_provider(tmp_path):
    from indicmeet.asr_provider import ImportAsrProvider
    raw=json.loads((ROOT/"fixtures/session.json").read_text(encoding="utf-8"))["segments"]
    file=tmp_path/"legacy.json";file.write_text(json.dumps(raw,ensure_ascii=False),encoding="utf-8")
    rows=ImportAsrProvider().transcribe(tmp_path/"unused.wav",asr_json_path=file)
    assert len(rows)==len(raw)
    assert all(Segment.model_validate(row) for row in rows)
