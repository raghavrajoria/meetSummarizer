from indicmeet.asr import IndicMeetASR


def _write_diarization(path):
    path.write_text("start,end,speaker\n0,10,SPEAKER_00\n2,5,SPEAKER_00\n", encoding="utf-8")


def test_build_turns_preserves_furthest_end_for_overlapping_same_speaker(tmp_path):
    source = tmp_path / "diarization.csv"
    _write_diarization(source)

    turns = IndicMeetASR.build_turns(source)

    assert turns == [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}]


def test_build_turns_clamps_to_audio_duration(tmp_path):
    source = tmp_path / "diarization.csv"
    _write_diarization(source)

    turns = IndicMeetASR.build_turns(source, audio_duration=8.0)

    assert turns == [{"start": 0.0, "end": 8.0, "speaker": "SPEAKER_00"}]
