"""Focused pipeline contract tests that do not require GPU models or network access."""

from pathlib import Path

from indicmeet import pipeline


def test_attendees_and_aliases_are_read_and_passed_to_summary(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    attendees_path = repo / "fixtures" / "attendees_scrum.json"
    attendees, aliases = pipeline._attendee_data(attendees_path)
    assert attendees == ["Shashank", "Neha", "Deepika", "KT", "Manoj", "Sindhu", "Sanam"]
    assert aliases == {"KT": ["Kitty", "Kirti", "Keith", "Keithy"], "Sindhu": ["Sendu"]}

    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    captured = {}

    def fake_summarize(asr, got_attendees, got_aliases):
        captured.update(attendees=got_attendees, aliases=got_aliases)
        return {"overview": "offline test"}

    from indicmeet import summary

    monkeypatch.setattr(summary, "summarize", fake_summarize)
    pipeline._summary_stage([], attendees, aliases, None, tmp_path, False)
    assert captured == {"attendees": attendees, "aliases": aliases}
