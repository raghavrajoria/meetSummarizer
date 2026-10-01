"""Offline regression test: mocked LLM, no network. Encodes the ground-truth checks from the 30 Sep handoff (section 5)."""
import json, tempfile, os, re
from indicmeet import summary as m

ATT = ["Shashank", "Neha", "Deepika", "KT", "Sindhu"]
AL = {"KT": ["Kitty", "Kirti", "Keith", "Keithy"], "Sindhu": ["Sendu"]}
T = [
 (0,  "S05", "Neha, are you blocked on anything?"),
 (1,  "S07", "I will work on the database creation of the login page."),
 (2,  "S05", "Thanks, Neha."),
 (3,  "S06", "I will try to investigate from my end."),
 (4,  "S05", "Thank you, Deepika."),
 (5,  "S00", "So, Shashank, I have a problem with the email validation regex."),
 (6,  "S08", "Yes, Sindhu, I can help you with that. I have done that in LastPrint itself."),
 (7,  "S00", "Thank you, KT."),
 (8,  "S05", "I'll work with Deepika on the dashboard integration and get back by end of the day."),
 (9,  "S06", "Yes, sure, Shashank."),
 (10, "S04", "The coffee machines are not working, I'll work on that soon."),
]
ASR = [{"idx": i, "start": i * 10.0, "end": i * 10.0 + 8, "speaker": f"SPEAKER_{s[1:]}", "text": t, "quality": "accepted"} for i, s, t in T]

calls = {"notes": 0, "act": 0, "rec": 0, "ov": 0}
def fake_llm(system, user, max_out, key, warn):
    if system.startswith("Extract notes"):
        calls["notes"] += 1
        return {"key_discussion": [
            {"point": "Shashank asked for help with email validation regex", "ev": ["5"]},
            {"point": "Neha is working on database creation for the login page", "ev": [1, 2]}],
            "decisions": [], "follow_ups": [], "questions": [], "concerns": []}, "ok"
    if system.startswith("Find every explicit"):
        calls["act"] += 1
        assert "CANDIDATES" in user
        return {"action_items": [
            {"task": "Work on database creation of the login page", "owner": None, "due": None, "ev": [1]},
            {"task": "Investigate the issue from own end", "owner": None, "due": None, "ev": [3]},
            {"task": "Work with Deepika on the dashboard integration and report back", "owner": "Shashank", "due": "end of the day", "ev": [8]},
            {"task": "Work on the coffee machines that are not working", "owner": None, "due": None, "ev": [10]}],
            "rejected": []}, "ok"           # deliberately misses KT (seg 6)
    if system.startswith("Each CANDIDATE"):
        calls["rec"] += 1
        assert "CANDIDATE 6" in user
        return {"action_items": [{"task": "Help Sindhu with the email validation regex", "owner": None, "due": None, "ev": [6]}], "rejected": []}, "ok"
    if system.startswith('Return ONLY JSON {"overview"'):
        calls["ov"] += 1
        assert "Shashank asked" not in user, "vocative item leaked into overview input"
        return {"overview": "The team gave standup updates and agreed follow-up work."}, "ok"
    raise AssertionError("unexpected prompt")

real_call_llm = m.call_llm
try:
    m.call_llm = fake_llm
    r = m.summarize(ASR, attendees=ATT, aliases=AL, api_key="x", log=lambda *a: None)
finally:
    m.call_llm = real_call_llm

assert r["speaker_hints"] == {"S07": "Neha", "S06": "Deepika", "S05": "Shashank", "S08": "KT", "S00": "Sindhu"}, r["speaker_hints"]
own = {a["task"].split()[0] + " " + a["task"].split()[1]: (a["owner"], a["owner_basis"]) for a in r["action_items"]}
by = {a["ev"][0]: a for a in r["action_items"]}
assert by[1]["owner"] == "Neha" and by[3]["owner"] == "Deepika", "reply owners"
assert by[8]["owner"] == "Shashank" and by[8]["owner_basis"] == "addressed in reply by another speaker", "REPLY lookahead fix ('Yes, sure, Shashank')"
assert by[8]["due"] == "end of the day"
assert by[6]["owner"] == "KT", "KT recovered via candidates-first + 2-letter name reply"
assert by[10]["owner"] is None
assert calls["rec"] == 1 and not r["uncovered_candidates"]
voc = [k for k in r["key_discussion"] if k["point"].startswith("Shashank asked")][0]
assert any(u.startswith("name_is_vocative:Shashank") for u in voc["unverified"]), voc
neha = [k for k in r["key_discussion"] if k["point"].startswith("Neha")][0]
assert "unverified" not in neha, neha
assert calls["ov"] == 1 and r["overview"]
assert not re.search(r"\bS\d{2}\b", json.dumps({k: v for k, v in r.items() if k != "speaker_hints"}))

# ---- 429 handling + cache (fake requests.post, fake sleep) ----
class R:
    def __init__(s, code, text="", js=None): s.status_code, s.text, s._js = code, text, js
    def json(s): return s._js
seq = [R(429, '{"error":{"message":"Rate limit reached ... Please try again in 2.5s."}}'),
       R(200, js={"choices": [{"finish_reason": "stop", "message": {"content": '{"a": 1}'}}]})]
posts, slept = [], []
real_post, real_sleep, real_cache_dir = m.requests.post, m.time.sleep, m.CACHE_DIR
real_cache_stats, real_window = m.CACHE_STATS.copy(), m._window[:]
try:
    m.requests.post = lambda *a, **k: (posts.append(1), seq.pop(0))[1]
    m.time.sleep = lambda s: slept.append(s)
    m.CACHE_DIR = tempfile.mkdtemp()
    m.call_llm = real_call_llm
    warns = []
    obj, why = m.call_llm("sys", "usr", 100, "k", warns.append)
    assert obj == {"a": 1} and why == "ok" and len(posts) == 2 and abs(slept[0] - 3.5) < 1e-6, (obj, why, posts, slept)
    assert any("429" in w for w in warns), "real 429 message must be shown"
    obj2, _ = m.call_llm("sys", "usr", 100, "k", warns.append)
    assert obj2 == {"a": 1} and len(posts) == 2 and m.CACHE_STATS["hit"] >= 1, "second identical call must hit the cache"
    seq[:] = [R(429, "Please try again in 5m10.2s.")]
    obj3, why3 = m.call_llm("sys2", "usr", 100, "k", warns.append)
    assert obj3 is None and why3 == "ratelimit"
    assert abs(m._retry_after("try again in 850ms") - 1.85) < 1e-6 and abs(m._retry_after("try again in 1h2m3s") - 3724) < 1e-6
    print("ALL OFFLINE CHECKS PASSED")
finally:
    m.requests.post, m.time.sleep, m.CACHE_DIR = real_post, real_sleep, real_cache_dir
    m.call_llm = real_call_llm
    m.CACHE_STATS.clear()
    m.CACHE_STATS.update(real_cache_stats)
    m._window[:] = real_window


def test_summary_regression_completed():
    # The detailed regression assertions above run offline during collection.
    assert r["overview"]
    assert calls["rec"] == 1


def test_unresolved_language_review_segments_are_excluded(monkeypatch):
    prompts = []

    def fake_llm(system, user, max_out, key, warn):
        prompts.append(user)
        if system.startswith("Extract notes"):
            return ({"key_discussion": [], "decisions": [], "follow_ups": [], "questions": [], "concerns": []}, "ok")
        if system.startswith("Find every explicit"):
            return ({"action_items": [], "rejected": []}, "ok")
        if system.startswith('Return ONLY JSON {"overview"'):
            return ({"overview": "The group discussed the project."}, "ok")
        raise AssertionError("unexpected summary prompt")

    monkeypatch.setattr(m, "call_llm", fake_llm)
    suspicious = ["veces", "Absolutamente.", "Sin duda, sin duda.", "Non, c'est jamais.",
                  "Iya, lihat dulu. Oke.", "Tchau, doutora Ivana."]
    asr = [{"idx": 0, "start": 1, "end": 3, "speaker": "SPEAKER_01", "text": "We discussed the project plan.",
            "quality": "accepted", "method": "whisper"}]
    asr.extend({"idx": i + 1, "start": 5 + i, "end": 5.5 + i, "speaker": "SPEAKER_02", "text": text,
                "quality": "review", "method": "whisper_latin_unresolved"} for i, text in enumerate(suspicious))
    result = m.summarize(asr, attendees=[], aliases={}, api_key="test-only", log=lambda *args: None)
    assert result["stats"]["segments"] == 1
    assert any("excluded 6 unresolved-language" in warning for warning in result["warnings"])
    assert all(text not in "\n".join(prompts) for text in suspicious)
