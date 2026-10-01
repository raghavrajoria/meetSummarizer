"""IndicMeet summary stage v3: ASR turns -> validated meeting summary. Self-contained, no notebook state.

v3 changes over v2:
  1. 429 handling: shows the real message, parses h/m/s/ms waits, gives up with reason "ratelimit" if the wait > MAX_WAIT_S.
  2. Disk cache keyed on (model, system, user, max_out): reruns cost no tokens. CACHE_DIR via env INDICMEET_LLM_CACHE.
  3. REPLY regex uses a lookahead so "Yes, sure, Shashank" yields both "sure" and "Shashank" (v2 swallowed the name).
  4. Vocative fix: speaker-name hints derived ONLY from "thanks, <attendee>" replies in the transcript are given to the model,
     and code flags any item whose name appears in evidence only as a vocative (addressee), or not at all.
  5. Candidates-first commitments: code finds commitment cues, the model must turn each into an item or reject it,
     uncovered candidates get a targeted recovery call, leftovers produce a warning.
  6. Name/number/pronoun guards apply to every list (key_discussion, decisions, follow_ups, questions, concerns, action_items).
  Short names (2 letters, e.g. KT) now work in reply detection (exact match only).
"""
import json, re, os, time, difflib, hashlib, requests
from collections import Counter, defaultdict
from .settings import get_settings

_settings = get_settings()
MODEL = _settings.groq_model
URL = _settings.groq_api_url
CACHE_DIR = str(_settings.llm_cache_dir)
CACHE_STATS = {"hit": 0, "api": 0}
MAX_WAIT_S = _settings.groq_max_wait_seconds
TEMPERATURE = _settings.groq_temperature
REQUEST_TIMEOUT_S = _settings.groq_request_timeout_seconds
SUMMARY_PROMPT_VERSION = 1
_window = []  # (timestamp, tokens) rolling 60s budget

def ntok(t): return len(t) // 3 + 1

def mmss(s):
    s = int(s); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

def _key():
    k = get_settings().groq_api_key
    if k: return k
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret("GROQ_API_KEY")

def _wait(need, cap=7000):
    while True:
        now = time.time(); _window[:] = [(t, k) for t, k in _window if now - t < 60]
        if not _window or sum(k for _, k in _window) + need <= cap: return
        time.sleep(max(1, 61 - (now - _window[0][0])))

def _retry_after(text):
    m = re.search(r"try again in\s*((?:\d+(?:\.\d+)?(?:ms|h|m|s)\s*)+)", text)
    if not m: return 10.0
    unit = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
    return sum(float(v) * unit[u] for v, u in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1))) + 1

def _call_llm_raw(system, user, max_out, key, warn):
    """One request -> (parsed JSON or None, reason). Never retries with a bigger budget."""
    need = ntok(system) + ntok(user) + max_out
    shown = False
    for _ in range(6):
        _wait(need)
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=REQUEST_TIMEOUT_S,
                json={"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                      "max_completion_tokens": max_out, "reasoning_effort": "low", "temperature": TEMPERATURE})
        except requests.RequestException as e:
            warn(f"network error: {e}"); time.sleep(3); continue
        if r.status_code == 429:
            if not shown: warn("429 from Groq: " + r.text[:200].replace("\n", " ")); shown = True
            wait = _retry_after(r.text)
            if wait > MAX_WAIT_S: warn(f"rate limit wait {wait:.0f}s > {MAX_WAIT_S}s, giving up on this call"); return None, "ratelimit"
            time.sleep(wait); continue
        _window.append((time.time(), need))
        if r.status_code != 200:
            warn(f"HTTP {r.status_code}: {r.text[:200]}"); return None, "http"
        ch = r.json()["choices"][0]
        if ch["finish_reason"] != "stop":
            warn(f"finish_reason={ch['finish_reason']}"); return None, "length"
        txt = ch["message"].get("content") or ""
        try: return json.loads(txt[txt.find("{"): txt.rfind("}") + 1]), "ok"
        except Exception: warn("unparseable JSON"); return None, "parse"
    warn("gave up after 6 attempts"); return None, "net"

def _cache_path(system, user, max_out):
    h = hashlib.sha256(json.dumps([SUMMARY_PROMPT_VERSION, MODEL, URL, TEMPERATURE, REQUEST_TIMEOUT_S,
                                  system, user, max_out],
                                  ensure_ascii=False).encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, h + ".json")

def call_llm(system, user, max_out, key, warn):
    """Cached wrapper. Only successful parsed results are cached."""
    p = _cache_path(system, user, max_out)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f: obj = json.load(f)
            CACHE_STATS["hit"] += 1; return obj, "ok"
        except Exception: pass
    obj, why = _call_llm_raw(system, user, max_out, key, warn)
    CACHE_STATS["api"] += 1
    if obj is not None:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False)
        except OSError as e: warn(f"cache write failed: {e}")
    return obj, why

BASE = ("The text is a meeting transcript with ASR errors. Use ONLY what is in the text; never invent. "
        "Do not add details that are not in the text: no numbers, durations or gender (never use he/she/his/her; use the person's name or 'they'). "
        "Speaker labels (S00...) are anonymous: never output a label, and never infer a person's name from a label unless SPEAKER NAME HINTS give it. "
        "A name inside a line is usually the person being ADDRESSED, not the speaker: the speaker of a line is only shown by its [id|time|label] prefix. "
        "Never write 'X asked/said/will' because X is named inside the line; use the name only if the hints or the text clearly make X the speaker or the doer. "
        "Every item needs 'ev' = a list of 1 to 3 segment IDs, the most relevant ones, copied from the [id|time|speaker] prefixes. "
        "Never list ranges and never more than 3 IDs.")
NOTES_SYS = ('Extract notes from this part of a meeting. Return ONLY JSON: {"key_discussion":[{"point":"","ev":[]}],'
    '"decisions":[{"decision":"","ev":[]}],"follow_ups":[{"item":"","ev":[]}],"questions":[{"q":"","ev":[]}],'
    '"concerns":[{"concern":"","ev":[]}]}. '
    'key_discussion: 3-6 specific points; if a speaker is teaching or explaining a concept, write what was taught, not something the team did. '
    'decisions: only explicit decisions, empty list if none. '
    'follow_ups: things explicitly deferred to later or to another forum ("let\'s discuss offline", "we\'ll revisit"), not work commitments. '
    'questions: real questions asked to someone in the meeting that need an answer, not example questions being taught. '
    'concerns: ONLY problems a participant states about their own or the team\'s work (blockers, impediments, missing help, low confidence). '
    'Not teaching content, not general advice, not things that might go wrong. Never write "potential", "risk that" or "could" unless the speaker said it. '
    'Write names as heard. ' + BASE)
ACT_SYS = ('Find every explicit commitment by a participant to do work after this part of the meeting. Return ONLY JSON: '
    '{"action_items":[{"task":"","owner":null,"due":null,"ev":[]}],"rejected":[{"id":0,"why":""}]}. '
    'Include "I will/I\'ll ...", "let me ...", "I can help you with ...", "we\'ll do it together", and accepted offers of help. '
    'Exclude status updates about past work, general advice, teaching content, and meeting logistics (sharing a screen, starting, greetings, thanks). '
    'CANDIDATES lists segment IDs that contain commitment phrases: for EACH candidate ID either create an action item whose ev includes that ID, '
    'or list the ID in "rejected" with a short reason. You may also add items for other segments. '
    'Every task must be self-contained: if the speaker says "I\'ll do that", replace "that" with the specific thing from the earlier line and list both IDs. '
    'owner: a personal name only if the text says who does it, else null (do not guess). due: only a time that is actually said, else null. '
    'Empty lists if none. ' + BASE)
REC_SYS = ('Each CANDIDATE segment (with surrounding lines) may contain a commitment to do work after the meeting. Return ONLY JSON: '
    '{"action_items":[{"task":"","owner":null,"due":null,"ev":[]}],"rejected":[{"id":0,"why":""}]}. '
    'For each candidate ID either create one action item (ev must include that ID) or list the ID in "rejected" with a reason. '
    'Task must be self-contained, owner only if the text says who does it else null, due only if actually said. ' + BASE)
OV_SYS = ('Return ONLY JSON {"overview":"3-4 sentences"}. Describe the meeting using only these validated notes. '
    'Say something was assigned only if it appears in action_items. If the notes are mostly explanations of a concept, say the recording includes teaching. '
    'Never mention speaker labels and never use he/she/his/her.')

FIELDS = {"key_discussion": "point", "decisions": "decision", "follow_ups": "item",
          "questions": "q", "concerns": "concern", "action_items": "task"}
LABEL = re.compile(r"\b(?:SPEAKER_\d+|S\d{2})\b")
PAREN = re.compile(r"\s*\(\s*(?:SPEAKER_\d+|S\d{2})\s*\)")
BAD_OWNER = re.compile(r"(?i)^(s\d+|speaker[_ ]?\d+|participants?|speaker|team|someone|unknown|everyone)$")
VAGUE = re.compile(r"(?i)^(do|does|did|handle|take care of|work on)\s+(that|this|it)\b|unspecified|\b(?:share|sharing)\s+(?:my|the)\s+screen\b")
_CUES = r"(?:thank you|thanks|sure|okay|ok|great|awesome|alright|yes|yeah)"
# lookahead: zero-width so "Yes, sure, Shashank" yields both "sure" and "Shashank"
REPLY = re.compile(rf"(?i)(?=\b{_CUES}(?:\s+so\s+much)?[,\s]+([A-Za-z]+))")
VOC_CUE = re.compile(r"(?i)\b(?:thanks?|thank you|sure|yes|yeah|okay|ok|hi|hey|hello|so|great)[,\s]+")
COMMIT = re.compile(r"(?i)\b(?:i'?ll|i will|i can help|i can do|i can take|let me|we'?ll|we will|i'?m going to|i am going to)\b")
GENDER = re.compile(r"\b(he|she|his|her|him|hers)\b", re.I)
NUMW = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()) if w != "one"}

def nums(t):
    t = t.lower()
    return set(re.findall(r"\d+", t)) | {NUMW[w] for w in re.findall(r"[a-z]+", t) if w in NUMW}

def specifics_check(text, ctx):
    bad = [f"number:{n}" for n in sorted(nums(text) - nums(ctx))]
    bad += [f"pronoun:{p.lower()}" for p in set(GENDER.findall(text)) if not re.search(rf"\b{p}\b", ctx, re.I)]
    return bad

def mk_chunks(segs, sline, cap=1500):
    out, cur, n = [], [], 0
    for s in segs:
        t = ntok(sline(s)) + 1
        if cur and n + t > cap: out.append(cur); cur, n = [], 0
        cur.append(s); n += t
    if cur: out.append(cur)
    return out

def make_canon(attendees, aliases):
    lut = {c.lower(): c for c in attendees or []}
    for c, al in (aliases or {}).items():
        lut[c.lower()] = c
        for a in al: lut[a.lower()] = c
    def canon(name):
        n = name.strip(); k = n.lower()
        if not lut: return n, False
        if k in lut: return lut[k], True
        if len(k) < 3: return n, False          # short words never fuzzy-match (KT etc. need an exact hit)
        best = max(lut, key=lambda x: difflib.SequenceMatcher(None, k, x).ratio())
        if difflib.SequenceMatcher(None, k, best).ratio() >= 0.7: return lut[best], True
        return n, False
    return canon

def make_alias_sub(aliases, seen):
    pairs = [(a, c) for c, al in (aliases or {}).items() for a in al if a.lower() != c.lower()]
    def sub(t):
        for a, c in pairs:
            t2 = re.sub(rf"\b{re.escape(a)}\b", c, t, flags=re.I)
            if t2 != t: seen.add((a, c)); t = t2
        return t
    return sub

def ev_ids(ev, valid):
    out = []
    for e in ev or []:
        m = re.match(r"\[?\s*(\d+)", str(e))
        if m and int(m.group(1)) in valid and int(m.group(1)) not in out: out.append(int(m.group(1)))
    return out

def norm(raw, field, valid, clean, warn, kind):
    out = []
    for it in raw:
        if not isinstance(it, dict) or not isinstance(it.get(field), str) or not it[field].strip(): continue
        ev = ev_ids(it.get("ev"), valid)
        if not ev: continue
        d = {k: v for k, v in it.items() if k not in (field, "ev")}
        d[field] = clean(it[field].strip()); d["ev"] = ev[:3]
        out.append(d)
    if len(out) < len(raw): warn(f"{kind}: dropped {len(raw) - len(out)} items (no text or no valid evidence)")
    return out

def dedupe(items, field, thr=0.75):
    out = []
    for it in items:
        for o in out:
            if difflib.SequenceMatcher(None, it[field].lower(), o[field].lower()).ratio() >= thr:
                o["ev"] = list(dict.fromkeys(o["ev"] + it["ev"]))[:3]; break
        else: out.append(it)
    return out

def ctx_text(ids, segs, pos, before, after):
    ps = sorted({p for e in ids for p in range(max(0, pos[e] - before), min(len(segs), pos[e] + after + 1))})
    return " ".join(segs[p]["text"] for p in ps)

# ---------- speaker hints and name checks (transcript-derived only, never audio) ----------
def speaker_hints(segs, canon, known):
    """label -> attendee name, only from 'thanks, <name>' style replies by the NEXT speaker. Conflicts/ties are dropped."""
    if not known: return {}
    votes = defaultdict(Counter)
    for p in range(1, len(segs)):
        a, b = segs[p - 1], segs[p]
        if a["spk"] == b["spk"]: continue
        for w in REPLY.findall(b["text"]):
            if len(w) < 2: continue
            name, ok = canon(w)
            if ok: votes[a["spk"]][name] += 1; break
    hints = {}
    for spk, c in votes.items():
        top = c.most_common(2)
        if len(top) == 1 or top[0][1] > top[1][1]: hints[spk] = top[0][0]
    return hints

def hints_line(hints):
    if not hints: return ""
    return ("SPEAKER NAME HINTS (derived from replies in this transcript, may be wrong): "
            + "; ".join(f"{k}={v}" for k, v in sorted(hints.items())) + "\n")

def _is_vocative(variant, text):
    v = re.escape(variant)
    return bool(re.search(rf"\b{v}\b\s*,", text, re.I) or re.search(rf"(?i)\b(?:thanks?|thank you|sure|yes|yeah|okay|ok|hi|hey|hello|so|great)[,\s]+{v}\b", text))

def name_check(text, ev, segs, pos, hints, variants):
    """Flag attendee names in item text that are absent from the evidence, or appear there only as a vocative (addressee)."""
    bad = []
    near = sorted({p for e in ev for p in range(max(0, pos[e] - 1), min(len(segs), pos[e] + 2))})
    ev_spk = {segs[pos[e]]["spk"] for e in ev}
    for name, vs in variants.items():
        if not re.search(rf"\b{re.escape(name)}\b", text, re.I): continue
        occ = [(p, v) for p in near for v in vs if re.search(rf"\b{re.escape(v)}\b", segs[p]["text"], re.I)]
        if not occ: bad.append(f"name_not_in_evidence:{name}"); continue
        if all(_is_vocative(v, segs[p]["text"]) for p, v in occ) and not any(hints.get(s) == name for s in ev_spk):
            bad.append(f"name_is_vocative:{name}")
    return bad

def finish_action(a, segs, pos, canon, known, warn):
    """Owner: from another speaker's reply ('thanks, <name>'), else LLM owner only if the name is in nearby text. Due: only if said nearby."""
    a["owner_basis"] = None
    last = max(pos[e] for e in a["ev"]); who = segs[last]["spk"]
    if known:
        for p in range(last + 1, min(last + 4, len(segs))):
            if segs[p]["spk"] == who: continue
            for w in REPLY.findall(segs[p]["text"]):
                if len(w) < 2: continue
                name, ok = canon(w)
                if ok: a["owner"], a["owner_basis"] = name, "addressed in reply by another speaker"; break
            if a["owner_basis"]: break
    words = set(re.findall(r"[a-z']+", ctx_text(a["ev"], segs, pos, 0, 2).lower()))
    if not a["owner_basis"]:
        owner = a.get("owner")
        if isinstance(owner, str) and owner.strip() and not BAD_OWNER.match(owner.strip()):
            name, _ = canon(owner)
            if any(difflib.SequenceMatcher(None, c, w).ratio() >= 0.7 for c in {owner.lower(), name.lower()} for w in words):
                a["owner"], a["owner_basis"] = name, "name appears in nearby transcript"
            else:
                warn(f"owner '{owner}' not in nearby text, set to null: {a['task'][:60]}"); a["owner"] = None
        else:
            a["owner"] = None
    due = a.get("due")
    if isinstance(due, str) and due.strip():
        sig = [w for w in re.findall(r"[a-z']+", due.lower()) if len(w) > 2]
        if not sig or not all(w in words for w in sig):
            warn(f"due '{due}' not in nearby text, set to null: {a['task'][:60]}"); a["due"] = None
    else:
        a["due"] = None

def summarize(asr, attendees=None, aliases=None, api_key=None, log=print, use_hints=True):
    warnings, failed = [], []
    def warn(m): warnings.append(m); log("WARN:", m)
    key = api_key or _key()
    canon = make_canon(attendees, aliases); known = bool(attendees or aliases)
    seen = set(); alias_sub = make_alias_sub(aliases, seen); nlabels = [0]
    variants = {c: [c] + list((aliases or {}).get(c, [])) for c in (list(attendees or []) + [k for k in (aliases or {}) if k not in (attendees or [])])}
    def clean(t):
        t = PAREN.sub("", t); nlabels[0] += len(LABEL.findall(t))
        return alias_sub(LABEL.sub(lambda m: "A participant" if m.start() == 0 else "a participant", t))

    segs = [{"idx": r["idx"], "start": r["start"], "spk": "S" + r["speaker"].split("_")[-1], "text": (r.get("text") or "").strip().replace("\u2019", "'")}
            for r in sorted(asr, key=lambda r: r["start"]) if r.get("quality") != "rejected"]
    segs = [s for s in segs if re.search(r"\w", s["text"]) and "thanks for watching" not in s["text"].lower()]
    # Whisper often hallucinates foreign-language filler on very short clips ("Absolutamente.", "Tchau, doutora Ivana.").
    # The ASR stage marks these method=whisper_latin_unresolved + quality=review; keep them out of the LLM input.
    bad_ids = {r["idx"] for r in asr if r.get("method") == "whisper_latin_unresolved" and r.get("quality") == "review"}
    if bad_ids:
        segs = [s for s in segs if s["idx"] not in bad_ids]
        warn(f"excluded {len(bad_ids)} unresolved-language review segments from the LLM input: {sorted(bad_ids)}")
    valid = {s["idx"] for s in segs}; when = {s["idx"]: mmss(s["start"]) for s in segs}
    pos = {s["idx"]: i for i, s in enumerate(segs)}
    sline = lambda s: f'[{s["idx"]}|{mmss(s["start"])}|{s["spk"]}] {s["text"]}'
    hints = speaker_hints(segs, canon, known) if use_hints else {}
    hl = hints_line(hints)
    if hints: log("speaker hints:", hints)

    def run_pass(system, ss, max_out, name, prefix="", depth=0):
        obj, why = call_llm(system, prefix + "\n".join(sline(s) for s in ss), max_out, key, warn)
        if obj is not None: return [obj]
        if why in ("length", "parse") and depth < 1 and len(ss) >= 8:
            h = len(ss) // 2
            return run_pass(system, ss[:h], max_out, name, prefix, depth + 1) + run_pass(system, ss[h:], max_out, name, prefix, depth + 1)
        span = f"{mmss(ss[0]['start'])}-{mmss(ss[-1]['start'])}"
        failed.append({"pass": name, "span": span, "reason": why}); warn(f"{name} failed for {span}")
        return []

    chunks = mk_chunks(segs, sline)
    raw = {k: [] for k in FIELDS}
    rejected = set(); candidates = []
    for i, ch in enumerate(chunks, 1):
        log(f"chunk {i}/{len(chunks)} ({mmss(ch[0]['start'])}-{mmss(ch[-1]['start'])})")
        for n in run_pass(NOTES_SYS, ch, 1000, "notes", hl):
            for k in ("key_discussion", "decisions", "follow_ups", "questions", "concerns"):
                v = n.get(k); raw[k] += v if isinstance(v, list) else []
        cand = [s["idx"] for s in ch if COMMIT.search(s["text"]) and len(s["text"].split()) >= 3]
        candidates += cand
        pre = hl + (f"CANDIDATES (segment IDs containing commitment phrases): {cand}\n" if cand else "")
        for a in run_pass(ACT_SYS, ch, 700, "actions", pre):
            v = a.get("action_items"); raw["action_items"] += v if isinstance(v, list) else []
            for r in (a.get("rejected") or []):
                if isinstance(r, dict): rejected.update(ev_ids([r.get("id")], valid))

    def covered(c):
        near = {segs[p]["idx"] for p in range(max(0, pos[c] - 1), min(len(segs), pos[c] + 2))}
        return c in rejected or any(set(ev_ids(it.get("ev"), valid)) & near for it in raw["action_items"] if isinstance(it, dict))

    uncovered = [c for c in candidates if not covered(c)]
    if uncovered:
        log(f"recovering {len(uncovered)} uncovered commitment candidates")
        for b in range(0, len(uncovered), 4):
            batch = uncovered[b:b + 4]
            parts = []
            for c in batch:
                lo, hi = max(0, pos[c] - 2), min(len(segs), pos[c] + 2)
                parts.append(f"CANDIDATE {c}:\n" + "\n".join(sline(s) for s in segs[lo:hi]))
            obj, why = call_llm(REC_SYS, hl + "\n\n".join(parts), 400, key, warn)
            if obj is None:
                failed.append({"pass": "recovery", "span": str(batch), "reason": why}); continue
            v = obj.get("action_items"); raw["action_items"] += v if isinstance(v, list) else []
            for r in (obj.get("rejected") or []):
                if isinstance(r, dict): rejected.update(ev_ids([r.get("id")], valid))
        uncovered = [c for c in uncovered if not covered(c)]
    for c in uncovered:
        warn(f"commitment candidate not covered [{c}] {when[c]}: {segs[pos[c]]['text'][:70]}")

    out = {}
    for k, f in FIELDS.items():
        items = norm(raw[k], f, valid, clean, warn, k)
        if k == "action_items":
            kept = [a for a in items if len(a["task"].split()) >= 3 and not VAGUE.search(a["task"])]
            if len(kept) < len(items): warn(f"action_items: dropped {len(items) - len(kept)} vague or logistics tasks")
            items = kept
        out[k] = dedupe(items, f)
    out["follow_ups"] = [f for f in out["follow_ups"] if all(
        difflib.SequenceMatcher(None, f["item"].lower(), a["task"].lower()).ratio() < 0.6 for a in out["action_items"])]
    for a in out["action_items"]: finish_action(a, segs, pos, canon, known, warn)

    nflag = 0
    for k, f in FIELDS.items():
        for it in out[k]:
            bad = specifics_check(it[f], ctx_text(it["ev"], segs, pos, 1, 1))
            if known: bad += name_check(it[f], it["ev"], segs, pos, hints, variants)
            if bad: it["unverified"] = bad; nflag += 1
            it["at"] = [when[e] for e in it["ev"]]
    if nflag: warn(f"{nflag} items contain numbers, pronouns or names not supported by their evidence; flagged as 'unverified'")

    ok = lambda k: [i[FIELDS[k]] for i in out[k] if "unverified" not in i]
    ov, _ = call_llm(OV_SYS, json.dumps({k: ok(k) for k in ("key_discussion", "decisions", "action_items", "concerns")}, ensure_ascii=False), 400, key, warn)
    overview = clean(ov["overview"]) if isinstance(ov, dict) and isinstance(ov.get("overview"), str) else ""
    if not overview: warn("overview missing")

    res = {"overview": overview, **out, "name_normalizations": [{"heard": h, "used": u} for h, u in sorted(seen)],
           "speaker_hints": hints, "uncovered_candidates": uncovered,
           "failed_chunks": failed, "warnings": warnings,
           "stats": {"segments": len(segs), "chunks": len(chunks), "labels_replaced": nlabels[0],
                     "commit_candidates": len(candidates), "cache_hits": CACHE_STATS["hit"], "api_calls": CACHE_STATS["api"]}}
    if LABEL.search(json.dumps({k: v for k, v in res.items() if k != "speaker_hints"}, ensure_ascii=False)): warn("speaker label still present in output")
    return res
