"""IndicMeet summary stage v2.1: ASR turns -> validated meeting summary. Self-contained, no notebook state.
v2 + 429 handling + disk cache + REPLY lookahead fix, merged into one file. Untested as a single file."""
import json, re, os, time, difflib, hashlib, requests

MODEL = "openai/gpt-oss-20b"
URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE_DIR = os.environ.get("INDICMEET_CACHE_DIR", "/kaggle/working/llm_cache")
CACHE_STATS = {"hits": 0, "api_calls": 0}
_window = []  # (timestamp, tokens) rolling 60s budget

def ntok(t): return len(t) // 3 + 1

def mmss(s):
    s = int(s); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

def _key():
    k = os.environ.get("GROQ_API_KEY")
    if k: return k
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret("GROQ_API_KEY")

def _wait(need, cap=7000):
    while True:
        now = time.time(); _window[:] = [(t, k) for t, k in _window if now - t < 60]
        if not _window or sum(k for _, k in _window) + need <= cap: return
        time.sleep(max(1, 61 - (now - _window[0][0])))

def _call_llm_raw(system, user, max_out, key, warn):
    """One request -> (parsed JSON or None, reason). Never retries with a bigger budget."""
    need = ntok(system) + ntok(user) + max_out
    for _ in range(3):
        _wait(need)
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=120,
                json={"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                      "max_completion_tokens": max_out, "reasoning_effort": "low", "temperature": 0.2})
        except requests.RequestException as e:
            warn(f"network error: {e}"); time.sleep(3); continue
        if r.status_code == 429:
            m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", r.text)
            wait = (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3)) + 1) if m else 10
            warn(f"429, wait {wait:.0f}s: {r.text[:220]}")
            if wait > 120: return None, "ratelimit"
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
    warn("gave up after 3 attempts"); return None, "net"

def call_llm(system, user, max_out, key, warn):
    """Cached wrapper: identical (model, prompt, input, budget) is never sent to the API twice."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    h = hashlib.sha256(json.dumps([MODEL, system, user, max_out], ensure_ascii=False).encode()).hexdigest()[:24]
    f = f"{CACHE_DIR}/{h}.json"
    if os.path.exists(f):
        try:
            obj = json.load(open(f)); CACHE_STATS["hits"] += 1; return obj, "ok"
        except Exception: pass
    CACHE_STATS["api_calls"] += 1
    obj, why = _call_llm_raw(system, user, max_out, key, warn)
    if obj is not None: json.dump(obj, open(f, "w"), ensure_ascii=False)
    return obj, why

BASE = ("The text is a meeting transcript with ASR errors. Use ONLY what is in the text; never invent. "
        "Do not add details that are not in the text: no numbers, durations or gender (never use he/she/his/her; use the person's name or 'they'). "
        "Speaker labels (S00...) are anonymous: never output a label and never infer a person's name from a label. "
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
    '{"action_items":[{"task":"","owner":null,"due":null,"ev":[]}]}. Include "I will/I\'ll ...", "let me ...", "I can help you with ...", '
    '"we\'ll do it together", and accepted offers of help. Exclude status updates about past work, general advice, teaching content, '
    'and meeting logistics (sharing a screen, starting, greetings, thanks). '
    'Every task must be self-contained: if the speaker says "I\'ll do that", replace "that" with the specific thing from the earlier line and list both IDs. '
    'owner: a personal name only if the text says who does it, else null (do not guess). due: only a time that is actually said, else null. '
    'Empty list if none. ' + BASE)
OV_SYS = ('Return ONLY JSON {"overview":"3-4 sentences"}. Describe the meeting using only these validated notes. '
    'Say something was assigned only if it appears in action_items. If the notes are mostly explanations of a concept, say the recording includes teaching. '
    'Never mention speaker labels and never use he/she/his/her.')

FIELDS = {"key_discussion": "point", "decisions": "decision", "follow_ups": "item",
          "questions": "q", "concerns": "concern", "action_items": "task"}
LABEL = re.compile(r"\b(?:SPEAKER_\d+|S\d{2})\b")
PAREN = re.compile(r"\s*\(\s*(?:SPEAKER_\d+|S\d{2})\s*\)")
BAD_OWNER = re.compile(r"(?i)^(s\d+|speaker[_ ]?\d+|participants?|speaker|team|someone|unknown|everyone)$")
VAGUE = re.compile(r"(?i)^(do|does|did|handle|take care of|work on)\s+(that|this|it)\b|unspecified|\b(?:share|sharing)\s+(?:my|the)\s+screen\b")
REPLY = re.compile(r"(?=\b(?:thank you|thanks|sure|okay|ok|great|awesome|alright|yes|yeah)(?:\s+so\s+much)?[,\s]+([A-Za-z]+))", re.I)
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

def norm(raw, field, valid, clean, warn, kind):
    out = []
    for it in raw:
        if not isinstance(it, dict) or not isinstance(it.get(field), str) or not it[field].strip(): continue
        ev = []
        for e in it.get("ev") or []:
            m = re.match(r"\[?\s*(\d+)", str(e))
            if m and int(m.group(1)) in valid and int(m.group(1)) not in ev: ev.append(int(m.group(1)))
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

def finish_action(a, segs, pos, canon, known, warn):
    """Owner: from another speaker's reply ('thanks, <name>'), else LLM owner only if the name is in nearby text. Due: only if said nearby."""
    a["owner_basis"] = None
    last = max(pos[e] for e in a["ev"]); who = segs[last]["spk"]
    if known:
        for p in range(last + 1, min(last + 4, len(segs))):
            if segs[p]["spk"] == who: continue
            for w in REPLY.findall(segs[p]["text"]):
                if len(w) < 3: continue
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

def summarize(asr, attendees=None, aliases=None, api_key=None, log=print):
    warnings, failed = [], []
    def warn(m): warnings.append(m); log("WARN:", m)
    key = api_key or _key()
    canon = make_canon(attendees, aliases); known = bool(attendees or aliases)
    seen = set(); alias_sub = make_alias_sub(aliases, seen); nlabels = [0]
    def clean(t):
        t = PAREN.sub("", t); nlabels[0] += len(LABEL.findall(t))
        return alias_sub(LABEL.sub(lambda m: "A participant" if m.start() == 0 else "a participant", t))

    segs = [{"idx": r["idx"], "start": r["start"], "spk": "S" + r["speaker"].split("_")[-1], "text": (r.get("text") or "").strip()}
            for r in sorted(asr, key=lambda r: r["start"]) if r.get("quality") != "rejected"]
    segs = [s for s in segs if re.search(r"\w", s["text"]) and "thanks for watching" not in s["text"].lower()]
    valid = {s["idx"] for s in segs}; when = {s["idx"]: mmss(s["start"]) for s in segs}
    pos = {s["idx"]: i for i, s in enumerate(segs)}
    sline = lambda s: f'[{s["idx"]}|{mmss(s["start"])}|{s["spk"]}] {s["text"]}'

    def run_pass(system, ss, max_out, name, depth=0):
        obj, why = call_llm(system, "\n".join(sline(s) for s in ss), max_out, key, warn)
        if obj is not None: return [obj]
        if why in ("length", "parse") and depth < 1 and len(ss) >= 8:
            h = len(ss) // 2
            return run_pass(system, ss[:h], max_out, name, depth + 1) + run_pass(system, ss[h:], max_out, name, depth + 1)
        span = f"{mmss(ss[0]['start'])}-{mmss(ss[-1]['start'])}"
        failed.append({"pass": name, "span": span, "reason": why}); warn(f"{name} failed for {span}")
        return []

    chunks = mk_chunks(segs, sline)
    raw = {k: [] for k in FIELDS}
    for i, ch in enumerate(chunks, 1):
        log(f"chunk {i}/{len(chunks)} ({mmss(ch[0]['start'])}-{mmss(ch[-1]['start'])})")
        for n in run_pass(NOTES_SYS, ch, 1000, "notes"):
            for k in ("key_discussion", "decisions", "follow_ups", "questions", "concerns"):
                v = n.get(k); raw[k] += v if isinstance(v, list) else []
        for a in run_pass(ACT_SYS, ch, 700, "actions"):
            v = a.get("action_items"); raw["action_items"] += v if isinstance(v, list) else []

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
            if bad: it["unverified"] = bad; nflag += 1
            it["at"] = [when[e] for e in it["ev"]]
    if nflag: warn(f"{nflag} items contain numbers or pronouns not found in their evidence; flagged as 'unverified'")

    ok = lambda k: [i[FIELDS[k]] for i in out[k] if "unverified" not in i]
    ov, _ = call_llm(OV_SYS, json.dumps({k: ok(k) for k in ("key_discussion", "decisions", "action_items", "concerns")}, ensure_ascii=False), 400, key, warn)
    overview = clean(ov["overview"]) if isinstance(ov, dict) and isinstance(ov.get("overview"), str) else ""
    if not overview: warn("overview missing")

    res = {"overview": overview, **out, "name_normalizations": [{"heard": h, "used": u} for h, u in sorted(seen)],
           "failed_chunks": failed, "warnings": warnings,
           "stats": {"segments": len(segs), "chunks": len(chunks), "labels_replaced": nlabels[0]}}
    if LABEL.search(json.dumps(res, ensure_ascii=False)): warn("speaker label still present in output")
    return res
