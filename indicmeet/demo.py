"""Explicit offline fake LLM. Only selected under DEMO_MODE or by tests."""
import json,re
class FakeGroq:
    def __init__(self): self.prompts=[]
    def __call__(self,system,user,max_out,key,warn):
        self.prompts.append((system,user))
        if system.startswith("Enrich"):
            rows=json.loads(user)
            return {"segments":[{"segment_id":s["segment_id"],"text_roman":s["text_native"] if s["language"]=="en" else "Demo transliteration unavailable", "text_english":s["text_native"] if s["language"]=="en" else "Demo translation unavailable"} for s in rows]},"ok"
        if system.startswith("Extract notes"):
            rows=re.findall(r"\[(\d+)\|[^\]]+\] ([^\n]+)",user)
            return {"key_discussion":[{"point":text,"ev":[int(idx)]} for idx,text in rows[:2]],"decisions":[],"follow_ups":[],"questions":[],"concerns":[]},"ok"
        if system.startswith("Find every explicit") or system.startswith("Each CANDIDATE"):
            return {"action_items":[],"rejected":[]},"ok"
        return {"overview":"Demo summary"},"ok"
