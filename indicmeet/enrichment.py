"""Bounded enrichment with explicit IDs. Native text is immutable."""
from copy import deepcopy
from .contract import review_gate
from .settings import get_settings
from .summary import call_llm

def enrich(segments, client=None, strict=None):
    settings=get_settings(); client=client or call_llm
    output=deepcopy(segments)
    admitted=review_gate(output, settings.strict_review if strict is None else strict)
    def process(batch, attempts=0):
        import json
        payload=[{"segment_id":s["segment_id"],"language":s["language"],"text_native":s["text_native"]} for s in batch]
        answer, _=client("Enrich without changing native text. Return JSON {segments:[{segment_id,text_roman,text_english}]}. Preserve meaning; do not invent facts.",json.dumps(payload,ensure_ascii=False),2000,settings.groq_api_key or "demo",lambda _:None)
        items=answer.get("segments",[]) if isinstance(answer,dict) else []
        found={x.get("segment_id"):x for x in items if isinstance(x,dict)}
        missing=[]
        for segment in batch:
            row=found.get(segment["segment_id"])
            if row and all(isinstance(row.get(k),str) for k in ("text_roman","text_english")):
                segment["text_roman"],segment["text_english"]=row["text_roman"],row["text_english"]
            else: missing.append(segment)
        if missing and attempts<2:
            half=max(1,len(missing)//2)
            for i in range(0,len(missing),half): process(missing[i:i+half],attempts+1)
        elif missing:
            for segment in missing: segment["reasons"].append("enrichment_failed")
    for i in range(0,len(admitted),8): process(admitted[i:i+8])
    return output
