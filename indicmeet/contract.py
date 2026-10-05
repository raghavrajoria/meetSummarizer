"""Canonical segments and explicit legacy import adapters. Original text is never rewritten."""
from __future__ import annotations
import hashlib
import json
import math
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

ISO_639_1 = frozenset('aa ab ae af ak am an ar as av ay az ba be bg bi bm bn bo br bs ca ce ch co cr cs cu cv cy da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht hu hy hz ia id ie ig ii ik io is it iu ja jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg mh mi mk ml mn mr ms mt my na nb nd ne ng nl nn no nr nv ny oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so sq sr ss st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo za zh zu'.split())

class AsrMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    method: str = "import"
    whisper_lang: str | None = None
    whisper_lang_conf: float | None = Field(default=None, ge=0, le=1)
    mms_lang: str | None = None
    mms_lang_conf: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)

class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_id: str = Field(min_length=1)
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(ge=0, allow_inf_nan=False)
    speaker: str = Field(min_length=1)
    speaker_name: str | None = None
    language: str = Field(min_length=2)
    text_native: str
    text_roman: str | None = None
    text_english: str | None = None
    quality: Literal["accepted", "review", "rejected"]
    reasons: list[str] = Field(default_factory=list)
    asr: AsrMetadata

    @model_validator(mode="after")
    def time_order(self):
        if self.language not in ISO_639_1 | {"mul", "und"}:
            raise ValueError("Language must be ISO 639-1, mul or und")
        if self.language in {"mul", "und"} and self.quality == "accepted":
            raise ValueError("mul/und cannot be accepted without language review")
        if self.end < self.start:
            raise ValueError("end precedes start")
        return self

def stable_id(start: float, speaker: str) -> str:
    value = json.dumps([round(float(start), 3), speaker], ensure_ascii=False, separators=(",", ":"))
    return "seg_" + hashlib.sha256(value.encode()).hexdigest()[:20]

def seconds(value: Any) -> float:
    if isinstance(value, str) and ":" in value:
        total = 0.0
        for part in value.split(":"):
            total = total * 60 + float(part)
        return total
    return float(value)

def canonicalize(data: Any) -> list[dict]:
    if isinstance(data, dict):
        data = data.get("segments", data.get("transcript"))
    if not isinstance(data, list):
        raise ValueError("Expected transcript array or segments/transcript object")
    result, seen = [], set()
    from collections import Counter
    counts = Counter((seconds(row.get("start", row.get("t", row.get("timestamp", 0)))), str(row.get("speaker") or "SPEAKER_00")) for row in data)
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Segment {index} must be an object")
        if "segment_id" in row and "text_native" in row:
            segment = dict(row)
            segment["asr"] = dict(row.get("asr", {}))
            segment["reasons"] = list(row.get("reasons", []))
        else:
            start = seconds(row.get("start", row.get("t", row.get("timestamp", 0))))
            next_start = seconds(data[index + 1].get("start", data[index + 1].get("t", data[index + 1].get("timestamp", start + 1)))) if index + 1 < len(data) else start + 1
            end = seconds(row.get("end", max(start, next_start)))
            speaker = str(row.get("speaker") or "SPEAKER_00")
            language = str(row.get("language", row.get("lang", "unknown")))
            reasons = list(row.get("reasons", row.get("quality_reasons", [])))
            status = row.get("quality", row.get("quality_flag", "review"))
            status = status.get("status", "review") if isinstance(status, dict) else status
            status = {"accept": "accepted", "reject": "rejected"}.get(status, status)
            if language == "ur":
                language = "hi"
            if language not in {"en", "hi", "bn", "gu", "mr", "ta", "te", "kn", "ml", "pa", "or", "as"}:
                reasons.append("language_outside_allowlist:" + language)
                status = "rejected" if status == "rejected" else "review"
            text = row.get("text_native", row.get("text", row.get("tx", "")))
            if not isinstance(text, str):
                raise ValueError(f"Segment {index} text must be a string")
            metadata = {key: row.get(key) for key in ("method", "whisper_lang", "whisper_lang_conf", "mms_lang", "mms_lang_conf", "confidence") if key in row}
            metadata.setdefault("method", "import")
            if row.get("lang") == "ur":
                metadata.setdefault("whisper_lang", "ur")
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError("Nonfinite timestamp")
            identifier = stable_id(start, speaker)
            if counts[(start, speaker)] > 1:
                identifier += "_" + hashlib.sha256(json.dumps([end, text], ensure_ascii=False).encode()).hexdigest()[:12]
            segment = {"segment_id": identifier, "start": start, "end": end,
                "speaker": speaker, "speaker_name": row.get("speaker_name"), "language": language,
                "text_native": text, "text_roman": row.get("text_roman", row.get("roman")),
                "text_english": row.get("text_english", row.get("english", row.get("en"))),
                "quality": status, "reasons": reasons, "asr": metadata}
        raw_language = str(segment["language"]).lower()
        language = {"mixed": "mul", "unknown": "und", "ur": "hi"}.get(raw_language, raw_language)
        if language not in ISO_639_1 | {"mul", "und"}:
            language = "und"
        if language != raw_language:
            segment["asr"]["source_language"] = raw_language
        segment["language"] = language
        if language in {"mul", "und"}:
            segment["quality"] = "rejected" if segment["quality"] == "rejected" else "review"
            if "unresolved_language" not in segment["reasons"]: segment["reasons"].append("unresolved_language")
        segment = Segment.model_validate(segment).model_dump()
        if segment["segment_id"] in seen:
            raise ValueError("Duplicate canonical segment ID")
        seen.add(segment["segment_id"])
        result.append(segment)
    return result

def legacy_rows(segments: list[dict]) -> list[dict]:
    """Private adapter for the pre-existing numeric-ID extraction engine only."""
    return [{"idx": i, "start": s["start"], "end": s["end"], "speaker": s["speaker"],
        "text": s["text_native"], "lang": s["language"], "quality": s["quality"],
        "reasons": s["reasons"], "duration": s["end"] - s["start"], **s["asr"]} for i, s in enumerate(segments)]

def review_gate(segments: list[dict], strict: bool) -> list[dict]:
    return [s for s in segments if s["text_native"].strip() and s["quality"] != "rejected" and (not strict or s["quality"] == "accepted")]
