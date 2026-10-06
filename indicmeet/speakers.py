"""Speaker sources and overlap-based LiveKit cluster naming."""
import json,math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str
class SpeakerSource(Protocol):
    def turns(self,session=None)->list[Turn]: ...
class Diarization:
    def __init__(self,path): self.path=path
    def turns(self,session=None):
        from .pipeline import _read_diar_csv
        return [Turn(row["start"],row["end"],row["speaker"]) for row in _read_diar_csv(self.path)]
def pyannote_turns(output):
    annotation=getattr(output,"speaker_diarization",output)
    if not hasattr(annotation,"itertracks"): raise ValueError("Unsupported pyannote output")
    return [Turn(float(span.start),float(span.end),str(speaker)) for span,_,speaker in annotation.itertracks(yield_label=True)]
class LiveKitEvents:
    def __init__(self,manifest,events):
        self.manifest=json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.events=[json.loads(line) for line in Path(events).read_text(encoding="utf-8").splitlines() if line.strip()]
        start=datetime.fromisoformat(self.manifest["recording_started_at_utc"].replace("Z","+00:00"))
        stop=datetime.fromisoformat(self.manifest["recording_stopped_at_utc"].replace("Z","+00:00"))
        if start.tzinfo is None or stop.tzinfo is None: raise ValueError("UTC timestamp requires timezone")
        self.epoch=start.timestamp()*1000; self.duration=(stop-start).total_seconds()
        if self.duration<0: raise ValueError("Recording stop precedes start")
        self.names={p["identity"]:p["display_name"] for p in self.manifest["participants"]}
    def timestamp(self,event):
        value=float(event["t_ms"])/1000 if "t_ms" in event else (float(event["timestamp_ms"])-self.epoch)/1000
        if not math.isfinite(value) or value<0 or value>self.duration: raise ValueError("Event outside recording")
        return value
    def turns(self,session=None):
        active,spans={},[]
        for event in sorted(self.events,key=self.timestamp):
            time=self.timestamp(event); kind=event.get("type",event.get("event"))
            if kind not in {"active_speakers","recording_stopped","participant_left","recording_paused"}: continue
            next_active={p["identity"] for p in event.get("speakers",[])} if kind=="active_speakers" else set(active)-{event.get("identity")} if kind=="participant_left" else set()
            if not next_active.issubset(self.names): raise ValueError("Unknown participant identity")
            for speaker in set(active)-next_active:
                if time>active[speaker]: spans.append(Turn(active[speaker],time,speaker))
                del active[speaker]
            for speaker in next_active-set(active): active[speaker]=time
        for speaker,start in active.items():
            if self.duration>start: spans.append(Turn(start,self.duration,speaker))
        return spans
def name_clusters(segments,source):
    from copy import deepcopy
    output=deepcopy(segments); spans=source.turns(); scores={}
    for row in output:
        for turn in spans:
            score=max(0,min(row["end"],turn.end)-max(row["start"],turn.start))
            key=(row["speaker"],turn.speaker); scores[key]=scores.get(key,0)+score
    for row in output:
        choices=sorted([(score,identity) for (speaker,identity),score in scores.items() if speaker==row["speaker"] and score>0],reverse=True)
        if choices and (len(choices)==1 or choices[0][0]>choices[1][0]):
            row["speaker_name"]=source.names[choices[0][1]]
            row['speaker_name_source']='inferred'
    return output


def transcribe_tracks(tracks,provider,extract_audio,temporary,duration,names):
    """Tracks must share the recording time origin (silence padding before join)."""
    from .contract import canonicalize,reidentify
    rows=[]
    for index,(identity,path) in enumerate(sorted(tracks.items())):
        wav=extract_audio(path,Path(temporary)/f"track-{index}.wav")
        segments=canonicalize(provider.transcribe(wav,turns=[{"start":0.,"end":duration,"speaker":identity}]))
        for row in segments:
            row["speaker"]=identity;row["speaker_name"]=names.get(identity)
            row['speaker_name_source']='attendee_list' if row['speaker_name'] else 'none'
            rows.append(row)
    return reidentify(sorted(rows,key=lambda s:(s["start"],s["speaker"])))
