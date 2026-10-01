import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "../css/style.css";

const API = window.MEETINGS_API_BASE || (import.meta.env.DEV ? "/api" : "http://127.0.0.1:8000");
const apiUrl = (path) => path.startsWith("http") ? path : `${API}${path.startsWith("/") ? "" : "/"}${path}`;

function fmtLong(value) {
  const sec = Number(value);
  if (!Number.isFinite(sec) || sec <= 0) return "—";
  const n = Math.round(sec);
  if (n < 3600) return `${Math.floor(n / 60)}m${n % 60 ? ` ${n % 60}s` : ""}`;
  return `${Math.floor(n / 3600)}h ${Math.round((n % 3600) / 60)}m`;
}
function fmtClock(value) {
  const n = Math.max(0, Math.floor(Number(value) || 0)), h = Math.floor(n / 3600), m = Math.floor((n % 3600) / 60), s = n % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
function sessionSegments(session) {
  if (Array.isArray(session?.segments)) return session.segments;
  return (Array.isArray(session?.transcript) ? session.transcript : []).map((s, i) => ({
    id: s.id || `segment-${i + 1}`, t: Number(s.t ?? s.start ?? 0), end: Number(s.end ?? s.t ?? s.start ?? 0),
    time: s.time || fmtClock(s.t ?? s.start), speaker: s.speaker || "SPEAKER_00", lang: s.lang || "unknown",
    tx: s.tx ?? s.text ?? "", en: s.en ?? s.english ?? "", quality: s.quality || "accepted", verified: s.verified,
  }));
}
function sessionIntelligence(session) {
  if (session?.intelligence) return session.intelligence;
  const data = session?.summaryData || {};
  const list = (key, field) => (data[key] || []).map((item) => ({ ...item, text: item[field] || item.text || item.task || item.point || item.item || "" }));
  return { discussed: data.overview || session?.summary || "", keyDiscussion: list("key_discussion", "point"),
    decisions: list("decisions", "decision"), actionItems: list("action_items", "task"), followUps: list("follow_ups", "item"),
    questions: list("questions", "q"), concerns: list("concerns", "concern") };
}
function useSessions() {
  const [sessions, setSessions] = useState([]), [loading, setLoading] = useState(true), [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API}/sessions`, { signal: controller.signal }).then((r) => { if (!r.ok) throw new Error(`API returned ${r.status}`); return r.json(); })
      .then((data) => { setSessions(Array.isArray(data) ? data : []); setError(""); })
      .catch((e) => { if (e.name !== "AbortError") setError("Cannot reach the meeting API. Start the backend and reload."); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, []);
  return { sessions, loading, error };
}
function Icon({ name, size = 16 }) {
  const common = { viewBox: "0 0 16 16", width: size, height: size, fill: "none", stroke: "currentColor", strokeWidth: 1.4, "aria-hidden": true };
  if (name === "search") return <svg {...common}><circle cx="7" cy="7" r="4.5"/><path d="m10.5 10.5 3.5 3.5"/></svg>;
  if (name === "logout") return <svg {...common}><path d="M6.5 13.5h-3a1 1 0 0 1-1-1v-9a1 1 0 0 1 1-1h3"/><path d="m10.5 10.5 3-2.5-3-2.5M13.5 8H6"/></svg>;
  if (name === "download") return <svg {...common}><path d="M8 2.5v8M5 7.5l3 3 3-3M3 13.5h10"/></svg>;
  if (name === "copy") return <svg {...common}><rect x="5.5" y="5.5" width="8" height="8" rx="1.5"/><path d="M10.5 5.5v-2A1.5 1.5 0 0 0 9 2H4a1.5 1.5 0 0 0-1.5 1.5V9A1.5 1.5 0 0 0 4 10.5h1.5"/></svg>;
  if (name === "edit") return <svg {...common}><path d="m11 2.5 2.5 2.5L6 12.5 3 13l.5-3z"/></svg>;
  return <svg {...common} fill="currentColor" stroke="none"><path d="M4.5 2.5v11l9-5.5z"/></svg>;
}
function Sidebar({ active }) {
  return <aside className="sidebar"><a className="brand" href="dashboard.html">meetReaderCB</a><nav aria-label="Main">
    <a className="nav-link" href="dashboard.html" aria-current={active === "overview" ? "page" : undefined}>Overview</a>
    <a className="nav-link" href="meetings.html" aria-current={active === "meetings" ? "page" : undefined}>Meetings</a>
  </nav><div className="sidebar-foot"><hr className="rule"/><div className="profile"><span className="avatar" aria-hidden="true">A</span>
    <div className="profile-id"><strong>Admin</strong><span>Account</span></div><button className="logout" type="button" aria-label="Sign out" title="Sign out" onClick={() => { window.location.href = "index.html"; }}><Icon name="logout" size={15}/></button>
  </div></div></aside>;
}
function Shell({ active, children }) {
  return <div className="app"><Sidebar active={active}/><main className="content"><header className="topbar"><div className="top-actions"><button className="icon-button" type="button" aria-label="Search"><Icon name="search"/></button></div></header>{children}</main></div>;
}
function LoadState({ loading, error }) {
  if (loading) return <p className="empty-state">Loading meetings…</p>;
  if (error) return <p className="empty-state" role="alert">{error}</p>;
  return null;
}
function MeetingRow({ meeting: m, mode, duration, onDuration }) {
  const media = m.media ? apiUrl(m.media) : "";
  return <div className={`m-row ${mode === "meetings" ? "m-row--link" : ""}`}>
    <div><span className="t-title">{m.title || "Untitled meeting"}</span><span className="t-sub">{m.group || ""}</span></div>
    <span className="cell">{m.dateLabel || m.date || "Date not recorded"}{m.time ? ` · ${m.time}` : ""}</span><span className="cell cell--dur">{duration == null ? "—" : fmtLong(duration)}</span>
    <span className="status status--done">Processed</span><div className="row-actions"><a className="button" href={`meeting.html?id=${encodeURIComponent(m.id)}&tab=transcript`}>Transcript</a><a className="button" href={`meeting.html?id=${encodeURIComponent(m.id)}&tab=summary`}>Overview</a></div>
    {media && <video className="media-probe" src={media} preload="metadata" aria-hidden="true" onLoadedMetadata={(e) => onDuration(m.id, e.currentTarget.duration)} onError={() => onDuration(m.id, NaN)}/>}
  </div>;
}
function SessionListPage({ mode }) {
  const { sessions, loading, error } = useSessions(), [date, setDate] = useState(""), [durations, setDurations] = useState({});
  const today = new Intl.DateTimeFormat(undefined, { dateStyle: "long" }).format(new Date()), overview = mode === "overview";
  const filtered = useMemo(() => sessions.filter((m) => !date || m.date === date), [sessions, date]);
  const setDuration = (id, n) => setDurations((old) => old[id] === n ? old : { ...old, [id]: n });
  const totalDuration = sessions.reduce((n, m) => n + (Number(durations[m.id]) || 0), 0);
  const speakers = sessions.reduce((n, m) => n + new Set(sessionSegments(m).map((s) => s.speaker)).size, 0);
  const lines = sessions.reduce((n, m) => n + sessionSegments(m).length, 0);
  return <Shell active={mode}><div className="page-head"><h1>{overview ? "Overview" : "Meetings"}</h1><div className="sub"><b>{overview ? "Your meetings" : `${sessions.length} recordings`}</b><span>{overview ? today : "Select a meeting to open its recording"}</span></div></div>
    {overview && <section className="stats" aria-label="Summary"><div className="stat"><strong>{loading ? "—" : sessions.length}</strong><span>Meetings</span></div><div className="stat"><strong>{loading || Object.keys(durations).length < sessions.length ? "—" : fmtLong(totalDuration)}</strong><span>Meeting time</span></div><div className="stat"><strong>{loading ? "—" : speakers}</strong><span>Speakers</span></div><div className="stat"><strong>{loading ? "—" : lines}</strong><span>Transcript lines</span></div></section>}
    <section className="section" style={overview ? { marginTop: "var(--s8)" } : undefined}><div className="section-head"><div><p className="eyebrow">{overview ? "Recordings" : "Library"}</p><h2 className="h2">{overview ? "Meetings" : "All recordings"}</h2></div><div className="filter-bar"><label htmlFor="dateFilter">Date</label><input className="date-input" type="date" id="dateFilter" value={date} onChange={(e) => setDate(e.target.value)}/><button className="button" type="button" onClick={() => setDate("")}>All dates</button></div></div>
      <LoadState loading={loading} error={error}/>{!loading && !error && <><div className="m-list">{filtered.map((m) => <MeetingRow key={m.id} meeting={m} mode={mode} duration={durations[m.id]} onDuration={setDuration}/>)}</div>{filtered.length === 0 && <p className="empty-state">No meetings on this date.</p>}</>}
    </section></Shell>;
}
function LoginPage() {
  return <main className="login"><div className="login-inner"><div className="login-card"><div className="login-mark">meetReaderCB</div><p className="login-tag">Understand every meeting.</p>
    <form className="login-form" onSubmit={(e) => { e.preventDefault(); window.location.href = "dashboard.html"; }}><div className="field"><label htmlFor="email">Email</label><input type="email" id="email" autoComplete="email" placeholder="you@company.com" required/></div><div className="field"><label htmlFor="password">Password</label><input type="password" id="password" autoComplete="current-password" placeholder="••••••••" required/></div><button className="button button--ink button--full" type="submit">Sign In</button><div className="login-foot"><a className="link-quiet" href="#" onClick={(e) => e.preventDefault()}>Forgot password?</a></div></form>
  </div></div><p className="login-legal">meetReaderCB · Meeting intelligence for business teams</p></main>;
}
function QualityBadge({ item }) {
  const status = item?.quality || item?.status;
  if (status === "review" || status === "rejected") return <span className={`quality-badge quality-badge--${status}`}>{status === "review" ? "Review" : "Rejected"}</span>;
  if (item && typeof item === "object" && (item.verified === false || item.unverified)) return <span className="quality-badge quality-badge--unverified">Unverified</span>;
  return null;
}
function Evidence({ item, segments, onSeek }) {
  const refs = Array.isArray(item?.evidence) ? item.evidence : (item?.t != null ? [{ t: item.t, time: item.time }] : []);
  return refs.map((ref, i) => {
    const source = ref.segmentId && segments.find((s) => s.id === ref.segmentId), t = Number(ref.t ?? source?.t);
    if (!Number.isFinite(t)) return null;
    const label = ref.time || source?.time || fmtClock(t);
    return <button className="evidence-chip js-seek" key={`${ref.segmentId || t}-${i}`} type="button" aria-label={`Play evidence at ${label}`} onClick={() => onSeek(t)}>{label}</button>;
  });
}
function Player({ src, controllerRef, onTime, onDuration }) {
  const mediaRef = useRef(null), canvasRef = useRef(null), scrubRef = useRef(null);
  const [mode, setMode] = useState("loading"), [playing, setPlaying] = useState(false), [muted, setMuted] = useState(false), [duration, setDuration] = useState(0), [current, setCurrent] = useState(0);
  useEffect(() => { const el = mediaRef.current; setMode(src ? "loading" : "empty"); setDuration(0); setCurrent(0); setPlaying(false); if (!el) return; el.pause(); el.src = src || ""; if (src) el.load(); return () => { el.pause(); el.removeAttribute("src"); el.load(); }; }, [src]);
  const toggle = () => { const el = mediaRef.current; if (!el) return; if (el.paused) el.play().catch(() => setMode("empty")); else el.pause(); };
  const seek = (time) => { const el = mediaRef.current; if (el && Number.isFinite(el.duration)) el.currentTime = Math.max(0, Math.min(el.duration, time)); };
  if (controllerRef) controllerRef.current = { toggle, seek };
  useEffect(() => { const canvas = canvasRef.current, scrub = scrubRef.current; if (!canvas || !scrub) return; const dpr = window.devicePixelRatio || 1, rect = scrub.getBoundingClientRect(); canvas.width = Math.max(1, Math.round(rect.width * dpr)); canvas.height = Math.max(1, Math.round(rect.height * dpr)); const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, rect.width, rect.height); const y = Math.round(rect.height / 2), progress = duration > 0 ? current / duration : 0; ctx.fillStyle = "rgba(255,255,255,.22)"; ctx.fillRect(0, y - 1, rect.width, 2); ctx.fillStyle = "#fff"; ctx.fillRect(0, y - 1, rect.width * progress, 2); ctx.fillRect(Math.max(0, rect.width * progress - 1), y - 6, 2, 12); }, [current, duration, mode]);
  const seekEvent = (e) => { const r = e.currentTarget.getBoundingClientRect(); if (r.width && duration) seek((e.clientX - r.left) / r.width * duration); };
  return <section className="player" data-mode={mode} data-playing={String(playing)} data-muted={String(muted)} aria-label="Meeting recording">
    <video ref={mediaRef} playsInline preload="metadata" onLoadedMetadata={(e) => { const d = e.currentTarget.duration || 0; setDuration(d); onDuration?.(d); setMode(e.currentTarget.videoWidth ? "video" : "audio"); }} onTimeUpdate={(e) => { setCurrent(e.currentTarget.currentTime); onTime?.(e.currentTarget.currentTime); }} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} onError={() => setMode("empty")}/>
    <div className="player-stage"><div className="wave-wrap" role="slider" tabIndex="0" aria-label="Seek within recording" aria-valuemin="0" aria-valuenow={Math.round(current)} aria-valuemax={Math.round(duration)} onClick={seekEvent} onKeyDown={(e) => { if (e.key === "ArrowRight") seek(current + 5); if (e.key === "ArrowLeft") seek(current - 5); if (e.key === "Enter" || e.key === " ") { toggle(); e.preventDefault(); } }}/><button className="play-big" type="button" aria-label={playing ? "Pause" : "Play"} onClick={toggle}><svg className="ico-play" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4.5 2.5v11l9-5.5z"/></svg><svg className="ico-pause" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4.5 3h2.5v10H4.5zM9 3h2.5v10H9z"/></svg></button></div>
    <div className="player-empty"><div><p className="eyebrow">Recording unavailable</p><p>The media file for this meeting could not be loaded.</p></div></div>
    <div className="player-bar"><span className="time tnum">{fmtClock(current)}</span><div className="scrub" ref={scrubRef} role="slider" tabIndex="0" aria-label="Seek within recording" aria-valuemin="0" aria-valuenow={Math.round(current)} aria-valuemax={Math.round(duration)} onClick={seekEvent} onKeyDown={(e) => { if (e.key === "ArrowRight") seek(current + 5); if (e.key === "ArrowLeft") seek(current - 5); }}><canvas ref={canvasRef}/></div><span className="time time--end tnum">{duration ? fmtClock(duration) : "--:--"}</span><button className="p-icon" type="button" aria-label={muted ? "Unmute" : "Mute"} onClick={() => { const el = mediaRef.current; if (el) { el.muted = !el.muted; setMuted(el.muted); } }}><svg className="ico-vol" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true"><path d="M8 3 5 5.5H3v5h2L8 13zM10.5 6a2.8 2.8 0 0 1 0 4M12.4 4.3a5.2 5.2 0 0 1 0 7.4"/></svg><svg className="ico-muted" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true"><path d="M8 3 5 5.5H3v5h2L8 13zM10.5 6.5l3 3M13.5 6.5l-3 3"/></svg></button></div>
  </section>;
}
function ActionItem({ item, segments, onSeek }) {
  return <article className="action-row"><div><span className="label">Owner</span><strong>{item?.owner ?? "Unassigned"}</strong></div><div><span className="label">Task</span><p>{item?.task ?? item?.text ?? item?.item ?? ""} <QualityBadge item={item}/><Evidence item={item} segments={segments} onSeek={onSeek}/></p></div><div><span className="label">Due date</span><span>{item?.due || "—"}</span></div></article>;
}
function MeetingPage() {
  const id = new URLSearchParams(window.location.search).get("id"), initialTab = new URLSearchParams(window.location.search).get("tab");
  const [meeting, setMeeting] = useState(null), [error, setError] = useState(""), [loading, setLoading] = useState(true), [tab, setTab] = useState(initialTab === "transcript" ? "transcript" : "summary");
  const [query, setQuery] = useState(""), [language, setLanguage] = useState("both"), [activeTime, setActiveTime] = useState(-1), [duration, setDuration] = useState(0), [editing, setEditing] = useState(false), [editedSummary, setEditedSummary] = useState("");
  const controllerRef = useRef(null), segmentRefs = useRef(new Map());
  useEffect(() => { if (!id) { setError("No meeting id was provided."); setLoading(false); return; } const controller = new AbortController();
    fetch(`${API}/sessions/${encodeURIComponent(id)}`, { signal: controller.signal }).then((r) => { if (!r.ok) throw new Error(r.status === 404 ? "Meeting not found." : `API returned ${r.status}`); return r.json(); })
      .then((data) => { setMeeting(data); setEditedSummary(data.intelligence?.discussed || data.summary || ""); }).catch((e) => { if (e.name !== "AbortError") setError(e.message || "Could not load this meeting."); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); }); return () => controller.abort();
  }, [id]);
  const segments = useMemo(() => sessionSegments(meeting || {}), [meeting]), intel = useMemo(() => sessionIntelligence(meeting || {}), [meeting]);
  const speakers = useMemo(() => new Map((meeting?.speakers || meeting?.participants || []).map((s) => [s.id, s])), [meeting]);
  const speakerName = (key) => speakers.get(key)?.name || key || "Participant";
  const speakerIds = useMemo(() => [...new Set(segments.map((s) => s.speaker || "SPEAKER_00"))], [segments]);
  const stats = useMemo(() => { const rows = new Map(); segments.forEach((seg, i) => { if (seg.quality === "rejected" || seg.status === "rejected") return; const key = seg.speaker || "SPEAKER_00", row = rows.get(key) || { id: key, intervals: [] }; const start = Number(seg.t ?? seg.start), end = Number(seg.end ?? segments[i + 1]?.t ?? start + 8); if (Number.isFinite(start) && Number.isFinite(end) && end > start) row.intervals.push([start, end]); rows.set(key, row); }); const result = [...rows.values()].map((r) => { r.intervals.sort((a,b) => a[0]-b[0]); let total=0,start=null,end=null; for (const [a,b] of r.intervals) { if(start===null)[start,end]=[a,b]; else if(a<=end)end=Math.max(end,b); else {total+=end-start;[start,end]=[a,b];} } if(start!==null)total+=end-start; return {id:r.id,seconds:total}; }); const total=result.reduce((n,r)=>n+r.seconds,0)||1; return result.map((r)=>({...r,pct:Math.round(100*r.seconds/total)})); }, [segments]);
  const langs = [...new Set(segments.map((s) => ({en:"English",hi:"Hindi",bn:"Bengali",mixed:"Mixed"}[s.lang] || s.lang)).filter(Boolean))];
  const filtered = segments.map((s, i) => ({ ...s, _i: i })).filter((s) => { const text = language === "original" ? (s.tx || "") : language === "english" ? (s.en || s.roman || "") : `${s.tx || ""} ${s.en || s.roman || ""}`; return `${speakerName(s.speaker)} ${text}`.toLowerCase().includes(query.trim().toLowerCase()); });
  const media = meeting?.media ? apiUrl(meeting.media) : "";
  const seekTo = (t) => { setTab("transcript"); setActiveTime(t); controllerRef.current?.seek(t); const target = segments.find((s) => Number(s.t) >= t) || segments[segments.length - 1]; if (target) requestAnimationFrame(() => segmentRefs.current.get(target.id)?.scrollIntoView({block:"center",behavior:"smooth"})); };
  const evidence = (item) => <><QualityBadge item={item}/><Evidence item={item} segments={segments} onSeek={seekTo}/></>;
  const textOf = (item) => typeof item === "string" ? item : item?.text || item?.task || item?.point || item?.decision || item?.item || item?.q || item?.concern || "";
  const copy = async (text) => { try { await navigator.clipboard.writeText(text); } catch { window.prompt("Copy this text", text); } };
  const transcriptText = segments.map((s) => `[${s.time || fmtClock(s.t)}] ${speakerName(s.speaker)}: ${language === "english" ? (s.en || s.roman || "") : language === "original" ? (s.tx || "") : `${s.tx || ""}${s.en ? `\n${s.en}` : ""}`}`).join("\n");
  const download = () => { const blob = new Blob([`${meeting.title}\n\nSUMMARY\n${editedSummary}\n\nTRANSCRIPT\n${transcriptText}\n`], {type:"text/plain;charset=utf-8"}), a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=`${meeting.id}-transcript.txt`; a.click(); URL.revokeObjectURL(a.href); };
  return <Shell active="meetings"><header className="topbar topbar--meeting"><a className="back-link" href="meetings.html">← Back to Meetings</a><div className="top-actions"><button className="button" type="button" disabled={!meeting} onClick={download}><Icon name="download" size={13}/>Download transcript</button></div></header>
    <LoadState loading={loading} error={error}/>{meeting && <><section className="m-head"><div><p className="eyebrow">{meeting.group || "Meeting"}</p><h1>{meeting.title || "Untitled meeting"}</h1><p className="meta"><span>{meeting.dateLabel || meeting.date || "Date not recorded"}</span>{meeting.time && <span>{meeting.time}</span>}<span>{fmtLong(duration || meeting.duration)}</span><span>{speakerIds.length} participants</span></p></div><button className="button button--ink" type="button" onClick={() => controllerRef.current?.toggle()}><Icon name="play" size={12}/>Play Recording</button></section>
      <Player src={media} controllerRef={controllerRef} onTime={setActiveTime} onDuration={setDuration}/><nav className="tabs" role="tablist" aria-label="Meeting sections"><button className="tab" role="tab" aria-selected={tab === "summary"} onClick={() => setTab("summary")}>Overview</button><button className="tab" role="tab" aria-selected={tab === "transcript"} onClick={() => setTab("transcript")}>Transcript</button></nav>
      {tab === "summary" ? <section className="panel-view" role="tabpanel"><div className="section-head"><div><p className="eyebrow">Meeting summary</p><h2 className="h2">Overview</h2></div><div className="section-actions"><button className="button" type="button" onClick={() => setEditing(!editing)}><Icon name="edit"/>{editing ? "Done" : "Edit"}</button><button className="button" type="button" onClick={() => copy(editedSummary)}><Icon name="copy"/>Copy</button></div></div>
        <div className="overview-grid"><section className="block"><p className="eyebrow">What was discussed</p>{editing ? <textarea className="summary-editor" value={editedSummary} onChange={(e)=>setEditedSummary(e.target.value)} aria-label="Edit meeting summary"/> : <p className="sum-lede">{editedSummary || "No summary is available for this meeting."}</p>}</section>
          <section className="block"><p className="eyebrow">Key discussion</p><div className="tag-list">{(intel.keyDiscussion||[]).map((x,i)=><span key={i}>{textOf(x)} {evidence(x)}</span>)}</div></section>
          {!!intel.decisions?.length && <section className="block"><p className="eyebrow">Decisions</p><ul className="decisions">{intel.decisions.map((x,i)=><li key={i}><span className="check" aria-hidden="true">✓</span><span>{textOf(x)} {evidence(x)}</span></li>)}</ul></section>}
          <section className="block"><p className="eyebrow">Action items</p><div className="action-table">{intel.actionItems?.length ? intel.actionItems.map((x,i)=><ActionItem key={i} item={x} segments={segments} onSeek={seekTo}/>) : <p className="muted">No action items were extracted.</p>}</div></section>
          {!!intel.followUps?.length && <section className="block"><p className="eyebrow">Follow-ups</p><ol className="followups">{intel.followUps.map((x,i)=><li key={i}><span className="n">{String(i+1).padStart(2,"0")}</span><span>{textOf(x)} {evidence(x)}</span></li>)}</ol></section>}
          {!!intel.questions?.length && <section className="block"><p className="eyebrow">Questions / Open Items</p><ul className="plain-list">{intel.questions.map((x,i)=><li key={i}>{textOf(x)} {evidence(x)}</li>)}</ul></section>}
          {!!intel.concerns?.length && <section className="block"><p className="eyebrow">Objections / Concerns</p><ul className="plain-list">{intel.concerns.map((x,i)=><li key={i}>{textOf(x)} {evidence(x)}</li>)}</ul></section>}
        </div><dl className="facts"><div><dt>Participants</dt><dd>{speakerIds.length}</dd></div><div><dt>Transcript lines</dt><dd>{segments.length}</dd></div><div><dt>Languages</dt><dd>{langs.join(", ") || "—"}</dd></div><div><dt>Duration</dt><dd>{fmtLong(duration || meeting.duration)}</dd></div></dl>
      </section> : <section className="panel-view" role="tabpanel"><div className="section-head"><div><p className="eyebrow">{segments.length} lines · {speakerIds.length} participants</p><h2 className="h2">Transcript</h2></div><div className="filter-bar"><div className="seg" role="group" aria-label="Transcript language">{["both","original","english"].map((v)=><button type="button" key={v} aria-pressed={language===v} onClick={()=>setLanguage(v)}>{v==="both"?"Both":v==="original"?"Original":segments.some((s)=>s.en)?"English":"Romanization"}</button>)}</div><label className="search-wrap"><Icon name="search"/><input className="search" type="search" placeholder="Search transcript" aria-label="Search transcript" value={query} onChange={(e)=>setQuery(e.target.value)}/></label><button className="button" type="button" onClick={()=>copy(transcriptText)}><Icon name="copy"/>Copy</button></div></div>
        <div className="t-scroll"><div>{filtered.map((s)=><article className={`t-line ${activeTime>=Number(s.t)&&activeTime<Number(s.end||s.t+1)?"is-active":""}`} data-t={s.t} key={s.id} ref={(el)=>el&&segmentRefs.current.set(s.id,el)}><button className="ts-chip" type="button" onClick={()=>controllerRef.current?.seek(Number(s.t)||0)}>{s.time||fmtClock(s.t)}</button><div><div className="who"><span className="avatar avatar--sm" aria-hidden="true">{String(speakerName(s.speaker)).slice(0,2).toUpperCase()}</span><strong>{speakerName(s.speaker)}</strong><span className="lang-pill">{s.lang||"unknown"}</span><QualityBadge item={s}/></div>{(language!=="english"||!s.en)&&<p className="t-tx">{s.tx||s.text||""}</p>}{language!=="original"&&(s.en||s.roman)&&<p className="t-en">{s.en||s.roman}</p>}</div><button className="mini-play" type="button" aria-label={`Play from ${s.time||fmtClock(s.t)}`} onClick={()=>controllerRef.current?.seek(Number(s.t)||0)}><Icon name="play"/></button></article>)}</div></div>{filtered.length===0&&<p className="no-results">No lines match that search.</p>}
      </section>}</>}
  </Shell>;
}
function App() {
  const path = window.location.pathname.toLowerCase();
  if (path.endsWith("meeting.html")) return <MeetingPage/>;
  if (path.endsWith("meetings.html")) return <SessionListPage mode="meetings"/>;
  if (path.endsWith("dashboard.html")) return <SessionListPage mode="overview"/>;
  return <LoginPage/>;
}

createRoot(document.getElementById("root")).render(<App/>);
