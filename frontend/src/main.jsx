import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, apiUrl, token, clearAuth, upload } from "./api.js";
import "../css/style.css";
import "./workflow.css";

const id = () => new URLSearchParams(location.search).get("id");
const clock = n => `${Math.floor(n / 60)}:${String(Math.floor(n % 60)).padStart(2, "0")}`;
function ErrorBox({ error }) { return error ? <p role="alert" className="error">{error}</p> : null; }
function Shell({ children }) {
  const [config, setConfig] = useState(null), [error, setError] = useState("");
  useEffect(() => { api("/config").then(setConfig).catch(e => setError(e.message)); }, []);
  async function logout() { try { await api("/auth/logout", { method: "POST" }); clearAuth(); location.assign("index.html"); } catch (e) { setError(e.message); } }
  return <div className="app"><aside className="sidebar"><a className="brand" href="dashboard.html">IndicMeet</a><nav><a className="nav-link" href="dashboard.html">Overview</a><a className="nav-link" href="meetings.html">Meetings</a><a className="nav-link" href="upload.html">Upload recording</a></nav><button className="button" onClick={logout}>Sign out</button></aside><main className="content"><ErrorBox error={error}/>{config?.demo_mode && <p className="demo-banner">Demo mode: stored transcript and fake summary; no models or Groq calls.</p>}{children}</main></div>;
}
function Login() {
  const [error, setError] = useState(""), [busy, setBusy] = useState(false), [demo, setDemo] = useState(false);
  useEffect(() => { api("/config").then(c => setDemo(c.demo_mode)).catch(e => setError(e.message)); }, []);
  async function submit(e) {
    e.preventDefault(); setBusy(true); setError("");
    const form = new FormData(e.currentTarget);
    try { const reply = await api("/auth/login", { method: "POST", json: Object.fromEntries(form) }); sessionStorage.setItem("access_token", reply.access_token); location.assign("dashboard.html"); }
    catch (e) { setError(e.message); } finally { setBusy(false); }
  }
  return <main className="login"><div className="login-card"><h1>IndicMeet</h1><p>Understand every meeting.</p>{demo && <p>Demo account: demo / demo-password</p>}<form className="login-form" onSubmit={submit}><label>Username<input name="username" autoComplete="username" required/></label><label>Password<input name="password" type="password" autoComplete="current-password" required/></label><ErrorBox error={error}/><button className="button button--ink" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button></form></div></main>;
}
function Library() {
  const [meetings, setMeetings] = useState(null), [query, setQuery] = useState(""), [date, setDate] = useState(""), [error, setError] = useState("");
  const load = () => api("/meetings").then(setMeetings).catch(e => setError(e.message));
  useEffect(() => { load(); }, []);
  async function remove(m) { if (!confirm(`Delete “${m.title}” and its recording?`)) return; try { await api(`/meetings/${m.id}`, { method: "DELETE" }); await load(); } catch (e) { setError(e.message); } }
  const filtered = meetings?.filter(m => (!date || m.date === date) && `${m.title} ${m.group}`.toLowerCase().includes(query.toLowerCase()));
  return <Shell><div className="page-head"><h1>Meetings</h1><a className="button button--ink" href="upload.html">Upload recording</a></div><div className="filter-bar"><label>Search meetings<input type="search" value={query} onChange={e => setQuery(e.target.value)}/></label><label>Date<input type="date" value={date} onChange={e => setDate(e.target.value)}/></label><button className="button" onClick={() => { setDate(""); setQuery(""); }}>Clear filters</button></div><ErrorBox error={error}/>{!meetings && !error && <p>Loading meetings…</p>}{filtered?.length === 0 && <p className="empty-state">No meetings found. Upload a recording to begin.</p>}<div className="m-list">{filtered?.map(m => <article className="meeting-row" key={m.id}><div><h2>{m.title}</h2><p>{m.date || "Date not recorded"} · {m.status}</p></div><a className="button" href={`${m.status === "processed" ? "meeting" : "processing"}.html?id=${m.id}`}>Open meeting</a><button className="button" onClick={() => remove(m)}>Delete</button></article>)}</div></Shell>;
}
function Upload() {
  const [config, setConfig] = useState(null), [recording, setRecording] = useState(null), [error, setError] = useState(""), [busy, setBusy] = useState(false), [progress, setProgress] = useState(0);
  useEffect(() => { api("/config").then(setConfig).catch(e => setError(e.message)); }, []);
  function pick(file) {
    setError("");
    if (!file) return;
    if (!/\.(mp4|m4a|mp3|wav|ogg|webm|flac|aac|mov)$/i.test(file.name)) { setError("Choose an audio or video recording."); return; }
    if (file.size > (config?.max_upload_mb || 512) * 1024 * 1024) { setError(`Recording exceeds ${config.max_upload_mb} MB.`); return; }
    setRecording(file);
  }
  async function submit(e) {
    e.preventDefault(); if (!recording) { setError("Select a recording."); return; }
    setBusy(true); setError(""); const data = new FormData(e.currentTarget); data.set("recording", recording);
    for (const [key, value] of [...data]) if (value instanceof File && !value.size) data.delete(key);
    try { const result = await upload(data, setProgress); location.assign(`processing.html?id=${result.id}`); } catch (e) { setError(e.message); setBusy(false); }
  }
  return <Shell><h1>Upload recording</h1><ErrorBox error={error}/>{!config ? <p>Loading upload settings…</p> : <form className="workflow-form" onSubmit={submit}><label>Meeting title<input name="title" required maxLength={500}/></label><label>Meeting date<input type="date" name="date" required defaultValue={new Date().toISOString().slice(0, 10)}/></label><label>Group<input name="group" maxLength={300}/></label><div className="drop-zone" onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); if (!busy) pick(e.dataTransfer.files[0]); }}><label>Recording<input type="file" accept="audio/*,video/*" disabled={busy} onChange={e => pick(e.target.files[0])}/></label><p>{recording ? recording.name : `Drop audio or video here (up to ${config.max_upload_mb} MB)`}</p></div>{(config.demo_mode || config.import_mode) && <><label>ASR JSON (optional in demo mode)<input type="file" name="asr_json" accept=".json"/></label><label>Speaker turns CSV (optional)<input type="file" name="diarization_csv" accept=".csv"/></label></>}{busy && <><progress value={progress} max="100" aria-label="Upload progress"/><p>Uploading: {progress}%</p></>}<button className="button button--ink" disabled={busy}>{busy ? "Uploading…" : "Upload and process"}</button></form>}</Shell>;
}
function Processing() {
  const [job, setJob] = useState(null), [error, setError] = useState("");
  useEffect(() => {
    let alive = true, timer;
    async function poll() { try { const result = await api(`/meetings/${id()}/job`); if (!alive) return; setJob(result); setError(""); if (result.status === "done") { location.replace(`meeting.html?id=${id()}`); return; } if (result.status !== "failed") timer = setTimeout(poll, 500); } catch (e) { if (alive) setError(e.message); } }
    poll(); return () => { alive = false; clearTimeout(timer); };
  }, []);
  async function retry() { try { await api(`/jobs/${job.id}/retry`, { method: "POST" }); location.reload(); } catch (e) { setError(e.message); } }
  return <Shell><h1>Processing meeting</h1><ErrorBox error={error}/>{job ? <><p role="status">{job.stage} · {job.status}</p><progress value={job.progress} max="100" aria-label="Processing progress"/><p>{job.progress}%</p>{job.status === "failed" && <><p role="alert">{job.error}</p><button className="button" onClick={retry}>Retry processing</button></>}</> : !error && <p>Loading job…</p>}{error && <button className="button" onClick={() => location.reload()}>Reload progress</button>}</Shell>;
}
function Detail() {
  const [meeting, setMeeting] = useState(null), [error, setError] = useState(""), [media, setMedia] = useState(""), [tab, setTab] = useState("Native"), [query, setQuery] = useState(""), [summary, setSummary] = useState(""), [saved, setSaved] = useState(""), [busy, setBusy] = useState(false);
  const player = useRef(null), timer = useRef(null);
  async function load() { const m = await api(`/meetings/${id()}`); setMeeting(m); setSummary(m.summary || ""); }
  async function refreshMedia() { const reply = await api(`/meetings/${id()}/media-url`); const position = player.current?.currentTime || 0, playing = player.current && !player.current.paused; setMedia(apiUrl(reply.url)); if (player.current) player.current.onloadedmetadata = () => { player.current.currentTime = position; if (playing) player.current.play().catch(() => {}); }; }
  useEffect(() => { load().then(refreshMedia).catch(e => setError(e.message)); timer.current = setInterval(() => refreshMedia().catch(e => setError(e.message)), 90000); return () => clearInterval(timer.current); }, []);
  function seek(segmentId) { const s = meeting.transcript.find(s => s.segment_id === segmentId); if (s && player.current) { player.current.currentTime = s.start; player.current.play().catch(e => setError(e.message)); } }
  async function save() { setBusy(true); setSaved(""); try { await api(`/meetings/${id()}/summary`, { method: "PUT", json: { text: summary } }); setSaved("Summary saved"); } catch (e) { setError(e.message); } finally { setBusy(false); } }
  async function revert() { try { await api(`/meetings/${id()}/summary`, { method: "DELETE" }); await load(); setSaved("Original summary restored"); } catch (e) { setError(e.message); } }
  async function download() { try { const blob = await api(`/meetings/${id()}/transcript`, { blob: true }); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = "transcript.json"; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000); } catch (e) { setError(e.message); } }
  const citations = item => (item.source_segment_ids || []).filter(ref => meeting.transcript.some(s => s.segment_id === ref)).map(ref => <button className="evidence-chip" aria-label={`Play evidence ${ref}`} key={ref} onClick={() => seek(ref)}>{clock(meeting.transcript.find(s => s.segment_id === ref).start)}</button>);
  const text = s => tab === "Native" ? s.text_native : tab === "Roman" ? s.text_roman || (s.language === "en" ? s.text_native : "Romanization unavailable") : s.text_english || (s.language === "en" ? s.text_native : "Translation unavailable");
  return <Shell><a href="meetings.html">← Meetings</a><ErrorBox error={error}/>{!meeting && !error && <p>Loading meeting…</p>}{meeting && <><div className="page-head"><h1>{meeting.title}</h1><button className="button" onClick={download}>Download transcript</button></div><video ref={player} src={media || undefined} controls preload="metadata" className="recording" onError={() => setError("Recording playback failed. Reload to refresh access.")}/><h2>Transcript</h2><nav className="tabs" aria-label="Transcript language">{["Native", "Roman", "English"].map(t => <button className="tab" aria-pressed={tab === t} key={t} onClick={() => setTab(t)}>{t}</button>)}</nav><label>Search transcript<input type="search" value={query} onChange={e => setQuery(e.target.value)}/></label><section className="transcript">{meeting.transcript?.filter(s => `${s.text_native} ${s.text_roman} ${s.text_english}`.toLowerCase().includes(query.toLowerCase())).map(s => <article className="t-line" key={s.segment_id}><button className="ts-chip" onClick={() => seek(s.segment_id)}>{clock(s.start)}</button><div><strong>{s.speaker_name || s.speaker}</strong> <span className="lang-pill">{s.language}</span> <span className={`quality-badge quality-badge--${s.quality}`}>{s.quality}</span><p>{text(s)}</p></div></article>)}</section><h2>Meeting summary</h2>{meeting.summary_edited && <p>Edited summary; original AI claims remain below.</p>}<label>Edit meeting summary<textarea className="summary-editor" value={summary} onChange={e => { setSummary(e.target.value); setSaved(""); }}/></label><div className="section-actions"><button className="button button--ink" disabled={busy} onClick={save}>Save summary</button><button className="button" onClick={revert}>Revert to original</button><p role="status">{saved}</p></div>{[["Overview claims", "overviewClaims"], ["Key discussion", "keyDiscussion"], ["Decisions", "decisions"], ["Actions", "actionItems"], ["Questions", "questions"], ["Risks", "concerns"]].map(([name, key]) => <section className="block" key={key}><h3>{name}</h3>{meeting.intelligence?.[key]?.length ? meeting.intelligence[key].map((item, index) => <p key={index}>{item.text}{item.owner && ` — ${item.owner}`}{item.due && ` · ${item.due}`} {citations(item)}</p>) : <p>No {name.toLowerCase()} extracted.</p>}</section>)}</>}</Shell>;
}
function App() {
  const path = location.pathname;
  const login = path === "/" || path.endsWith("index.html");
  if (!login && !token()) { location.replace("index.html"); return <p>Sign in required…</p>; }
  if (login) return <Login/>;
  if (path.endsWith("upload.html")) return <Upload/>;
  if (path.endsWith("processing.html")) return <Processing/>;
  if (path.endsWith("meeting.html")) return <Detail/>;
  return <Library/>;
}
createRoot(document.getElementById("root")).render(<App/>);
