import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, apiUrl, token, clearAuth, upload } from "./api.js";
import "../css/style.css";
import "./workflow.css";
import { speakerLabels } from "./speakerNames.js";

const id = () => new URLSearchParams(location.search).get("id");
const clock = n => `${Math.floor(n / 60)}:${String(Math.floor(n % 60)).padStart(2, "0")}`;
function ErrorBox({ error }) { return error ? <p role="alert" className="error">{error}</p> : null; }
function Shell({ children }) {
 const [config,setConfig]=useState(null),[error,setError]=useState('');
 useEffect(()=>{api('/config').then(setConfig).catch(e=>setError(e.message));},[]);
 async function logout(){try{await api('/auth/logout',{method:'POST'});clearAuth();location.assign('index.html');}catch(e){setError(e.message);}}
 return <div className="app"><aside className="sidebar"><a className="brand" href="dashboard.html">meetReaderCB</a><nav aria-label="Main"><a className="nav-link" aria-current={location.pathname.endsWith('dashboard.html')?'page':undefined} href="dashboard.html">Overview</a><a className="nav-link" aria-current={/meeting/.test(location.pathname)?'page':undefined} href="meetings.html">Meetings</a>{config && !config.precomputed_only && <a className="nav-link" href="upload.html">Upload recording</a>}</nav><div className="sidebar-foot"><hr className="rule"/><div className="profile"><span className="avatar">A</span><div className="profile-id"><strong>Admin</strong><span>Account</span></div><button className="logout" onClick={logout} aria-label="Sign out">↪</button></div></div></aside><main className="content"><ErrorBox error={error}/>{config?.demo_mode && <p className="demo-banner">Pre-computed results from earlier pipeline runs. Live hosted-model processing is not connected yet.</p>}{children}</main></div>;
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
  return <main className="login"><div className="login-inner"><div className="login-card"><header><div className="login-mark">meetReaderCB</div><p className="login-tag">Understand every meeting.</p></header><form className="login-form" onSubmit={submit}><div className="field"><label htmlFor="username">Email or username</label><input id="username" name="username" placeholder="you@company.com" autoComplete="username" required/></div><div className="field"><label htmlFor="password">Password</label><input id="password" name="password" type="password" placeholder="••••••••" autoComplete="current-password" required/></div><ErrorBox error={error}/><button className="button button--ink button--full" disabled={busy}>{busy?'Signing in…':'Sign In'}</button>{demo && <p className="muted">Demo account: demo / demo-password</p>}<div className="login-foot"><button className="link-quiet" type="button" onClick={()=>setError('Contact your account administrator to reset your password.')}>Forgot password?</button></div></form></div></div><p className="login-legal">meetReaderCB · Meeting intelligence for business teams</p></main>;
}
function Library() {
 const [meetings,setMeetings]=useState(null),[query,setQuery]=useState(''),[date,setDate]=useState(''),[error,setError]=useState('');
 useEffect(()=>{api('/meetings').then(setMeetings).catch(e=>setError(e.message));},[]);
 const overview=location.pathname.endsWith('dashboard.html');
 const filtered=meetings?.filter(m=>(!date||m.date===date)&&`${m.title} ${m.group}`.toLowerCase().includes(query.toLowerCase()));
 async function remove(m){if(!confirm(`Delete “${m.title}” and its recording?`))return;try{await api(`/meetings/${m.id}`,{method:'DELETE'});setMeetings(items=>items.filter(x=>x.id!==m.id));}catch(e){setError(e.message);}}
 const seconds=m=>m.duration || Math.max(0,...(m.transcript||[]).map(s=>s.end));
 const duration=n=>`${Math.floor(n/60)}m ${Math.round(n%60)}s`;
 return <Shell><div className="page-head"><h1>{overview?'Overview':'Meetings'}</h1><div className="sub"><b>{overview?'Your meetings':`${meetings?.length||0} meetings`}</b><span>{overview?'Recorded meeting library':'Select a meeting to open its recording'}</span></div></div>{overview && <section className="stats" aria-label="Summary"><div className="stat"><strong>{meetings?.length??'—'}</strong><span>Meetings</span></div><div className="stat"><strong>{meetings?duration(meetings.reduce((n,m)=>n+seconds(m),0)):'—'}</strong><span>Meeting time</span></div></section>}<section className="section library-section"><div className="section-head"><div><p className="eyebrow">{overview?'Recordings':'Library'}</p><h2 className="h2">{overview?'Meetings':'All recordings'}</h2></div></div><div className="library-toolbar"><label className="library-search">Search meetings<input type="search" placeholder="Search meetings" value={query} onChange={e=>setQuery(e.target.value)}/></label><div className="library-filters"><label>Date<input type="date" value={date} onChange={e=>setDate(e.target.value)}/></label><button className="button" onClick={()=>setDate('')}>All dates</button></div></div><ErrorBox error={error}/>{!meetings&&!error&&<p>Loading meetings…</p>}{filtered?.length===0&&<p className="empty-state">No meetings found.</p>}<div className="m-list">{filtered?.map(m=><article className="m-row meeting-row" key={m.id}><div><h2 className="t-title">{m.title}</h2><span className="t-sub">{m.group||'Recording'}</span></div><span className="cell">{m.date||'Date not recorded'}</span><span className="cell">{duration(seconds(m))}</span><span className={`status ${m.status==='processed'?'status--done':''}`}>{m.status}</span><div className="row-actions"><a className="button" href={`meeting.html?id=${m.id}&view=transcript`}>Transcript</a><a className="button" href={`meeting.html?id=${m.id}&view=overview`}>Overview</a><button className="icon-button" aria-label="Delete" title="Delete meeting" onClick={()=>remove(m)}>×</button></div></article>)}</div></section></Shell>;
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
  const [meeting, setMeeting] = useState(null), [error, setError] = useState(""), [media, setMedia] = useState(""), [query, setQuery] = useState(""), [summary, setSummary] = useState(""), [saved, setSaved] = useState(""), [busy, setBusy] = useState(false);
  const player = useRef(null), timer = useRef(null);
  const [view,setView]=useState(new URLSearchParams(location.search).get('view')||'overview'),[editing,setEditing]=useState(false),[demo,setDemo]=useState(false);
  useEffect(()=>{api('/config').then(c=>setDemo(c.demo_mode)).catch(e=>setError(e.message));},[]);
  function switchView(v){setView(v);const url=new URL(location.href);url.searchParams.set('view',v);history.replaceState(null,'',url);}
  const names=meeting?speakerLabels(meeting,demo):{};
  const display=s=>names[s.speaker] || {name:s.speaker_name||s.speaker,source:s.speaker_name_source||'none'};
  async function copy(value){try{await navigator.clipboard.writeText(value);setSaved('Copied');}catch{setError('Copy failed. Use Download transcript instead.');}}
  const MediaElement = meeting?.kind === "audio" ? "audio" : "video";
  async function load() { const m = await api(`/meetings/${id()}`); setMeeting(m); setSummary(m.summary || ""); }
  async function refreshMedia() { const reply = await api(`/meetings/${id()}/media-url`); const position = player.current?.currentTime || 0, playing = player.current && !player.current.paused; setMedia(apiUrl(reply.url)); if (player.current) player.current.onloadedmetadata = () => { player.current.currentTime = position; if (playing) player.current.play().catch(() => {}); }; }
  useEffect(() => { load().then(refreshMedia).catch(e => setError(e.message)); timer.current = setInterval(() => refreshMedia().catch(e => setError(e.message)), 90000); return () => clearInterval(timer.current); }, []);
  function seek(segmentId) { const s = meeting.transcript.find(s => s.segment_id === segmentId); if (s && player.current) { player.current.currentTime = s.start; player.current.play().catch(e => setError(e.message)); } }
  async function save() { setBusy(true); setSaved(""); try { await api(`/meetings/${id()}/summary`, { method: "PUT", json: { text: summary } }); setSaved("Summary saved"); } catch (e) { setError(e.message); } finally { setBusy(false); } }
  async function revert() { try { await api(`/meetings/${id()}/summary`, { method: "DELETE" }); await load(); setSaved("Original summary restored"); } catch (e) { setError(e.message); } }
  async function download() { try { const blob = await api(`/meetings/${id()}/transcript`, { blob: true }); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = "transcript.json"; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000); } catch (e) { setError(e.message); } }
  const citations = item => (item.source_segment_ids || []).filter(ref => meeting.transcript.some(s => s.segment_id === ref)).map(ref => <button className="evidence-chip" aria-label={`Play evidence ${ref}`} key={ref} onClick={() => seek(ref)}>{clock(meeting.transcript.find(s => s.segment_id === ref).start)}</button>);
  return <Shell><ErrorBox error={error}/>{!meeting&&!error&&<p>Loading meeting…</p>}{meeting&&<><div className="page-head meeting-heading"><h1>{meeting.title}</h1><div className="sub"><span>{meeting.date||'Date not recorded'}</span></div></div><MediaElement ref={player} src={media||undefined} controls preload="metadata" className="recording" onError={()=>setError('Recording playback failed. Reload to refresh access.')}/><details className="provenance"><summary>Recording provenance</summary><p className="model-metadata">ASR model: {meeting.provenance?.model||(meeting.asr_model_versions||[]).join(', ')||'unknown'} · Run date: {meeting.provenance?.run_date||'unknown'}</p>{meeting.provenance&&<p className="provenance-metadata">{meeting.provenance.verdict} · {meeting.provenance.evidence}</p>}</details><nav className="tabs meeting-tabs" aria-label="Meeting view">{['overview','transcript'].map(v=><button key={v} className="tab" aria-selected={view===v} onClick={()=>switchView(v)}>{v==='overview'?'Overview':'Transcript'}</button>)}</nav><section hidden={view!=='transcript'} className="panel-view"><div className="section-head"><div><p className="eyebrow">Recorded conversation</p><h2 className="h2">Transcript</h2></div><div className="transcript-tools"><label className="search-wrap"><span className="sr-only">Search transcript</span><input className="search" type="search" placeholder="Search transcript" value={query} onChange={e=>setQuery(e.target.value)}/></label><button className="button" onClick={()=>copy(meeting.transcript.map(s=>`${clock(s.start)} ${display(s).name}: ${s.text_native}`).join('\n'))}>Copy</button><button className="button" aria-label="Download transcript" onClick={download}>Download</button></div></div>{demo&&<p className="alias-note">Names marked inferred come from transcript evidence. Demo aliases are invented display names, not participant identities.</p>}<section className="transcript t-scroll">{meeting.transcript?.filter(s=>`${display(s).name} ${s.text_native} ${s.text_roman||''} ${s.text_english||''}`.toLowerCase().includes(query.toLowerCase())).map(s=><article className="t-line" key={s.segment_id}><button className="ts-chip evidence-chip" aria-label={`Play evidence ${s.segment_id}`} onClick={()=>seek(s.segment_id)}>{clock(s.start)}</button><div><div className="who"><span className="avatar">{display(s).name.split(' ').map(n=>n[0]).join('').slice(0,2)}</span><strong>{display(s).name}</strong>{display(s).source==='inferred'&&<span className="lang-pill">inferred</span>}{display(s).source==='demo_alias'&&<span className="lang-pill">demo alias</span>}<span className="lang-pill">{s.language}</span><span className={`quality-badge quality-badge--${s.quality}`}>{s.quality}</span></div><p className="t-tx">{s.text_native}</p></div><button className="mini-play" aria-label={`Play transcript ${s.segment_id}`} onClick={()=>seek(s.segment_id)}>▶</button></article>)}</section></section><section hidden={view!=='overview'} className="panel-view"><div className="section-head"><div><p className="eyebrow">Meeting summary</p><h2 className="h2">Overview</h2></div>{meeting.summary_status!=='not_generated'&&<div className="section-actions"><button className="button" onClick={()=>setEditing(!editing)}>Edit</button><button className="button" onClick={()=>copy(summary)}>Copy</button></div>}</div>{meeting.summary_status==='not_generated'?<p>summary not generated</p>:<><section className="block"><p className="eyebrow">What was discussed</p>{editing?<><label>Edit meeting summary<textarea className="summary-editor" value={summary} onChange={e=>setSummary(e.target.value)}/></label><div className="section-actions"><button className="button button--ink" disabled={busy} onClick={save}>Save summary</button><button className="button" onClick={revert}>Revert to original</button></div></>:<p className="lede">{summary}</p>}{meeting.summary_edited&&<p className="muted">Edited summary; original AI claims remain below.</p>}</section>{[['Overview claims','overviewClaims'],['Key discussion','keyDiscussion'],['Decisions','decisions'],['Actions','actionItems'],['Follow-ups','followUps'],['Questions','questions'],['Risks','concerns']].map(([name,key])=><section className="block" key={key}><h3 className="eyebrow">{name}</h3>{meeting.intelligence?.[key]?.length?meeting.intelligence[key].map((item,index)=>key==='actionItems'?<div className="action-row" key={index}><div><span className="label">Owner</span><strong>{item.owner||'Unassigned'}</strong>{item.owner_name_source==='inferred'&&<span className="lang-pill">inferred</span>}</div><div><span className="label">Task</span><p>{item.text}</p></div><div><span className="label">Due date</span><span>{item.due||'Not specified'}</span></div><div><span className="label">Timestamp</span>{citations(item)}</div></div>:<p className="intelligence-row" key={index}>{key==='decisions'&&<span className="check">✓ </span>}{item.text} {citations(item)}</p>):<p className="muted">No {name.toLowerCase()} extracted.</p>}</section>)}</>}<p role="status">{saved}</p></section></>}</Shell>;
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
