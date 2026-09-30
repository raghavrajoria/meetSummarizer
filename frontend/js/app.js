/* ==========================================================================
   meetReaderCB — interface behaviour
   Renders the meeting list and meeting page from the IndicMeet API.
   ========================================================================== */

(function () {
  "use strict";

  const API_BASE = window.MEETINGS_API_BASE || "http://127.0.0.1:8000";
  let DATA = [];
  const $ = (id) => document.getElementById(id);

  /* ------------------------------------------------------------------ utils */

  function fmtLong(sec) {
    if (!isFinite(sec) || sec <= 0) return "—";
    const total = Math.round(sec);
    if (total < 3600) {
      const m = Math.floor(total / 60), s = total % 60;
      return s ? `${m}m ${s}s` : `${m}m`;
    }
    const h = Math.floor(total / 3600), m = Math.round((total % 3600) / 60);
    return `${h}h ${m}m`;
  }

  const speakerMapOf = (mt) => new Map((mt.speakers || []).map((s) => [s.id, s]));
  const speakerIdsOf = (mt) => [...new Set(mt.segments.map((s) => s.speaker))];
  const speakersOf = (mt) => {
    const map = speakerMapOf(mt);
    return speakerIdsOf(mt).map((id) => map.get(id)?.name || "Participant");
  };

  function speakerFor(mt, id) {
    const found = speakerMapOf(mt).get(id);
    const fallbackNames = ["Amit Kumar", "Priya Singh", "Rohit Mehta", "Nikhil Sharma", "Sneha Iyer", "Vikram Rao", "Neha Gupta"];
    const n = Number(String(id).replace(/\D/g, "")) || 1;
    return found || { id, name: fallbackNames[(n - 1) % fallbackNames.length], role: id, join: "—", leave: "—" };
  }

  function initials(name) {
    const clean = (name || "Participant").replace(/[^A-Za-z\s]/g, "").trim();
    if (!clean) return "P";
    const parts = clean.split(/\s+/).filter(Boolean);
    return (parts.length > 1 ? parts[0][0] + parts[parts.length - 1][0] : parts[0].slice(0, 2)).toUpperCase();
  }

  function segmentEnd(mt, index) {
    const explicitEnd = Number(mt.segments[index].end);
    if (Number.isFinite(explicitEnd) && explicitEnd > mt.segments[index].t) return explicitEnd;
    const next = mt.segments[index + 1];
    return next ? next.t : mt.segments[index].t + 8;
  }

  function speakerStats(mt) {
    const stats = new Map();
    mt.segments.forEach((seg, i) => {
      if (seg.quality === "rejected" || seg.status === "rejected") return;
      const current = stats.get(seg.speaker) || { id: seg.speaker, lines: 0, intervals: [] };
      current.lines += 1;
      const start = Number(seg.t), end = segmentEnd(mt, i);
      if (Number.isFinite(start) && end > start) current.intervals.push([start, end]);
      stats.set(seg.speaker, current);
    });
    const rows = [...stats.values()].map(({ id, lines, intervals }) => {
      intervals.sort((a, b) => a[0] - b[0]);
      let seconds = 0, start = null, end = null;
      for (const interval of intervals) {
        if (start === null) [start, end] = interval;
        else if (interval[0] <= end) end = Math.max(end, interval[1]);
        else { seconds += end - start; [start, end] = interval; }
      }
      if (start !== null) seconds += end - start;
      return { id, lines, seconds };
    });
    const total = rows.reduce((n, s) => n + s.seconds, 0) || 1;
    return rows.map((s) => ({ ...s, pct: Math.round((s.seconds / total) * 100) }));
  }

  function langsOf(mt) {
    const names = { en: "English", hi: "Hindi", bn: "Bengali", mixed: "Mixed" };
    return [...new Set(mt.segments.map((s) => s.lang))].map((l) => names[l] || l);
  }

  function mediaURL(src) {
    if (!src) return "";
    if (/^https?:\/\//i.test(src)) return src;
    return src.startsWith("/") ? `${API_BASE}${src}` : new URL(src, location.href).href;
  }

  /** Read the true duration out of the media file itself.
   *  The probe elements are kept in _probes: a detached <video> with no
   *  reference can be collected before loadedmetadata ever fires. */
  const _probes = [];
  const _durCache = new Map();

  function probeDuration(src, cb) {
    if (_durCache.has(src)) { cb(_durCache.get(src)); return; }
    const v = document.createElement("video");
    _probes.push(v);
    v.preload = "metadata";
    v.muted = true;
    const done = (d) => {
      _durCache.set(src, d);
      cb(d);
      const i = _probes.indexOf(v);
      if (i > -1) _probes.splice(i, 1);
    };
    v.addEventListener("loadedmetadata", () => done(v.duration));
    v.addEventListener("error", () => done(NaN));
    v.src = src;
  }

  const toastEl = $("toast");
  let toastTimer = 0;
  function toast(msg) {
    if (!toastEl) return;
    toastEl.textContent = msg;
    toastEl.classList.add("is-on");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove("is-on"), 1800);
  }

  async function copyText(text, label) {
    try {
      await navigator.clipboard.writeText(text);
      toast(label + " copied");
    } catch (_) {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); toast(label + " copied"); }
      catch (e) { toast("Copy unavailable — select the text manually"); }
      document.body.removeChild(ta);
    }
  }

  const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  const icon = {
    play: '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4.5 2.5v11l9-5.5z"/></svg>',
  };

  function itemText(item) {
    return typeof item === "string" ? item : (item?.text || item?.task || "");
  }

  function reviewBadge(status) {
    if (status !== "review" && status !== "rejected") return "";
    const label = status === "review" ? "Review" : "Rejected";
    return `<span class="quality-badge quality-badge--${status}">${label}</span>`;
  }

  function verifiedBadge(item) {
    return item && typeof item === "object" && item.verified === false
      ? '<span class="quality-badge quality-badge--unverified">Unverified</span>' : "";
  }

  function evidenceChips(item, mt) {
    if (!item || typeof item !== "object") return "";
    const evidence = Array.isArray(item.evidence) ? item.evidence :
      (item.t != null ? [{ t: item.t, time: item.time }] : []);
    return evidence.map((ref) => {
      const source = ref.segmentId && mt.segments.find((s) => s.id === ref.segmentId);
      const at = Number(ref.t ?? source?.t);
      if (!Number.isFinite(at)) return "";
      const label = ref.time || source?.time || `${Math.floor(at / 60)}:${String(Math.floor(at % 60)).padStart(2, "0")}`;
      return `<button class="evidence-chip js-seek" type="button" data-t="${at}" aria-label="Play evidence at ${esc(label)}">${esc(label)}</button>`;
    }).join("");
  }

  function startApp() {

  /* ---------------------------------------------------------------- logout */

  const logout = $("logout");
  if (logout) {
    logout.addEventListener("click", () => { window.location.href = "index.html"; });
  }

  /* ====================================================== list pages ====== */

  const listEl = $("meetingList");

  if (listEl) {
    const mode     = listEl.dataset.mode;           // "overview" | "meetings"
    const dateIn   = $("dateFilter");
    const clearBtn = $("clearDate");
    const emptyEl  = $("listEmpty");

    const dates  = [...new Set(DATA.map((m) => m.date))].sort().reverse();
    const latest = dates[0] || "";

    function rowHTML(mt) {
      return `
        <div class="m-row ${mode === "meetings" ? "m-row--link" : ""}" data-id="${mt.id}"
             ${mode === "meetings" ? 'role="link" tabindex="0"' : ""}>
          <div>
            <span class="t-title">${esc(mt.title)}</span>
            <span class="t-sub">${esc(mt.group)}</span>
          </div>
          <span class="cell">${mt.dateLabel} · ${mt.time}</span>
          <span class="cell cell--dur" data-dur="${mt.id}">—</span>
          <span class="status status--done">Processed</span>
          <div class="row-actions">
            <a class="button" href="meeting.html?id=${mt.id}&tab=transcript">Transcript</a>
            <a class="button" href="meeting.html?id=${mt.id}&tab=summary">Overview</a>
          </div>
        </div>`;
    }

    /** Stats always describe what is currently on screen. */
    let statsToken = 0;
    function renderStats(rows) {
      const stM = $("stMeetings"), stS = $("stSpeakers"), stL = $("stLines"), stT = $("stTime");
      if (stM) stM.textContent = rows.length;
      // sum per meeting: "Speaker 1" in one meeting is not the same person as
      // "Speaker 1" in another, so a distinct union would undercount badly
      if (stS) stS.textContent = rows.reduce((n, m) => n + speakerIdsOf(m).length, 0);
      if (stL) stL.textContent = rows.reduce((n, m) => n + m.segments.length, 0);
      if (stT) {
        const token = ++statsToken;
        if (!rows.length) { stT.textContent = "—"; return; }
        let total = 0, done = 0;
        rows.forEach((mt) => probeDuration(mt.media, (d) => {
          if (token !== statsToken) return;       // a newer filter won the race
          if (isFinite(d)) total += d;
          if (++done === rows.length) stT.textContent = fmtLong(total);
        }));
      }
    }

    function render(filterDate) {
      const rows = DATA.filter((m) => !filterDate || m.date === filterDate);
      listEl.innerHTML = rows.map(rowHTML).join("");
      if (emptyEl) emptyEl.hidden = rows.length !== 0;

      rows.forEach((mt) => {
        probeDuration(mediaURL(mt.media), (d) => {
          const cell = listEl.querySelector(`[data-dur="${mt.id}"]`);
          if (cell) cell.textContent = isFinite(d) ? fmtLong(d) : "—";
        });
      });

      if (mode === "meetings") {
        listEl.querySelectorAll(".m-row--link").forEach((row) => {
          const go = (ev) => {
            // the Transcript / Summary buttons handle their own navigation
            if (ev && ev.target.closest(".row-actions")) return;
            location.href = "meeting.html?id=" + row.dataset.id;
          };
          row.addEventListener("click", go);
          row.addEventListener("keydown", (e) => {
            if (e.target.closest(".row-actions")) return;
            if (e.key === "Enter" || e.key === " ") { go(); e.preventDefault(); }
          });
        });
      }

      const countEl = $("countLabel");
      if (countEl) countEl.textContent = rows.length + (rows.length === 1 ? " meeting" : " meetings");

      const headDate = $("headDate");
      if (headDate) headDate.textContent = filterDate ? (rows[0] ? rows[0].dateLabel : filterDate) : "All dates";

      renderStats(rows);
    }

    if (dateIn) {
      dateIn.value = latest;
      dateIn.addEventListener("change", () => render(dateIn.value));
    }
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        if (dateIn) dateIn.value = "";
        render("");
      });
    }
    render(latest);
  }

  /* ==================================================== meeting page ====== */

  const titleEl = $("mTitle");

  if (titleEl && DATA.length) {
    const params = new URLSearchParams(location.search);
    const mt = DATA.find((m) => m.id === params.get("id")) || DATA[0];

    document.title = "meetReaderCB — " + mt.title;
    $("mGroup").textContent = mt.group;
    titleEl.textContent = mt.title;

    const speakerIds = speakerIdsOf(mt);
    const speakers = speakersOf(mt);
    const intel = mt.intelligence || {};
    $("mMeta").innerHTML =
      `<span>${mt.dateLabel}</span><span>${mt.time}</span>` +
      `<span id="metaDur">—</span><span>${speakerIds.length} participants</span>`;

    // --- summary
    $("sumText").textContent = intel.discussed || mt.summary;
    if (mt.summaryVerified === false) {
      $("sumText").insertAdjacentHTML("afterend", '<span class="quality-badge quality-badge--unverified">Unverified</span>');
    }
    $("keyDiscussion").innerHTML = (intel.keyDiscussion || []).map((item) =>
      `<span>${esc(itemText(item))}${verifiedBadge(item)}${evidenceChips(item, mt)}</span>`).join("");
    $("decisionsList").innerHTML = (intel.decisions || []).map((item) =>
      `<li><span class="check" aria-hidden="true">✓</span><span>${esc(itemText(item))}${verifiedBadge(item)}${evidenceChips(item, mt)}</span></li>`).join("");
    $("actionItems").innerHTML = (intel.actionItems || []).map((item) => `
      <article class="action-row">
        <div><span class="label">Owner</span><strong>${esc(item.owner ?? "Unassigned")}</strong></div>
        <div><span class="label">Task</span><p>${esc(item.task)}${verifiedBadge(item)}${evidenceChips(item, mt)}</p></div>
        <div><span class="label">Due date</span><span>${esc(item.due || "—")}</span></div>
        ${item.t == null ? "" : `<div><span class="label">Timestamp</span><button class="ts-chip js-seek" type="button" data-t="${item.t}" aria-label="Play from ${esc(item.time)}">${esc(item.time || "—")}</button></div>`}
      </article>`).join("");
    $("followUpsList").innerHTML = (intel.followUps || []).map((item, i) =>
      `<li><span class="n">${String(i + 1).padStart(2, "0")}</span><span>${esc(itemText(item))}${verifiedBadge(item)}${evidenceChips(item, mt)}</span>${item.t == null ? "" : `<button class="ts-chip js-seek" type="button" data-t="${item.t}" aria-label="Play from ${esc(item.time)}">${esc(item.time || "—")}</button>`}</li>`).join("");
    $("questionsList").innerHTML = (intel.questions || []).map((item) => `<li>${esc(itemText(item))}${verifiedBadge(item)}${evidenceChips(item, mt)}</li>`).join("");
    $("concernsList").innerHTML = (intel.concerns || []).map((item) => `<li>${esc(itemText(item))}${verifiedBadge(item)}${evidenceChips(item, mt)}</li>`).join("");
    $("followUpsBlock").hidden = !(intel.followUps || []).length;
    $("questionsBlock").hidden = !(intel.questions || []).length;
    $("concernsBlock").hidden = !(intel.concerns || []).length;
    $("facts").innerHTML = `
      <div><dt>Participants</dt><dd>${speakerIds.length}</dd></div>
      <div><dt>Transcript lines</dt><dd>${mt.segments.length}</dd></div>
      <div><dt>Languages</dt><dd>${langsOf(mt).join(", ")}</dd></div>
      <div><dt>Duration</dt><dd id="factDur">—</dd></div>`;

    $("insightSummary").textContent = mt.summary;
    $("insightKeyPoints").innerHTML = (intel.keyDiscussion || []).map((item) => `<li>${esc(itemText(item))}${verifiedBadge(item)}${evidenceChips(item, mt)}</li>`).join("");
    $("insightDecisions").innerHTML = (intel.decisions || []).map((item) => `<li>${esc(itemText(item))}${verifiedBadge(item)}${evidenceChips(item, mt)}</li>`).join("");
    $("insightActions").innerHTML = (intel.actionItems || []).map((item) => `<li><strong>${esc(item.owner ?? "Unassigned")}</strong>: ${esc(item.task)}${verifiedBadge(item)}${evidenceChips(item, mt)}</li>`).join("");
    $("insightFollowUps").innerHTML = (intel.followUps || []).map((item, i) =>
      `<li><span class="n">${String(i + 1).padStart(2, "0")}</span><span>${esc(itemText(item))}${verifiedBadge(item)}${evidenceChips(item, mt)}</span>${item.t == null ? "" : `<button class="ts-chip js-seek" type="button" data-t="${item.t}" aria-label="Play from ${esc(item.time)}">${esc(item.time || "—")}</button>`}</li>`).join("");

    const stats = speakerStats(mt);
    $("participantsList").innerHTML = `
      <div class="th">
        <span>Name</span><span>Join</span><span>Leave</span><span class="col-speak">Speaking</span><span>%</span>
      </div>` + stats.map((s) => {
        const person = speakerFor(mt, s.id);
        return `
          <div class="tr">
            <div class="who">
              <span class="avatar avatar--sm" aria-hidden="true">${initials(person.name)}</span>
              <div><strong>${esc(person.name)}</strong><span>${esc(person.role || s.id)}</span></div>
            </div>
            <span class="cell">${esc(person.join || "—")}</span>
            <span class="cell">${esc(person.leave || "—")}</span>
            <span class="cell col-speak">${fmtLong(s.seconds)}</span>
            <span class="pct">${s.pct}%</span>
          </div>`;
      }).join("");
    $("speakingBars").innerHTML = stats.map((s, i) => {
      const person = speakerFor(mt, s.id);
      return `
        <div class="bar-row">
          <span>${esc(person.name)}</span>
          <span class="track"><i class="${i > 3 ? "is-dim" : ""}" style="width:${Math.max(4, s.pct)}%"></i></span>
          <b>${s.pct}%</b>
        </div>`;
    }).join("");

    // --- transcript
    $("tMeta").textContent = `${mt.segments.length} lines · ${speakerIds.length} participants`;

    $("transcriptList").innerHTML = mt.segments.map((s, i) => `
      <article class="t-line" data-t="${s.t}" data-i="${i}" data-view="both">
        <button class="ts-chip" type="button" aria-label="Play from ${s.time}">${s.time}</button>
        <div>
          <div class="who">
            <span class="avatar avatar--sm" aria-hidden="true">${initials(speakerFor(mt, s.speaker).name)}</span>
            <strong>${esc(speakerFor(mt, s.speaker).name)}</strong>
        <span class="lang-pill">${esc(s.lang)}</span>
        ${reviewBadge(s.quality || s.status)}
        ${s.verified === false ? '<span class="quality-badge quality-badge--unverified">Unverified</span>' : ""}
      </div>
          <p class="t-tx">${esc(s.tx)}</p>
          ${(s.en || s.roman) ? `<p class="t-en">${esc(s.en || s.roman)}</p>` : ""}
        </div>
        <button class="mini-play" type="button" aria-label="Play from ${s.time}">${icon.play}</button>
      </article>`).join("");

    const lines = Array.from(document.querySelectorAll(".t-line"));

    // --- player
    if (window.MeetlyPlayer) window.MeetlyPlayer.load(mediaURL(mt.media));

    $("media").addEventListener("loadedmetadata", function () {
      const d = this.duration;
      [$("metaDur"), $("factDur")].forEach((el) => {
        if (el) el.textContent = isFinite(d) ? fmtLong(d) : "—";
      });
    });

    /* ----------------------------------------------------------- tabs */

    const tabs  = Array.from(document.querySelectorAll(".tab"));
    const views = Array.from(document.querySelectorAll(".panel-view"));

    function showTab(name) {
      if (name === "overview") name = "summary";
      const tab = tabs.find((t) => t.dataset.tab === name) || tabs[0];
      tabs.forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
      views.forEach((v) => { v.hidden = v.id !== tab.dataset.tab; });
    }
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => showTab(tab.dataset.tab));
      tab.addEventListener("keydown", (ev) => {
        const i = tabs.indexOf(tab);
        let next = null;
        if (ev.key === "ArrowRight") next = tabs[(i + 1) % tabs.length];
        if (ev.key === "ArrowLeft")  next = tabs[(i - 1 + tabs.length) % tabs.length];
        if (next) { next.focus(); showTab(next.dataset.tab); ev.preventDefault(); }
      });
    });
    showTab(params.get("tab") || "summary");

    /* ------------------------------------------- original / english */

    const toggle = $("langToggle");
    if (toggle && !mt.segments.some((s) => s.en) && mt.segments.some((s) => s.roman)) {
      const romanButton = toggle.querySelector('[data-view="english"]');
      if (romanButton) romanButton.textContent = "Romanization";
    }
    let view = "both";

    function setView(next) {
      view = next;
      lines.forEach((el) => { el.dataset.view = next; });
      if (toggle) {
        toggle.querySelectorAll("button").forEach((b) =>
          b.setAttribute("aria-pressed", String(b.dataset.view === next)));
      }
      if (search && search.value.trim()) runSearch();
    }
    if (toggle) {
      toggle.addEventListener("click", (ev) => {
        const b = ev.target.closest("button");
        if (b) setView(b.dataset.view);
      });
    }

    /* ------------------------------------------------- jump + follow */

    // Some browsers (and anyone with reduced motion enabled) ignore
    // behavior:"smooth" entirely, which would leave the active line off-screen.
    const reduceMotion = window.matchMedia
      ? window.matchMedia("(prefers-reduced-motion: reduce)")
      : { matches: false };

    function activate(el, scroll) {
      lines.forEach((l) => l.classList.toggle("is-active", l === el));
      if (!scroll || !el) return;
      const box = $("tScroll");
      if (!box) return;
      const top = Math.max(0, el.offsetTop - box.clientHeight / 2 + el.clientHeight / 2);
      if (reduceMotion.matches) {
        box.scrollTop = top;                  // guaranteed to apply
      } else {
        box.scrollTo({ top, behavior: "smooth" });
        // if smooth was a no-op, land it anyway
        const was = box.scrollTop;
        setTimeout(() => { if (box.scrollTop === was) box.scrollTop = top; }, 220);
      }
    }

    lines.forEach((el) => {
      const at = Number(el.dataset.t || 0);
      const jump = () => {
        activate(el, false);
        showTab("transcript");
        if (window.MeetlyPlayer) window.MeetlyPlayer.seek(at, true);
      };
      el.querySelector(".ts-chip").addEventListener("click", jump);
      el.querySelector(".mini-play").addEventListener("click", jump);
    });

    document.querySelectorAll(".js-seek").forEach((btn) => {
      btn.addEventListener("click", () => {
        const at = Number(btn.dataset.t || 0);
        showTab("transcript");
        const line = lines.find((el) => Number(el.dataset.t || 0) >= at) || lines[lines.length - 1];
        if (line) activate(line, true);
        if (window.MeetlyPlayer) window.MeetlyPlayer.seek(at, true);
      });
    });

    let lastActive = null;
    window.addEventListener("meetly:time", (ev) => {
      const t = ev.detail.t;
      let current = null;
      for (const el of lines) {
        if (Number(el.dataset.t || 0) <= t) current = el; else break;
      }
      if (current && current !== lastActive) {
        lastActive = current;
        activate(current, true);
      }
    });

    /* ----------------------------------------------------- search */

    const search = $("tSearch"), tEmpty = $("tEmpty");

    function runSearch() {
      const q = search.value.trim();
      const needle = q.toLowerCase();
      let shown = 0;

      lines.forEach((el, i) => {
        const seg = mt.segments[i];
        const hay = (speakerFor(mt, seg.speaker).name + " " + seg.tx + " " + seg.en + " " + seg.roman).toLowerCase();
        const match = !q || hay.includes(needle);
        el.hidden = !match;
        if (match) shown++;

        [["t-tx", seg.tx], ["t-en", seg.en || seg.roman || ""]].forEach(([cls, text]) => {
          const p = el.querySelector("." + cls);
          if (!p) return;
          p.innerHTML = q && text.toLowerCase().includes(needle)
            ? esc(text).replace(new RegExp(escapeRe(esc(q)), "gi"), (m) => `<mark>${m}</mark>`)
            : esc(text);
        });
      });

      if (tEmpty) tEmpty.hidden = shown !== 0;
    }

    if (search) search.addEventListener("input", runSearch);

    /* ------------------------------------------------ edit + copy */

    const sumText = $("sumText"), sumEdit = $("sumEdit");
    sumEdit && sumEdit.addEventListener("click", () => {
      const on = sumText.isContentEditable;
      sumText.contentEditable = on ? "false" : "true";
      sumText.style.outline = on ? "" : "1px solid var(--border-firm)";
      sumText.style.borderRadius = "4px";
      sumText.style.padding = on ? "" : "8px";
      sumEdit.lastChild.textContent = on ? "Edit" : "Done";
      if (!on) sumText.focus(); else toast("Summary updated locally");
    });

    $("sumCopy") && $("sumCopy").addEventListener("click",
      () => copyText(sumText.textContent.trim(), "Summary"));

    /** Respects the Original / English / Both toggle. */
    function transcriptText() {
      return mt.segments.map((s) => {
        const head = `[${s.time}] ${speakerFor(mt, s.speaker).name}`;
        if (view === "original") return `${head}: ${s.tx}`;
        if (view === "english")  return `${head}: ${s.en}`;
        return `${head}: ${s.tx}\n${" ".repeat(head.length + 2)}${s.en}`;
      }).join("\n");
    }

    $("trCopy") && $("trCopy").addEventListener("click",
      () => copyText(transcriptText(), "Transcript"));

    $("dlBtn") && $("dlBtn").addEventListener("click", () => {
      const body =
        `${mt.title}\n${mt.dateLabel} · ${mt.time}\n` +
        `${speakerIds.length} participants · ${mt.segments.length} lines\n\n` +
        `SUMMARY\n${mt.summary}\n\nTRANSCRIPT\n${transcriptText()}\n`;
      const blob = new Blob([body], { type: "text/plain;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = mt.id + "-transcript.txt";
      a.click();
      URL.revokeObjectURL(a.href);
      toast("Transcript downloaded");
    });
  }

  /* ------------------------------------------------------------------ login */

  const loginForm = $("loginForm");
  if (loginForm) {
    loginForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      window.location.href = "dashboard.html";
    });
  }
  }

  fetch(`${API_BASE}/sessions`)
    .then((response) => {
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      return response.json();
    })
    .then(async (sessions) => {
      DATA = Array.isArray(sessions) ? sessions : [];
      const detailId = new URLSearchParams(location.search).get("id");
      if (detailId && document.getElementById("mTitle")) {
        const detailResponse = await fetch(`${API_BASE}/sessions/${encodeURIComponent(detailId)}`);
        if (detailResponse.ok) {
          const detail = await detailResponse.json();
          DATA = [detail, ...DATA.filter((session) => session.id !== detail.id)];
        }
      }
      startApp();
    })
    .catch((error) => {
      console.error("Could not load meeting sessions", error);
      const toastEl = document.getElementById("toast");
      if (toastEl) { toastEl.textContent = "Cannot reach the meeting API. Start the backend and reload."; toastEl.classList.add("is-visible"); }
    });
})();
