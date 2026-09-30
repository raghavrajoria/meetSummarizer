/* ==========================================================================
   Meetly — media player
   A real HTML5 player with custom monochrome controls.

   Modes, decided from the actual file:
     video  — the file has a picture track
     audio  — audio only; renders a waveform derived from the real audio
     empty  — no file resolved; shows a quiet placeholder

   Waveform derivation, in order of preference:
     1. fetch + decodeAudioData -> real peaks for the whole track (needs http/https)
     2. MediaElementSource + AnalyserNode -> live levels while playing (works on file://)
     3. plain timeline, if neither is available
   Nothing here is a decorative squiggle: if no real audio data can be read,
   no waveform is drawn.
   ========================================================================== */

(function () {
  "use strict";

  const player = document.getElementById("player");
  if (!player) return;

  const media    = document.getElementById("media");
  const playBig  = document.getElementById("playBig");
  const heroPlay = document.getElementById("heroPlay");
  const heroLbl  = document.getElementById("heroPlayLabel");
  const tCur     = document.getElementById("tCur");
  const tDur     = document.getElementById("tDur");
  const scrub    = document.getElementById("scrub");
  const scrubCv  = document.getElementById("scrubCanvas");
  const waveWrap = document.getElementById("waveWrap");
  const waveCv   = document.getElementById("waveCanvas");
  const muteBtn  = document.getElementById("muteBtn");

  /** Real peaks for the whole track, 0..1, or null. */
  let peaks = null;
  /** Rolling level history from the analyser, or null. */
  let live = null;
  let analyser = null;
  let analyserBuf = null;
  let analyserDead = false;
  let rafId = 0;

  const LIVE_SLOTS = 640;

  /* ------------------------------------------------------------------ utils */

  function fmt(sec) {
    if (!isFinite(sec) || sec < 0) return "--:--";
    const s = Math.floor(sec % 60);
    const m = Math.floor((sec / 60) % 60);
    const h = Math.floor(sec / 3600);
    const pad = (n) => String(n).padStart(2, "0");
    return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
  }

  function fitCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  const progress = () => {
    const d = media.duration;
    return isFinite(d) && d > 0 ? Math.min(1, media.currentTime / d) : 0;
  };

  /* ------------------------------------------------------------------- draw */

  const INK_ON   = "#FFFFFF";
  const INK_OFF  = "rgba(255,255,255,0.22)";
  const INK_FAINT= "rgba(255,255,255,0.16)";

  function drawBars(ctx, w, h, data, played, opts) {
    const barW = opts.barW, gap = opts.gap, amp = opts.amp, minH = opts.minH;
    const step = barW + gap;
    const count = Math.floor(w / step);
    const mid = h / 2;
    const cut = played * w;

    for (let i = 0; i < count; i++) {
      const x = i * step;
      // map this bar to the data array
      const v = data.length ? data[Math.floor((i / count) * data.length)] || 0 : 0;
      const barH = Math.max(minH, v * amp);
      ctx.fillStyle = x + barW <= cut ? INK_ON : opts.off;
      ctx.fillRect(x, mid - barH / 2, barW, barH);
    }
  }

  function drawScrub() {
    if (!scrubCv) return;
    const { ctx, w, h } = fitCanvas(scrubCv);
    const played = progress();
    const data = peaks || live;

    if (data && data.length) {
      drawBars(ctx, w, h, data, played, {
        barW: 2, gap: 2, amp: h * 0.78, minH: 2, off: INK_OFF,
      });
    } else {
      const y = Math.round(h / 2) - 1;
      ctx.fillStyle = INK_OFF;
      ctx.fillRect(0, y, w, 2);
      ctx.fillStyle = INK_ON;
      ctx.fillRect(0, y, w * played, 2);
      // position marker
      const x = Math.min(w - 2, Math.max(0, w * played));
      ctx.fillRect(x, y - 4, 2, 10);
    }
  }

  function drawWave() {
    if (!waveCv || player.dataset.mode !== "audio") return;
    const { ctx, w, h } = fitCanvas(waveCv);
    const played = progress();
    const data = peaks || live;
    if (!data || !data.length) return;

    drawBars(ctx, w, h, data, played, {
      barW: 2, gap: 2, amp: h * 0.5, minH: 1, off: INK_FAINT,
    });
  }

  function render() {
    drawScrub();
    drawWave();
  }

  /* --------------------------------------------------------- peak extraction */

  async function loadPeaks() {
    const src = media.currentSrc;
    if (!src) return false;
    // fetch is blocked for file:// in most browsers; that's expected, we fall back.
    if (location.protocol === "file:") return false;
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return false;
      const res = await fetch(src);
      if (!res.ok) return false;
      const buf = await res.arrayBuffer();
      const ctx = new AC();
      const audio = await ctx.decodeAudioData(buf);
      const ch = audio.getChannelData(0);
      const buckets = 900;
      const size = Math.floor(ch.length / buckets) || 1;
      const out = new Float32Array(buckets);
      let max = 0;
      for (let i = 0; i < buckets; i++) {
        const start = i * size;
        const end = Math.min(start + size, ch.length);
        const stride = Math.max(1, Math.floor((end - start) / 256));
        let peak = 0;
        for (let j = start; j < end; j += stride) {
          const v = Math.abs(ch[j]);
          if (v > peak) peak = v;
        }
        out[i] = peak;
        if (peak > max) max = peak;
      }
      if (max > 0) for (let i = 0; i < buckets; i++) out[i] /= max;
      peaks = out;
      ctx.close && ctx.close();
      return true;
    } catch (_) {
      return false;
    }
  }

  /* ------------------------------------------------------- live analyser mode */

  function initAnalyser() {
    if (analyser || analyserDead || peaks) return;
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) { analyserDead = true; return; }
      const ctx = new AC();
      const source = ctx.createMediaElementSource(media);
      analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      analyser.connect(ctx.destination);   // keep audible
      analyserBuf = new Uint8Array(analyser.fftSize);
      live = new Float32Array(LIVE_SLOTS);
      if (ctx.state === "suspended") ctx.resume();
    } catch (_) {
      analyser = null;
      analyserDead = true;
    }
  }

  let silentFrames = 0;
  function sampleAnalyser() {
    if (!analyser || !live) return;
    analyser.getByteTimeDomainData(analyserBuf);
    let sum = 0;
    for (let i = 0; i < analyserBuf.length; i++) {
      const v = (analyserBuf[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / analyserBuf.length);

    // Write into the slot matching the current playhead, so the waveform
    // builds up in the right place and survives seeking.
    const d = media.duration;
    const slot = isFinite(d) && d > 0
      ? Math.min(LIVE_SLOTS - 1, Math.floor((media.currentTime / d) * LIVE_SLOTS))
      : 0;
    live[slot] = Math.max(live[slot], Math.min(1, rms * 2.6));

    // If the element is tainted, the analyser reads pure silence forever.
    // Detect that and stop pretending we have a waveform.
    if (!media.muted && media.volume > 0) {
      silentFrames = rms < 0.0005 ? silentFrames + 1 : 0;
      if (silentFrames > 150) { analyserDead = true; analyser = null; live = null; }
    }
  }

  function loop() {
    if (!media.paused && !media.ended) {
      sampleAnalyser();
      render();
      rafId = requestAnimationFrame(loop);
    } else {
      rafId = 0;
    }
  }

  /* ---------------------------------------------------------------- controls */

  function setPlaying(on) {
    player.dataset.playing = on ? "true" : "false";
    if (playBig) playBig.setAttribute("aria-label", on ? "Pause" : "Play");
    if (heroLbl) heroLbl.textContent = on ? "Pause Recording" : "Play Recording";
  }

  function toggle() {
    if (player.dataset.mode === "empty") return;
    if (media.paused) {
      initAnalyser();
      media.play().catch(() => {});
    } else {
      media.pause();
    }
  }

  function seekTo(sec, andPlay) {
    if (player.dataset.mode === "empty") return;
    const d = media.duration;
    if (!isFinite(d) || d <= 0) return;
    media.currentTime = Math.max(0, Math.min(d - 0.05, sec));
    render();
    if (andPlay && media.paused) { initAnalyser(); media.play().catch(() => {}); }
  }

  function seekFromEvent(el, ev) {
    const rect = el.getBoundingClientRect();
    const x = (ev.clientX ?? 0) - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    const d = media.duration;
    if (isFinite(d) && d > 0) seekTo(ratio * d, false);
  }

  function bindSeek(el) {
    if (!el) return;
    let dragging = false;
    el.addEventListener("pointerdown", (ev) => {
      if (player.dataset.mode === "empty") return;
      dragging = true;
      el.setPointerCapture && el.setPointerCapture(ev.pointerId);
      seekFromEvent(el, ev);
    });
    el.addEventListener("pointermove", (ev) => { if (dragging) seekFromEvent(el, ev); });
    const stop = () => { dragging = false; };
    el.addEventListener("pointerup", stop);
    el.addEventListener("pointercancel", stop);
  }

  /* ------------------------------------------------------------------ wiring */

  playBig && playBig.addEventListener("click", toggle);
  heroPlay && heroPlay.addEventListener("click", toggle);
  bindSeek(scrub);
  bindSeek(waveWrap);

  waveWrap && waveWrap.addEventListener("keydown", (ev) => {
    const d = media.duration || 0;
    if (ev.key === "ArrowRight") { seekTo(media.currentTime + 5, false); ev.preventDefault(); }
    if (ev.key === "ArrowLeft")  { seekTo(media.currentTime - 5, false); ev.preventDefault(); }
    if (ev.key === " " || ev.key === "Enter") { toggle(); ev.preventDefault(); }
    waveWrap.setAttribute("aria-valuemax", String(Math.round(d)));
  });

  muteBtn && muteBtn.addEventListener("click", () => {
    media.muted = !media.muted;
    player.dataset.muted = media.muted ? "true" : "false";
    muteBtn.setAttribute("aria-label", media.muted ? "Unmute" : "Mute");
  });

  media.addEventListener("play",  () => { setPlaying(true);  if (!rafId) loop(); });
  media.addEventListener("pause", () => { setPlaying(false); render(); });
  media.addEventListener("ended", () => { setPlaying(false); render(); });

  media.addEventListener("timeupdate", () => {
    tCur.textContent = fmt(media.currentTime);
    if (waveWrap) waveWrap.setAttribute("aria-valuenow", String(Math.round(media.currentTime)));
    if (!rafId) render();
    window.dispatchEvent(new CustomEvent("meetly:time", { detail: { t: media.currentTime } }));
  });

  let posterNudged = false;
  media.addEventListener("loadedmetadata", async () => {
    const isVideo = media.videoWidth > 0;
    player.dataset.mode = isVideo ? "video" : "audio";
    // metadata alone paints nothing; nudge once so a real frame shows
    if (isVideo && !posterNudged && media.currentTime === 0) {
      posterNudged = true;
      try { media.currentTime = 0.1; } catch (_) {}
    }
    tDur.textContent = fmt(media.duration);
    tCur.textContent = fmt(media.currentTime);
    if (waveWrap) waveWrap.setAttribute("aria-valuemax", String(Math.round(media.duration || 0)));
    render();
    // real peaks are a bonus; the player is fully usable without them
    if (await loadPeaks()) render();
  });

  media.addEventListener("error", () => { player.dataset.mode = "empty"; render(); });

  /** Point the player at a file. Resets peaks/live data for the new track. */
  function load(src) {
    peaks = null;
    live = null;
    analyser = null;
    analyserDead = false;
    silentFrames = 0;
    posterNudged = false;
    player.dataset.mode = "loading";
    player.dataset.playing = "false";
    tCur.textContent = "00:00";
    tDur.textContent = "--:--";
    media.src = src;
    media.load();
    render();
    // if nothing resolves, fall back to the placeholder rather than hanging
    setTimeout(() => {
      if (player.dataset.mode === "loading" &&
          (!isFinite(media.duration) || media.duration === 0)) {
        player.dataset.mode = "empty";
        render();
      }
    }, 4000);
  }

  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(render, 120);
  });

  document.addEventListener("keydown", (ev) => {
    const tag = (ev.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || ev.target.isContentEditable) return;
    if (ev.key === " ") { toggle(); ev.preventDefault(); }
    else if (ev.key === "ArrowRight") { seekTo(media.currentTime + 5, false); }
    else if (ev.key === "ArrowLeft")  { seekTo(media.currentTime - 5, false); }
    else if (ev.key.toLowerCase() === "m" && muteBtn) { muteBtn.click(); }
  });

  render();

  /* --------------------------------------------------------------- public API */

  window.MeetlyPlayer = {
    load,
    seek: seekTo,
    toggle,
    get currentTime() { return media.currentTime; },
    get duration()    { return media.duration; },
    get ready()       { return player.dataset.mode !== "empty" && player.dataset.mode !== "loading"; },
  };
})();
