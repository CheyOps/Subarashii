"use strict";
const LANG_NAMES = { ja: "Japanese", zh: "Chinese", ko: "Korean" };
const $app = document.getElementById("app");
const player = document.getElementById("player");
const COLORS = ["#e76f51", "#2a9d8f", "#8e7dbe", "#e9c46a", "#4ea8de", "#f4a261", "#90be6d", "#f28482", "#577590", "#b5838d"];
let timer = null;
let current = null; // job being viewed
let filters = { speaker: "", q: "", onlyIssues: false, showJa: false };
let peak = { id: null, v: 0 }; // highest overall progress shown for the running job

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (t) => { const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, "0"); return `${m}:${s}`; };
const syl = (text) => (text.match(/[A-Za-z']+|\d+/g) || []).reduce((n, w) => {
  w = w.toLowerCase().replace(/'/g, "");
  if (/^\d+$/.test(w)) return n + Math.max(1, Math.round(w.length * 1.6));
  if (w.length <= 3) return n + 1;
  let g = (w.match(/[aeiouy]+/g) || []).length;
  if (w.endsWith("e") && !/(le|ee|ye)$/.test(w) && g > 1) g--;
  return n + Math.max(1, g);
}, 0);

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    alert(msg);
    throw new Error(msg);
  }
  return res.json();
}
const json = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
function play(url) { player.src = url + (url.includes("?") ? "&" : "?") + "t=" + Date.now(); player.play(); }
function colorOf(job, sid) { const ids = Object.keys(job.speakers); return COLORS[ids.indexOf(sid) % COLORS.length] || "#999"; }

// ------------------------------------------------------------------ home
async function renderHome() {
  clearInterval(timer);
  current = null;
  const [jobs, inputs, cfg] = await Promise.all([api("/api/jobs"), api("/api/inputs"), api("/api/config")]);
  $app.innerHTML = `
    <h2>New dub</h2>
    <form class="card" id="newjob">
      <div class="row">
        <label>Upload <input type="file" name="file" accept="video/*,.mkv"></label>
        <span class="muted">or</span>
        <label>From input folder
          <select name="input_name"><option value="">—</option>${inputs.map((f) => `<option>${esc(f)}</option>`).join("")}</select>
        </label>
      </div>
      <div class="row" style="margin-top:10px">
        <label>Spoken language
          <select name="language">
            ${[["auto", "Detect automatically"], ["ja", "Japanese"], ["zh", "Chinese (Mandarin)"], ["ko", "Korean"]].map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}
          </select>
        </label>
        <label title="Optional. Everything is filled in automatically; no knowledge of the original language or the source material needed."><input type="checkbox" name="review"> Let me check characters and lines before voicing (optional)</label>
        <label>Translator
          <select name="provider">
            ${["llamacpp", "openai", "anthropic"].map((p) => `<option value="${p}" ${p === cfg.provider ? "selected" : ""}>${p}${cfg.models[p] ? " (" + esc(cfg.models[p]) + ")" : ""}</option>`).join("")}
          </select>
        </label>
        <label>Model override <input name="model" placeholder="default"></label>
      </div>
      <details style="margin-top:10px"><summary class="muted">Advanced settings (key=value per line)</summary>
        <textarea name="settings" rows="3" placeholder="translation.honorifics=keep&#10;diarization.distance_threshold=0.55"></textarea>
      </details>
      <div class="row" style="margin-top:10px"><button class="primary" type="submit">Start</button><span id="upmsg" class="muted"></span></div>
    </form>
    <h2>Jobs</h2>
    <div class="card"><table>
      <thead><tr><th>Video</th><th>Status</th><th>Stage</th><th>Progress</th><th></th></tr></thead>
      <tbody id="jobrows">${jobRows(jobs)}</tbody>
    </table></div>`;
  document.getElementById("newjob").onsubmit = async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    fd.set("review", e.target.review.checked ? "true" : "false");
    if (!fd.get("file") || !fd.get("file").name) fd.delete("file");
    document.getElementById("upmsg").textContent = "uploading…";
    const { id } = await api("/api/jobs", { method: "POST", body: fd });
    location.hash = `#/job/${id}`;
  };
  timer = setInterval(async () => {
    const rows = document.getElementById("jobrows");
    if (rows) rows.innerHTML = jobRows(await api("/api/jobs"));
  }, 3000);
}

function fmtEta(sec) {
  if (sec == null) return "";
  if (sec < 60) return "less than a minute";
  const m = Math.round(sec / 60);
  if (m < 90) return `${m} minute${m === 1 ? "" : "s"}`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

function jobRows(jobs) {
  if (!jobs.length) return `<tr><td colspan="5" class="muted">No jobs yet.</td></tr>`;
  return jobs.map((j) => {
    const live = ["running", "queued"].includes(j.status);
    return `<tr>
    <td><a href="#/job/${esc(j.id)}">${esc(j.source_name)}</a><div class="muted small">${esc(j.created)}</div></td>
    <td><span class="badge ${esc(j.status)}">${esc(j.status)}</span></td>
    <td>${j.step ? `Step ${j.step_index}/${j.step_count}: ${esc(j.step_label)}` : ""}<div class="muted small">${esc(j.message)}</div></td>
    <td><div class="bar"><div style="width:${Math.round((j.overall ?? j.progress) * 100)}%"></div></div>
      <div class="muted small">${Math.round((j.overall ?? 0) * 100)}%${live && j.eta_seconds != null ? ` · ${fmtEta(j.eta_seconds)} left` : ""}</div></td>
    <td>${j.output ? `<a class="btn" href="/api/jobs/${esc(j.id)}/download">Download</a>` : ""}</td></tr>`;
  }).join("");
}

// ------------------------------------------------------------------ job
async function renderJob(id) {
  clearInterval(timer);
  current = await api(`/api/jobs/${id}`);
  draw();
  timer = setInterval(async () => {
    if (!["running", "queued"].includes(current.status)) return;
    const fresh = await api(`/api/jobs/${id}`);
    const structural = fresh.status !== current.status || fresh.stages_done.length !== current.stages_done.length;
    current = fresh;
    if (structural) draw(); else drawStatus();
  }, 2000);
}

function busy() { return ["running", "queued"].includes(current.status); }

function drawStatus() {
  const j = current;
  const el = document.getElementById("status");
  if (!el) return;
  const e = j.estimate || {};
  // never let the overall bar move backwards while a run is in progress
  if (!["running", "queued"].includes(j.status) || peak.id !== j.id) peak = { id: j.id, v: 0 };
  peak.v = Math.max(peak.v, e.overall || 0);
  const overall = j.status === "done" ? 100 : Math.round(peak.v * 100);
  let overallNote = "";
  if (["running", "queued"].includes(j.status) && e.eta_seconds != null) {
    overallNote = `Estimated time left: ${fmtEta(e.eta_seconds)}`;
    if (e.eta_review_seconds != null) overallNote += ` · pauses for your review in about ${fmtEta(e.eta_review_seconds)}`;
  } else if (j.status === "review") overallNote = "Paused for your review — check the characters and script, then click “Approve & voice it”.";
  else if (j.status === "done") overallNote = "Finished";
  else if (j.status === "error") overallNote = "Stopped on an error";
  else if (j.status === "cancelled") overallNote = "Cancelled";
  else if (j.status === "paused") overallNote = "Paused. Click Resume to continue from the next step.";
  const showStep = ["running", "queued"].includes(j.status) && e.step;
  el.innerHTML = `
    <div class="row"><span class="badge ${esc(j.status)}">${esc(j.status)}</span><div class="grow"></div>
      <span class="muted small">Stages done: ${j.stages_done.length} of ${e.step_count || 12}</span></div>
    <div class="prog">
      <div class="prog-head"><strong>Overall</strong><span class="pct">${overall}%</span><span class="muted">${esc(overallNote)}</span></div>
      <div class="bar big"><div style="width:${overall}%"></div></div>
    </div>
    ${showStep ? `<div class="prog">
      <div class="prog-head"><strong>Step ${e.step_index} of ${e.step_count}: ${esc(e.step_label)}</strong><span class="pct">${Math.round(e.step_progress * 100)}%</span><span class="muted">${esc(j.message)}</span></div>
      <div class="bar step"><div style="width:${Math.round(e.step_progress * 100)}%"></div></div>
    </div>` : ""}
    ${j.error ? `<div class="err" style="margin-top:8px">${esc(j.error)}</div>` : ""}`;
  const log = document.getElementById("log");
  if (log) { log.textContent = j.log.join("\n"); log.scrollTop = log.scrollHeight; }
}

function draw() {
  const j = current;
  const stages = ["ingest", "separate", "transcribe", "diarize", "profile", "proofread", "characters", "translate", "check", "synthesize", "mix", "export"];
  $app.innerHTML = `
    <div class="row"><a href="#/">← Jobs</a><h2 class="grow" style="margin:0 0 0 10px">${esc(j.source_name)}</h2>${j.language ? `<span class="muted">${esc(LANG_NAMES[j.language] || j.language)} → English</span>` : ""}</div>
    <div class="card" id="status" style="margin-top:12px"></div>
    <div class="row toolbar">
      ${j.status === "review" ? `<button class="primary" id="continue">Approve & voice it</button>` : ""}
      ${j.status === "done" || j.status === "error" ? `<button class="primary" id="rerender">Re-render changed lines</button>` : ""}
      <label>Re-run from <select id="fromstage">${stages.map((s) => `<option>${s}</option>`).join("")}</select></label>
      <button id="rerun" ${busy() ? "disabled" : ""}>Run</button>
      ${busy() ? `<button id="pause">Pause</button><button id="cancel">Cancel</button>` : ""}
      ${j.status === "paused" ? `<button class="primary" id="resume">Resume</button>` : ""}
      ${j.output ? `<a class="btn" href="/api/jobs/${esc(j.id)}/download">Download video</a>` : ""}
      <div class="grow"></div>
      <button class="ghost" id="del" ${busy() ? "disabled" : ""}>Delete job</button>
    </div>
    ${j.output && j.status === "done" ? `<details><summary class="muted">Preview</summary><video controls preload="none" src="/api/jobs/${esc(j.id)}/download"></video></details>` : ""}
    ${j.status === "review" ? reviewHelp() : ""}
    ${Object.keys(j.speakers).length ? speakersHtml(j) : ""}
    ${j.lines.length ? linesHtml(j) : ""}
    <h2>Log</h2><div class="log" id="log"></div>`;
  drawStatus();
  bindJob();
}

function reviewHelp() {
  return `<div class="card help"><strong>Optional check before voicing.</strong> Everything below was filled in automatically,
  and you don't need to know the original language or the source material. If you want to improve the result:
  <ul>
    <li>Play each character's <em>voice sample</em>. If two cards sound like the same person, use <em>Same person as…</em> to combine them.</li>
    <li>If you hear two different people in one sample, find the odd lines in the script below (filter by that character, press ▶ Original) and move them to the right character.</li>
    <li>Names you know can be typed in; otherwise the descriptive labels are fine.</li>
    <li>Lines marked <span class="flag">⚠</span> are ones the system was unsure about.</li>
  </ul>
  When you're happy, click <strong>Approve & voice it</strong>. Nothing here is required.</div>`;
}

function speakersHtml(j) {
  const ids = Object.keys(j.speakers);
  return `<h2>Characters <span class="muted small">— each voice is cloned from the original actor</span></h2>
  <div class="speakers">${ids.map((sid) => {
    const s = j.speakers[sid];
    const desc = [s.gender, s.age_group].filter((x) => x && x !== "unknown").join(", ");
    return `<div class="card speaker" data-sid="${esc(sid)}">
      <div class="row"><span class="dot" style="background:${colorOf(j, sid)}"></span><h3>${esc(s.name || sid)}</h3>
        ${s.name && !s.name_confident ? `<span class="muted small" title="The name isn't said clearly in the dialogue, so this is a description">(description)</span>` : ""}
        <div class="grow"></div><span class="muted small">${s.line_count} lines</span></div>
      ${(s.thumbs || []).length ? `<div class="thumbs">${s.thumbs.map((t) => `<img loading="lazy" src="/api/jobs/${esc(j.id)}/file/${esc(t)}" alt="">`).join("")}</div>
        <div class="muted small">Scenes where this voice is heard (the speaker may be off-screen)</div>` : ""}
      <div class="row" style="margin-top:8px">
        ${s.ref_audio ? `<button class="play" data-play="/api/jobs/${esc(j.id)}/file/${esc(s.ref_audio)}">▶ Voice sample</button>` : ""}
        <span class="muted small">${esc(desc)}</span>
      </div>
      ${s.personality ? `<p class="small" style="margin:8px 0 0">${esc(s.personality)}</p>` : ""}
      <details style="margin-top:8px"><summary class="muted small">Edit</summary>
        <div class="fields">
          <label>Name</label><input data-f="name" value="${esc(s.name)}">
          <label>Gender</label><select data-f="gender">${["", "male", "female", "unknown"].map((v) => `<option ${v === s.gender ? "selected" : ""}>${v}</option>`).join("")}</select>
          <label>Age</label><select data-f="age_group">${["", "child", "teen", "young adult", "adult", "elderly", "unknown"].map((v) => `<option ${v === s.age_group ? "selected" : ""}>${v}</option>`).join("")}</select>
          <label>Personality</label><textarea data-f="personality" rows="2">${esc(s.personality)}</textarea>
          <label>Speech style</label><textarea data-f="speech_style" rows="2">${esc(s.speech_style)}</textarea>
          <label title="Line numbers whose original audio is used to clone this voice">Voice from lines</label><input data-f="ref_clips" value="${esc((s.ref_clips || []).join(","))}">
        </div>
        <button class="primary save-spk">Save</button>
      </details>
      <div class="row" style="margin-top:8px">
        <select class="merge"><option value="">Same person as…</option>${ids.filter((o) => o !== sid).map((o) => `<option value="${esc(o)}">${esc(j.speakers[o].name || o)}</option>`).join("")}</select>
      </div>
    </div>`;
  }).join("")}</div>`;
}

function linesHtml(j) {
  const ids = Object.keys(j.speakers);
  return `<h2>Script</h2>
  <div class="row" style="margin-bottom:8px">
    <select id="fspk"><option value="">All characters</option>${ids.map((s) => `<option value="${esc(s)}" ${filters.speaker === s ? "selected" : ""}>${esc(j.speakers[s].name || s)}</option>`).join("")}</select>
    <input id="fq" placeholder="search" value="${esc(filters.q)}">
    <label><input type="checkbox" id="fissues" ${filters.onlyIssues ? "checked" : ""}> Only lines that need attention</label>
    <label><input type="checkbox" id="fja" ${filters.showJa ? "checked" : ""}> Show original text</label>
    <span class="muted small">Edit any English line; changed lines are re-voiced when you click “Re-render”.</span>
  </div>
  <div class="card" style="padding:0"><table class="lines"><thead><tr>
    <th>#</th><th>Time</th><th>Character</th><th>English</th><th>Listen</th><th></th>
  </tr></thead><tbody>${j.lines.filter(lineVisible).map((l) => lineRow(j, l, ids)).join("")}</tbody></table></div>`;
}

function lineFlags(l) {
  const f = [];
  if (l.speaker_conf < 0.4) f.push("Unsure who is speaking — listen to the original");
  if (l.sep && l.sep.length) f.push("Two people talk at once here — listen to check the voice and words");
  if (l.tts && l.tts.fit === "overflow") f.push("Runs past its time slot");
  else if (l.tts && l.tts.fit === "stretched") f.push("Sped up a lot, may sound rushed");
  else if (!l.tts && l.syl_max && syl(l.en) > l.syl_max) f.push("May be too long to say in time");
  if (l.lyric) f.push("Song lyric: left in the original language");
  if (l.check_unresolved) f.push(l.check_note);
  return f;
}

function lineVisible(l) {
  if (filters.speaker && l.speaker !== filters.speaker) return false;
  if (filters.q) { const q = filters.q.toLowerCase(); if (!(l.ja + " " + l.en).toLowerCase().includes(q)) return false; }
  if (filters.onlyIssues && !lineFlags(l).length) return false;
  return true;
}

function lineRow(j, l, ids) {
  const n = syl(l.en);
  const cands = Object.entries(l.candidates || {}).filter(([, t]) => t && t !== l.en);
  const flags = lineFlags(l);
  const candName = { natural: "longer", concise: "shorter", minimal: "shortest", refit: "fitted" };
  return `<tr data-lid="${l.id}" class="${l.skip || l.lyric ? "skipped" : ""}">
    <td class="muted small">${l.id}</td>
    <td class="time">${fmt(l.start)}<br>${(l.end - l.start).toFixed(1)}s</td>
    <td><span class="dot" style="background:${colorOf(j, l.speaker)}"></span>
      <select class="lspk">${ids.map((s) => `<option value="${esc(s)}" ${s === l.speaker ? "selected" : ""}>${esc(j.speakers[s].name || s)}</option>`).join("")}</select></td>
    <td class="en"><textarea class="len" rows="2">${esc(l.en)}</textarea>
      ${flags.map((f) => `<div class="flag small">⚠ ${esc(f)}</div>`).join("")}
      ${l.check_note && !l.check_unresolved ? `<div class="muted small">✎ ${esc(l.check_note)}</div>` : ""}
      ${filters.showJa ? `<div class="muted small ja">${esc(l.ja)}</div>` : ""}
      <div class="row small"><span class="syl ${l.syl_max && n > l.syl_max ? "over" : ""}" title="Syllables: the line's length versus what fits in the time">${n} / ${l.syl_max} syllables</span>${l.en_edited ? `<span class="badge">edited</span>` : ""}${l.dirty ? `<span class="badge review">will be re-voiced</span>` : ""}</div>
      ${cands.length ? `<div class="chips"><span class="muted small">Other versions:</span>${cands.map(([k, t]) => `<button class="chip" data-cand="${esc(t)}" title="Use this ${esc(candName[k] || k)} version">${esc(t)}</button>`).join("")}</div>` : ""}</td>
    <td><div class="listen">${l.ja_clip ? `<button class="play" data-play="/api/jobs/${esc(j.id)}/file/${esc(l.ja_clip)}">▶ Original</button>` : ""}
      ${l.tts && l.tts.file ? `<button class="play" data-play="/api/jobs/${esc(j.id)}/file/${esc(l.tts.file)}">▶ English</button>` : ""}</div></td>
    <td><label class="small" title="Keep the original audio for this line"><input type="checkbox" class="lskip" ${l.skip ? "checked" : ""}> keep original</label>
      <button class="ghost small newspk" title="This line is a different character from the others in its group">different person</button></td>
  </tr>`;
}

function bindJob() {
  // toolbar buttons are re-created on every draw(); per-row/card actions use the delegated handlers below
  const base = `/api/jobs/${current.id}`;
  const refresh = async () => { current = await api(base); draw(); };
  const byId = (id) => document.getElementById(id);
  if (byId("continue")) byId("continue").onclick = async () => { await api(`${base}/run`, json("POST", { approve: true })); refresh(); };
  if (byId("rerender")) byId("rerender").onclick = async () => { await api(`${base}/rerender`, { method: "POST" }); refresh(); };
  byId("rerun").onclick = async () => {
    const st = byId("fromstage").value;
    if (!confirm(`Re-run from "${st}"? Later stages will be redone (your edited English lines are kept).`)) return;
    await api(`${base}/run`, json("POST", { from_stage: st })); refresh();
  };
  if (byId("pause")) byId("pause").onclick = async (e) => {
    const btn = e.target;
    const fresh = await api(base);
    const est = fresh.estimate || {};
    const send = async (now) => {
      await api(`${base}/pause?now=${now}`, { method: "POST" });
      if (!now) { btn.disabled = true; btn.textContent = "Pausing after this step…"; }
    };
    if (fresh.status !== "running" || !est.step) return send(false); // still waiting in the queue: nothing to lose
    const dlg = document.getElementById("pausedlg");
    dlg.innerHTML = `
      <h2 style="margin-top:0">Pause this job?</h2>
      <p><strong>Step ${est.step_index} of ${est.step_count}: ${esc(est.step_label)}</strong> (${Math.round(est.step_progress * 100)}% done)</p>
      <table class="small"><tbody>
        <tr><td class="muted">Spent on this step</td><td>${fmtEta(est.step_elapsed_seconds)}</td></tr>
        <tr><td class="muted">Left on this step</td><td>about ${fmtEta(est.step_eta_seconds)}</td></tr>
        <tr><td class="muted">Left on the whole job</td><td>about ${fmtEta(est.eta_seconds)}</td></tr>
      </tbody></table>
      <div class="pauseopts">
        <button class="primary" id="pauseafter">Pause when this step finishes</button>
        <span class="muted small">Nothing is lost. Resume continues from the next step.</span>
        <button id="pausenow">Pause now</button>
        <span class="muted small">Stops this step immediately. Its ${fmtEta(est.step_elapsed_seconds)} of work is thrown away and the step starts over on resume.</span>
      </div>
      <div class="row" style="margin-top:14px"><button class="ghost" id="pausekeep">Keep running</button></div>`;
    dlg.querySelector("#pauseafter").onclick = async () => { dlg.close(); await send(false); };
    dlg.querySelector("#pausenow").onclick = async () => { dlg.close(); await send(true); };
    dlg.querySelector("#pausekeep").onclick = () => dlg.close();
    dlg.showModal();
  };
  if (byId("resume")) byId("resume").onclick = async () => { await api(`${base}/run`, json("POST", {})); refresh(); };
  if (byId("cancel")) byId("cancel").onclick = async (e) => {
    await api(`${base}/cancel`, { method: "POST" });
    e.target.disabled = true; e.target.textContent = "Cancelling…";
  };
  byId("del").onclick = async () => {
    if (!confirm("Delete this job and its files? The output video in the output folder is kept.")) return;
    await api(base, { method: "DELETE" }); location.hash = "#/";
  };
}

const lidOf = (el) => Number(el.closest("tr").dataset.lid);
async function patchLine(el, body) {
  const updated = await api(`/api/jobs/${current.id}/lines/${lidOf(el)}`, json("PATCH", body));
  current.lines[current.lines.findIndex((l) => l.id === updated.id)] = updated;
  el.closest("tr").outerHTML = lineRow(current, updated, Object.keys(current.speakers));
}

$app.addEventListener("click", async (e) => {
  const t = e.target.closest("button");
  if (!t || !current) return;
  const base = `/api/jobs/${current.id}`;
  if (t.dataset.play) return play(t.dataset.play);
  if (t.classList.contains("chip")) return patchLine(t, { en: t.dataset.cand });
  if (t.classList.contains("newspk")) { current = { ...current, ...(await api(`${base}/speakers`, json("POST", { from_line: lidOf(t) }))) }; return draw(); }
  if (t.classList.contains("save-spk")) {
    const card = t.closest(".speaker"), body = {};
    card.querySelectorAll("[data-f]").forEach((el) => {
      if (el.dataset.f === "ref_clips") {
        const ids = el.value.split(/[ ,]+/).filter(Boolean).map(Number);
        if (ids.join(",") !== (current.speakers[card.dataset.sid].ref_clips || []).join(",")) body.ref_clips = ids;
      } else body[el.dataset.f] = el.value;
    });
    current = { ...current, ...(await api(`${base}/speakers/${card.dataset.sid}`, json("PATCH", body))) };
    draw();
  }
});

$app.addEventListener("change", async (e) => {
  const t = e.target;
  if (!current) return;
  if (t.classList.contains("len")) return patchLine(t, { en: t.value });
  if (t.classList.contains("lspk")) return patchLine(t, { speaker: t.value });
  if (t.classList.contains("lskip")) return patchLine(t, { skip: t.checked });
  if (t.classList.contains("llyric")) return patchLine(t, { lyric: t.checked });
  if (t.classList.contains("merge")) {
    const sid = t.closest(".speaker").dataset.sid, to = t.value;
    if (!to || !confirm(`Combine these two into one character? All lines of "${current.speakers[sid].name || sid}" move to "${current.speakers[to].name || to}".`)) { t.value = ""; return; }
    current = { ...current, ...(await api(`/api/jobs/${current.id}/speakers/${sid}`, json("PATCH", { merge_into: to }))) };
    return draw();
  }
  if (t.id === "fspk") { filters.speaker = t.value; return draw(); }
  if (t.id === "fq") { filters.q = t.value; return draw(); }
  if (t.id === "fissues") { filters.onlyIssues = t.checked; return draw(); }
  if (t.id === "fja") { filters.showJa = t.checked; return draw(); }
});

$app.addEventListener("input", (e) => {
  const t = e.target;
  if (!current || !t.classList.contains("len")) return;
  const l = current.lines.find((x) => x.id === lidOf(t)), n = syl(t.value);
  const s = t.parentElement.querySelector(".syl");
  s.textContent = `${n} / ${l.syl_max} syllables`;
  s.classList.toggle("over", !!(l.syl_max && n > l.syl_max));
});

// ------------------------------------------------------------------ settings
const PROVIDER_LABELS = { llamacpp: "llama.cpp (local router)", openai: "OpenAI-compatible server", anthropic: "Anthropic (Claude)" };
const FIELD_LABELS = { url: "Endpoint URL", api_key: "API key", model: "Model", effort: "Effort" };

async function openSettings() {
  const dlg = document.getElementById("settings");
  await renderSettings();
  if (!dlg.open) dlg.showModal();
}

async function renderSettings() {
  const s = await api("/api/settings/llm");
  const field = (prov, f) => {
    const v = s.providers[prov];
    if (f === "api_key") return `<label>${FIELD_LABELS[f]} <input type="password" data-prov="${prov}" data-f="${f}" autocomplete="off" placeholder="${v.api_key_set ? "saved (leave blank to keep)" : "not set"}">
      ${v.api_key_set ? `<button type="button" class="ghost clearkey" data-prov="${prov}">Clear</button>` : ""}</label>`;
    if (f === "effort") return `<label>${FIELD_LABELS[f]} <select data-prov="${prov}" data-f="${f}">${["low", "medium", "high"].map((e) => `<option ${e === v.effort ? "selected" : ""}>${e}</option>`).join("")}</select></label>`;
    if (f === "model") return `<label>${FIELD_LABELS[f]} <span class="modelbox" data-prov="${prov}">${modelControl(prov, v.model, [])}</span></label>`;
    return `<label>${FIELD_LABELS[f]} <input data-prov="${prov}" data-f="${f}" value="${esc(v[f])}" size="40"></label>`;
  };
  // a drop-down of the provider's models once they are known, a plain text box until then
  const modelControl = (prov, cur, models) => models.length
    ? `<select data-prov="${prov}" data-f="model">${(models.includes(cur) || !cur ? models : [cur, ...models]).map((m) => `<option value="${esc(m)}" ${m === cur ? "selected" : ""}>${esc(m)}</option>`).join("")}</select>`
    : `<input data-prov="${prov}" data-f="model" value="${esc(cur)}" size="28" placeholder="model id">`;
  const loadModels = async (prov) => {
    const box = document.querySelector(`.modelbox[data-prov="${prov}"]`);
    const cur = box.querySelector("[data-f]").value;
    const res = await fetch(`/api/settings/llm/test?provider=${prov}`, { method: "POST" });
    if (!res.ok) { let d = res.statusText; try { d = (await res.json()).detail || d; } catch (_) {} return d; }
    box.innerHTML = modelControl(prov, cur, (await res.json()).models);
    return "";
  };
  document.getElementById("settingsbody").innerHTML = `
    <h2 style="margin-top:0">Translator settings</h2>
    <form id="llmform">
      <div class="row"><label>Provider
        <select id="llmprov">${Object.keys(PROVIDER_LABELS).map((p) => `<option value="${p}" ${p === s.provider ? "selected" : ""}>${PROVIDER_LABELS[p]}</option>`).join("")}</select></label></div>
      ${Object.keys(PROVIDER_LABELS).map((p) => `<div class="prov" data-prov="${p}" style="margin-top:10px"><div class="row" style="flex-direction:column;align-items:flex-start">
        ${Object.keys(s.providers[p]).map((f) => field(p, f.replace(/_set$/, ""))).join("")}</div></div>`).join("")}
      <p class="muted small">Saved on the server and used for new jobs. API keys are stored in the data folder and never sent back to the browser.</p>
      <div class="row"><button class="primary" type="submit">Save</button><button type="button" id="llmtest">Save and test connection</button><button type="button" class="ghost" id="llmclose">Close</button><span id="llmmsg" class="muted"></span></div>
    </form>`;
  const loaded = new Set();
  const showProv = () => document.querySelectorAll(".prov").forEach((d) => { d.hidden = d.dataset.prov !== document.getElementById("llmprov").value; });
  const fill = () => { const p = document.getElementById("llmprov").value; if (!loaded.has(p)) { loaded.add(p); loadModels(p); } };
  showProv();
  fill();
  document.getElementById("llmprov").onchange = () => { showProv(); fill(); };
  const save = async () => {
    const providers = {};
    document.querySelectorAll("#llmform [data-f]").forEach((el) => { (providers[el.dataset.prov] ??= {})[el.dataset.f] = el.value; });
    await api("/api/settings/llm", json("PUT", { provider: document.getElementById("llmprov").value, providers }));
  };
  document.getElementById("llmclose").onclick = () => document.getElementById("settings").close();
  const msg = document.getElementById("llmmsg");
  document.getElementById("llmform").onsubmit = async (e) => { e.preventDefault(); await save(); msg.textContent = "saved"; renderSettingsKeepMsg(); };
  document.getElementById("llmtest").onclick = async () => {
    await save();
    msg.textContent = "testing…";
    const err = await loadModels(document.getElementById("llmprov").value);
    msg.textContent = err ? `test failed: ${err}` : "connected; pick a model from the list";
  };
  document.querySelectorAll(".clearkey").forEach((b) => b.onclick = async () => {
    await api(`/api/settings/llm/${b.dataset.prov}/api_key`, { method: "DELETE" });
    renderSettings();
  });
}
async function renderSettingsKeepMsg() { await renderSettings(); document.getElementById("llmmsg").textContent = "saved"; }

// ------------------------------------------------------------------ router
function route() {
  const m = location.hash.match(/^#\/job\/(.+)$/);
  if (m) renderJob(decodeURIComponent(m[1])); else renderHome();
}
window.addEventListener("hashchange", route);
route();
document.getElementById("opensettings").onclick = openSettings;
for (const id of ["settings", "pausedlg"]) document.getElementById(id).addEventListener("click", (e) => { if (e.target.id === id) e.target.close(); }); // click on the backdrop
