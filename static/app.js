"use strict";

const $ = (id) => document.getElementById(id);
const state = { files: [], mode: "service_account", jobId: null, since: 0, matches: [], running: false, modelReady: false };

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

function setError(message) {
  $("formError").textContent = message || "";
}

function updateGo() {
  $("go").disabled = !state.modelReady || state.running;
  $("go").textContent = state.running ? "Working…" : "Find photos";
}

// ---------------------------------------------------------------- model status
async function pollHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    const pill = $("modelPill");
    state.modelReady = h.state === "ready";
    pill.textContent = state.modelReady ? "Ready" : h.message;
    pill.className = "pill " + (h.state === "ready" ? "ready" : h.state === "error" ? "error" : "");
    updateGo();
    if (h.state === "error") return;
    if (!state.modelReady) setTimeout(pollHealth, 1500);
  } catch {
    $("modelPill").textContent = "Can't reach the server";
    $("modelPill").className = "pill error";
    setTimeout(pollHealth, 3000);
  }
}

// ---------------------------------------------------------------- face photos
function addPhotos(list) {
  for (const file of list) {
    if (!file.type.startsWith("image/")) continue;
    const key = `${file.name}|${file.size}|${file.lastModified}`;
    if (state.files.some((f) => f.key === key)) continue;
    state.files.push({ key, file, url: URL.createObjectURL(file) });
  }
  renderPreviews();
}

function renderPreviews() {
  $("previews").replaceChildren(...state.files.map((f, i) => {
    const remove = el("button", { type: "button", title: "Remove", textContent: "×" });
    remove.addEventListener("click", () => {
      URL.revokeObjectURL(f.url);
      state.files.splice(i, 1);
      renderPreviews();
    });
    const img = el("img", { src: f.url, alt: f.file.name, title: f.file.name });
    img.addEventListener("error", () => { img.style.visibility = "hidden"; }); // e.g. HEIC the browser can't show
    return el("div", { className: "thumb" }, img, remove);
  }));
}

$("drop").addEventListener("click", () => $("photos").click());
$("drop").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("photos").click(); } });
$("photos").addEventListener("change", (e) => { addPhotos(e.target.files); e.target.value = ""; });
for (const zone of [$("drop"), $("jsonDrop")]) {
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
}
$("drop").addEventListener("drop", (e) => { e.preventDefault(); $("drop").classList.remove("over"); addPhotos(e.dataTransfer.files); });

// ---------------------------------------------------------------- credentials
function setMode(mode) {
  state.mode = mode;
  for (const b of document.querySelectorAll(".tabs button")) b.classList.toggle("on", b.dataset.mode === mode);
  $("panelSa").hidden = mode !== "service_account";
  $("panelKey").hidden = mode !== "api_key";
}
for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => setMode(b.dataset.mode));

function showStatus(ok, ...content) {
  const box = $("jsonStatus");
  box.hidden = false;
  box.className = "status " + (ok ? "ok" : "err");
  box.replaceChildren(...content);
}

function updateShareHint() {
  const email = $("client_email").value.trim();
  $("shareHint").replaceChildren(...(email
    ? ["Share the Drive folder with ", el("b", { textContent: email }), " (as Viewer), then paste the folder link above."]
    : ["Share the Drive folder with the service account's email (as ", el("b", { textContent: "Viewer" }), "), then paste the folder link above."]));
}

/** Read pasted text / a dropped file and fill in the right fields. */
function handleCredentialText(text) {
  const r = Creds.parseCredentials(text);
  if (r.kind === "service_account") {
    setMode("service_account");
    for (const [name, value] of Object.entries(r.fields)) $(name).value = value;
    $("private_key").classList.add("masked");
    $("toggleKey").textContent = "Show";
    $("jsonText").value = "";
    showStatus(true, "Service account loaded: ", el("b", { textContent: r.fields.client_email || "(no email found)" }),
      ...r.notes.map((n) => " · " + n));
    updateShareHint();
  } else if (r.kind === "api_key") {
    setMode("api_key");
    $("api_key").value = r.apiKey;
    $("jsonText").value = "";
    showStatus(true, "API key detected and filled in.");
  } else if (r.kind === "private_key") {
    setMode("service_account");
    $("private_key").value = r.privateKey;
    $("jsonText").value = "";
    showStatus(true, "Private key loaded. Now add the client email.");
  } else {
    showStatus(false, r.message);
  }
}

let pasteTimer;
$("jsonText").addEventListener("input", () => {
  clearTimeout(pasteTimer);
  if ($("jsonText").value.trim()) pasteTimer = setTimeout(() => handleCredentialText($("jsonText").value), 500);
});
$("jsonText").addEventListener("paste", () => setTimeout(() => {
  clearTimeout(pasteTimer);
  handleCredentialText($("jsonText").value);
}, 0));
$("pickJson").addEventListener("click", () => $("jsonFile").click());
$("jsonFile").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) handleCredentialText(await file.text());
  e.target.value = "";
});
$("jsonDrop").addEventListener("drop", async (e) => {
  e.preventDefault();
  $("jsonDrop").classList.remove("over");
  const file = e.dataTransfer.files[0];
  if (file) handleCredentialText(await file.text());
  else if (e.dataTransfer.getData("text")) handleCredentialText(e.dataTransfer.getData("text"));
});

$("client_email").addEventListener("input", updateShareHint);
$("private_key").addEventListener("blur", () => { // tidy a key pasted on one line, with "\n" text, etc.
  if ($("private_key").value.trim()) $("private_key").value = Creds.normalizePrivateKey($("private_key").value);
});
$("toggleKey").addEventListener("click", () => {
  const hidden = $("private_key").classList.toggle("masked");
  $("toggleKey").textContent = hidden ? "Show" : "Hide";
});
$("threshold").addEventListener("input", () => { $("thresholdOut").textContent = Number($("threshold").value).toFixed(2); });

function collectCredentials() {
  if (state.mode === "api_key") {
    const key = $("api_key").value.trim();
    return key ? { value: key } : { problem: "Enter your API key.", focus: "api_key" };
  }
  const f = Object.fromEntries(["client_email", "private_key", "project_id", "private_key_id", "token_uri"].map((n) => [n, $(n).value]));
  const problems = Creds.validateServiceAccount(f);
  if (problems.length) return { problem: problems[0], focus: f.client_email ? "private_key" : "client_email" };
  return { value: JSON.stringify(Creds.buildServiceAccount(f)) };
}

// ---------------------------------------------------------------- submit
$("form").addEventListener("input", () => setError("")); // a stale error shouldn't outlive the fix

function fail(message, focusId) {
  setError(message);
  if (focusId) $(focusId).focus();
  return null;
}

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  setError("");
  const name = $("name").value.trim();
  if (!name) return fail("Enter the person's name.", "name");
  if (!state.files.length) return fail("Add at least one photo of the person's face.", "drop");
  const folder = $("folder").value.trim();
  if (!Creds.extractFolderId(folder)) return fail("That doesn't look like a Google Drive folder link.", "folder");
  const creds = collectCredentials();
  if (creds.problem) return fail(creds.problem, creds.focus);

  const fd = new FormData();
  fd.append("name", name);
  fd.append("folder_link", folder);
  fd.append("auth_mode", state.mode);
  fd.append("credentials", creds.value);
  fd.append("threshold", $("threshold").value);
  fd.append("quality", $("quality").value);
  fd.append("include_subfolders", $("recursive").checked);
  for (const f of state.files) fd.append("images", f.file, f.file.name);

  state.running = true;
  updateGo();
  showProgress("Checking your photos and Drive access…", null);
  try {
    const res = await fetch("/api/search", { method: "POST", body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(describeError(data, res));
    state.jobId = data.job_id;
    state.since = 0;
    state.matches = [];
    $("gallery").replaceChildren();
    $("results").hidden = true;
    $("cancel").hidden = false;
    poll();
  } catch (err) {
    $("progress").hidden = true;
    state.running = false;
    updateGo();
    setError(err.message);
  }
});

function describeError(data, res) {
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).join("; ");
  return `Request failed (${res.status}).`;
}

$("cancel").addEventListener("click", async () => {
  $("cancel").disabled = true;
  if (state.jobId) await fetch(`/api/jobs/${state.jobId}/cancel`, { method: "POST" }).catch(() => {});
});

// ---------------------------------------------------------------- progress
function showProgress(title, fraction, text = "") {
  $("progress").hidden = false;
  $("progressTitle").textContent = title;
  $("progressText").textContent = text;
  $("bar").classList.toggle("indeterminate", fraction === null);
  $("bar").style.width = fraction === null ? "" : `${Math.round(fraction * 100)}%`;
}

function duration(seconds) {
  seconds = Math.max(0, Math.round(seconds));
  return seconds >= 60 ? `${Math.floor(seconds / 60)} min ${seconds % 60} s` : `${seconds} s`;
}

async function poll() {
  let job;
  try {
    const res = await fetch(`/api/jobs/${state.jobId}?since=${state.since}`);
    if (!res.ok) throw new Error(describeError(await res.json().catch(() => ({})), res));
    job = await res.json();
  } catch (err) {
    if (/Unknown job/.test(err.message)) return finish(null, err.message);
    $("progressText").textContent = "Lost contact with the server, retrying…";
    return setTimeout(poll, 2000);
  }
  render(job);
  if (["done", "cancelled", "error"].includes(job.status)) return finish(job);
  setTimeout(poll, 700);
}

const TITLES = {
  queued: "Waiting for the previous search to finish…", listing: "Reading the Drive folder…",
  scanning: "Scanning photos…", saving: "Saving matched photos…",
};

function render(job) {
  const total = job.total || 0;
  if (job.status === "queued" || job.status === "listing" || !total) {
    showProgress(TITLES[job.status] || job.message, null, job.message);
  } else {
    const parts = [`${job.scanned} of ${total} photos`];
    if (job.cached) parts.push(`${job.cached} already known from earlier searches`);
    if (job.status === "scanning" && job.fetched >= 5 && job.todo > job.fetched) {
      parts.push(`about ${duration(job.scan_elapsed / job.fetched * (job.todo - job.fetched))} left`);
    }
    showProgress(TITLES[job.status] || job.message, job.status === "saving" ? 1 : job.scanned / total, parts.join(" · "));
  }
  $("sScanned").textContent = job.scanned;
  $("sFaces").textContent = job.faces;
  $("sMatched").textContent = job.matched;
  $("sFailed").textContent = job.failed;

  const notes = [...job.warnings.map((w) => el("li", { textContent: w })),
                 ...job.errors.slice(-3).map((m) => el("li", { className: "err", textContent: "Skipped " + m }))];
  $("notes").replaceChildren(...notes);

  if (job.matches.length) {
    state.matches.push(...job.matches);
    state.since = job.match_count;
  }
  if (state.matches.length) renderGallery(job);
}

function renderGallery(job) {
  $("results").hidden = false;
  $("resultsTitle").textContent = `Photos of ${job.name} (${state.matches.length})`;
  $("savedPath").textContent = `Saved in: ${job.output_folder}`;
  const sorted = [...state.matches].sort((a, b) => b.score - a.score); // best match first
  $("gallery").replaceChildren(...sorted.map((m) => el("figure", {},
    el("a", { href: m.url, target: "_blank", rel: "noopener" },
      el("img", { src: m.thumb || m.url, alt: m.file, loading: "lazy" })),
    el("figcaption", { textContent: `${m.file} · ${m.score.toFixed(2)}`, title: m.file }))));
}

function finish(job, problem) {
  state.running = false;
  $("cancel").hidden = true;
  $("cancel").disabled = false;
  updateGo();
  if (!job) return setError(problem || "Something went wrong.");
  if (job.status === "done") {
    showProgress(job.message, 1, `Scanned ${job.scanned} photos in ${duration(job.elapsed)}.`);
  } else if (job.status === "cancelled") {
    showProgress("Stopped", job.total ? job.scanned / job.total : 0, "Photos found so far are saved.");
  } else {
    showProgress("Search failed", job.total ? job.scanned / job.total : 0, job.message);
    setError(job.message);
  }
  const zip = $("zipLink");
  zip.hidden = !state.matches.length;
  zip.href = `/api/results/${encodeURIComponent(job.folder)}/zip`;
  loadSaved();
}

// ---------------------------------------------------------------- earlier results
async function loadSaved() {
  try {
    const rows = (await (await fetch("/api/results")).json()).filter((r) => r.count);
    $("saved").hidden = !rows.length;
    $("savedList").replaceChildren(...rows.map((r) => el("li", {},
      el("div", {}, `${r.name} · ${r.count} photo${r.count === 1 ? "" : "s"}`, el("small", { textContent: r.path })),
      el("a", { className: "button", href: `/api/results/${encodeURIComponent(r.name)}/zip`, textContent: "ZIP" }))));
  } catch { /* not important */ }
}

updateShareHint();
updateGo();
pollHealth();
loadSaved();
