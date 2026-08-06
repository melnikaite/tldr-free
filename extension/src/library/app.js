// Library page controller — browse mode.
//
// Behavior:
//   - On load: daemon.listJobs(), render rows.
//   - Subscribe to GET /events for realtime updates (job created/updated/
//     deleted, workers state). No polling.
//   - Status/kind filters trigger refetch from daemon.
//   - Per-row actions: open in side panel (writes activeJobId to session storage,
//     attempts chrome.sidePanel.open), delete (with confirm), retry (failed only).

import { daemon } from "../lib/daemon-client.js";
import { openEventStream } from "../lib/event-stream.js";
import { escapeHtml, stringifyError } from "../lib/utils.js";
import { openSidePanel } from "../lib/browser-compat.js";

// Library only renders status badges + queue counter — skip the high-volume
// stage/delta chatter from running pipelines.
const eventStream = openEventStream({ types: ["job", "workers", "done", "error"] });

/** @import { JobImportResponse, JobSummary, JobStatus } from "../lib/api-types.js" */

const tbody = /** @type {HTMLElement} */ (
  document.querySelector("#jobs tbody")
);
const filterStatus = /** @type {HTMLSelectElement} */ (
  document.getElementById("filter-status")
);
const filterKind = /** @type {HTMLSelectElement} */ (
  document.getElementById("filter-kind")
);
/** @type {JobSummary[]} */
let allJobs = [];

filterStatus.addEventListener("change", () => refetch());
filterKind.addEventListener("change", () => refetch());

// ---------------------------------------------------------------------------
// Multi-select (export bundle) — selection is a Set of job ids kept here in
// module state. render() rebuilds tbody.innerHTML from scratch each time, so
// the checked state is re-applied (renderRow) and the set pruned (render)
// on every pass rather than trusting the DOM to remember it.
// ---------------------------------------------------------------------------

/** @type {Set<string>} */
const selectedIds = new Set();
// Job id of the last row checkbox the user clicked directly (not via
// shift-range) — the anchor for shift-click range selection. Stored as an
// id, not an index: the Library is live (handleJobEvent does
// `allJobs.unshift(j)` for newly created jobs), so an index captured at one
// click can point at a different row by the next click. Resolved back to an
// index at click time instead.
/** @type {string | null} */
let lastClickedId = null;

const selectAllCheckbox = /** @type {HTMLInputElement | null} */ (
  document.getElementById("select-all")
);
const selectionBar = document.getElementById("selection-bar");
const selectionCountEl = document.getElementById("selection-count");
const selectionWarningEl = document.getElementById("selection-warning");
const exportBtn = /** @type {HTMLButtonElement | null} */ (
  document.getElementById("export-btn")
);
const clearSelectionBtn = document.getElementById("clear-selection-btn");
let exportInFlight = false;

selectAllCheckbox?.addEventListener("change", () => {
  if (selectAllCheckbox.checked) {
    for (const j of allJobs) selectedIds.add(j.id);
  } else {
    for (const j of allJobs) selectedIds.delete(j.id);
  }
  render();
});

clearSelectionBtn?.addEventListener("click", () => {
  selectedIds.clear();
  lastClickedId = null;
  render();
});

exportBtn?.addEventListener("click", async () => {
  const exportableIds = allJobs
    .filter((j) => selectedIds.has(j.id) && j.status === "done")
    .map((j) => j.id);
  if (exportableIds.length === 0 || exportInFlight) return;
  exportInFlight = true;
  updateSelectionBar();
  try {
    const blob = await daemon.exportJobs(exportableIds);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `tldr-export-${todayDateStamp()}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    // Don't revoke immediately: `a.click()` only *starts* the download —
    // Chrome hands it off to the download manager asynchronously, and an
    // object URL revoked before that handoff completes can end up pointing
    // at nothing, silently dropping the download (worst on the large
    // bundles this matters most for). A short delay lets the browser grab
    // the bytes first; the object URL is cheap to keep alive that long.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    selectedIds.clear();
    lastClickedId = null;
    render();
  } catch (err) {
    alert(`Export failed: ${stringifyError(err)}`);
  } finally {
    exportInFlight = false;
    updateSelectionBar();
  }
});

function todayDateStamp() {
  return new Date().toISOString().slice(0, 10);
}

/**
 * Recompute the header checkbox's checked/indeterminate state and the
 * selection bar's visibility/labels from `selectedIds` + the currently
 * rendered `allJobs`. Called after every render() and after the selection
 * changes without a full re-render (export in-flight toggle).
 */
function updateSelectionBar() {
  if (selectAllCheckbox) {
    if (allJobs.length === 0) {
      selectAllCheckbox.checked = false;
      selectAllCheckbox.indeterminate = false;
    } else {
      const selectedCount = allJobs.filter((j) => selectedIds.has(j.id)).length;
      selectAllCheckbox.checked = selectedCount === allJobs.length;
      selectAllCheckbox.indeterminate =
        selectedCount > 0 && selectedCount < allJobs.length;
    }
  }
  if (!selectionBar || !selectionCountEl || !exportBtn || !selectionWarningEl) return;
  const n = selectedIds.size;
  if (n === 0) {
    selectionBar.classList.add("hidden");
    return;
  }
  selectionBar.classList.remove("hidden");
  const selectedJobs = allJobs.filter((j) => selectedIds.has(j.id));
  const doneCount = selectedJobs.filter((j) => j.status === "done").length;
  const unfinishedCount = selectedJobs.length - doneCount;
  selectionCountEl.textContent = `${n} selected`;
  exportBtn.textContent = `Export ${doneCount}`;
  exportBtn.disabled = doneCount === 0 || exportInFlight;
  selectionWarningEl.textContent =
    unfinishedCount > 0 ? `${unfinishedCount} unfinished won't be exported` : "";
}

/**
 * Row checkbox click handler — plain click toggles just that row; shift-click
 * extends the last explicitly-clicked row's new state across the range
 * between it and this row (inclusive), same as the familiar file-manager
 * convention.
 * @param {MouseEvent} ev
 */
function onRowCheckboxClick(ev) {
  const cb = /** @type {HTMLInputElement} */ (ev.target);
  const id = cb.dataset.id;
  if (!id) return;
  const idx = allJobs.findIndex((j) => j.id === id);
  if (idx === -1) return;
  const checked = cb.checked;
  // Resolve the anchor to an index NOW, not when it was recorded — allJobs
  // can shift (new jobs unshift onto the front) between the anchor click
  // and this one, so a stale index would range over the wrong rows.
  const anchorIdx =
    lastClickedId !== null ? allJobs.findIndex((j) => j.id === lastClickedId) : -1;
  if (ev.shiftKey && anchorIdx !== -1) {
    const [start, end] = [anchorIdx, idx].sort((a, b) => a - b);
    for (let i = start; i <= end; i++) {
      const jid = allJobs[i].id;
      if (checked) selectedIds.add(jid);
      else selectedIds.delete(jid);
    }
  } else if (checked) {
    selectedIds.add(id);
  } else {
    selectedIds.delete(id);
  }
  lastClickedId = id;
  render();
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

const importBtn = /** @type {HTMLButtonElement | null} */ (
  document.getElementById("import-btn")
);
const importFileInput = /** @type {HTMLInputElement | null} */ (
  document.getElementById("import-file")
);

importBtn?.addEventListener("click", () => importFileInput?.click());

importFileInput?.addEventListener("change", async () => {
  const file = importFileInput.files?.[0];
  if (!file) return;
  if (importBtn) importBtn.disabled = true;
  try {
    const resp = await daemon.importJobs(file);
    await refetch();
    reportImportResult(resp);
  } catch (err) {
    alert(`Import failed: ${stringifyError(err)}`);
  } finally {
    if (importBtn) importBtn.disabled = false;
    // Reset so re-picking the exact same file still fires `change`.
    importFileInput.value = "";
  }
});

/** @param {JobImportResponse} resp */
function reportImportResult(resp) {
  const importedN = resp.imported?.length || 0;
  const skippedN = resp.skipped?.length || 0;
  const failedN = resp.failed?.length || 0;
  const lines = [`Imported ${importedN} job${importedN === 1 ? "" : "s"}`];
  if (skippedN) {
    lines.push(`${skippedN} duplicate${skippedN === 1 ? "" : "s"} skipped`);
  }
  if (failedN) {
    lines.push(`${failedN} failed`);
  }
  showToast(lines.join("\n"), 6000);
}

// ---------------------------------------------------------------------------
// Whisper queue pause/resume
// ---------------------------------------------------------------------------

const queueToggle = /** @type {HTMLButtonElement | null} */ (
  document.getElementById("queue-toggle")
);
const queueToggleLabel = /** @type {HTMLElement | null} */ (
  document.getElementById("queue-toggle-label")
);
/** @type {{paused: boolean, queue_size: number, running: number} | null} */
let queueState = null;

if (queueToggle) {
  queueToggle.addEventListener("click", async () => {
    if (!queueState) return;
    queueToggle.disabled = true;
    try {
      const next = queueState.paused
        ? await daemon.resumeWorkers()
        : await daemon.pauseWorkers();
      applyQueueState(next);
    } catch (err) {
      alert(`Queue control failed: ${stringifyError(err)}`);
    } finally {
      queueToggle.disabled = false;
    }
  });
}

function applyQueueState(state) {
  queueState = state;
  if (!queueToggleLabel || !queueToggle) return;
  const whisperBacklog = (state.queue_size || 0) + (state.running || 0);
  if (state.paused) {
    queueToggleLabel.textContent = whisperBacklog > 0
      ? `Paused — Resume (${whisperBacklog} waiting)`
      : "Paused — Resume";
    queueToggle.classList.add("queue-paused");
  } else {
    queueToggleLabel.textContent = whisperBacklog > 0
      ? `Pause processing (${whisperBacklog} in queue)`
      : "Pause processing";
    queueToggle.classList.remove("queue-paused");
  }
}

// Initial render + initial workers state. After this, /events drives all
// updates — no polling.
refetch();
daemon.workersStatus().then(applyQueueState).catch(() => {});

eventStream.subscribe((event) => {
  if (event.type === "workers") {
    applyQueueState(event.state);
  } else if (event.type === "job") {
    handleJobEvent(event);
  } else if (event.job_id && (event.type === "done" || event.type === "error")) {
    patchJobInPlace(event);
  }
});

/**
 * Insert / update / remove a row based on a job event from the daemon.
 * @param {{action: string, job: JobSummary}} event
 */
function handleJobEvent(event) {
  if (event.action === "deleted") {
    const id = event.job?.id;
    if (id) {
      allJobs = allJobs.filter((j) => j.id !== id);
      render();
    }
    return;
  }
  // created | updated — replace the matching row in place (or prepend).
  const j = event.job;
  if (!j) return;
  const idx = allJobs.findIndex((x) => x.id === j.id);
  if (idx >= 0) {
    allJobs[idx] = { ...allJobs[idx], ...j };
  } else {
    allJobs.unshift(j);
  }
  render();
}

/**
 * Patch local row from a done/error event so the status badge reflects
 * pipeline completion without a fetch. Stage transitions are intentionally
 * excluded from the Library subscription (server filters them out).
 * @param {{type: string, job_id: string, error?: string}} event
 */
function patchJobInPlace(event) {
  const j = allJobs.find((x) => x.id === event.job_id);
  if (!j) return;
  if (event.type === "done") {
    j.status = /** @type {JobStatus} */ ("done");
    j.progress_stage = null;
  } else if (event.type === "error") {
    j.status = /** @type {JobStatus} */ ("failed");
    j.progress_stage = null;
  }
  render();
}

async function refetch() {
  try {
    /** @type {JobStatus[] | undefined} */
    let statuses;
    if (filterStatus.value) {
      statuses = /** @type {JobStatus[]} */ (
        filterStatus.value.split(",").map((s) => s.trim()).filter(Boolean)
      );
    }
    const params = {
      status: statuses,
      kind: filterKind.value || undefined,
      limit: 500,
    };
    const resp = await daemon.listJobs(params);
    allJobs = resp.items || [];
    render();
  } catch (err) {
    renderError(err);
  }
}

function render() {
  // Prune ids that no longer exist (deleted jobs, or filtered out of the
  // currently fetched set) before recomputing anything selection-related.
  const existingIds = new Set(allJobs.map((j) => j.id));
  for (const id of selectedIds) {
    if (!existingIds.has(id)) selectedIds.delete(id);
  }
  if (allJobs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty">No jobs found.</td></tr>`;
    updateSelectionBar();
    return;
  }
  tbody.innerHTML = allJobs.map(renderRow).join("");
  // Wire up button handlers (event delegation would also work).
  for (const btn of tbody.querySelectorAll("button[data-action]")) {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const id = /** @type {HTMLElement} */ (btn).dataset.id;
      const action = /** @type {HTMLElement} */ (btn).dataset.action;
      if (!id || !action) return;
      handleAction(action, id).catch((err) =>
        alert(`${action} failed: ${stringifyError(err)}`),
      );
    });
  }
  // Row selection checkboxes.
  for (const cb of tbody.querySelectorAll("input.row-select")) {
    cb.addEventListener("click", (ev) => {
      ev.stopPropagation();
      onRowCheckboxClick(/** @type {MouseEvent} */ (ev));
    });
  }
  // Row click → open.
  for (const row of tbody.querySelectorAll("tr[data-id]")) {
    row.addEventListener("click", (ev) => {
      // Ignore clicks that originated on a button, link, checkbox, or label
      // (the leftmost select-column shouldn't open the side panel).
      const target = /** @type {HTMLElement} */ (ev.target);
      if (target.closest("button, a, input, label")) return;
      const id = /** @type {HTMLElement} */ (row).dataset.id;
      if (id) openInSidePanel(id).catch((err) => alert(stringifyError(err)));
    });
  }
  updateSelectionBar();
}

/** @param {JobSummary} j */
function renderRow(j) {
  const kindIcon = j.kind === "youtube" ? "▶" : "📄";
  const created = formatDate(j.created_at);
  const titleText = j.title || j.url;
  const actions = renderActions(j);
  const titleAttr = escapeHtml(titleText);
  const urlAttr = escapeHtml(j.url);
  const { label, cls } = renderStatusBadge(j);
  const checked = selectedIds.has(j.id) ? "checked" : "";
  return `
    <tr data-id="${escapeHtml(j.id)}">
      <td class="select-col">
        <input type="checkbox" class="row-select" data-id="${escapeHtml(j.id)}" ${checked} />
      </td>
      <td class="kind" title="${j.kind}">${kindIcon}</td>
      <td class="title">
        <div class="title-text">${titleAttr}</div>
        <div class="url muted small" title="${urlAttr}">${urlAttr}</div>
      </td>
      <td><span class="status-badge status-${cls}">${label}</span></td>
      <td class="muted small">${escapeHtml(created)}</td>
      <td class="actions">${actions}</td>
    </tr>
  `;
}

const STAGE_LABELS = {
  extracting: "Extracting",
  fetching_captions: "Fetching captions",
  downloading: "Downloading",
  transcribing: "Transcribing",
  ready: "Preparing",
  summarizing: "Summarizing",
  paused: "Paused",
  queued: "Queued",
};

/**
 * Pick the human-friendly status badge for a row.
 * For ``status=running`` we surface ``progress_stage`` so users see what the
 * pipeline is actually doing right now (downloading / transcribing /
 * summarizing / paused) instead of the catch-all "running".
 * @param {JobSummary} j
 * @returns {{label: string, cls: string}}
 */
function renderStatusBadge(j) {
  if (j.status === "running" && j.progress_stage) {
    const cls = j.progress_stage === "paused" ? "paused" : "running";
    return {
      label: escapeHtml(STAGE_LABELS[j.progress_stage] || j.progress_stage),
      cls,
    };
  }
  return { label: escapeHtml(j.status), cls: escapeHtml(j.status) };
}

/** @param {JobSummary} j */
function renderActions(j) {
  const open = `<button data-action="open" data-id="${escapeHtml(j.id)}">Open</button>`;
  const del = `<button data-action="delete" data-id="${escapeHtml(j.id)}" class="danger">Delete</button>`;
  const retry =
    j.status === "failed"
      ? `<button data-action="retry" data-id="${escapeHtml(j.id)}">Retry</button>`
      : "";
  return `${open}${retry}${del}`;
}

/**
 * @param {string} action
 * @param {string} id
 */
async function handleAction(action, id) {
  if (action === "open") {
    await openInSidePanel(id);
  } else if (action === "delete") {
    const job = allJobs.find((x) => x.id === id);
    const what = job?.title || job?.url || id;
    if (!confirm(`Delete "${what}"? This cannot be undone.`)) return;
    await daemon.deleteJob(id);
    allJobs = allJobs.filter((x) => x.id !== id);
    render();
  } else if (action === "retry") {
    const job = allJobs.find((x) => x.id === id);
    if (!job) return;
    if (!confirm(`Re-run "${job.title || job.url}"?`)) return;
    try {
      await daemon.retryJob(id);
      await refetch();
      await openInSidePanel(id);
    } catch (err) {
      alert(`Retry failed: ${stringifyError(err)}`);
    }
  }
}

/** @param {string} id */
async function openInSidePanel(id) {
  // Kick off the panel open FIRST, synchronously within the button-click
  // gesture — Firefox's sidebarAction.open() rejects if any await runs
  // before it. The promise is awaited below so the existing
  // toast-on-failure path still works. (The Chrome path inside the shim
  // resolves the current window itself, same as before.)
  const opening = openSidePanel();
  await chrome.storage.session.set({ activeJobId: id });
  // Try to broadcast so an open side panel switches.
  try {
    // shouldSwitch=true: Library "open in side panel" / "retry" is an
    // explicit request to follow this job — bypass the source-tab guard
    // background.js applies to toolbar-click submissions.
    await chrome.runtime.sendMessage({
      type: "job-created", jobId: id, shouldSwitch: true,
    });
  } catch {
    // No side panel listening — that's fine, it'll read storage on next open.
  }
  // Best-effort attempt to open the side panel. Requires a user gesture,
  // which the button click satisfies (the call was issued synchronously
  // above; only the await is deferred).
  try {
    await opening;
  } catch (err) {
    // Common case: opening a side panel from a tab page sometimes requires the
    // user gesture to come through the action button. Show a friendly hint.
    console.warn("[TLDR] sidePanel.open from library failed", err);
    showToast("Open the side panel from the toolbar to view this job.");
  }
}

/**
 * @param {string} text  - may contain `\n`; CSS (`white-space: pre-line`)
 *   renders it as a line break, for the multi-part import summary.
 * @param {number} [duration]  - ms before auto-dismiss; default matches the
 *   original single-line 3s toast.
 */
function showToast(text, duration = 3000) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = text;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

/** @param {unknown} err */
function renderError(err) {
  tbody.innerHTML = `<tr><td colspan="6" class="empty error">Failed to load jobs: ${escapeHtml(stringifyError(err))}</td></tr>`;
}

/** @param {string} iso */
function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const now = new Date();
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  if (sameDay) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleDateString([], { year: "numeric", month: "short", day: "numeric" });
}

