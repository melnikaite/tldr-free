// Library page controller — browse mode.
//
// Behavior:
//   - On load: daemon.listJobs(), render rows.
//   - Subscribe to GET /events for realtime updates (job created/updated/
//     deleted, workers state). No polling.
//   - Filters trigger refetch. Search filters the loaded list locally by title.
//   - Per-row actions: open in side panel (writes activeJobId to session storage,
//     attempts chrome.sidePanel.open), delete (with confirm), retry (failed only).

import { daemon } from "../lib/daemon-client.js";
import { openEventStream } from "../lib/event-stream.js";
import { escapeHtml, stringifyError } from "../lib/utils.js";

// Library only renders status badges + queue counter — skip the high-volume
// stage/delta chatter from running pipelines.
const eventStream = openEventStream({ types: ["job", "workers", "done", "error"] });

/** @import { JobSummary, JobStatus } from "../lib/api-types.js" */

const tbody = /** @type {HTMLElement} */ (
  document.querySelector("#jobs tbody")
);
const filterStatus = /** @type {HTMLSelectElement} */ (
  document.getElementById("filter-status")
);
const filterKind = /** @type {HTMLSelectElement} */ (
  document.getElementById("filter-kind")
);
const filterSearch = /** @type {HTMLInputElement} */ (
  document.getElementById("filter-search")
);

/** @type {JobSummary[]} */
let allJobs = [];

filterStatus.addEventListener("change", () => refetch());
filterKind.addEventListener("change", () => refetch());
let searchDebounce = 0;
filterSearch.addEventListener("input", () => {
  clearTimeout(searchDebounce);
  searchDebounce = window.setTimeout(() => render(), 150);
});

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
  const q = filterSearch.value.trim().toLowerCase();
  const items = q
    ? allJobs.filter((j) => (j.title || "").toLowerCase().includes(q) || j.url.toLowerCase().includes(q))
    : allJobs;

  if (items.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty">No jobs found.</td></tr>`;
    return;
  }
  tbody.innerHTML = items.map(renderRow).join("");
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
  // Row click → open.
  for (const row of tbody.querySelectorAll("tr[data-id]")) {
    row.addEventListener("click", (ev) => {
      // Ignore clicks that originated on a button or link.
      const target = /** @type {HTMLElement} */ (ev.target);
      if (target.closest("button, a")) return;
      const id = /** @type {HTMLElement} */ (row).dataset.id;
      if (id) openInSidePanel(id).catch((err) => alert(stringifyError(err)));
    });
  }
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
  return `
    <tr data-id="${escapeHtml(j.id)}">
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
  await chrome.storage.session.set({ activeJobId: id });
  // Try to broadcast so an open side panel switches.
  try {
    await chrome.runtime.sendMessage({ type: "job-created", jobId: id });
  } catch {
    // No side panel listening — that's fine, it'll read storage on next open.
  }
  // Best-effort attempt to open the side panel. Requires a user gesture, which
  // a button click satisfies. Need a windowId — use the current window.
  try {
    const win = await chrome.windows.getCurrent();
    if (win?.id !== undefined) {
      await chrome.sidePanel.open({ windowId: win.id });
    }
  } catch (err) {
    // Common case: opening a side panel from a tab page sometimes requires the
    // user gesture to come through the action button. Show a friendly hint.
    console.warn("[TLDR] sidePanel.open from library failed", err);
    showToast("Open the side panel from the toolbar to view this job.");
  }
}

/** @param {string} text */
function showToast(text) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = text;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

/** @param {unknown} err */
function renderError(err) {
  tbody.innerHTML = `<tr><td colspan="5" class="empty error">Failed to load jobs: ${escapeHtml(stringifyError(err))}</td></tr>`;
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

