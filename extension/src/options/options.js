// Options page.
//
// Two independent concerns:
//   - Daemon URL: client-side only, persisted to chrome.storage.local
//     (never sent to the daemon's own /config).
//   - Everything else (LLM backend, Whisper backend, output language):
//     lives in the daemon's config file, read/written via GET/PATCH
//     /config. That endpoint may not exist yet (older daemon) or the
//     daemon may be unreachable — in either case the settings fieldset
//     is disabled and the Daemon URL section keeps working on its own.

import { daemon } from "../lib/daemon-client.js";

const DEFAULT_URL = "http://127.0.0.1:8765";
const API_KEY_STORAGE_OPTIONS = ["file", "keychain", "inline"];

// --- Daemon URL section (unchanged behavior, new element ids) ---------

const daemonUrlInput = /** @type {HTMLInputElement} */ (document.getElementById("daemon-url"));
const saveDaemonBtn = /** @type {HTMLButtonElement} */ (document.getElementById("save-daemon"));
const daemonStatusEl = /** @type {HTMLElement} */ (document.getElementById("daemon-status"));

// --- Settings section (LLM / Whisper / Output) -------------------------

const settingsFieldset = /** @type {HTMLFieldSetElement} */ (
  document.getElementById("settings-fieldset")
);
const configUnavailableEl = /** @type {HTMLElement} */ (
  document.getElementById("config-unavailable")
);

const llmBaseUrlInput = /** @type {HTMLInputElement} */ (document.getElementById("llm-base-url"));
const llmModelInput = /** @type {HTMLInputElement} */ (document.getElementById("llm-model"));
const llmContextLengthInput = /** @type {HTMLInputElement} */ (
  document.getElementById("llm-context-length")
);
const llmSinglePassLimitInput = /** @type {HTMLInputElement} */ (
  document.getElementById("llm-single-pass-limit")
);
const llmMaxConcurrentInput = /** @type {HTMLInputElement} */ (
  document.getElementById("llm-max-concurrent")
);
const llmReasoningEffortInput = /** @type {HTMLInputElement} */ (
  document.getElementById("llm-reasoning-effort")
);
const llmApiKeyInput = /** @type {HTMLInputElement} */ (document.getElementById("llm-api-key"));
const llmApiKeySourceEl = /** @type {HTMLElement} */ (
  document.getElementById("llm-api-key-source")
);
const llmApiKeyStorageSelect = /** @type {HTMLSelectElement} */ (
  document.getElementById("llm-api-key-storage")
);

const testBtn = /** @type {HTMLButtonElement} */ (document.getElementById("test-connection"));
const testResultEl = /** @type {HTMLElement} */ (document.getElementById("test-result"));

const whisperBaseUrlInput = /** @type {HTMLInputElement} */ (
  document.getElementById("whisper-base-url")
);
const whisperModelInput = /** @type {HTMLInputElement} */ (
  document.getElementById("whisper-model")
);
const whisperMaxUploadInput = /** @type {HTMLInputElement} */ (
  document.getElementById("whisper-max-upload")
);

const outputLanguageInput = /** @type {HTMLInputElement} */ (
  document.getElementById("output-language")
);

const saveSettingsBtn = /** @type {HTMLButtonElement} */ (
  document.getElementById("save-settings")
);
const settingsStatusEl = /** @type {HTMLElement} */ (document.getElementById("settings-status"));
const restartNoticeEl = /** @type {HTMLElement} */ (document.getElementById("restart-notice"));

/** Last config snapshot returned by the daemon; used as the diff baseline for PATCH. */
let lastConfig = null;
/** api_key_storage value implied by the last loaded config's api_key_source. */
let initialApiKeyStorage = "file";

/**
 * Turn an Error thrown by daemon-client's `request()` (shape:
 * "<status> <statusText>: <body>") into a readable message, pulling out
 * the `detail` field from a JSON error body (e.g. FastAPI's 422) when
 * present.
 *
 * @param {unknown} err
 * @returns {string}
 */
function formatRequestError(err) {
  const msg = err instanceof Error ? err.message : String(err);
  const sep = msg.indexOf(": ");
  if (sep === -1) return msg;
  const prefix = msg.slice(0, sep);
  const bodyText = msg.slice(sep + 2);
  try {
    const body = JSON.parse(bodyText);
    if (body && typeof body.detail === "string") return `${prefix}: ${body.detail}`;
    if (body && body.detail !== undefined) return `${prefix}: ${JSON.stringify(body.detail)}`;
  } catch {
    // Not JSON — fall through to the raw message.
  }
  return msg;
}

/**
 * Populate the settings form from a GET/PATCH /config response. Never
 * writes anything into the password field — only updates its placeholder.
 */
function renderConfig(cfg) {
  llmBaseUrlInput.value = cfg.llm?.base_url ?? "";
  llmModelInput.value = cfg.llm?.model ?? "";
  llmContextLengthInput.value = cfg.llm?.context_length ?? "";
  llmSinglePassLimitInput.value = cfg.llm?.single_pass_token_limit ?? "";
  llmMaxConcurrentInput.value = cfg.llm?.max_concurrent_calls ?? "";
  llmReasoningEffortInput.value = cfg.llm?.reasoning_effort ?? "";

  llmApiKeyInput.value = "";
  llmApiKeyInput.placeholder = cfg.llm?.api_key_set
    ? `••••${cfg.llm.api_key_hint || ""}`
    : "no API key set";
  llmApiKeySourceEl.textContent = cfg.llm?.api_key_set
    ? `Current key source: ${cfg.llm.api_key_source}`
    : "";

  initialApiKeyStorage = API_KEY_STORAGE_OPTIONS.includes(cfg.llm?.api_key_source)
    ? cfg.llm.api_key_source
    : "file";
  llmApiKeyStorageSelect.value = initialApiKeyStorage;

  whisperBaseUrlInput.value = cfg.whisper?.base_url ?? "";
  whisperModelInput.value = cfg.whisper?.model ?? "";
  whisperMaxUploadInput.value = cfg.whisper?.max_upload_mb ?? "";

  outputLanguageInput.value = cfg.output?.language ?? "";
}

/** Diff helper: string field, changed only if the trimmed value differs. */
function addStringDiff(target, key, rawValue, oldValue) {
  const newValue = rawValue.trim();
  const old = oldValue ?? "";
  if (newValue !== old) target[key] = newValue;
}

/**
 * Diff helper: numeric field. An empty input means "leave unchanged"
 * (we never send a blank number); a non-numeric input is ignored rather
 * than sent as NaN.
 */
function addNumberDiff(target, key, rawValue, oldValue) {
  const trimmed = rawValue.trim();
  if (trimmed === "") return;
  const num = Number(trimmed);
  if (Number.isNaN(num)) return;
  if (num !== oldValue) target[key] = num;
}

/** Build a PATCH /config body containing only fields the user changed. */
function buildPatch() {
  const patch = {};

  const llmPatch = {};
  addStringDiff(llmPatch, "base_url", llmBaseUrlInput.value, lastConfig.llm?.base_url);
  addStringDiff(llmPatch, "model", llmModelInput.value, lastConfig.llm?.model);
  addNumberDiff(
    llmPatch,
    "context_length",
    llmContextLengthInput.value,
    lastConfig.llm?.context_length,
  );
  addNumberDiff(
    llmPatch,
    "single_pass_token_limit",
    llmSinglePassLimitInput.value,
    lastConfig.llm?.single_pass_token_limit,
  );
  addNumberDiff(
    llmPatch,
    "max_concurrent_calls",
    llmMaxConcurrentInput.value,
    lastConfig.llm?.max_concurrent_calls,
  );

  const reasoningRaw = llmReasoningEffortInput.value.trim();
  const reasoningNew = reasoningRaw === "" ? null : reasoningRaw;
  const reasoningOld = lastConfig.llm?.reasoning_effort ?? null;
  if (reasoningNew !== reasoningOld) llmPatch.reasoning_effort = reasoningNew;

  // Empty API key field means "do not change" — the field is simply omitted.
  const apiKeyRaw = llmApiKeyInput.value.trim();
  if (apiKeyRaw !== "") llmPatch.api_key = apiKeyRaw;

  const storageValue = llmApiKeyStorageSelect.value;
  if (storageValue !== initialApiKeyStorage) llmPatch.api_key_storage = storageValue;

  if (Object.keys(llmPatch).length) patch.llm = llmPatch;

  const whisperPatch = {};
  addStringDiff(whisperPatch, "base_url", whisperBaseUrlInput.value, lastConfig.whisper?.base_url);
  addStringDiff(whisperPatch, "model", whisperModelInput.value, lastConfig.whisper?.model);
  addNumberDiff(
    whisperPatch,
    "max_upload_mb",
    whisperMaxUploadInput.value,
    lastConfig.whisper?.max_upload_mb,
  );
  if (Object.keys(whisperPatch).length) patch.whisper = whisperPatch;

  const outputPatch = {};
  addStringDiff(outputPatch, "language", outputLanguageInput.value, lastConfig.output?.language);
  if (Object.keys(outputPatch).length) patch.output = outputPatch;

  return patch;
}

/** Render a POST /config/test result into #test-result. */
function renderTestResult(result) {
  testResultEl.textContent = "";

  const summary = document.createElement("div");
  summary.className = result.ok ? "ok" : "err";
  const parts = [result.ok ? "OK" : "Failed", `step: ${result.step}`];
  if (typeof result.latency_ms === "number") parts.push(`${result.latency_ms} ms`);
  if (result.status_code !== undefined && result.status_code !== null) {
    parts.push(`status ${result.status_code}`);
  }
  summary.textContent = parts.join(" — ");
  testResultEl.appendChild(summary);

  if (!result.ok && result.detail) {
    const pre = document.createElement("pre");
    pre.className = "err";
    pre.textContent = result.detail;
    testResultEl.appendChild(pre);
  }

  if (Array.isArray(result.models) && result.models.length) {
    const details = document.createElement("details");
    const summaryEl = document.createElement("summary");
    summaryEl.textContent = `${result.models.length} model${result.models.length === 1 ? "" : "s"} available`;
    details.appendChild(summaryEl);
    const list = document.createElement("ul");
    for (const m of result.models) {
      const li = document.createElement("li");
      li.textContent = String(m);
      list.appendChild(li);
    }
    details.appendChild(list);
    testResultEl.appendChild(details);
  }
}

/** Load the daemon URL (always available, independent of /config). */
async function loadDaemonUrl() {
  const stored = await chrome.storage.local.get("daemonUrl");
  daemonUrlInput.value = stored.daemonUrl || DEFAULT_URL;
}

/**
 * Load GET /config and populate the settings form. On any failure — the
 * daemon is unreachable, or it's an older build without /config yet —
 * disable the whole settings fieldset and show an inline notice, without
 * throwing.
 */
async function loadSettings() {
  try {
    const cfg = await daemon.getConfig();
    lastConfig = cfg;
    renderConfig(cfg);
    settingsFieldset.disabled = false;
    configUnavailableEl.hidden = true;
    configUnavailableEl.textContent = "";
  } catch (err) {
    lastConfig = null;
    settingsFieldset.disabled = true;
    configUnavailableEl.hidden = false;
    configUnavailableEl.textContent =
      `Settings API unavailable — daemon unreachable or too old (GET /config failed): ${formatRequestError(err)}`;
  }
}

testBtn.addEventListener("click", async () => {
  testResultEl.textContent = "Testing…";
  testBtn.disabled = true;
  try {
    /** @type {{ base_url?: string, model?: string, api_key?: string }} */
    const llmOverrides = {};
    const baseUrl = llmBaseUrlInput.value.trim();
    const model = llmModelInput.value.trim();
    const apiKey = llmApiKeyInput.value.trim();
    if (baseUrl) llmOverrides.base_url = baseUrl;
    if (model) llmOverrides.model = model;
    if (apiKey) llmOverrides.api_key = apiKey;
    const overrides = Object.keys(llmOverrides).length ? { llm: llmOverrides } : {};

    const result = await daemon.testConfig(overrides);
    renderTestResult(result);
  } catch (err) {
    testResultEl.textContent = "";
    const pre = document.createElement("pre");
    pre.className = "err";
    pre.textContent = `Request failed: ${formatRequestError(err)}`;
    testResultEl.appendChild(pre);
  } finally {
    testBtn.disabled = false;
  }
});

saveSettingsBtn.addEventListener("click", async () => {
  settingsStatusEl.textContent = "";
  settingsStatusEl.className = "";
  restartNoticeEl.hidden = true;

  if (!lastConfig) {
    settingsStatusEl.textContent = "Settings not loaded — nothing to save.";
    settingsStatusEl.className = "err";
    return;
  }

  const patch = buildPatch();
  if (Object.keys(patch).length === 0) {
    settingsStatusEl.textContent = "Nothing changed.";
    return;
  }

  saveSettingsBtn.disabled = true;
  try {
    const result = await daemon.updateConfig(patch);
    lastConfig = result;
    renderConfig(result);
    settingsStatusEl.textContent = "Saved.";
    settingsStatusEl.className = "ok";
    if (result.restart_required) {
      restartNoticeEl.textContent =
        "Some changes only take effect after the daemon restarts. Run: " +
        "tldr-daemon service uninstall && tldr-daemon service install";
      restartNoticeEl.hidden = false;
      restartNoticeEl.className = "notice";
    }
  } catch (err) {
    settingsStatusEl.textContent = `Save failed: ${formatRequestError(err)}`;
    settingsStatusEl.className = "err";
  } finally {
    saveSettingsBtn.disabled = false;
  }
});

saveDaemonBtn.addEventListener("click", async () => {
  daemonStatusEl.textContent = "";
  daemonStatusEl.className = "";

  const url = daemonUrlInput.value.trim() || DEFAULT_URL;
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    daemonStatusEl.textContent = "Invalid URL.";
    daemonStatusEl.className = "err";
    return;
  }
  if (!/^https?:$/.test(parsed.protocol)) {
    daemonStatusEl.textContent = "URL must start with http:// or https://";
    daemonStatusEl.className = "err";
    return;
  }
  // Strip trailing slash for consistency.
  const cleanUrl = url.replace(/\/+$/, "");

  await chrome.storage.local.set({ daemonUrl: cleanUrl });

  daemonStatusEl.textContent = "Saved. Checking daemon…";
  daemonStatusEl.className = "";

  try {
    const health = await daemon.health();
    const llmStatus = health.llm_backend_reachable
      ? "ok"
      : `unreachable${health.llm_backend_error ? ` — ${health.llm_backend_error}` : ""}`;
    daemonStatusEl.textContent = `Saved — daemon ${health.status} (LLM backend: ${llmStatus}).`;
    daemonStatusEl.className = "ok";
  } catch (err) {
    daemonStatusEl.textContent = `Saved, but daemon check failed: ${err instanceof Error ? err.message : String(err)}`;
    daemonStatusEl.className = "err";
  }

  // The daemon URL changed — reload settings from the (possibly
  // different) daemon at the new address.
  await loadSettings();
});

(async function init() {
  try {
    await loadDaemonUrl();
  } catch (err) {
    daemonStatusEl.textContent = `Failed to load daemon URL: ${err instanceof Error ? err.message : String(err)}`;
    daemonStatusEl.className = "err";
  }
  await loadSettings();
})();
