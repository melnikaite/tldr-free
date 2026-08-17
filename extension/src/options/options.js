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
const llmModelListEl = /** @type {HTMLDataListElement} */ (
  document.getElementById("llm-model-list")
);
const llmApiKeyInput = /** @type {HTMLInputElement} */ (document.getElementById("llm-api-key"));
const llmApiKeySourceEl = /** @type {HTMLElement} */ (
  document.getElementById("llm-api-key-source")
);
const llmApiKeyStorageSelect = /** @type {HTMLSelectElement} */ (
  document.getElementById("llm-api-key-storage")
);
const llmApiKeyStorageHintEl = /** @type {HTMLElement} */ (
  document.getElementById("llm-api-key-storage-hint")
);
const apiKeyVerifyResultEl = /** @type {HTMLElement} */ (
  document.getElementById("api-key-verify-result")
);

const llmApiKeyStorageKeychainOptionEl = /** @type {HTMLOptionElement} */ (
  document.getElementById("llm-api-key-storage-keychain-option")
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
const whisperApiKeyInput = /** @type {HTMLInputElement} */ (
  document.getElementById("whisper-api-key")
);
const whisperApiKeySourceEl = /** @type {HTMLElement} */ (
  document.getElementById("whisper-api-key-source")
);
const whisperApiKeyStorageSelect = /** @type {HTMLSelectElement} */ (
  document.getElementById("whisper-api-key-storage")
);
const whisperApiKeyStorageHintEl = /** @type {HTMLElement} */ (
  document.getElementById("whisper-api-key-storage-hint")
);
const whisperApiKeyVerifyResultEl = /** @type {HTMLElement} */ (
  document.getElementById("whisper-api-key-verify-result")
);
const whisperApiKeyStorageKeychainOptionEl = /** @type {HTMLOptionElement} */ (
  document.getElementById("whisper-api-key-storage-keychain-option")
);
const whisperTestBtn = /** @type {HTMLButtonElement} */ (
  document.getElementById("whisper-test-connection")
);
const whisperTestResultEl = /** @type {HTMLElement} */ (
  document.getElementById("whisper-test-result")
);

const outputLanguageInput = /** @type {HTMLInputElement} */ (
  document.getElementById("output-language")
);

const storageRetentionDaysInput = /** @type {HTMLInputElement} */ (
  document.getElementById("storage-retention-days")
);
const storageNeverDeleteCheckbox = /** @type {HTMLInputElement} */ (
  document.getElementById("storage-never-delete")
);
// Fallback value offered when the user unchecks "Never delete automatically"
// starting from retention_days === 0 (nothing sensible to restore to).
const DEFAULT_RETENTION_DAYS = 365;

const saveSettingsBtn = /** @type {HTMLButtonElement} */ (
  document.getElementById("save-settings")
);
const settingsStatusEl = /** @type {HTMLElement} */ (document.getElementById("settings-status"));
const restartNoticeEl = /** @type {HTMLElement} */ (document.getElementById("restart-notice"));

/**
 * Per-section API-key UI elements, keyed the same way the config API nests
 * them ("llm" / "whisper") — lets renderApiKeySection/addApiKeyPatchFields
 * be written once and reused for both sections instead of copy-pasted.
 */
const API_KEY_SECTIONS = {
  llm: {
    apiKeyInput: llmApiKeyInput,
    apiKeySourceEl: llmApiKeySourceEl,
    storageSelect: llmApiKeyStorageSelect,
    storageHintEl: llmApiKeyStorageHintEl,
    keychainOptionEl: llmApiKeyStorageKeychainOptionEl,
    baseUrlInput: llmBaseUrlInput,
    modelInput: llmModelInput,
  },
  whisper: {
    apiKeyInput: whisperApiKeyInput,
    apiKeySourceEl: whisperApiKeySourceEl,
    storageSelect: whisperApiKeyStorageSelect,
    storageHintEl: whisperApiKeyStorageHintEl,
    keychainOptionEl: whisperApiKeyStorageKeychainOptionEl,
    baseUrlInput: whisperBaseUrlInput,
    modelInput: whisperModelInput,
  },
};

/** Last config snapshot returned by the daemon; used as the diff baseline for PATCH. */
let lastConfig = null;
/** api_key_storage value implied by the last loaded config's api_key_source, per section. */
const initialApiKeyStorage = { llm: "file", whisper: "file" };

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
 * Populate one section's ("llm" | "whisper") API-key UI from its slice of
 * a GET/PATCH /config response. Never writes anything into the password
 * field — only updates its placeholder. Records the effective initial
 * storage mode in `initialApiKeyStorage[section]` so `buildPatch()` can
 * diff against it later.
 *
 * @param {"llm" | "whisper"} section
 * @param {{ api_key_set?: boolean, api_key_hint?: string | null, api_key_source?: string }} [cfgSection]
 * @param {boolean} keychainAvailable
 */
function renderApiKeySection(section, cfgSection, keychainAvailable) {
  const els = API_KEY_SECTIONS[section];

  els.apiKeyInput.value = "";
  els.apiKeyInput.placeholder = cfgSection?.api_key_set
    ? `••••${cfgSection.api_key_hint || ""}`
    : "no API key set";
  els.apiKeySourceEl.textContent = cfgSection?.api_key_set
    ? `Current key source: ${cfgSection.api_key_source}`
    : "";

  // OS keychain is the recommended default, but only offer it when the
  // daemon reports a real, usable backend — otherwise disable the option
  // with an explanatory hint and fall back to File.
  els.keychainOptionEl.disabled = !keychainAvailable;
  if (keychainAvailable) {
    els.keychainOptionEl.title = "";
    els.storageHintEl.hidden = true;
    els.storageHintEl.textContent = "";
  } else {
    els.keychainOptionEl.title = "No usable OS keychain backend on this daemon's machine.";
    els.storageHintEl.hidden = false;
    els.storageHintEl.textContent =
      "OS keychain unavailable here (no usable backend — e.g. no Secret Service " +
      "running in this session on Linux). Falling back to File.";
  }

  const defaultStorage = keychainAvailable ? "keychain" : "file";
  let initial = API_KEY_STORAGE_OPTIONS.includes(cfgSection?.api_key_source)
    ? /** @type {string} */ (cfgSection?.api_key_source)
    : defaultStorage;
  if (initial === "keychain" && !keychainAvailable) initial = "file";
  initialApiKeyStorage[section] = initial;
  els.storageSelect.value = initial;
}

/**
 * Populate the settings form from a GET/PATCH /config response.
 */
function renderConfig(cfg) {
  llmBaseUrlInput.value = cfg.llm?.base_url ?? "";
  llmModelInput.value = cfg.llm?.model ?? "";
  llmContextLengthInput.value = cfg.llm?.context_length ?? "";
  llmSinglePassLimitInput.value = cfg.llm?.single_pass_token_limit ?? "";
  llmMaxConcurrentInput.value = cfg.llm?.max_concurrent_calls ?? "";
  llmReasoningEffortInput.value = cfg.llm?.reasoning_effort ?? "";

  const keychainAvailable = cfg.keychain_available === true;
  renderApiKeySection("llm", cfg.llm, keychainAvailable);

  whisperBaseUrlInput.value = cfg.whisper?.base_url ?? "";
  whisperModelInput.value = cfg.whisper?.model ?? "";
  whisperMaxUploadInput.value = cfg.whisper?.max_upload_mb ?? "";
  renderApiKeySection("whisper", cfg.whisper, keychainAvailable);

  outputLanguageInput.value = cfg.output?.language ?? "";

  renderStorageSection(cfg);
}

/**
 * Populate the Storage section's "never delete" checkbox + retention-days
 * input from a GET/PATCH /config response. `retention_days === 0` means
 * automatic deletion is off — surfaced as a checked "Never delete
 * automatically" box (disabling the number input) rather than expecting
 * the user to know 0 is the magic off value.
 */
function renderStorageSection(cfg) {
  const days = cfg.storage?.retention_days;
  if (days === 0) {
    storageNeverDeleteCheckbox.checked = true;
    storageRetentionDaysInput.value = String(DEFAULT_RETENTION_DAYS);
    storageRetentionDaysInput.disabled = true;
  } else {
    storageNeverDeleteCheckbox.checked = false;
    storageRetentionDaysInput.value = days != null ? String(days) : "";
    storageRetentionDaysInput.disabled = false;
  }
}

storageNeverDeleteCheckbox.addEventListener("change", () => {
  const checked = storageNeverDeleteCheckbox.checked;
  storageRetentionDaysInput.disabled = checked;
  if (!checked) {
    const current = Number(storageRetentionDaysInput.value);
    if (!storageRetentionDaysInput.value || !Number.isFinite(current) || current <= 0) {
      storageRetentionDaysInput.value = String(
        lastConfig?.storage?.retention_days || DEFAULT_RETENTION_DAYS,
      );
    }
  }
});

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

/**
 * Add `api_key`/`api_key_storage` to `sectionPatch` if the user changed
 * either — shared by both the llm and whisper PATCH builders below.
 *
 * @param {"llm" | "whisper"} section
 * @param {object} sectionPatch
 */
function addApiKeyPatchFields(section, sectionPatch) {
  const els = API_KEY_SECTIONS[section];
  // Empty API key field means "do not change" — the field is simply omitted.
  const apiKeyRaw = els.apiKeyInput.value.trim();
  if (apiKeyRaw !== "") sectionPatch.api_key = apiKeyRaw;

  const storageValue = els.storageSelect.value;
  if (storageValue !== initialApiKeyStorage[section]) sectionPatch.api_key_storage = storageValue;
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

  addApiKeyPatchFields("llm", llmPatch);

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
  addApiKeyPatchFields("whisper", whisperPatch);

  if (Object.keys(whisperPatch).length) patch.whisper = whisperPatch;

  const outputPatch = {};
  addStringDiff(outputPatch, "language", outputLanguageInput.value, lastConfig.output?.language);
  if (Object.keys(outputPatch).length) patch.output = outputPatch;

  const storagePatch = {};
  if (storageNeverDeleteCheckbox.checked) {
    // "Never delete automatically" sends 0 regardless of whatever is
    // (disabled) in the number input.
    if ((lastConfig.storage?.retention_days ?? null) !== 0) storagePatch.retention_days = 0;
  } else {
    addNumberDiff(
      storagePatch,
      "retention_days",
      storageRetentionDaysInput.value,
      lastConfig.storage?.retention_days,
    );
  }
  if (Object.keys(storagePatch).length) patch.storage = storagePatch;

  return patch;
}

/**
 * Render a POST /config/test result into the given container element
 * (#test-result for llm, #whisper-test-result for whisper).
 *
 * @param {HTMLElement} container
 * @param {object} result
 */
function renderTestResult(container, result) {
  container.textContent = "";

  const summary = document.createElement("div");
  summary.className = result.ok ? "ok" : "err";
  const parts = [result.ok ? "OK" : "Failed", `step: ${result.step}`];
  if (typeof result.latency_ms === "number") parts.push(`${result.latency_ms} ms`);
  if (result.status_code !== undefined && result.status_code !== null) {
    parts.push(`status ${result.status_code}`);
  }
  summary.textContent = parts.join(" — ");
  container.appendChild(summary);

  if (!result.ok && result.detail) {
    const pre = document.createElement("pre");
    pre.className = "err";
    pre.textContent = result.detail;
    container.appendChild(pre);
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
    container.appendChild(details);
  }
}

/**
 * Build the `{ base_url?, model?, api_key? }` overrides object POSTed to
 * `/config/test` for one section, from whatever the user has currently
 * typed into that section's fields (not necessarily saved yet).
 *
 * @param {"llm" | "whisper"} section
 */
function buildTestOverrides(section) {
  const els = API_KEY_SECTIONS[section];
  /** @type {{ base_url?: string, model?: string, api_key?: string }} */
  const overrides = {};
  const baseUrl = els.baseUrlInput.value.trim();
  const model = els.modelInput.value.trim();
  const apiKey = els.apiKeyInput.value.trim();
  if (baseUrl) overrides.base_url = baseUrl;
  if (model) overrides.model = model;
  if (apiKey) overrides.api_key = apiKey;
  return overrides;
}

/** Load the daemon URL (always available, independent of /config). */
async function loadDaemonUrl() {
  const stored = await chrome.storage.local.get("daemonUrl");
  daemonUrlInput.value = stored.daemonUrl || DEFAULT_URL;
}

/**
 * Refresh the `<datalist>` backing the LLM model input from GET /health's
 * `llm_backend_models` (itself fetched by the daemon from the configured
 * backend's own /v1/models). Best-effort only: if the daemon is
 * unreachable, too old to report it, or the backend didn't list any
 * models, silently leave the datalist empty — the model field just
 * behaves like a plain text input, no error shown.
 */
async function refreshLlmModelList() {
  llmModelListEl.replaceChildren();
  try {
    const health = await daemon.health();
    for (const model of health.llm_backend_models ?? []) {
      const option = document.createElement("option");
      option.value = model;
      llmModelListEl.appendChild(option);
    }
  } catch {
    // Daemon unreachable or /health failed — leave the datalist empty.
  }
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
    const llmOverrides = buildTestOverrides("llm");
    const body = Object.keys(llmOverrides).length
      ? { target: "llm", llm: llmOverrides }
      : { target: "llm" };

    const result = await daemon.testConfig(body);
    renderTestResult(testResultEl, result);
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

whisperTestBtn.addEventListener("click", async () => {
  whisperTestResultEl.textContent = "Testing…";
  whisperTestBtn.disabled = true;
  try {
    const whisperOverrides = buildTestOverrides("whisper");
    const body = Object.keys(whisperOverrides).length
      ? { target: "whisper", whisper: whisperOverrides }
      : { target: "whisper" };

    const result = await daemon.testConfig(body);
    renderTestResult(whisperTestResultEl, result);
  } catch (err) {
    whisperTestResultEl.textContent = "";
    const pre = document.createElement("pre");
    pre.className = "err";
    pre.textContent = `Request failed: ${formatRequestError(err)}`;
    whisperTestResultEl.appendChild(pre);
  } finally {
    whisperTestBtn.disabled = false;
  }
});

/**
 * Render one section's write-then-read-back verification result into its
 * hint element, given the PATCH response's `<prefix>api_key_verified` /
 * `<prefix>api_key_verify_error` fields.
 *
 * @param {HTMLElement} el
 * @param {boolean} verified
 * @param {string | null | undefined} verifyError
 */
function renderApiKeyVerifyResult(el, verified, verifyError) {
  if (verified) {
    el.textContent = "API key verified — read back successfully.";
    el.className = "hint ok";
  } else {
    el.textContent = `API key saved, but verification failed: ${verifyError || "unknown reason"}`;
    el.className = "hint err";
  }
}

saveSettingsBtn.addEventListener("click", async () => {
  settingsStatusEl.textContent = "";
  settingsStatusEl.className = "";
  restartNoticeEl.hidden = true;
  apiKeyVerifyResultEl.textContent = "";
  apiKeyVerifyResultEl.className = "hint";
  whisperApiKeyVerifyResultEl.textContent = "";
  whisperApiKeyVerifyResultEl.className = "hint";

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
  // Only the API key write path is write-then-read-back verified —
  // don't show a verification line for a save that didn't touch it.
  const llmKeyWasWritten =
    patch.llm !== undefined &&
    (patch.llm.api_key !== undefined || patch.llm.api_key_storage !== undefined);
  const whisperKeyWasWritten =
    patch.whisper !== undefined &&
    (patch.whisper.api_key !== undefined || patch.whisper.api_key_storage !== undefined);

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
    if (llmKeyWasWritten) {
      renderApiKeyVerifyResult(apiKeyVerifyResultEl, result.api_key_verified, result.api_key_verify_error);
    }
    if (whisperKeyWasWritten) {
      renderApiKeyVerifyResult(
        whisperApiKeyVerifyResultEl,
        result.whisper_api_key_verified,
        result.whisper_api_key_verify_error,
      );
    }
    // base_url/model may have changed to point at a different backend —
    // refresh the model suggestions to match.
    await refreshLlmModelList();
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
  await refreshLlmModelList();
});

(async function init() {
  try {
    await loadDaemonUrl();
  } catch (err) {
    daemonStatusEl.textContent = `Failed to load daemon URL: ${err instanceof Error ? err.message : String(err)}`;
    daemonStatusEl.className = "err";
  }
  await loadSettings();
  await refreshLlmModelList();
})();
