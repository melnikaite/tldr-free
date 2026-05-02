// Options page — daemon URL persisted to chrome.storage.local.

import { daemon } from "../lib/daemon-client.js";

const DEFAULT_URL = "http://127.0.0.1:8765";

const urlInput = /** @type {HTMLInputElement} */ (document.getElementById("daemon-url"));
const saveBtn = /** @type {HTMLButtonElement} */ (document.getElementById("save"));
const statusEl = /** @type {HTMLElement} */ (document.getElementById("status"));

(async function load() {
  const stored = await chrome.storage.local.get("daemonUrl");
  urlInput.value = stored.daemonUrl || DEFAULT_URL;
})();

saveBtn.addEventListener("click", async () => {
  statusEl.textContent = "";
  statusEl.className = "";

  const url = urlInput.value.trim() || DEFAULT_URL;
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    statusEl.textContent = "Invalid URL.";
    statusEl.className = "err";
    return;
  }
  if (!/^https?:$/.test(parsed.protocol)) {
    statusEl.textContent = "URL must start with http:// or https://";
    statusEl.className = "err";
    return;
  }
  // Strip trailing slash for consistency.
  const cleanUrl = url.replace(/\/+$/, "");

  await chrome.storage.local.set({ daemonUrl: cleanUrl });

  statusEl.textContent = "Saved. Checking daemon…";
  statusEl.className = "";

  try {
    const health = await daemon.health();
    statusEl.textContent = `Saved — daemon ${health.status} (LLM backend: ${health.llm_backend_reachable ? "ok" : "unreachable"}).`;
    statusEl.className = "ok";
  } catch (err) {
    statusEl.textContent = `Saved, but daemon check failed: ${err instanceof Error ? err.message : String(err)}`;
    statusEl.className = "err";
  }
});
