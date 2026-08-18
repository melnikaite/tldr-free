// Sidepanel first-run / welcome screen.
//
// Shown INSTEAD of the normal idle view ("no summary yet" placeholder) when
// GET /health says either the daemon itself, or the configured model
// backend, isn't reachable — see app.js's `_gateIdleOnHealth` for exactly
// when this fires, and extension.md's "Side panel lifecycle" section for
// the reasoning.
//
// Honest structure, not "install or don't": the daemon is ALWAYS required —
// a cloud backend only ever replaces the model, never the daemon. So there
// are exactly two steps, never a "skip the daemon" option:
//   - step "daemon": the daemon itself didn't answer /health at all.
//   - step "model": the daemon answered, but its configured LLM backend
//     isn't reachable (`health.llm_backend_reachable === false`).
//
// Framework-free except plain DOM APIs — no chrome.* here; callbacks own
// that (mirrors error-hints.js's separation). Built entirely with
// createElement/textContent, never innerHTML, because
// `health.llm_backend_error` is backend-controlled text.

const INSTALL_COMMAND =
  "curl -fsSL https://raw.githubusercontent.com/melnikaite/tldr-free/main/scripts/install-uv.sh | sh";

// Anchor verified against the rendered README on GitHub (heading "Install —
// native, no Docker (recommended)"); update this alongside the heading text
// in README.md if that heading is ever reworded.
const INSTALL_DOCS_URL =
  "https://github.com/melnikaite/tldr-free#install--native-no-docker-recommended";

/**
 * @param {"daemon" | "model"} step
 * @param {import("../lib/api-types.js").HealthResponse | null | undefined} health
 * @param {{ onCheckAgain: () => Promise<void> | void, onOpenOptions: () => void }} callbacks
 * @returns {HTMLElement}
 */
export function buildWelcomeView(step, health, callbacks) {
  const root = document.createElement("div");
  root.className = "welcome-block";

  const heading = document.createElement("h2");
  heading.className = "welcome-title";
  heading.textContent = step === "daemon" ? "Welcome to TLDR" : "Almost there";
  root.appendChild(heading);

  if (step === "daemon") {
    root.appendChild(_buildDaemonStep());
  } else {
    root.appendChild(_buildModelStep(health, callbacks));
  }

  root.appendChild(_buildCheckAgainButton(callbacks.onCheckAgain));

  return root;
}

/**
 * Step 1 — the daemon itself never answered /health.
 * @returns {HTMLElement}
 */
function _buildDaemonStep() {
  const frag = document.createElement("div");

  const p1 = document.createElement("p");
  p1.textContent =
    "TLDR needs a small local program — the daemon — running on this " +
    "machine. It does the actual work (fetching pages, transcribing " +
    "video, talking to a model); the browser extension just captures " +
    "pages and shows the results.";
  frag.appendChild(p1);

  const p2 = document.createElement("p");
  p2.textContent = "Install it with one command (macOS or Linux):";
  frag.appendChild(p2);

  frag.appendChild(_buildCommandBlock(INSTALL_COMMAND));

  const p3 = document.createElement("p");
  p3.className = "muted small";
  p3.textContent =
    "This registers a background service that starts automatically, so " +
    "you won't need to run it by hand again.";
  frag.appendChild(p3);

  const docsLink = document.createElement("a");
  docsLink.className = "welcome-docs-link";
  docsLink.href = INSTALL_DOCS_URL;
  docsLink.target = "_blank";
  docsLink.rel = "noopener";
  docsLink.textContent = "Full instructions in the repository (other platforms, updating, troubleshooting)";
  frag.appendChild(docsLink);

  return frag;
}

/**
 * A `<pre><code>` command block with a "Copy" button next to it.
 * @param {string} command
 * @returns {HTMLElement}
 */
function _buildCommandBlock(command) {
  const wrap = document.createElement("div");
  wrap.className = "welcome-command";

  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.textContent = command;
  pre.appendChild(code);
  wrap.appendChild(pre);

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "welcome-copy-btn";
  copyBtn.textContent = "Copy";
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(command);
      copyBtn.textContent = "Copied!";
    } catch {
      copyBtn.textContent = "Copy failed";
    } finally {
      setTimeout(() => {
        copyBtn.textContent = "Copy";
      }, 1500);
    }
  });
  wrap.appendChild(copyBtn);

  return wrap;
}

/**
 * Step 2 — the daemon is up, but its configured LLM backend isn't
 * reachable. Two honest options, never a "works without installing
 * anything" option: the daemon (already confirmed running) stays in the
 * loop either way — a cloud model only replaces the model.
 *
 * @param {import("../lib/api-types.js").HealthResponse | null | undefined} health
 * @param {{ onOpenOptions: () => void }} callbacks
 * @returns {HTMLElement}
 */
function _buildModelStep(health, { onOpenOptions }) {
  const frag = document.createElement("div");

  const p1 = document.createElement("p");
  p1.textContent =
    "The daemon is running, but it has no model to talk to yet. Choose " +
    "where the model that reads and summarizes runs — the daemon you " +
    "already have stays in the loop either way, only the model changes.";
  frag.appendChild(p1);

  frag.appendChild(
    _buildOptionCard(
      "Run it locally — free and private",
      "The model runs on this machine; nothing leaves it. Needs some " +
        "free disk and memory: about 3.1 GB for the smallest tested " +
        "model (Gemma 4 E2B), up to 7-8 GB for the bundled Qwen3-VL 8B " +
        "+ Whisper setup. Works with Ollama, LM Studio, mlx-openai-server " +
        "(Apple Silicon), llama-server, and other OpenAI-compatible servers.",
    ),
  );

  frag.appendChild(
    _buildOptionCard(
      "Use a cloud provider — your key, your account",
      "Bring your own API key (OpenAI, Groq, and other OpenAI-compatible " +
        "providers). No local memory needed, but whatever you summarize " +
        "leaves this machine and goes to that provider.",
    ),
  );

  if (health?.llm_backend_error) {
    const detail = document.createElement("p");
    detail.className = "muted small welcome-health-detail";
    detail.textContent = `GET /health reports: ${health.llm_backend_error}`;
    frag.appendChild(detail);
  }

  const optionsBtn = document.createElement("button");
  optionsBtn.type = "button";
  optionsBtn.className = "welcome-options-btn";
  optionsBtn.textContent = "Open Options";
  optionsBtn.addEventListener("click", () => onOpenOptions());
  frag.appendChild(optionsBtn);

  const hint = document.createElement("p");
  hint.className = "muted small";
  hint.textContent =
    'Configure either one in Options, then use its "Test setup" button ' +
    "to confirm it works.";
  frag.appendChild(hint);

  return frag;
}

/**
 * @param {string} title
 * @param {string} body
 * @returns {HTMLElement}
 */
function _buildOptionCard(title, body) {
  const card = document.createElement("div");
  card.className = "welcome-option-card";

  const h3 = document.createElement("h3");
  h3.textContent = title;
  card.appendChild(h3);

  const p = document.createElement("p");
  p.className = "muted small";
  p.textContent = body;
  card.appendChild(p);

  return card;
}

/**
 * @param {() => Promise<void> | void} onCheckAgain
 * @returns {HTMLElement}
 */
function _buildCheckAgainButton(onCheckAgain) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "welcome-check-btn";
  btn.textContent = "Check again";
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Checking…";
    try {
      await onCheckAgain();
    } finally {
      // If `onCheckAgain` caused a re-render, this element is already
      // detached and these writes are harmless; if the state didn't
      // change, this restores the button for another try.
      btn.disabled = false;
      btn.textContent = "Check again";
    }
  });
  return btn;
}
