#!/usr/bin/env bash
# Build the Firefox variant of the extension into dist/firefox/.
#
# The Chrome tree in extension/ is the source of truth; the only Firefox
# difference is the manifest (sidebar_action instead of side_panel, event
# page instead of service worker — see extension/manifest.firefox.json).
# All JS is shared: extension/src/lib/browser-compat.js picks the right
# panel API at runtime.
#
# Idempotent: re-running fully syncs dist/firefox/ with the current tree
# (rsync --delete removes anything stale).
#
# Prerequisite: `task install` must have populated extension/vendor/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/extension"
DEST="$ROOT/dist/firefox"

if [[ ! -d "$SRC/vendor" ]] || [[ -z "$(ls -A "$SRC/vendor" 2>/dev/null)" ]]; then
  echo "error: $SRC/vendor is missing or empty — run 'task install' first." >&2
  exit 1
fi

mkdir -p "$DEST"
rsync -a --delete \
  --exclude "manifest.json" \
  --exclude "manifest.firefox.json" \
  "$SRC/" "$DEST/"
cp "$SRC/manifest.firefox.json" "$DEST/manifest.json"

echo "Firefox build ready: $DEST"
echo "Load via about:debugging → This Firefox → Load Temporary Add-on → $DEST/manifest.json"
