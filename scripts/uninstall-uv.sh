#!/bin/sh
#
# TLDR — native (uv) uninstaller. Mirror of install-uv.sh.
# POSIX sh, macOS + Linux. Idempotent.
#
#   1. Stop + deregister the autostart service.
#   2. uv tool uninstall the daemon.
#   3. WITH CONFIRMATION: delete the data dir (SQLite library) and config.
#
# Flags:
#   --purge    non-interactive full removal (skips the confirmation)

set -eu

PURGE=0
for arg in "$@"; do
  case "$arg" in
    --purge) PURGE=1 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

ok()   { printf '\033[32m+\033[0m %s\n' "$1"; }
warn() { printf '\033[33m!\033[0m %s\n' "$1"; }
hdr()  { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

PATH="$HOME/.local/bin:$PATH"
export PATH

hdr "Stop + remove the autostart service"
if command -v tldr-daemon >/dev/null 2>&1; then
  tldr-daemon service uninstall || warn "service uninstall failed (was it installed?)"
else
  warn "tldr-daemon not on PATH — skipping service uninstall"
fi

hdr "Uninstall the daemon"
if command -v uv >/dev/null 2>&1; then
  uv tool uninstall tldr-daemon 2>/dev/null && ok "tldr-daemon uninstalled" \
    || warn "tldr-daemon was not installed via uv tool"
else
  warn "uv not found — nothing to uninstall"
fi

hdr "Data + config"
if [ "$(uname -s)" = "Darwin" ]; then
  cfg_dir="$HOME/Library/Application Support/tldr"
  data_dir="$cfg_dir/data"
else
  cfg_dir="${XDG_CONFIG_HOME:-$HOME/.config}/tldr"
  data_dir="${XDG_DATA_HOME:-$HOME/.local/share}/tldr"
fi

if [ ! -d "$cfg_dir" ] && [ ! -d "$data_dir" ]; then
  ok "no data or config found — nothing to delete"
  exit 0
fi

if [ "$PURGE" = 1 ]; then
  answer=y
else
  printf 'Delete the SQLite library (%s) and config (%s)? [y/N] ' "$data_dir" "$cfg_dir"
  read -r answer || answer=n
fi

case "$answer" in
  y|Y|yes|YES)
    rm -rf "$data_dir" "$cfg_dir"
    ok "data and config removed"
    ;;
  *)
    warn "kept data and config (re-run with --purge for full removal)"
    ;;
esac
