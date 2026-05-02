#!/bin/sh
# Auto-update YouTube-related libraries on container start.
#
# Google plays cat-and-mouse with yt-dlp and youtube-transcript-api; restarting
# the daemon (`task down && task up`, or `docker compose restart daemon`) pulls
# the latest fixes without needing an image rebuild.
#
# Skipped for non-server commands (tests, linters, ad-hoc exec) so `task test`
# stays fast. The pyproject.toml pins act as a floor; pip will only ever go
# upward from there.
#
# If pip can't reach the network (offline), we log a warning and continue with
# whatever was baked into the image.

set -e

if [ "$1" = "uvicorn" ]; then
  echo "==> entrypoint: refreshing yt-dlp + youtube-transcript-api ..."
  if pip install --upgrade --disable-pip-version-check --quiet \
       --root-user-action=ignore yt-dlp youtube-transcript-api; then
    yt_ver=$(pip show yt-dlp                 | awk '/^Version:/{print $2}')
    yta_ver=$(pip show youtube-transcript-api | awk '/^Version:/{print $2}')
    echo "==> yt-dlp=$yt_ver  youtube-transcript-api=$yta_ver"
  else
    echo "WARN pip upgrade failed (offline?); continuing with image versions"
  fi
fi

exec "$@"
