"""Console entrypoint for native (uv) installs: ``tldr-daemon``.

  tldr-daemon                      run the server (uvicorn on 127.0.0.1:8765)
  tldr-daemon service install      autostart unit (launchd / systemd / schtasks)
  tldr-daemon service uninstall    stop + remove the unit
  tldr-daemon service status       unit installed? daemon healthy?

The Docker path doesn't use this module — the container runs uvicorn
directly via docker-entrypoint.sh.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src import selfupdate, service
from src.config import DAEMON_VERSION


def _serve(host: str, port: int) -> int:
    import threading

    import uvicorn

    # Native counterpart of the docker-entrypoint upgrade; no-op in Docker
    # (entrypoint sets TLDR_SKIP_PKG_UPDATE=1) and under pytest.
    selfupdate.refresh_youtube_libs()
    # Warm the ffmpeg + deno caches in the background so the first media job
    # doesn't wait on their downloads (~80 MB ffmpeg, ~40 MB deno). Daemon
    # threads: never block startup, die with the process. No-op when system
    # binaries are already present.
    from src.workers.ffmpeg import prefetch_ffmpeg
    from src.workers.jsruntime import prefetch_deno

    threading.Thread(target=prefetch_ffmpeg, name="ffmpeg-prefetch", daemon=True).start()
    threading.Thread(target=prefetch_deno, name="deno-prefetch", daemon=True).start()
    uvicorn.run("src.main:app", host=host, port=port)
    return 0


def _service(action: str) -> int:
    if action == "install":
        unit = service.install_service()
        print(f"Service installed: {unit if unit else 'schtasks logon task (experimental)'}")
        return 0
    if action == "uninstall":
        service.uninstall_service()
        print("Service stopped and removed.")
        return 0
    status = service.service_status()
    print(f"unit installed: {'yes' if status['installed'] else 'no'}")
    print(f"daemon healthy ({service.HEALTH_URL}): {'yes' if status['healthy'] else 'no'}")
    return 0 if status["installed"] and status["healthy"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tldr-daemon",
        description="TLDR daemon — run without arguments to start the server.",
    )
    parser.add_argument("--version", action="version", version=f"tldr-daemon {DAEMON_VERSION}")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="bind port (default: 8765)")
    sub = parser.add_subparsers(dest="command")
    svc = sub.add_parser("service", help="manage the user-level autostart service")
    svc.add_argument("action", choices=["install", "uninstall", "status"])

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")

    if args.command == "service":
        return _service(args.action)
    return _serve(args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
