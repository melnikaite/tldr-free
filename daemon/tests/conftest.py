"""Shared pytest fixtures.

Ensures `src.config.get_config()` has a valid file to read in CI/test
environments. We point TLDR_CONFIG at the example YAML if no real config
is mounted at the default path.
"""

from __future__ import annotations

import os
from pathlib import Path

# Try the docker-mounted path first; if it's missing (e.g. local pytest run
# outside the container, or a stripped CI image), fall back to the repo's
# example file.
_DEFAULT_CONFIG = Path("/app/config/tldr.yaml")
_REPO_EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "tldr.yaml.example"

if not _DEFAULT_CONFIG.is_file():
    if _REPO_EXAMPLE.is_file():
        os.environ.setdefault("TLDR_CONFIG", str(_REPO_EXAMPLE))
    else:
        # As a last resort, ship a tiny inline config so tests can still
        # import src.config.get_config().
        import tempfile

        tmp = Path(tempfile.gettempdir()) / "tldr_test_config.yaml"
        tmp.write_text(
            """
llm:
  base_url: http://localhost:18000/v1
  api_key: dummy
  model: qwen
  context_length: 32768
  single_pass_token_limit: 24000
whisper:
  base_url: http://localhost:18000/v1
  api_key: dummy
  model: whisper
output:
  language: ru
youtube:
  fast_path_max_attempts: 4
  fast_path_backoff_seconds: [1, 4, 16, 60]
  segment_window_seconds: 30
storage:
  data_dir: /tmp
  db_filename: tldr.db
logging:
  level: INFO
""".strip(),
            encoding="utf-8",
        )
        os.environ.setdefault("TLDR_CONFIG", str(tmp))
