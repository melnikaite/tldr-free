#!/usr/bin/env python3
"""Apply TLDR's mlx-openai-server patches.

Run after every install/upgrade of ``mlx-openai-server``. ``task install:mlx``
and ``bash scripts/mlx.sh patch`` both call this script.

Why patch and not fork the package: keeping the venv on the canonical
PyPI release means apt/brew/pip updates land cleanly; we just re-apply
this file's small diffs each time. If upstream accepts the equivalent
PR, this script becomes a no-op (anchors won't match the new code,
script prints "already patched" and exits 0).

What we change
==============

``app/schemas/openai.py``
    - Add ``VERBOSE_JSON = "verbose_json"`` to ``TranscriptionResponseFormat``.
    - Add a ``TranscriptionSegment`` model.
    - Add optional ``language`` / ``segments`` / ``duration`` to
      ``TranscriptionResponse`` and ``TranscriptionResponseStreamChoice``.

``app/handler/mlx_whisper.py``
    - Import ``TranscriptionSegment``.
    - Add a ``_coerce_segments`` helper.
    - Populate ``segments`` + ``language`` in ``generate_transcription_response``
      when the caller asked for ``verbose_json``.

Anchor-string approach (not unified diff) — works as long as the anchor
text exists somewhere in the file. If upstream restructures around the
anchor, the script prints a clear "anchor not found in <file>" and
refuses to half-apply.

Idempotent: if the marker string we INSERT is already present, the file
is left untouched and we count that as success.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
from pathlib import Path
import sys

TESTED_VERSION = "1.8.1"

# ---------------------------------------------------------------------------
# Patch fragments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Patch:
    """One anchor-based string replacement on a single file.

    ``anchor`` is a unique substring of the file we expect to find. We
    replace it with ``replacement``. ``done_marker`` is a string that
    we know will be present after the patch is applied — used to
    short-circuit when re-running (idempotent).
    """

    relpath: str
    anchor: str
    replacement: str
    done_marker: str
    label: str


_SCHEMAS_ENUM = Patch(
    relpath="app/schemas/openai.py",
    label="schemas: add VERBOSE_JSON to TranscriptionResponseFormat",
    done_marker='    VERBOSE_JSON = "verbose_json"',
    anchor=(
        '    JSON = "json"\n'
        '    TEXT = "text"\n'
    ),
    replacement=(
        '    JSON = "json"\n'
        '    TEXT = "text"\n'
        '    VERBOSE_JSON = "verbose_json"\n'
    ),
)


_SCHEMAS_MODELS = Patch(
    relpath="app/schemas/openai.py",
    label="schemas: TranscriptionSegment + extend TranscriptionResponse + Stream",
    done_marker="class TranscriptionSegment(OpenAIBaseModel):",
    anchor=(
        'class TranscriptionResponse(OpenAIBaseModel):\n'
        '    """Represents a transcription response."""\n'
        '\n'
        '    text: str = Field(..., description="The transcribed text.")\n'
        '    usage: TranscriptionUsageAudio = Field(..., description="The usage of the transcription.")\n'
        '\n'
        '\n'
        'class TranscriptionResponseStreamChoice(OpenAIBaseModel):\n'
        '    """Represents a choice in a streaming transcription response."""\n'
        '\n'
        '    delta: Delta = Field(..., description="The delta for this streaming choice.")\n'
        '    finish_reason: str | None = None\n'
        '    stop_reason: int | str | None = None\n'
    ),
    replacement=(
        'class TranscriptionSegment(OpenAIBaseModel):\n'
        '    """A single segment from Whisper output (verbose_json only)."""\n'
        '\n'
        '    id: int = Field(..., description="Sequential segment id from the model.")\n'
        '    start: float = Field(..., description="Segment start time in seconds (absolute).")\n'
        '    end: float = Field(..., description="Segment end time in seconds (absolute).")\n'
        '    text: str = Field(..., description="Transcribed text for this segment.")\n'
        '\n'
        '\n'
        'class TranscriptionResponse(OpenAIBaseModel):\n'
        '    """Represents a transcription response."""\n'
        '\n'
        '    text: str = Field(..., description="The transcribed text.")\n'
        '    usage: TranscriptionUsageAudio = Field(..., description="The usage of the transcription.")\n'
        '    # verbose_json fields. Populated only when the request asked for\n'
        '    # verbose output. Remain None on plain ``json`` for back-compat\n'
        '    # with existing clients that don\'t expect them.\n'
        '    language: str | None = Field(\n'
        '        None,\n'
        '        description="ISO 639-1 language code detected by the model (verbose_json only).",\n'
        '    )\n'
        '    segments: list[TranscriptionSegment] | None = Field(\n'
        '        None,\n'
        '        description="Per-segment timing + text from the model (verbose_json only).",\n'
        '    )\n'
        '    duration: float | None = Field(\n'
        '        None,\n'
        '        description="Audio duration in seconds (verbose_json only).",\n'
        '    )\n'
        '\n'
        '\n'
        'class TranscriptionResponseStreamChoice(OpenAIBaseModel):\n'
        '    """Represents a choice in a streaming transcription response."""\n'
        '\n'
        '    delta: Delta = Field(..., description="The delta for this streaming choice.")\n'
        '    finish_reason: str | None = None\n'
        '    stop_reason: int | str | None = None\n'
        '    # Verbose-mode extras per streamed chunk. Segments are absolute\n'
        '    # (chunk offset already applied), language is constant across\n'
        '    # the stream but echoed in every frame so clients can pick it\n'
        '    # up from the first one.\n'
        '    segments: list[TranscriptionSegment] | None = None\n'
        '    language: str | None = None\n'
    ),
)


_HANDLER_IMPORT = Patch(
    relpath="app/handler/mlx_whisper.py",
    label="handler: import TranscriptionSegment",
    done_marker="TranscriptionSegment,",
    anchor=(
        "from ..schemas.openai import (\n"
        "    Delta,\n"
        "    TranscriptionRequest,\n"
        "    TranscriptionResponse,\n"
        "    TranscriptionResponseFormat,\n"
        "    TranscriptionResponseStream,\n"
        "    TranscriptionResponseStreamChoice,\n"
        "    TranscriptionUsageAudio,\n"
        ")\n"
    ),
    replacement=(
        "from ..schemas.openai import (\n"
        "    Delta,\n"
        "    TranscriptionRequest,\n"
        "    TranscriptionResponse,\n"
        "    TranscriptionResponseFormat,\n"
        "    TranscriptionResponseStream,\n"
        "    TranscriptionResponseStreamChoice,\n"
        "    TranscriptionSegment,\n"
        "    TranscriptionUsageAudio,\n"
        ")\n"
        "\n"
        "\n"
        "def _coerce_segments(\n"
        "    raw_segments, time_offset: float = 0.0\n"
        "):\n"
        '    """Best-effort conversion of mlx_whisper\'s segment dicts.\n'
        "\n"
        "    ``time_offset`` is added to start/end — used in the streaming path\n"
        "    where segments are produced relative to a sub-chunk of the audio.\n"
        "    Returns ``None`` when input isn't a list, so the response field\n"
        "    stays ``None`` rather than empty-list (less noise for clients).\n"
        '    """\n'
        "    if not isinstance(raw_segments, list):\n"
        "        return None\n"
        "    out = []\n"
        "    for i, seg in enumerate(raw_segments):\n"
        "        if not isinstance(seg, dict):\n"
        "            continue\n"
        "        try:\n"
        "            out.append(\n"
        "                TranscriptionSegment(\n"
        '                    id=int(seg.get("id", i)),\n'
        '                    start=float(seg.get("start", 0.0)) + time_offset,\n'
        '                    end=float(seg.get("end", 0.0)) + time_offset,\n'
        '                    text=str(seg.get("text", "")),\n'
        "                )\n"
        "            )\n"
        "        except (TypeError, ValueError):\n"
        "            # Malformed segment — skip rather than fail the response.\n"
        "            continue\n"
        "    return out or None\n"
    ),
)


_HANDLER_RESPONSE = Patch(
    relpath="app/handler/mlx_whisper.py",
    label="handler: populate language+segments in non-stream response",
    done_marker="is_verbose = (\n                request.response_format",
    anchor=(
        "            response_data = TranscriptionResponse(\n"
        '                text=response["text"],\n'
        "                usage=TranscriptionUsageAudio(\n"
        '                    type="duration", seconds=int(calculate_audio_duration(temp_file_path))\n'
        "                ),\n"
        "            )\n"
        "            if request.response_format == TranscriptionResponseFormat.JSON:\n"
        "                return response_data\n"
        "            # dump to string for text response\n"
        "            return json.dumps(response_data.model_dump())\n"
    ),
    replacement=(
        "            duration_seconds = int(calculate_audio_duration(temp_file_path))\n"
        "            # mlx_whisper.transcribe() always returns {text, segments,\n"
        "            # language}. We only surface segments/language when the\n"
        "            # caller asked for verbose_json, so existing plain-JSON\n"
        "            # clients don't suddenly see new fields.\n"
        "            is_verbose = (\n"
        "                request.response_format == TranscriptionResponseFormat.VERBOSE_JSON\n"
        "            )\n"
        "            response_data = TranscriptionResponse(\n"
        '                text=response["text"],\n'
        "                usage=TranscriptionUsageAudio(\n"
        '                    type="duration", seconds=duration_seconds\n'
        "                ),\n"
        '                language=response.get("language") if is_verbose else None,\n'
        '                segments=_coerce_segments(response.get("segments"))\n'
        "                if is_verbose\n"
        "                else None,\n"
        "                duration=float(duration_seconds) if is_verbose else None,\n"
        "            )\n"
        "            if request.response_format in (\n"
        "                TranscriptionResponseFormat.JSON,\n"
        "                TranscriptionResponseFormat.VERBOSE_JSON,\n"
        "            ):\n"
        "                return response_data\n"
        "            # dump to string for text response\n"
        "            return json.dumps(response_data.model_dump())\n"
    ),
)


PATCHES: list[Patch] = [
    _SCHEMAS_ENUM,
    _SCHEMAS_MODELS,
    _HANDLER_IMPORT,
    _HANDLER_RESPONSE,
]


# ---------------------------------------------------------------------------
# Apply machinery
# ---------------------------------------------------------------------------


class PatchError(RuntimeError):
    pass


def find_site_packages(venv: Path) -> Path:
    """Resolve ``<venv>/lib/python3.*/site-packages``. Errors on multiple matches."""
    lib = venv / "lib"
    if not lib.is_dir():
        raise PatchError(f"no lib/ inside venv {venv}")
    candidates = sorted(lib.glob("python3.*/site-packages"))
    if not candidates:
        raise PatchError(f"no python3.*/site-packages under {lib}")
    if len(candidates) > 1:
        raise PatchError(
            f"multiple site-packages under {lib}: " + ", ".join(str(c) for c in candidates)
        )
    return candidates[0]


def check_installed_version(site_packages: Path) -> str:
    """Read the installed mlx-openai-server version from its dist-info.

    We use importlib.metadata against the venv's site-packages so we don't
    need to import the package (which would pull mlx into the wrong python).
    """
    # importlib.metadata in stdlib resolves against sys.path; we add the
    # target site-packages temporarily to query the right install.
    sys.path.insert(0, str(site_packages))
    try:
        return importlib.metadata.version("mlx-openai-server")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PatchError(
            f"mlx-openai-server not installed in {site_packages}"
        ) from exc
    finally:
        sys.path.remove(str(site_packages))


def apply_patch(site_packages: Path, patch: Patch) -> str:
    """Apply ``patch`` to ``site_packages/<relpath>``. Returns one of:
    ``"applied"``, ``"already"``, raises ``PatchError`` if anchor not found.
    """
    path = site_packages / patch.relpath
    if not path.is_file():
        raise PatchError(f"target file missing: {path}")
    src = path.read_text(encoding="utf-8")
    if patch.done_marker in src:
        return "already"
    if patch.anchor not in src:
        raise PatchError(
            f"anchor not found in {patch.relpath} for: {patch.label}\n"
            "Upstream likely changed around this region — patches need updating."
        )
    patched = src.replace(patch.anchor, patch.replacement, 1)
    path.write_text(patched, encoding="utf-8")
    return "applied"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--venv",
        default="~/.venvs/mlx-server",
        help="mlx-openai-server virtualenv path (default ~/.venvs/mlx-server)",
    )
    p.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Skip the installed-version check (use at your own risk).",
    )
    args = p.parse_args()

    venv = Path(args.venv).expanduser().resolve()
    if not venv.is_dir():
        print(f"✗ venv not found: {venv}", file=sys.stderr)
        return 2

    try:
        site = find_site_packages(venv)
        version = check_installed_version(site)
    except PatchError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2

    if version != TESTED_VERSION and not args.allow_version_mismatch:
        print(
            f"✗ mlx-openai-server {version} is installed; "
            f"this patch was tested against {TESTED_VERSION}.\n"
            "  Re-test the patches and update TESTED_VERSION, or rerun with "
            "--allow-version-mismatch.",
            file=sys.stderr,
        )
        return 2

    print(f"==> mlx-openai-server {version} at {site}")

    applied = 0
    already = 0
    try:
        for patch in PATCHES:
            result = apply_patch(site, patch)
            symbol = "✓" if result == "applied" else "↷"
            print(f"  {symbol} {patch.label} [{result}]")
            if result == "applied":
                applied += 1
            else:
                already += 1
    except PatchError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        return 1

    print(f"\nDone: {applied} applied, {already} already in place.")
    print("Restart mlx-server: bash scripts/mlx.sh stop && bash scripts/mlx.sh start")
    return 0


if __name__ == "__main__":
    sys.exit(main())
