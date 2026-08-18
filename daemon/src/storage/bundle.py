"""Export processed jobs to a portable zip bundle, and import one back.

Motivation: moving a library between machines, and letting a machine that
can't run local models still read summaries/transcripts imported from
elsewhere (see ``.claude/architecture.md`` — this module doesn't change
that story, it just gives the finished output a portable shape).

Pure storage-layer packing/unpacking — no FastAPI imports, so this module
is usable from a script or a test without pulling in a web framework. The
route layer (``api/jobs.py``) owns the HTTP-specific parts: turning
``BundleError`` into a 400, streaming the request body in, and returning a
``FileResponse`` over the zip this module writes to a tempfile.

Bundle layout
-------------
::

    manifest.json
    jobs/<original_job_id>/job.json
    jobs/<original_job_id>/frames/<file>                    (no moment, flat)
    jobs/<original_job_id>/frames/<segment>/<file>           (nested, one level)

``manifest.json`` — ``{"format": "tldr-export", "version": 1,
"exported_at": <iso8601>, "daemon_api_version": <int>, "jobs": [<id>, ...]}``.

``job.json`` — the whole job, machine-independent fields only. Deliberately
NOT included: ``id``, ``status``, ``error``, ``progress_stage``,
``audio_path``, ``audio_duration_seconds`` (host-local), ``added_at``
(machine-local — "when this row appeared on THIS machine"; the importing
machine sets its own, see ``repo.insert_imported_job``), and translations
whose status isn't ``done`` (they carry no text worth shipping).

Frame nesting
-------------
Frames for a job live on disk as ``<data_dir>/frames/<job_id>/t<seconds>/
frame_NN.jpg`` (see ``workers.frames``) — exactly one level of nesting.
The exporter writes that exact relative path into the zip
(``jobs/<job_id>/frames/t<seconds>/frame_NN.jpg``), and the import-side
member-name guard (``_MEMBER_FRAME_NESTED_RE``) accepts precisely one
nested segment — no deeper, no ``..``, no absolute paths — as well as a
flat ``jobs/<job_id>/frames/<file>`` shape (``_MEMBER_FRAME_FLAT_RE``) for
a bundle that never had a nested moment. Importing recreates whichever
shape the zip member had under the freshly-minted job id, which is what
makes the rewritten ``frame_url`` values (see ``_rewrite_frame_url``)
resolve to a real file again on the importing machine.

Import safety
-------------
``import_bundle`` validates the WHOLE zip (format/version, every member
name, every member's declared and streamed size) before writing anything
to the database or disk. Per-job work then happens one job at a time, each
as its own DB transaction (``repo.insert_imported_job``) — a job that
fails partway through is recorded under ``failed`` and does not block the
rest of the bundle, and never leaves a half-written row behind.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from src.api.schemas import ImportedJob, ImportIssue, JobImportResponse, JobKind
from src.config import DAEMON_API_VERSION
from src.storage import repo
from src.storage.db import Job
from src.workers import frames

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

BUNDLE_FORMAT = "tldr-export"
BUNDLE_VERSION = 1

# Zip-bomb / abuse guards (see module docstring "Import safety").
MAX_UPLOAD_BYTES = 512 * 1024 * 1024          # 512 MiB
MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024   # 2 GiB
MAX_MEMBER_UNCOMPRESSED_BYTES = 64 * 1024 * 1024        # 64 MiB per member

_COPY_CHUNK_BYTES = 1024 * 1024

# Member-name shapes accepted on import. Anything else (including deeper
# nesting, an absolute path, or a ``..`` component) is rejected wholesale
# before any file is written — see ``_validate_members``.
_JOB_ID_PART = r"[A-Za-z0-9_-]{1,64}"
_FRAME_SEGMENT_PART = r"[A-Za-z0-9._-]{1,64}"
_FRAME_BASENAME_PART = r"[A-Za-z0-9._-]+\.(?:jpg|jpeg|png)"
_MEMBER_JOB_JSON_RE = re.compile(rf"^jobs/(?P<id>{_JOB_ID_PART})/job\.json$")
# Flat: jobs/<id>/frames/<file> — a bundle with no nested moment directory.
_MEMBER_FRAME_FLAT_RE = re.compile(
    rf"^jobs/(?P<id>{_JOB_ID_PART})/frames/(?P<file>{_FRAME_BASENAME_PART})$",
    re.IGNORECASE,
)
# Nested, exactly one level: jobs/<id>/frames/<segment>/<file> — matches the
# real on-disk t<seconds>/frame_NN.jpg layout (see module docstring).
_MEMBER_FRAME_NESTED_RE = re.compile(
    rf"^jobs/(?P<id>{_JOB_ID_PART})/frames/(?P<segment>{_FRAME_SEGMENT_PART})"
    rf"/(?P<file>{_FRAME_BASENAME_PART})$",
    re.IGNORECASE,
)


class BundleError(ValueError):
    """Raised for anything wrong with a bundle — either building one (no
    exportable job in the request) or reading one (bad zip, bad manifest,
    unsafe member name, oversized). The route layer maps this to HTTP 400."""


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_jobs(job_ids: list[str]) -> tuple[Path, int]:
    """Pack the ``status == "done"`` subset of ``job_ids`` into a zip on
    disk. Returns ``(path, count)`` — the caller (``api/jobs.py``) streams
    ``path`` back as a ``FileResponse`` and is responsible for unlinking it
    afterwards (e.g. via ``BackgroundTask``).

    Unknown ids and non-done jobs are silently dropped — the client is
    expected to have filtered its own list already; this is a defensive
    re-check, not a validation error. Raises ``BundleError`` only if
    NOTHING in ``job_ids`` turned out exportable.
    """
    jobs: list[Job] = []
    for job_id in job_ids:
        job = repo.get_job(job_id)
        if job is None or job.status != "done":
            continue
        jobs.append(job)

    if not jobs:
        raise BundleError(
            "no exportable jobs: every id was unknown or not in status=done"
        )

    fd, tmp_name = tempfile.mkstemp(prefix="tldr-export-", suffix=".zip")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            manifest = {
                "format": BUNDLE_FORMAT,
                "version": BUNDLE_VERSION,
                "exported_at": datetime.utcnow().isoformat() + "Z",
                "daemon_api_version": DAEMON_API_VERSION,
                "jobs": [job.id for job in jobs],
            }
            zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for job in jobs:
                _write_job_entry(zf, job)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path, len(jobs)


def _write_job_entry(zf: zipfile.ZipFile, job: Job) -> None:
    messages = repo.list_messages(job.id)
    translations = repo.list_done_translations(job.id)

    payload: dict[str, Any] = {
        "url": job.url,
        "kind": job.kind,
        "title": job.title,
        "duration_seconds": job.duration_seconds,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "raw_text": job.raw_text,
        "summary_md": job.summary_md,
        "transcript_source": job.transcript_source,
        "video_id": job.video_id,
        "transcript_language": job.transcript_language,
        "raw_segments_json": job.raw_segments_json,
        "alt_media_candidates_json": job.alt_media_candidates_json,
        "transcript_missing_seconds": job.transcript_missing_seconds,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
                "frame_refs": _parse_frame_refs_json(m.frame_refs_json),
            }
            for m in messages
        ],
        "translations": [
            {
                "language_code": t["language_code"],
                "text": t["text"],
                "status": t["status"],
                "error": t["error"],
                "created_at": t["created_at"].isoformat(),
                "updated_at": t["updated_at"].isoformat(),
            }
            for t in translations
        ],
    }
    zf.writestr(
        f"jobs/{job.id}/job.json",
        json.dumps(payload, ensure_ascii=False),
    )

    frame_dir = frames.job_frames_dir_if_exists(job.id)
    if frame_dir is None:
        return
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for frame_path in sorted(frame_dir.rglob(ext)):
            rel = frame_path.relative_to(frame_dir).as_posix()
            zf.write(frame_path, f"jobs/{job.id}/frames/{rel}")


def _parse_frame_refs_json(raw: str | None) -> list[Any]:
    """Parse ``Message.frame_refs_json`` back into a list of dicts for the
    export payload. A row whose JSON doesn't parse exports as ``[]`` rather
    than failing the whole export — same defensive style as
    ``api/jobs.py._parse_frame_refs``."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def import_bundle(zip_path: Path) -> JobImportResponse:
    """Validate and import a zip bundle produced by ``export_jobs`` (or
    anything claiming to be one), reading it from ``zip_path`` rather than
    a bytes blob — the route layer streams the upload straight to a
    tempfile so an oversized upload never has to be fully buffered in
    memory before the size guard can fire (see ``api/jobs.py``'s
    ``import_jobs``). Raises ``BundleError`` (-> HTTP 400) for anything
    wrong with the bundle as a whole; per-job problems (duplicate URL,
    malformed job entry) are reported in the returned response instead —
    see ``JobImportResponse``.

    The size check here is a defensive second line, not the primary one:
    it covers direct callers of this function (tests, a future script)
    that hand it a path without going through the streaming route.
    """
    size = zip_path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise BundleError(f"bundle file is {size} bytes, over the {MAX_UPLOAD_BYTES}-byte limit")

    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise BundleError(f"not a valid zip file: {exc}") from exc

    with zf:
        manifest = _read_manifest(zf)
        members = _validate_members(zf)

        manifest_job_ids = manifest.get("jobs")
        if not isinstance(manifest_job_ids, list):
            raise BundleError("manifest.json 'jobs' must be a list")

        imported: list[ImportedJob] = []
        skipped: list[ImportIssue] = []
        failed: list[ImportIssue] = []

        for job_id in manifest_job_ids:
            if not isinstance(job_id, str):
                continue
            entry = members.get(job_id)
            if entry is None or entry["job_json"] is None:
                failed.append(
                    ImportIssue(
                        url="", title=None,
                        reason=f"bundle is missing jobs/{job_id}/job.json",
                    )
                )
                continue
            _import_one_job(zf, entry, imported=imported, skipped=skipped, failed=failed)

    return JobImportResponse(imported=imported, skipped=skipped, failed=failed)


def _read_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    if "manifest.json" not in zf.namelist():
        raise BundleError("bundle is missing manifest.json")
    raw = _read_zip_member(zf, zf.getinfo("manifest.json"))
    try:
        manifest = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise BundleError(f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BundleError("manifest.json must be a JSON object")
    if manifest.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"unsupported bundle format: {manifest.get('format')!r}")
    version = manifest.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version > BUNDLE_VERSION:
        raise BundleError(f"unsupported bundle version: {version!r}")
    return manifest


def _validate_members(zf: zipfile.ZipFile) -> dict[str, dict[str, Any]]:
    """Validate every member name + size BEFORE anything is written to
    disk or the database. Returns ``{job_id: {"job_json": ZipInfo | None,
    "frames": [(rel_path, ZipInfo), ...]}}`` where ``rel_path`` is either
    ``"<file>"`` (flat) or ``"<segment>/<file>"`` (one level of nesting —
    see module docstring "Frame nesting").
    """
    members: dict[str, dict[str, Any]] = {}
    total_uncompressed = 0

    for info in zf.infolist():
        name = info.filename
        if name == "manifest.json" or info.is_dir():
            continue

        # Defence in depth ahead of the regex matches below: an absolute
        # path or a ``..`` component is rejected outright regardless of
        # whether it would also fail every regex (it always would, since
        # none of their character classes allow a ``..`` segment or a
        # leading ``/``), but this check names the exact reason in the
        # error message.
        if name.startswith("/") or ".." in Path(name).parts:
            raise BundleError(f"unsafe member name in bundle: {name!r}")

        m_job = _MEMBER_JOB_JSON_RE.match(name)
        m_flat = _MEMBER_FRAME_FLAT_RE.match(name)
        m_nested = _MEMBER_FRAME_NESTED_RE.match(name)
        if m_job is None and m_flat is None and m_nested is None:
            raise BundleError(f"unexpected member name in bundle: {name!r}")

        if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise BundleError(
                f"member {name!r} is {info.file_size} bytes, over the "
                f"{MAX_MEMBER_UNCOMPRESSED_BYTES}-byte per-member limit"
            )
        total_uncompressed += info.file_size
        if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise BundleError(
                f"bundle's total uncompressed size exceeds the "
                f"{MAX_TOTAL_UNCOMPRESSED_BYTES}-byte limit"
            )

        if m_job is not None:
            entry = members.setdefault(m_job.group("id"), {"job_json": None, "frames": []})
            entry["job_json"] = info
        elif m_flat is not None:
            entry = members.setdefault(m_flat.group("id"), {"job_json": None, "frames": []})
            entry["frames"].append((m_flat.group("file"), info))
        else:
            assert m_nested is not None
            entry = members.setdefault(m_nested.group("id"), {"job_json": None, "frames": []})
            rel = f"{m_nested.group('segment')}/{m_nested.group('file')}"
            entry["frames"].append((rel, info))

    return members


def _read_zip_member(
    zf: zipfile.ZipFile, info: zipfile.ZipInfo, *, max_bytes: int = MAX_MEMBER_UNCOMPRESSED_BYTES
) -> bytes:
    """Stream-read a member with a hard cap enforced against the ACTUAL
    bytes produced, not just the (attacker-controllable) declared
    ``file_size`` — the zip-bomb guard ``_validate_members`` runs is a
    fast pre-check on declared sizes; this is the real enforcement."""
    chunks: list[bytes] = []
    total = 0
    with zf.open(info) as src:
        while True:
            chunk = src.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise BundleError(
                    f"member {info.filename!r} exceeded the {max_bytes}-byte "
                    "guard while streaming"
                )
            chunks.append(chunk)
    return b"".join(chunks)


def _rewrite_frame_url(frame_url: Any, new_job_id: str) -> str | None:
    """``/jobs/<old_id>/frames/t12/frame_01.jpg`` -> ``/jobs/<new_id>/
    frames/t12/frame_01.jpg``. Returns ``None`` (caller drops the ref) if
    ``frame_url`` isn't a string shaped like that at all."""
    if not isinstance(frame_url, str):
        return None
    marker = "/frames/"
    idx = frame_url.find(marker)
    if idx == -1:
        return None
    rel = frame_url[idx + len(marker):]
    if not rel:
        return None
    return f"/jobs/{new_job_id}/frames/{rel}"


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        # Tolerate a trailing "Z" (our own exported_at/completed_at use
        # plain UTC isoformat without one, but be lenient either way).
        return datetime.fromisoformat(value[:-1] if value.endswith("Z") else value)
    except ValueError:
        return None


def _import_one_job(
    zf: zipfile.ZipFile,
    entry: dict[str, Any],
    *,
    imported: list[ImportedJob],
    skipped: list[ImportIssue],
    failed: list[ImportIssue],
) -> None:
    job_json_info: zipfile.ZipInfo = entry["job_json"]
    url = ""
    title: str | None = None
    try:
        raw = _read_zip_member(zf, job_json_info)
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise BundleError(f"{job_json_info.filename} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise BundleError(f"{job_json_info.filename} must be a JSON object")

        url = payload.get("url") or ""
        title = payload.get("title")
        if not isinstance(url, str) or not url:
            raise BundleError(f"{job_json_info.filename} is missing 'url'")

        # Filter on status="done" directly rather than fetching the newest
        # row for this URL and checking ITS status — a machine can have
        # both a failed and a done job for the same URL (a retry that left
        # the failed row behind), and the newest row is not necessarily
        # the done one. Asking "does a done row exist" is the question
        # that actually matches the duplicate rule.
        existing_done, _ = repo.list_jobs(url=url, status="done", limit=1)
        if existing_done:
            skipped.append(ImportIssue(url=url, title=title, reason="duplicate"))
            return

        # Validate against the actual JobKind values rather than trusting
        # the bundle — this string is written straight into a column the
        # UI switches on (page/youtube/media/pdf), so a bad or missing
        # value fails this job entry instead of silently defaulting.
        kind_raw = payload.get("kind")
        if not isinstance(kind_raw, str):
            raise BundleError(f"{job_json_info.filename} has invalid 'kind': {kind_raw!r}")
        try:
            kind = JobKind(kind_raw).value
        except ValueError as exc:
            raise BundleError(
                f"{job_json_info.filename} has invalid 'kind': {kind_raw!r}"
            ) from exc

        new_id = repo.generate_job_id()
        frame_root: Path | None = None
        try:
            frame_root = _copy_frames(zf, entry["frames"], new_id)
            messages = _build_messages(payload.get("messages"), new_id)
            translations = _build_translations(payload.get("translations"))

            created_at = _parse_iso(payload.get("created_at")) or datetime.utcnow()
            completed_at = _parse_iso(payload.get("completed_at"))

            repo.insert_imported_job(
                job_id=new_id,
                url=url,
                kind=kind,
                title=title,
                duration_seconds=payload.get("duration_seconds"),
                created_at=created_at,
                completed_at=completed_at,
                raw_text=payload.get("raw_text"),
                summary_md=payload.get("summary_md"),
                transcript_source=payload.get("transcript_source"),
                video_id=payload.get("video_id"),
                transcript_language=payload.get("transcript_language"),
                raw_segments_json=payload.get("raw_segments_json"),
                alt_media_candidates_json=payload.get("alt_media_candidates_json"),
                transcript_missing_seconds=payload.get("transcript_missing_seconds"),
                messages=messages,
                translations=translations,
            )
        except Exception:
            # The DB insert (or something before it) failed after frame
            # files may already have been written under new_id — clean
            # those up so a failed job doesn't leave an orphaned directory.
            if frame_root is not None:
                with contextlib.suppress(OSError):
                    shutil.rmtree(frame_root, ignore_errors=True)
            raise

        imported.append(ImportedJob(job_id=new_id, url=url, title=title))
    except Exception as exc:
        log.warning("bundle: import failed for %s: %s", job_json_info.filename, exc)
        failed.append(ImportIssue(url=url, title=title, reason=str(exc)))


def _copy_frames(
    zf: zipfile.ZipFile, frame_entries: list[tuple[str, zipfile.ZipInfo]], new_job_id: str
) -> Path | None:
    """Copy every frame member for a job under the freshly-minted
    ``new_job_id``, recreating whichever shape (flat or one-level nested)
    ``_validate_members`` recorded for it — ``rel`` is already the correct
    on-disk relative path (e.g. ``"t12/frame_01.jpg"``), no decoding
    needed. The containment check is a backstop, not the primary defence:
    ``_validate_members``'s member-name regexes already reject anything
    that could resolve outside ``frame_root`` before this function ever
    runs.
    """
    if not frame_entries:
        return None
    frame_root = frames.ensure_job_frames_dir(new_job_id)
    resolved_root = frame_root.resolve()
    for rel, info in frame_entries:
        dest = (frame_root / rel).resolve()
        if dest != resolved_root and resolved_root not in dest.parents:
            raise BundleError(f"frame member escapes its job directory: {info.filename!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with zf.open(info) as src, open(dest, "wb") as out:
            while True:
                chunk = src.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_MEMBER_UNCOMPRESSED_BYTES:
                    raise BundleError(
                        f"member {info.filename!r} exceeded the "
                        f"{MAX_MEMBER_UNCOMPRESSED_BYTES}-byte guard while streaming"
                    )
                out.write(chunk)
    return frame_root


def _build_messages(raw_messages: Any, new_job_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list):
        return []
    out: list[dict[str, Any]] = []
    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        rewritten_refs: list[Any] = []
        raw_refs = m.get("frame_refs")
        if isinstance(raw_refs, list):
            for ref in raw_refs:
                if not isinstance(ref, dict):
                    continue
                new_url = _rewrite_frame_url(ref.get("frame_url"), new_job_id)
                if new_url is None:
                    continue
                rewritten_refs.append({**ref, "frame_url": new_url})
        frame_refs_json = (
            json.dumps(rewritten_refs, ensure_ascii=False, separators=(",", ":"))
            if rewritten_refs
            else None
        )
        out.append(
            {
                "role": role,
                "content": content,
                "created_at": _parse_iso(m.get("created_at")),
                "frame_refs_json": frame_refs_json,
            }
        )
    return out


# The only two statuses a stored TranscriptTranslation row may safely
# carry text under (see db.py's docstring). This is a WHITELIST, not a
# best-effort filter: ``TranscriptTranslationSummary.status`` is a
# pydantic ``Literal``, and `GET /jobs/{id}` (``response_model=JobDetails``)
# rejects an out-of-set value with a 500 — a hostile or corrupt bundle
# writing an arbitrary string would 500 that endpoint for the imported
# job PERMANENTLY, with no way to fix it from the UI. Same class of issue
# as the zip member-name containment check in this module — untrusted
# bundle content must never reach storage unvalidated (see
# .claude/contract.md).
_VALID_IMPORT_TRANSLATION_STATUSES = ("done", "partial")


def _build_translations(raw_translations: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_translations, list):
        return []
    out: list[dict[str, Any]] = []
    for t in raw_translations:
        if not isinstance(t, dict):
            continue
        lang = t.get("language_code")
        text = t.get("text")
        if not isinstance(lang, str) or not lang or not isinstance(text, str):
            continue
        # Anything other than "done"/"partial" — absent (bundles written
        # before "partial" existed exported only "done" translations by
        # construction), malformed, or a LIVE status like "queued"/
        # "running" (which would make ``re_enqueue_running_on_startup``
        # start re-translating an imported job on the next daemon
        # restart) — silently becomes "done".
        raw_status = t.get("status")
        status = (
            raw_status
            if raw_status in _VALID_IMPORT_TRANSLATION_STATUSES
            else "done"
        )
        raw_error = t.get("error")
        # error only ever accompanies "partial" in a genuine row (see
        # db.py) — drop it otherwise rather than trust a bundle to have
        # kept the two in sync.
        error = raw_error if status == "partial" and isinstance(raw_error, str) else None
        out.append(
            {
                "language_code": lang,
                "text": text,
                "status": status,
                "error": error,
                "created_at": _parse_iso(t.get("created_at")),
                "updated_at": _parse_iso(t.get("updated_at")),
            }
        )
    return out


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_VERSION",
    "MAX_UPLOAD_BYTES",
    "BundleError",
    "export_jobs",
    "import_bundle",
]
