"""PDF text extraction with multimodal vision fallback.

Server-side parsing keeps the extension content-agnostic — it just hands
over a URL (http(s)) or raw bytes (file://) and the daemon does the rest.
The output is plain text which feeds straight into the same
``_summarize_and_finish`` step that pages and YouTube transcripts use, so
the downstream contract is unchanged.

Two-step strategy:

1. **Text-first via pypdf**. Fast (ms per page), accurate for native
   text PDFs. Most academic papers / reports / web-saved PDFs land here.

2. **Vision fallback via the multimodal LLM** when pypdf returns ~nothing
   (``_TEXT_THRESHOLD_CHARS`` total). pymupdf rasterizes each page to PNG;
   we send each PNG through ``llm.client.complete_with_messages`` with
   ``image_url`` content and the ``pdf_vision_page.txt`` prompt; the
   concatenated transcriptions become ``raw_text``.

The fallback is genuinely slow on a local backend (10–60 s per page on an
Apple-Silicon mlx Gemma 4 E4B) — that's the cost of OCR-via-LLM on consumer
hardware, and the text-first path is what saves us in the common case.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any

import fitz
import httpx
import pypdf

from src.api.schemas import Cookie, TranscriptSource
from src.llm import client as llm_client
from src.workers.broker import get_broker, stage_event

log = logging.getLogger(__name__)

# Below this many characters of pypdf output across the whole PDF we
# assume the document is scanned / image-only and fall back to vision.
# A typical text-PDF page is ~1500–3000 chars, so 200 is a sane "got
# nothing useful" floor even for short documents.
_TEXT_THRESHOLD_CHARS = 200

# Rendering DPI for vision OCR. 150 dpi is a good balance for Latin text:
# small enough to keep image tokens reasonable, large enough that the LLM
# reads body type comfortably. Bump to 200 for very small fonts.
_RENDER_DPI = 150

# Cap on pages we'll OCR via vision before bailing. A 1000-page scanned
# book would otherwise take days locally and the user is better served by
# external OCR (e.g. ocrmypdf) first.
_MAX_VISION_PAGES = 100


async def process_pdf(
    *,
    job_id: str,
    url: str,
    pdf_bytes: bytes | None,
    cookies: list[Cookie],
) -> tuple[str, TranscriptSource]:
    """Return ``(text, source)`` for a PDF job.

    ``pdf_bytes`` is non-None when the extension uploaded the file (the
    ``file://`` case); otherwise we fetch ``url`` ourselves with the
    supplied cookies. Raises on permanent failure (e.g. fetch error, can't
    open PDF) — caller turns that into ``mark_failed`` + error event.
    """
    broker = get_broker()

    if pdf_bytes is None:
        broker.publish(job_id, stage_event("extracting", detail="fetching PDF"))
        pdf_bytes = await _fetch(url, cookies)

    text = await asyncio.to_thread(_pypdf_extract, pdf_bytes)
    if len(text) >= _TEXT_THRESHOLD_CHARS:
        log.info("pdf %s: pypdf returned %d chars — text-first path", job_id, len(text))
        return text, TranscriptSource.PDF_TEXT

    log.info(
        "pdf %s: pypdf returned only %d chars — falling back to vision OCR",
        job_id, len(text),
    )
    vision_text = await _vision_ocr(job_id=job_id, data=pdf_bytes)
    return vision_text, TranscriptSource.PDF_VISION


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


async def _fetch(url: str, cookies: list[Cookie]) -> bytes:
    """HTTP GET the PDF, forwarding cookies if any.

    Daemon-side fetch covers http(s) URLs. ``file://`` URLs can't be fetched
    from inside the daemon container (no host fs mount + no file scheme on
    httpx) — the caller must have supplied ``pdf_bytes`` for those, and
    that's enforced upstream in ``api/jobs.create_job``.
    """
    cookies_dict = {c.name: c.value for c in cookies} if cookies else None
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        cookies=cookies_dict,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; TLDR/0.1)"},
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


# ---------------------------------------------------------------------------
# Text-first path
# ---------------------------------------------------------------------------


def _pypdf_extract(data: bytes) -> str:
    """Plain text from a PDF via pypdf. Returns ``""`` on parse failure.

    pypdf swallows decryption errors silently in some versions so we
    also try opening with no password and treat any exception as
    "no text", letting the vision path take over.
    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Try empty password — pypdf returns a PasswordType enum whose
            # truthy variants mean a successful decrypt. Falsy → real
            # encryption, give up and let the vision path try.
            try:
                ok = bool(reader.decrypt(""))
            except Exception:
                ok = False
            if not ok:
                log.warning("pypdf: PDF is encrypted — text path skipped")
                return ""
    except Exception as exc:
        log.warning("pypdf failed to open PDF: %s", exc)
        return ""

    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception as exc:
            log.warning("pypdf page %d extract failed: %s", i, exc)
            continue
        t = t.strip()
        if t:
            pages.append(t)
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Vision fallback path
# ---------------------------------------------------------------------------


async def _vision_ocr(*, job_id: str, data: bytes) -> str:
    """Stream PDF pages through the multimodal LLM, one at a time.

    Per-page calls (not one giant message) for three reasons:
      * Failure isolation — one bad page shouldn't kill the whole job.
      * Token budget — every page is ~1500-2000 image tokens; concatenating
        a 50-page PDF would blow past most context windows.
      * Per-page progress events to the side panel timeline.

    Memory shape: render-then-OCR happens **one page at a time**. The
    per-page PNG (~0.5–2 MB rendered + ~0.7–3 MB base64) goes out of
    scope before the next page is rendered, so peak overhead above the
    input bytes is bounded to ~one page. A naive "render all pages,
    then OCR all" would hold every page's PNG simultaneously — 100 MB+
    for a long scanned PDF.
    """
    broker = get_broker()

    page_count = await asyncio.to_thread(_pdf_page_count, data)
    if page_count <= 0:
        raise ValueError("PDF could not be opened for OCR (corrupt or empty)")
    if page_count > _MAX_VISION_PAGES:
        raise ValueError(
            f"PDF has {page_count} pages — over the {_MAX_VISION_PAGES}-page "
            "vision OCR cap. Run an OCR tool (ocrmypdf, Adobe) over it "
            "first so the daemon's text-first path can read it.",
        )
    log.info("pdf %s: vision OCR over %d page(s)", job_id, page_count)

    prompt_template = _load_vision_prompt()
    parts: list[str] = []
    for i in range(page_count):
        broker.publish(
            job_id,
            stage_event("extracting", detail=f"OCR page {i + 1}/{page_count}"),
        )
        # Render this single page in a thread (pymupdf is CPU-bound C code),
        # then OCR. The base64 string goes out of scope at the end of the
        # iteration — gc reclaims before the next render.
        b64 = await asyncio.to_thread(_render_one_page, data, i)
        try:
            text = await _ocr_one_page(prompt_template, i + 1, b64)
        except Exception as exc:
            log.warning("pdf %s: OCR failed on page %d: %s", job_id, i + 1, exc)
            text = ""
        del b64
        text = text.strip()
        if text:
            parts.append(f"[page {i + 1}]\n{text}")
    if not parts:
        raise ValueError(
            "vision OCR produced no text — model may not support image input, "
            "or all pages were blank/illegible",
        )
    return "\n\n".join(parts)


def _pdf_page_count(data: bytes) -> int:
    """Open a PDF from bytes just to read the page count. Fast — pymupdf
    only parses the xref table, doesn't rasterize anything."""
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return int(doc.page_count)
    except Exception as exc:
        log.warning("pymupdf could not open PDF for page count: %s", exc)
        return 0


def _render_one_page(data: bytes, index: int) -> str:
    """Render a single PDF page (0-indexed) to a base64-encoded PNG.

    Opens the doc per-call so pymupdf releases its internal copy of the
    PDF after rendering — keeps memory bounded across long iterations.
    The repeated open is cheap (~ms) because pymupdf only re-parses
    the xref/trailer, not the page content streams that haven't been
    accessed yet.

    PNG is preferred over JPEG: lossless preservation of small text,
    fewer artifacts that confuse OCR. Size overhead vs JPEG is real but
    the LLM is the bottleneck anyway.
    """
    zoom = _RENDER_DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(stream=data, filetype="pdf") as doc:
        page = doc[index]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return base64.b64encode(pix.tobytes("png")).decode("ascii")


async def _ocr_one_page(prompt_template: str, page_num: int, b64: str) -> str:
    """Send one page image to the LLM and return the transcription."""
    prompt = prompt_template.format(page=page_num)
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
        },
    ]
    resp = await llm_client.complete_with_messages(
        messages,
        max_tokens=3000,    # full-page transcription can be substantial
        temperature=0.0,    # deterministic OCR
        respect_pause=True,
    )
    msg = resp.choices[0].message
    return getattr(msg, "content", None) or ""


def _load_vision_prompt() -> str:
    p = Path(__file__).resolve().parent.parent / "prompts" / "pdf_vision_page.txt"
    return p.read_text(encoding="utf-8")


__all__ = ["process_pdf"]
