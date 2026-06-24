"""Tests for the PDF worker.

Covers both paths through ``process_pdf``:

  - text-first via pypdf for native text PDFs (the common case)
  - vision OCR fallback for ~empty PDFs (scanned / image-only)

The vision path uses a mocked LLM — we just verify that ``process_pdf``
dispatches into it when pypdf's output is below threshold and that the
returned text comes back as ``PDF_VISION`` source.
"""

from __future__ import annotations

import io

import fitz  # type: ignore[import-untyped]
import pytest

from src.api.schemas import TranscriptSource
from src.workers import pdf as pdf_worker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_pdf(*pages: str) -> bytes:
    """Generate a minimal in-memory PDF with the given pages of text."""
    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((50, 72), body, fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_image_only_pdf() -> bytes:
    """PDF that has page geometry but no text — pypdf returns "" for these.

    A 1×1 transparent image is enough to give the page some content without
    embedding any text objects.
    """
    doc = fitz.open()
    page = doc.new_page()
    # Insert a tiny opaque rectangle so the page isn't strictly empty —
    # pure-blank PDFs occasionally trip pdf parsers in weird ways.
    page.draw_rect((10, 10, 11, 11), color=(0, 0, 0), fill=(0, 0, 0))
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# pypdf path
# ---------------------------------------------------------------------------


def test_pypdf_extract_returns_text_for_native_pdf() -> None:
    pdf = _make_text_pdf(
        "The quick brown fox jumps over the lazy dog.\n"
        "Several lines of body text that exceeds the threshold.\n"
        * 10,
    )
    out = pdf_worker._pypdf_extract(pdf)
    assert "quick brown fox" in out
    assert len(out) > 200


def test_pypdf_extract_empty_on_image_only_pdf() -> None:
    pdf = _make_image_only_pdf()
    out = pdf_worker._pypdf_extract(pdf)
    # Image-only PDFs may yield "" or a tiny amount of whitespace; either
    # way it's below the threshold that triggers the vision fallback.
    assert len(out.strip()) < 200


def test_pypdf_extract_handles_garbage_bytes() -> None:
    """A non-PDF blob shouldn't crash; ``_pypdf_extract`` returns ``""``."""
    out = pdf_worker._pypdf_extract(b"this is not a pdf")
    assert out == ""


# ---------------------------------------------------------------------------
# Full process_pdf — text-first path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_pdf_text_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal text PDF should never touch the LLM."""
    pdf = _make_text_pdf(
        "The first page of body text. " * 30,
        "The second page of body text. " * 30,
    )

    called = {"vision": 0}

    async def _explode(*a: object, **kw: object) -> str:
        called["vision"] += 1
        raise AssertionError("vision OCR called but text path should have won")

    monkeypatch.setattr(pdf_worker, "_vision_ocr", _explode)

    text, source = await pdf_worker.process_pdf(
        job_id="t", url="http://x/a.pdf", pdf_bytes=pdf, cookies=[],
    )
    assert source == TranscriptSource.PDF_TEXT
    assert "first page of body text" in text
    assert "second page of body text" in text
    assert called["vision"] == 0


# ---------------------------------------------------------------------------
# Full process_pdf — vision fallback path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_pdf_falls_back_to_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    """When pypdf returns ~nothing, OCR via the multimodal LLM kicks in."""
    pdf = _make_image_only_pdf()

    async def _fake_vision(*, job_id: str, data: bytes) -> str:
        return "OCR'd content from the scanned PDF."

    monkeypatch.setattr(pdf_worker, "_vision_ocr", _fake_vision)

    text, source = await pdf_worker.process_pdf(
        job_id="t", url="http://x/a.pdf", pdf_bytes=pdf, cookies=[],
    )
    assert source == TranscriptSource.PDF_VISION
    assert text == "OCR'd content from the scanned PDF."
