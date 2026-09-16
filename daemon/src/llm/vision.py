"""Ask a multimodal LLM about a moment's video frames — the reusable half
of the QA LOOK step, and (since stage 2) the summary-time frame-analysis
step too.

Public surface:
    async def inspect_moment(*, job, candidate: DeixisCandidate, question: str,
                              output_language: str) -> MomentInspection
        QA's entry point. Fetches frames for one deixis candidate
        (`workers.frames.fetch_frames`, resolution picked per
        `DeixisCategory` — see `_HEIGHT_BY_CATEGORY`) and asks the
        multimodal LLM the QA-anchored `qa_frames.txt` question about them
        via a FORCED `report_frame_findings` tool call, returning a
        structured `VisionResult` — never free prose, so relevance is a
        boolean field to read, not something guessed by regex/keyword
        matching over an answer (see `_parse_vision_result`). Composed from
        the two seam functions below; unchanged behaviour from before they
        existed.

    async def fetch_moment_frames(*, job, candidate: DeixisCandidate,
                                   cookies=None) -> list[Path]
        The download half of `inspect_moment`, pulled out on its own so a
        caller juggling SEVERAL moments (the summarization step, unlike a
        QA turn's at-most-2) can start fetching moment i+1's frames while
        awaiting moment i's vision call — see `workers.pipeline`'s
        conveyor. Raises on failure (`FrameExtractionError` or anything
        `workers.frames.fetch_frames` raises); degrading a failure into
        "no contribution" is the CALLER's job now that fetch and analyze
        are two separate calls (`inspect_moment` still does that itself,
        below).

    async def analyze_summary_frames(frame_paths, *, candidate, output_language,
                                      job_id=None) -> VisionResult
        The summary-time counterpart to `inspect_moment`'s (private)
        QA vision call: same forced-tool-call shape, but anchored to
        `prompts/summary_frames.txt` — no question, just "what does this
        show and is it worth including" (see that prompt's own reasoning).
        Also raises on failure rather than degrading; same reasoning as
        `fetch_moment_frames`.

    class VisionResult(finding: str, relevant: bool, best_frame_index: int | None)
        One vision call's structured outcome. Shared by both the QA and
        summary paths — `relevant` means something different for each (see
        `qa_frames.txt` vs `summary_frames.txt`), but the shape a caller
        needs back is identical either way.

    class MomentInspection(finding: str, frame_path: Path | None)
        What `inspect_moment`'s QA caller needs to fold a moment into its
        own prompt/response: `finding` text (worth keeping even when
        nothing was relevant — "we checked and there was nothing to see"
        is still useful context), and `frame_path` set only when the model
        both rated the moment relevant AND named a valid frame. The
        summary-time caller builds its own equivalent directly from
        `VisionResult` + the frame paths it already has (see
        `workers.pipeline._run_frame_analysis`) rather than going through
        this dataclass — it needs to persist more fields (category,
        seconds, timecode) than this shape carries.

Why this is its own module: `llm/qa.py`'s Q&A flow was the first caller of
this machinery, but it was never Q&A-specific — fetching frames around a
moment and asking a vision model a narrow question about them is exactly
what the summarization-time "describe what's on screen" step
(`workers/pipeline.py`) also needs, just anchored to a different prompt and
without QA's one-at-a-time-then-answer shape (the summary step inspects
several moments with a download/analyze conveyor — see its own docstring).

Degrade, don't break: `inspect_moment` returns `MomentInspection("", None)`
— never raises — on ANY failure along the way: `FrameExtractionError`, an
empty/budget-spent frame list (`workers.frames.MAX_FRAMES_PER_JOB`), a
vision-call error, or a malformed tool-call response. Every failure is
logged as a warning carrying the job id and the moment's timestamp, so a
degraded moment leaves a trace instead of vanishing silently. Callers are
expected to just skip a degraded moment — no prompt contribution, no
thumbnail — rather than treat it as fatal to the surrounding turn, the same
"degrade, don't break" spirit `qa.py` already applies to a failed PLAN call
(falls back to search) or a failed web search (answers without it).
`fetch_moment_frames`/`analyze_summary_frames` themselves do NOT degrade —
they raise like any ordinary function — because the summary step's own
conveyor loop is what decides how to degrade a failed moment (see
`workers.pipeline._run_frame_analysis`'s docstring for why that decision
moved to the caller once fetch and analyze became two separate awaits).

This module has no opinion on `DeixisCategory.EXTERNAL` beyond picking a
frame resolution for the categories that DO want one (`_HEIGHT_BY_CATEGORY`
deliberately has no entry for it) — the actual guarantee that an EXTERNAL
candidate never reaches `fetch_frames` at all lives in each caller's own
loop (`qa.stream_answer`'s LOOK loop, and `workers.pipeline._run_frame_
analysis`'s candidate filter), not here.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.llm import client as llm_client
from src.workers import frames as _frames
from src.workers.deixis import DeixisCandidate, DeixisCategory
from src.workers.errors import FrameExtractionError

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=4)
def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


# Category -> section download resolution (workers/frames.py constants).
# OBJECT candidates are worth reading a label off, ACTION candidates only
# need to be seen. EXTERNAL is deliberately absent: it must never reach
# fetch_frames at all (see the LOOK loop in stream_answer), so there is no
# resolution to pick for it.
_HEIGHT_BY_CATEGORY: dict[DeixisCategory, int] = {
    DeixisCategory.OBJECT: _frames.SECTION_MAX_HEIGHT_READABLE_PX,
    DeixisCategory.ACTION: _frames.SECTION_MAX_HEIGHT_PX,
}


# Forced single tool for the LOOK step's vision call — same technique as
# `_PLAN_TOOL`/`_parse_plan` in `qa.py`: the model must fill in structured
# fields rather than free prose, so `stream_answer` can decide WITHOUT any
# regex/keyword guessing whether a frame actually contributed (see
# `VisionResult`/`_parse_vision_result`).
_VISION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "report_frame_findings",
        "description": (
            "Report what the frames show, and whether any of them actually "
            "help answer the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "finding": {
                    "type": "string",
                    "description": (
                        "Your full answer per the instructions above: what's "
                        "on screen, any readable text, any demonstrated "
                        "action, and the one closing sentence on how it "
                        "bears on the question (or that nothing relevant is "
                        "visible). If the question names something you "
                        "cannot actually see, say so plainly instead of "
                        "describing it — writing this field is not evidence "
                        "that the thing exists."
                    ),
                },
                "relevant": {
                    "type": "boolean",
                    "description": (
                        "true ONLY if the closing sentence you just wrote "
                        "genuinely bears on the question — false for the "
                        "'no information relevant to the question' case."
                    ),
                },
                "best_frame_index": {
                    "type": "integer",
                    "description": (
                        "The number, from the frame numbering you were "
                        "given, of the single frame to SHOW A PERSON as "
                        "evidence. Judge it on two things, in this order: "
                        "(1) it must actually show the thing — the object, "
                        "the readable text, the moment of the action; "
                        "(2) among those, pick the well-captured one — not "
                        "a mid-blink, eyes shut, head turned away, mouth "
                        "open mid-word, awkward halfway pose, motion-smeared "
                        "frame, or one where the thing shown is cut off by "
                        "the frame edge. If none passes (2), still pick the "
                        "one that shows the thing best. Give your best guess "
                        "even when relevant is false."
                    ),
                },
            },
            "required": ["finding", "relevant", "best_frame_index"],
        },
    },
}


# Summary-time counterpart to `_VISION_TOOL` — same tool NAME
# ("report_frame_findings", deliberately unchanged) so `_parse_vision_result`
# needs no branching to handle either caller, but reworded fields: there is
# no question to anchor `relevant` against here, so it means "the picture
# adds something the transcript alone doesn't" (see prompts/summary_frames.txt).
_SUMMARY_VISION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "report_frame_findings",
        "description": (
            "Report what the frames show, and whether it adds anything a "
            "transcript of the audio alone would not already convey."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "finding": {
                    "type": "string",
                    "description": (
                        "A plain, concrete description of what's on screen "
                        "per the instructions above: the setting, any "
                        "readable text, any demonstrated action. Nothing "
                        "else — do NOT include a judgment of whether this "
                        "is worth passing on; that verdict belongs only in "
                        "the 'relevant' field below, never restated or "
                        "justified in this text. Name a thing only as "
                        "specifically as the picture supports — writing "
                        "this field is not evidence that a specific object "
                        "exists beyond what you can actually see."
                    ),
                },
                "relevant": {
                    "type": "boolean",
                    "description": (
                        "true ONLY if the picture adds something the "
                        "transcript alone does not — a shown object, a "
                        "demonstrated action, readable on-screen text, a "
                        "chart, a diagram, a specific place or person. "
                        "false for a generic shot with nothing being shown "
                        "(most commonly: just the speaker talking to camera)."
                    ),
                },
                "best_frame_index": {
                    "type": "integer",
                    "description": (
                        "The number, from the frame numbering you were "
                        "given, of the single frame to SHOW A PERSON as "
                        "evidence. Judge it on two things, in this order: "
                        "(1) it must actually show the thing — the object, "
                        "the readable text, the moment of the action; "
                        "(2) among those, pick the well-captured one — not "
                        "a mid-blink, eyes shut, head turned away, mouth "
                        "open mid-word, awkward halfway pose, motion-smeared "
                        "frame, or one where the thing shown is cut off by "
                        "the frame edge. If none passes (2), still pick the "
                        "one that shows the thing best. Give your best guess "
                        "even when relevant is false."
                    ),
                },
            },
            "required": ["finding", "relevant", "best_frame_index"],
        },
    },
}


@dataclass(frozen=True)
class VisionResult:
    """Structured outcome of one LOOK-step vision call over a moment's frames.

    ``relevant`` gates whether the client gets shown a thumbnail at all (see
    `stream_answer`'s LOOK loop) — `finding` still goes into the synthesis
    prompt's VISUAL FINDINGS either way, since "we checked and there was
    nothing to see" is useful context for the answer even when
    ``relevant`` is false.

    ``best_frame_index`` is 1-based into the frame list the model was
    actually shown, in the order it was given them (matching the numbering
    `qa_frames.txt` tells the model about) — or ``None`` when nothing
    usable came back (missing, non-numeric, or out of range). ``None``
    means no thumbnail even when ``relevant`` is true, since there is
    nothing valid to point a thumbnail at.
    """

    finding: str
    relevant: bool
    best_frame_index: int | None


def _parse_best_frame_index(raw: Any, num_frames: int) -> int | None:
    """Validate the model's chosen frame index into ``[1, num_frames]``.
    Anything else (missing, non-numeric, out of range) is dropped rather
    than failing the whole parse — mirrors `_parse_look_at_indices`'s
    "drop rather than reject" spirit."""
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        return None
    return idx if 1 <= idx <= num_frames else None


def _parse_vision_result(
    response: Any,
    num_frames: int,
    *,
    job_id: Any = None,
    timestamp: float | None = None,
) -> VisionResult:
    """Extract a `VisionResult` from the forced `report_frame_findings` tool
    call. Mirrors `_parse_plan`'s defensive shape exactly: ANY malformed or
    missing call degrades to ``VisionResult(finding="", relevant=False,
    best_frame_index=None)`` — never raises. `inspect_moment` already
    promises to degrade to "no contribution" on any failure; this is what
    keeps that promise even when the tool call itself comes back garbled.

    Every degrade path below logs a warning — job id + moment timestamp,
    same context the neighbouring log calls in `inspect_moment` use — so
    "the user got no thumbnail and no finding" leaves a trace instead of
    vanishing silently. ``job_id``/``timestamp`` are optional (default
    ``None``) purely for callers that don't have them (e.g. the harness
    scripts in scratch dirs that call this directly); real QA turns always
    pass both via `_ask_vision_about_frames`.
    """
    ts_label = f"{timestamp:.1f}s" if isinstance(timestamp, (int, float)) else "?"
    try:
        tool_calls = response.choices[0].message.tool_calls or []
    except (AttributeError, IndexError):
        log.warning(
            "QA LOOK step: vision response had no tool_calls at all "
            "(job %s at %s)", job_id, ts_label,
        )
        return VisionResult("", False, None)
    for tc in tool_calls:
        if tc.function.name != "report_frame_findings":
            continue
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            log.warning(
                "QA LOOK step: vision tool call arguments failed to parse "
                "as JSON — possibly truncated by max_tokens (job %s at %s)",
                job_id, ts_label,
            )
            return VisionResult("", False, None)
        finding = str(args.get("finding") or "").strip()
        # Strict: only an explicit boolean True counts as relevant — same
        # bar `_parse_plan` holds `material_sufficient` to.
        relevant = args.get("relevant") is True
        best_frame_index = _parse_best_frame_index(args.get("best_frame_index"), num_frames)
        return VisionResult(finding, relevant, best_frame_index)
    log.warning(
        "QA LOOK step: vision response's tool call wasn't "
        "'report_frame_findings' (job %s at %s)", job_id, ts_label,
    )
    return VisionResult("", False, None)


@dataclass(frozen=True)
class MomentInspection:
    """What `inspect_moment` learned about one LOOK-step moment — everything
    `stream_answer` needs to both extend the synthesis prompt's VISUAL
    FINDINGS and, when warranted, hand the client a `FrameRef`.

    ``finding`` is ``""`` when nothing usable came back at all (any of the
    degrade points in `inspect_moment`'s docstring) — the caller skips
    contributing it to VISUAL FINDINGS in that case, same as before this
    feature. ``frame_path`` is the single frame the vision model singled
    out as most informative, and is only ever set when the model reported
    ``relevant=True`` AND a valid ``best_frame_index`` came back — that
    combination is what decides whether a thumbnail is shown at all.
    """

    finding: str
    frame_path: Path | None


def _frame_to_data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# Budget for the vision call's generated tokens. Before the LOOK step's
# vision output became a forced tool call, this only had to cover the
# finding prose itself; now the finding text has to fit INSIDE the tool
# call's JSON arguments alongside the field names/punctuation, and a
# truncated arguments string is invalid JSON — under `_parse_vision_result`
# that drops the WHOLE moment (thumbnail AND finding), where truncated free
# prose used to just cut a sentence short. Measured against three real
# findings this feature's harness produced for a Russian `output_language`
# (Cyrillic costs noticeably more cl100k_base tokens per word than English):
# finding text alone ranged 193-301 tokens, the full `{"finding": ...,
# "relevant": ..., "best_frame_index": ...}` JSON 210-318 tokens. 900 keeps
# roughly 3x headroom over the longest of those while staying well short of
# runaway generation.
_VISION_MAX_TOKENS = 900


async def _call_vision_tool(
    frame_paths: Sequence[Path],
    *,
    prompt_text: str,
    tool: dict[str, Any],
    timestamp: float,
    job_id: Any = None,
) -> VisionResult:
    """Shared plumbing for a forced-tool-call vision request over ALL of one
    moment's frames in a SINGLE call — deliberately unlike
    ``workers/pdf.py._ocr_one_page``, which sends exactly one image per call.

    That difference is deliberate, not an oversight: pdf.py's pages are
    independent documents-within-a-document (page N+1 has no bearing on
    page N), so it isolates failures and spends tokens per page, one at a
    time, by design. Here every frame in ``frame_paths`` is a ~1-frame/s
    sample of the SAME few-second moment ``workers.frames.fetch_frames``
    windowed around one deixis candidate — so cross-frame reasoning ("the
    hand moves from A to B across these frames") is exactly what the
    ACTION category exists to capture, and splitting them into separate
    calls would throw that continuity away for no benefit. The token cost
    stays bounded regardless: at most ``MAX_FRAMES_PER_CALL`` frames exist
    per moment.

    Both callers below (`_ask_vision_about_frames` for QA,
    `analyze_summary_frames` for the summary step) differ only in which
    prompt text and tool wording they pass in — the message shape, the
    forced tool_choice, and the parse step are identical either way.

    ``job_id`` is optional and used only to give `_parse_vision_result`'s
    warning logs the same job context callers' neighbouring log calls
    already carry — it plays no role in the call itself.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for path in frame_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _frame_to_data_uri(path)},
            }
        )
    resp = await llm_client.complete_with_messages(
        [{"role": "user", "content": content}],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "report_frame_findings"}},
        max_tokens=_VISION_MAX_TOKENS,
        temperature=0.0,
    )
    return _parse_vision_result(resp, len(frame_paths), job_id=job_id, timestamp=timestamp)


async def _ask_vision_about_frames(
    frame_paths: Sequence[Path],
    *,
    candidate: DeixisCandidate,
    question: str,
    output_language: str,
    job_id: Any = None,
) -> VisionResult:
    """QA's vision call: the narrow, question-anchored ``qa_frames.txt``."""
    prompt = _load_prompt("qa_frames.txt").format(
        output_language=output_language,
        phrase=candidate.phrase,
        question=question,
    )
    return await _call_vision_tool(
        frame_paths,
        prompt_text=prompt,
        tool=_VISION_TOOL,
        timestamp=candidate.timestamp,
        job_id=job_id,
    )


async def analyze_summary_frames(
    frame_paths: Sequence[Path],
    *,
    candidate: DeixisCandidate,
    output_language: str,
    job_id: Any = None,
) -> VisionResult:
    """Summary-time vision call: the neutral, no-question ``summary_frames.
    txt`` — "what does this show, and is it worth including" rather than
    QA's "does this help answer the question" (see that prompt's own
    reasoning, and the module docstring's note on what ``relevant`` means
    here).

    Raises on failure (an LLM-call error, same as any ordinary async call)
    rather than degrading — unlike `inspect_moment`, which degrades
    QA-side. The summary step's own conveyor loop
    (``workers.pipeline._run_frame_analysis``) is what decides how to
    degrade a failed moment, since it also owns the corresponding
    `fetch_moment_frames` call and needs one consistent place to do so for
    both halves.
    """
    prompt = _load_prompt("summary_frames.txt").format(
        output_language=output_language,
        phrase=candidate.phrase,
    )
    return await _call_vision_tool(
        frame_paths,
        prompt_text=prompt,
        tool=_SUMMARY_VISION_TOOL,
        timestamp=candidate.timestamp,
        job_id=job_id,
    )


async def fetch_moment_frames(
    *, job: Any, candidate: DeixisCandidate, cookies: list[Any] | None = None
) -> list[Path]:
    """Fetch frames for one deixis candidate at the category-appropriate
    resolution (`_HEIGHT_BY_CATEGORY`) — the download half of
    `inspect_moment`, pulled out so a caller juggling several moments (the
    summary step) can prefetch moment i+1's frames while awaiting moment
    i's vision call (see `workers.pipeline._run_frame_analysis`'s conveyor).

    Raises whatever ``workers.frames.fetch_frames`` raises
    (``FrameExtractionError``, or any other exception) rather than
    degrading — the caller decides how to degrade now that fetch and
    analyze are two separate awaits (see this module's docstring). Returns
    ``[]`` (not an error) when the job's per-job frame budget
    (``workers.frames.MAX_FRAMES_PER_JOB``) is already spent, same as
    ``fetch_frames`` itself.

    ``cookies`` defaults to ``None`` and is never persisted anywhere — it
    only ever lives as long as the caller's own stack frame. The two
    callers genuinely differ here, which is the whole reason this stayed a
    parameter instead of a hardcoded ``None``:

    - The QA LOOK step (``inspect_moment``, below) calls this long after
      ingestion finished, from a Q&A turn that has no job-creation request
      in scope at all — cookies only ever arrive on that original request
      (``api.schemas.CreateJobRequest.cookies``) and are never persisted
      on ``Job`` (see ``storage/db.py``), so by the time a question is
      asked there is genuinely nothing left to forward. It keeps passing
      none, unchanged.
    - The summary-time step (``workers.pipeline._run_frame_analysis``)
      runs INSIDE ``run_pipeline``, while the triggering request's cookies
      are still a live local variable threaded down through
      ``_summarize_and_finish`` — see that module's docstring. Forwarding
      them there is what makes frame analysis work at all on a video
      YouTube gates behind a signed-in session; without them every frame
      download 403s, the per-moment guard swallows it, and the step
      silently produces zero findings.

    A cookie-gated video's frame fetch still fails like any other network
    error when no cookies are available (or the ones given aren't enough),
    degrading the same way at the caller either way.

    The URL to fetch from is resolved via ``workers.frames.
    resolve_frame_source_url`` — for a ``kind=media`` job that's
    ``job.media_url`` (the actual video, separate from the page it's
    embedded in), for everything else it's ``job.url``. A media job with no
    stored ``media_url`` (every job created before migration v12) resolves
    to ``None`` here and raises the same as a job missing its id — the
    caller's degrade-to-``[]``/``[]``-findings behaviour applies uniformly,
    never a silent fall back to the page URL.
    """
    job_id = getattr(job, "id", None)
    url = _frames.resolve_frame_source_url(job)
    if not job_id or not url:
        raise FrameExtractionError("job is missing id/url; cannot fetch frames")
    max_height = _HEIGHT_BY_CATEGORY.get(candidate.category, _frames.SECTION_MAX_HEIGHT_PX)
    return await _frames.fetch_frames(
        job_id=job_id,
        url=url,
        timestamp_seconds=candidate.timestamp,
        cookies=cookies,
        max_height_px=max_height,
    )


async def inspect_moment(
    *,
    job: Any,
    candidate: DeixisCandidate,
    question: str,
    output_language: str,
) -> MomentInspection:
    """Fetch frames for one chosen deixis candidate and ask the vision model
    about them. Returns ``MomentInspection("", None)`` — never raises — on
    ANY failure: ``FrameExtractionError``, an empty frame list (the job's
    per-job frame budget is spent, see ``workers.frames.MAX_FRAMES_PER_JOB``),
    or the vision call itself erroring (including a malformed tool-call
    response — see `_parse_vision_result`). Each is logged as a warning and
    just means this one moment contributes nothing to the synthesis prompt
    and no thumbnail — the exact "degrade, don't break" spirit `qa.py`'s
    Q&A flow already applies to a failed PLAN call (falls back to search)
    or a failed web search (answers without it).

    Callers must never pass an EXTERNAL candidate here — see the guard in
    ``stream_answer``'s LOOK loop, which is `qa.py`'s own enforcement that
    EXTERNAL never triggers a frame fetch, independent of whether the model
    honoured the plan prompt's instruction not to pick one.
    """
    job_id = getattr(job, "id", None)
    url = getattr(job, "url", None)
    if not job_id or not url:
        return MomentInspection("", None)

    try:
        frame_paths = await fetch_moment_frames(job=job, candidate=candidate)
    except FrameExtractionError:
        log.warning(
            "QA LOOK step: frame fetch failed for job %s at %.1fs",
            job_id, candidate.timestamp, exc_info=True,
        )
        return MomentInspection("", None)
    except Exception:
        log.warning(
            "QA LOOK step: frame fetch raised unexpectedly for job %s at %.1fs",
            job_id, candidate.timestamp, exc_info=True,
        )
        return MomentInspection("", None)

    if not frame_paths:
        log.warning(
            "QA LOOK step: no frames returned for job %s at %.1fs "
            "(per-job frame budget likely spent)",
            job_id, candidate.timestamp,
        )
        return MomentInspection("", None)

    try:
        result = await _ask_vision_about_frames(
            frame_paths,
            candidate=candidate,
            question=question,
            output_language=output_language,
            job_id=job_id,
        )
    except Exception:
        log.warning(
            "QA LOOK step: vision call failed for job %s at %.1fs",
            job_id, candidate.timestamp, exc_info=True,
        )
        return MomentInspection("", None)

    frame_path = (
        frame_paths[result.best_frame_index - 1]
        if result.relevant and result.best_frame_index is not None
        else None
    )
    return MomentInspection(result.finding, frame_path)


__all__ = [
    "MomentInspection",
    "VisionResult",
    "analyze_summary_frames",
    "fetch_moment_frames",
    "inspect_moment",
]
