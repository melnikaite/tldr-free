"""Single-job Q&A — layered answering: material → video frames → AI knowledge → web.

Public surface:
    async def stream_answer(*, job, question: str, output_language: str, from_audio: bool)
        -> AsyncIterator[str | dict[str, Any]]

        Builds context = job.raw_text if it fits, else job.summary_md.

        Flow (the model never gets to veto a search with a single tool pick —
        that is the failure mode small models like Gemma 4 abuse, refusing to
        search at the slightest excuse):

          1. PLAN — one forced `plan` tool call returns
             {material_sufficient: bool, search_query: str}. The bar for
             `material_sufficient=true` is high: the material must explicitly
             and completely answer. Anything else — uncertainty, a missing
             specific detail, a parse error, a backend without tool support —
             defaults to SEARCHING. An extra web trip is cheap; a missed one
             leaves the user short.

             For jobs with a timestamped transcript (see below), the same
             call also offers a numbered list of `workers.deixis` candidates
             — moments where the speech points at the video's picture — and
             an extra `look_at_indices` tool field the model uses to name
             (at most 2) worth a frame download. With no candidates, the
             tool/prompt are byte-identical to before this feature existed —
             see `_plan_tool` / `_plan_messages`.
          2. LOOK — for every index the model picked, fetch that moment's
             frames (`workers.frames.fetch_frames`) at a resolution driven by
             the candidate's `DeixisCategory` (`_HEIGHT_BY_CATEGORY`) and ask
             the multimodal LLM the narrow `qa_frames.txt` question via a
             FORCED `report_frame_findings` tool call — a structured
             `VisionResult` (finding text, relevant: bool, best_frame_index),
             not free prose, so relevance is never guessed by regex/keyword
             matching (see `_parse_vision_result`). A moment the model rates
             `relevant=True` also contributes a `FrameRef` — daemon-side, an
             actual thumbnail the client can show — collected into one
             `{"type": "frames", "items": [...]}` event yielded once after
             the loop (never per-moment, and never at all when nothing came
             back relevant). Any failure (`FrameExtractionError`, an
             empty/budget-spent frame list, a vision-call error, a malformed
             tool-call response) just drops that moment's contribution —
             logged, never raised — the same "degrade, don't break" spirit
             as a failed PLAN call falling back to search. EXTERNAL
             candidates are never fetched, even if named: the daemon guards
             this independently of the model's compliance (see
             `stream_answer`'s LOOK loop).
          3. SEARCH — when not sufficient, fetch + trafilatura-clean the DDG
             results (real page content, not just snippets).
          4. SYNTHESIS — stream the answer grounded in, by priority, the
             material, then the LOOK step's visual findings (attributed by
             timecode), then the model's own knowledge, then the web results
             (cited). The prompt actively encourages going beyond the material.

        Yields:
          - str — token delta for the final answer
          - dict — stage event, e.g. {"type":"stage","stage":"searching","detail":"<query>"}
            or {"type":"stage","stage":"looking","detail":"<timecode> — <phrase>"}
          - dict — {"type":"frames","items":[FrameRef, ...]}, at most once,
            only when at least one LOOK-step moment was rated relevant

        Only jobs with a timestamped transcript get the LOOK step —
        `Job.raw_segments_json` present AND `Job.transcript_source` in
        `api.schemas.AUDIO_TRANSCRIPT_SOURCES` (see `_deixis_candidates_for_job`).
        Web pages and PDFs take a completely unchanged path: no candidates,
        so the plan tool/prompt revert to their pre-feature shape and the
        LOOK step never runs.

Called from api/ai.py POST /ai/qa.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import get_config
from src.llm import client as llm_client
from src.llm.tokens import count_tokens
from src.workers import deixis as _deixis
from src.workers import frames as _frames
from src.workers import search as _search
from src.workers import timecodes as _timecodes
from src.workers.deixis import DeixisCandidate, DeixisCategory
from src.workers.errors import FrameExtractionError

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Reserve room for the prompt scaffolding, tool results, and the answer.
_PROMPT_OVERHEAD_TOKENS = 4000

# Single forced tool. The model must fill in the plan — it cannot refuse to
# decide. material_sufficient is deliberately strict; we treat anything other
# than a clean True as "search".
_PLAN_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "plan",
        "description": (
            "Decide whether the provided material alone fully and specifically "
            "answers the user's question, and propose a web search query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "material_sufficient": {
                    "type": "boolean",
                    "description": (
                        "true ONLY if the material explicitly contains the "
                        "complete, specific answer to every part of the "
                        "question. false if the material only mentions the "
                        "topic but lacks the specific fact, number, name, or "
                        "detail asked for, or if current/external/additional "
                        "information would help. When in doubt, false."
                    ),
                },
                "search_query": {
                    "type": "string",
                    "description": (
                        "A concise web search query in the user's language to "
                        "find or enrich the answer online. Always provide one."
                    ),
                },
            },
            "required": ["material_sufficient", "search_query"],
        },
    },
}

# A model naming more than this many moments just gets the extras dropped —
# each one costs a frame download (workers/frames.py) and, downstream, a
# vision LLM call, so the budget is small and non-negotiable in code, not
# left to the model's judgment.
_MAX_LOOK_AT_MOMENTS = 2

# Category -> section download resolution (workers/frames.py constants).
# OBJECT candidates are worth reading a label off, ACTION candidates only
# need to be seen. EXTERNAL is deliberately absent: it must never reach
# fetch_frames at all (see the LOOK loop in stream_answer), so there is no
# resolution to pick for it.
_HEIGHT_BY_CATEGORY: dict[DeixisCategory, int] = {
    DeixisCategory.OBJECT: _frames.SECTION_MAX_HEIGHT_READABLE_PX,
    DeixisCategory.ACTION: _frames.SECTION_MAX_HEIGHT_PX,
}

_LOOK_AT_INDICES_PROPERTY: dict[str, Any] = {
    "type": "array",
    "items": {"type": "integer"},
    "description": (
        "Indices (from the numbered VIDEO MOMENTS list below) of moments "
        "worth visually inspecting to answer the question — at most "
        f"{_MAX_LOOK_AT_MOMENTS}, since each costs a video download. Only "
        "include a moment tagged [action] or [object] where seeing the "
        "picture would plausibly help (a demonstrated action, a shown "
        "object or label the question is about). NEVER include a moment "
        "tagged [external] — those defer to something OUTSIDE the video "
        "(a link, the description, an article number), which a web search "
        "answers, not a frame. Omit the field or return an empty array if "
        "no moment applies."
    ),
}


def _plan_tool(candidates: Sequence[DeixisCandidate]) -> dict[str, Any]:
    """Build the forced `plan` tool schema.

    With no candidates, returns `_PLAN_TOOL` itself, unchanged — byte-for-byte
    identical to the tool this daemon offered before the LOOK step existed, so
    a page/PDF job (or a job whose transcript has no deixis moments) sees
    exactly the old behaviour. With candidates, returns a deep copy — never
    mutates the shared module-level constant — with `look_at_indices` added
    as an extra optional property. The daemon never guesses relevance by
    keyword-matching the question against a candidate's phrase; the model
    sees the phrases (and categories) and chooses.
    """
    if not candidates:
        return _PLAN_TOOL
    tool = copy.deepcopy(_PLAN_TOOL)
    tool["function"]["parameters"]["properties"]["look_at_indices"] = (
        _LOOK_AT_INDICES_PROPERTY
    )
    return tool


def _format_timecode(seconds: float) -> str:
    """``MM:SS`` / ``H:MM:SS`` via the one place timecodes are formatted
    (``workers.timecodes``) — see ``.claude/llm.md`` "Timecodes are
    formatted in ONE place"."""
    total = max(0, int(round(seconds)))
    return _timecodes._format_marker(total, use_hours=total >= 3600)


def _format_candidates_block(candidates: Sequence[DeixisCandidate]) -> str:
    """Numbered list of deixis candidates for the PLAN prompt: index,
    timecode, category, and the exact phrase that triggered the candidate —
    everything the model needs to judge relevance itself. Appended to the
    plan prompt only when ``candidates`` is non-empty (see ``_plan_messages``)
    so the no-candidate case never touches this at all.
    """
    lines = [
        f"{i}: {_format_timecode(c.timestamp)} [{c.category.value}] — {c.phrase}"
        for i, c in enumerate(candidates, start=1)
    ]
    return (
        "VIDEO MOMENTS — points where the speaker may be pointing at the "
        "video's picture (index: timecode [category] — what was said):\n"
        + "\n".join(lines)
        + "\n\n"
        "If one of these moments would help answer the question, call "
        f"look_at_indices with its index (at most {_MAX_LOOK_AT_MOMENTS}). "
        "[object] moments are a shown item/label worth reading; [action] "
        "moments are a demonstrated action worth seeing. Never pick an "
        "[external] moment — it points to something outside the video "
        "(a link, the description, an article number); rely on "
        "search_query for those instead."
    )


# A line that is nothing but HTML tags + whitespace — a run of <br>, a stray
# </blockquote>, etc. A small local model degenerating at the end of an answer
# emits these as filler. A line with real text between tags (e.g. a link,
# "<a ...>Источник</a>") does NOT match, so genuine content survives; the
# frequency penalty on generation is what stops the loop producing them.
_MARKUP_ONLY_LINE = re.compile(r"^\s*(?:<[^>]+>\s*)+$")

# Timestamp instructions for the synthesis prompt — chosen per source type so
# a document-only job never even sees a [MM:SS] example to copy verbatim.
_TIMESTAMP_RULES_TRANSCRIPT = (
    "Timestamps: include a [MM:SS] or [HH:MM:SS] marker ONLY inline, right "
    "after a sentence taken from the material's transcript, and ONLY when "
    "the material itself contains that marker at that point. Put each "
    'timestamp in its own bracket (e.g. "[02:15] ... [05:47]") — never '
    'combine several inside one bracket like "[02:15, 05:47]". Never attach '
    "a timestamp to a fact that came from your knowledge or the web, never "
    "put one on a line by itself, and never output a list of bare timestamps."
)
_TIMESTAMP_RULES_DOCUMENT = (
    "The material is a document (web page or PDF) and has NO timestamps. "
    "Never output [MM:SS] or [HH:MM:SS] markers."
)


def clean_answer(text: str) -> str:
    """Strip degenerate filler a small model appends to a Q&A answer.

    Observed tail-noise that carries no information:
      - a dump of bare ``[MM:SS]`` markers, one per line (the whole transcript);
      - lines made only of HTML tags (runs of ``<br>``, a stray ``</blockquote>``).
    We drop both, collapse the blank runs left behind, and trim. Lines with real
    text — an inline timecode next to a sentence, a link with a label — are kept.

    Pure: same input → same output.
    """
    if not text:
        return text
    cleaned = _timecodes.strip_bare_timecode_lines(text)
    if "<" in cleaned:
        kept = [ln for ln in cleaned.split("\n") if not _MARKUP_ONLY_LINE.match(ln)]
        cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned


@lru_cache(maxsize=4)
def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _select_context(job: Any) -> str:
    """Pick raw_text if it fits in the model's context; otherwise summary_md."""
    raw = getattr(job, "raw_text", None) or ""
    summary = getattr(job, "summary_md", None) or ""
    budget = get_config().llm.context_length - _PROMPT_OVERHEAD_TOKENS
    if raw and count_tokens(raw) <= budget:
        return raw
    return summary


def _plan_context(job: Any, *, fallback: str) -> str:
    """Compact context for the cheap PLAN call — prefer the summary over the
    full transcript. The plan only judges sufficiency + drafts a query, so the
    summary is enough; it also halves the plan-call prefill and biases toward
    searching (the summary omits detail), which is the behaviour we want."""
    return getattr(job, "summary_md", None) or fallback


def _plan_messages(
    *,
    title: str,
    context: str,
    question: str,
    candidates: Sequence[DeixisCandidate] = (),
) -> list[dict[str, Any]]:
    prompt = _load_prompt("qa_plan.txt").format(
        title=title, context=context, question=question
    )
    # Appended, never interpolated into qa_plan.txt itself: with no
    # candidates this is a no-op, so the prompt sent to the model stays
    # byte-identical to what it was before the LOOK step existed.
    if candidates:
        prompt = f"{prompt}\n\n{_format_candidates_block(candidates)}"
    return [{"role": "user", "content": prompt}]


def _answer_messages(
    *,
    output_language: str,
    title: str,
    context: str,
    question: str,
    web_results: str,
    frame_findings: str,
    from_audio: bool,
) -> list[dict[str, Any]]:
    timestamp_rules = (
        _TIMESTAMP_RULES_TRANSCRIPT if from_audio else _TIMESTAMP_RULES_DOCUMENT
    )
    prompt = _load_prompt("qa.txt").format(
        output_language=output_language,
        title=title,
        context=context,
        question=question,
        web_results=web_results or "(no web search was run)",
        frame_findings=frame_findings or "(no frames were examined)",
        timestamp_rules=timestamp_rules,
    )
    return [{"role": "user", "content": prompt}]


def _parse_plan(response: Any, num_candidates: int = 0) -> tuple[bool, str, list[int]]:
    """Extract (material_sufficient, search_query, look_at_indices) from the
    plan tool call.

    Defaults to (False, "", []) — i.e. SEARCH, no frames — on any malformed
    / missing call, so a flaky tool response biases toward searching rather
    than dodging it. ``look_at_indices`` entries are validated against
    ``num_candidates`` (1-based, matching ``_format_candidates_block``) and
    capped at ``_MAX_LOOK_AT_MOMENTS`` — an out-of-range or excess index is
    silently dropped rather than treated as a parse failure, and duplicates
    collapse to one.
    """
    try:
        tool_calls = response.choices[0].message.tool_calls or []
    except (AttributeError, IndexError):
        return False, "", []
    for tc in tool_calls:
        if tc.function.name != "plan":
            continue
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            return False, "", []
        sufficient = args.get("material_sufficient")
        query = args.get("search_query") or ""
        indices = _parse_look_at_indices(args.get("look_at_indices"), num_candidates)
        # Strict: only an explicit boolean True counts as sufficient.
        return (sufficient is True), str(query).strip(), indices
    return False, "", []


def _parse_look_at_indices(raw: Any, num_candidates: int) -> list[int]:
    """Validate/cap the model's chosen indices. Not a list -> []. Each entry
    must parse as an int and land in ``[1, num_candidates]``; anything else
    (out of range, non-numeric, a repeat) is dropped rather than failing the
    whole plan. Stops collecting past ``_MAX_LOOK_AT_MOMENTS``."""
    if not isinstance(raw, list):
        return []
    indices: list[int] = []
    for value in raw:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= num_candidates and idx not in indices:
            indices.append(idx)
        if len(indices) >= _MAX_LOOK_AT_MOMENTS:
            break
    return indices



# Forced single tool for the LOOK step's vision call — same technique as
# `_PLAN_TOOL`/`_parse_plan` above: the model must fill in structured fields
# rather than free prose, so `stream_answer` can decide WITHOUT any
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
    best_frame_index=None)`` — never raises. `_inspect_moment` already
    promises to degrade to "no contribution" on any failure; this is what
    keeps that promise even when the tool call itself comes back garbled.

    Every degrade path below logs a warning — job id + moment timestamp,
    same context the neighbouring log calls in `_inspect_moment` use — so
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
    """What `_inspect_moment` learned about one LOOK-step moment — everything
    `stream_answer` needs to both extend the synthesis prompt's VISUAL
    FINDINGS and, when warranted, hand the client a `FrameRef`.

    ``finding`` is ``""`` when nothing usable came back at all (any of the
    degrade points in `_inspect_moment`'s docstring) — the caller skips
    contributing it to VISUAL FINDINGS in that case, same as before this
    feature. ``frame_path`` is the single frame the vision model singled
    out as most informative, and is only ever set when the model reported
    ``relevant=True`` AND a valid ``best_frame_index`` came back — that
    combination is what decides whether a thumbnail is shown at all.
    """

    finding: str
    frame_path: Path | None


# Deixis candidates for the LOOK step, or ``[]`` when this job doesn't
# qualify — which also keeps the PLAN tool/prompt byte-identical to before
# the feature existed (see ``_plan_tool`` / ``_plan_messages``). Shared with
# ``GET /jobs/{id}/moments`` (``api/jobs.py``, the on-demand "look"
# affordance next to a summary line) via ``workers.deixis.candidates_for_job``
# — see that function's docstring for the exact qualification rule.
_deixis_candidates_for_job = _deixis.candidates_for_job


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


async def _ask_vision_about_frames(
    frame_paths: Sequence[Path],
    *,
    candidate: DeixisCandidate,
    question: str,
    output_language: str,
    job_id: Any = None,
) -> VisionResult:
    """Ask the multimodal LLM the narrow ``qa_frames.txt`` question about
    ALL of one moment's frames in a SINGLE call — deliberately unlike
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
    per moment, and ``stream_answer`` inspects at most
    ``_MAX_LOOK_AT_MOMENTS`` (2) moments per QA turn.

    ``job_id`` is optional and used only to give `_parse_vision_result`'s
    warning logs the same job context `_inspect_moment`'s neighbouring log
    calls already carry — it plays no role in the call itself.
    """
    prompt = _load_prompt("qa_frames.txt").format(
        output_language=output_language,
        phrase=candidate.phrase,
        question=question,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in frame_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _frame_to_data_uri(path)},
            }
        )
    resp = await llm_client.complete_with_messages(
        [{"role": "user", "content": content}],
        tools=[_VISION_TOOL],
        tool_choice={"type": "function", "function": {"name": "report_frame_findings"}},
        max_tokens=_VISION_MAX_TOKENS,
        temperature=0.0,
    )
    return _parse_vision_result(
        resp, len(frame_paths), job_id=job_id, timestamp=candidate.timestamp
    )


async def _inspect_moment(
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
    and no thumbnail — the exact "degrade, don't break" spirit the rest of
    this module already applies to a failed PLAN call (falls back to
    search) or a failed web search (answers without it).

    Callers must never pass an EXTERNAL candidate here — see the guard in
    ``stream_answer``'s LOOK loop, which is this module's OWN enforcement
    that EXTERNAL never triggers a frame fetch, independent of whether the
    model honoured the plan prompt's instruction not to pick one.
    """
    job_id = getattr(job, "id", None)
    url = getattr(job, "url", None)
    if not job_id or not url:
        return MomentInspection("", None)

    max_height = _HEIGHT_BY_CATEGORY.get(candidate.category, _frames.SECTION_MAX_HEIGHT_PX)
    try:
        frame_paths = await _frames.fetch_frames(
            job_id=job_id,
            url=url,
            timestamp_seconds=candidate.timestamp,
            # No cookies: cookies only ever arrive on the original job-creation
            # request (api.schemas.CreateJobRequest.cookies) and are never
            # persisted on Job (see storage/db.py) — by QA time, long after
            # ingestion, there is nothing stored to forward. A cookie-gated
            # video's frame fetch just fails like any other network error
            # below, degrading the same way.
            cookies=None,
            max_height_px=max_height,
        )
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


async def stream_answer(
    *,
    job: Any,
    question: str,
    output_language: str,
    from_audio: bool,
) -> AsyncIterator[str | dict[str, Any]]:
    """Yield token deltas (str) or stage dicts (dict) for a QA turn.

    See module docstring for the plan → look → search → synthesis flow.
    """
    context = _select_context(job)
    title = getattr(job, "title", None) or ""
    candidates = _deixis_candidates_for_job(job)

    # Step 1: PLAN. Forced single tool — the model must decide; we bias to
    # search on any failure. Judged against the compact summary, not the full
    # transcript, to keep this call cheap. With no deixis candidates,
    # `_plan_tool`/`_plan_messages` are byte-identical to before the LOOK
    # step existed.
    sufficient = False
    query = question
    look_at_indices: list[int] = []
    try:
        plan = await llm_client.complete_with_messages(
            _plan_messages(
                title=title,
                context=_plan_context(job, fallback=context),
                question=question,
                candidates=candidates,
            ),
            tools=[_plan_tool(candidates)],
            tool_choice={"type": "function", "function": {"name": "plan"}},
            max_tokens=300,
            temperature=0.0,
        )
        sufficient, parsed_query, look_at_indices = _parse_plan(plan, len(candidates))
        if parsed_query:
            query = parsed_query
    except Exception:
        # Backend lacks tool support / errored — default to searching.
        log.warning("QA plan call failed; defaulting to web search", exc_info=True)
        sufficient = False

    # Step 2: LOOK — inspect the moments the plan named, if any. Each
    # inspection degrades to "no contribution" rather than raising (see
    # `_inspect_moment`'s docstring), so a bad frame fetch or vision call
    # never breaks the rest of the QA turn.
    frame_findings: list[str] = []
    # One entry per moment the vision model reported as actually relevant —
    # never "just because we fetched something" (see MomentInspection /
    # VisionResult). Emitted as a single `frames` event below, after the
    # loop, and forwarded by api/ai.py to be persisted on the assistant
    # message so history reload renders the identical thumbnail.
    frame_refs: list[dict[str, Any]] = []
    job_id_for_frames = getattr(job, "id", None)
    for idx in look_at_indices:
        candidate = candidates[idx - 1]
        if candidate.category == DeixisCategory.EXTERNAL:
            # Defence in depth: the plan tool/prompt both tell the model
            # never to pick an EXTERNAL index, but a model can misbehave —
            # this daemon-side guard is what actually GUARANTEES an EXTERNAL
            # candidate never reaches fetch_frames, not the model's compliance.
            log.warning(
                "QA plan picked an EXTERNAL deixis candidate (job %s); "
                "skipping frame fetch",
                getattr(job, "id", "?"),
            )
            continue
        timecode = _format_timecode(candidate.timestamp)
        yield {
            "type": "stage",
            "stage": "looking",
            "detail": f"{timecode} — {candidate.phrase}",
        }
        inspection = await _inspect_moment(
            job=job,
            candidate=candidate,
            question=question,
            output_language=output_language,
        )
        if inspection.finding:
            frame_findings.append(f"[{timecode}] {inspection.finding}")
        if inspection.frame_path is not None and job_id_for_frames:
            # Read the "t<second>" directory straight off the actual path
            # `workers.frames.fetch_frames` wrote to, rather than
            # re-deriving it from `candidate.timestamp` — the two agree
            # today only by coincidence (same rounding, and frames.py
            # clamps negative timestamps to 0 while a naive `int(...)` here
            # wouldn't), and there must be exactly one place that knows the
            # on-disk layout. Matches the shape `frames.resolve_frame_path`
            # / `GET /jobs/{id}/frames/{rel_path}` expect.
            rel_path = f"{inspection.frame_path.parent.name}/{inspection.frame_path.name}"
            frame_refs.append(
                {
                    "seconds": candidate.timestamp,
                    "timecode": timecode,
                    "phrase": candidate.phrase,
                    "frame_url": f"/jobs/{job_id_for_frames}/frames/{rel_path}",
                }
            )

    if frame_refs:
        yield {"type": "frames", "items": frame_refs}

    # Step 3: SEARCH unless the material clearly suffices on its own.
    web_results = ""
    if not sufficient:
        yield {"type": "stage", "stage": "searching", "detail": query}
        try:
            results = await _search.ddg_search_with_content(query)
            web_results = _search.format_results(results, full_content=True)
        except Exception:
            log.warning("QA web search failed; answering without it", exc_info=True)
            web_results = ""

    # Step 4: SYNTHESIS. Stream the grounded answer (material → frames → knowledge → web).
    messages = _answer_messages(
        output_language=output_language,
        title=title,
        context=context,
        question=question,
        web_results=web_results,
        frame_findings="\n\n".join(frame_findings),
        from_audio=from_audio,
    )
    async for delta in llm_client.stream_with_messages(
        messages,
        max_tokens=2000,
        temperature=0.3,
    ):
        yield delta


__all__ = ["clean_answer", "stream_answer"]
