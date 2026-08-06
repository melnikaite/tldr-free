// Build a small thumbnail row for a list of FrameRef — one visual language
// (`.chat-frame-row` / `.chat-frame-thumb` in sidepanel/style.css) shared by
// TWO callers so neither reinvents it:
//   - sidepanel/chat.js — a QA answer's LOOK-step frames (see AIFramesEvent).
//   - sidepanel/app.js — the on-demand "look" affordance next to a summary
//     line whose [MM:SS] timecode sits on a deixis moment (see
//     GET/POST /jobs/{id}/moments|frames).
//
// Clicking a thumbnail seeks exactly like clicking a `[MM:SS]` timecode
// link: same `data-tldr-*` dataset, same delegated click handler in app.js
// — not a second seek mechanism.

import { resolveVideoId } from "./url.js";

/** @import { FrameRef } from "./api-types.js" */

/**
 * Minimal job shape this module needs to build a seek target — a subset of
 * JobDetails, kept loose so callers can pass whatever job-like object they
 * already have on hand.
 *
 * @typedef {{ video_id?: string | null, url?: string | null, kind?: string | null }} SeekJob
 */

/**
 * Build the seek href for a frame thumbnail: the canonical YouTube URL with
 * `&t=Ns`, or the job's media page URL with `#t=Ns`. Same rule
 * markdown.js/transcript.js use for `[MM:SS]` timecode links.
 *
 * @param {SeekJob | null | undefined} job
 * @param {number} seconds
 * @returns {string}
 */
export function frameSeekHref(job, seconds) {
  if (!job) return "#";
  const videoId = resolveVideoId(job);
  if (videoId) {
    return `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}&t=${seconds}s`;
  }
  if (job.kind === "media" && job.url) {
    const u = job.url;
    const i = u.indexOf("#");
    return `${i === -1 ? u : u.slice(0, i)}#t=${seconds}`;
  }
  return "#";
}

/**
 * Tag an anchor with the same dataset keys app.js's delegated click handler
 * (`a[data-tldr-seconds]`) expects, so clicking it seeks exactly like a
 * `[MM:SS]` timecode link.
 *
 * @param {HTMLAnchorElement} a
 * @param {SeekJob | null | undefined} job
 * @param {number} seconds
 */
export function setFrameTimecodeTarget(a, job, seconds) {
  a.dataset.tldrSeconds = String(seconds);
  if (!job) return;
  const videoId = resolveVideoId(job);
  if (videoId) {
    a.dataset.tldrVideoId = videoId;
  } else if (job.kind === "media" && job.url) {
    a.dataset.tldrMediaPageUrl = job.url;
  }
}

/**
 * Group consecutive FrameRefs that describe the SAME moment (identical
 * timecode + phrase) — the on-demand "look" affordance can hand back
 * several frames for one moment (see app.js's FRAMES_SHOWN_BY_CATEGORY),
 * and those describe one moment, not several: one caption belongs under
 * the whole group, not repeated under every image in it. The QA LOOK step
 * only ever sends one FrameRef per moment, so every group there is size 1
 * — same per-image-caption look as before this grouping existed.
 *
 * @param {FrameRef[]} frameRefs
 * @returns {FrameRef[][]}
 */
function _groupByMoment(frameRefs) {
  /** @type {FrameRef[][]} */
  const groups = [];
  for (const ref of frameRefs) {
    const last = groups[groups.length - 1];
    if (last && last[0].timecode === ref.timecode && last[0].phrase === ref.phrase) {
      last.push(ref);
    } else {
      groups.push([ref]);
    }
  }
  return groups;
}

/**
 * Build a `.chat-frame-row` element for `frameRefs`. Does NOT insert it
 * anywhere — the caller decides placement (after a chat bubble, after a
 * summary line's "look" affordance, …).
 *
 * Frames from the same moment (see `_groupByMoment`) are wrapped in one
 * `.chat-frame-group` with a SINGLE caption underneath every image in it,
 * not one per image.
 *
 * @param {SeekJob | null | undefined} job
 * @param {FrameRef[]} frameRefs
 * @param {string} baseUrl daemon base URL (frame_url is daemon-rooted)
 * @returns {HTMLElement}
 */
export function buildFrameRow(job, frameRefs, baseUrl) {
  const row = document.createElement("div");
  row.className = "chat-frame-row";
  for (const group of _groupByMoment(frameRefs)) {
    const groupEl = document.createElement("div");
    groupEl.className = "chat-frame-group";
    if (group.length > 1) groupEl.classList.add("chat-frame-group--multi");

    const imagesEl = document.createElement("div");
    imagesEl.className = "chat-frame-group-images";
    for (const ref of group) {
      const a = document.createElement("a");
      a.className = "chat-frame-thumb";
      a.href = frameSeekHref(job, ref.seconds);
      a.target = "_blank";
      a.rel = "noopener";
      setFrameTimecodeTarget(a, job, ref.seconds);

      const img = document.createElement("img");
      img.src = `${baseUrl}${ref.frame_url}`;
      img.alt = ref.phrase;
      img.loading = "lazy";
      a.appendChild(img);

      imagesEl.appendChild(a);
    }
    groupEl.appendChild(imagesEl);

    const first = group[0];
    const cap = document.createElement("span");
    cap.className = "chat-frame-caption";
    cap.textContent = `[${first.timecode}] ${first.phrase}`;
    groupEl.appendChild(cap);

    row.appendChild(groupEl);
  }
  return row;
}
