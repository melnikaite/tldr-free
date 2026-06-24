// URL normalization for tab matching and storage.
//
// Why this exists:
//   The user expects "same page" to mean "same article". Browsers add stuff
//   to URLs that doesn't change identity:
//   - Tracking params (?utm_source=, ?fbclid=, etc.) on regular pages.
//   - Hash fragments (#section).
//   - On YouTube specifically: clicking a [12:34] timecode opens the same
//     video with `&t=754s` appended, which would otherwise look like a
//     different page to our daemon.
//
// Strategy: normalize before sending to the daemon AND before looking up
// "is the current tab summarized?". Storage and lookup go through the same
// canonical form, so they match.
//
// The helper is intentionally conservative — when in doubt, leave the URL
// alone. Bad output (matching unrelated pages together) is worse than
// missing a match.

const TRACKING_PARAMS = new Set([
  "utm_source",
  "utm_medium",
  "utm_campaign",
  "utm_term",
  "utm_content",
  "fbclid",
  "gclid",
  "msclkid",
  "yclid",
  "ref",
  "ref_src",
  "_ga",
]);

const YT_HOST_RE =
  /^(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be|youtube-nocookie\.com)$/i;

/**
 * Resolve the YouTube video id for a job. Prefers `job.video_id` (the
 * canonical value the daemon writes mid-pipeline) and falls back to parsing
 * the URL — so timecode links work from the very first delta event, before
 * the daemon has finished filling in `video_id`.
 *
 * One helper, one place where the policy lives. Callers should never write
 * `job.video_id || extractYoutubeVideoId(job.url)` directly.
 *
 * @param {{ video_id?: string | null, url?: string | null } | null | undefined} job
 * @returns {string | null}
 */
export function resolveVideoId(job) {
  if (!job) return null;
  return job.video_id || extractYoutubeVideoId(job.url);
}

/**
 * Extract a YouTube video id from a URL, or null if the URL isn't a
 * recognisable YouTube video page. Pure function — does not touch storage
 * or the daemon, so it's safe to call during render without race conditions
 * (the alternative — reading `job.video_id` — depends on the daemon having
 * already written that field, which happens mid-pipeline).
 *
 * @param {string | null | undefined} rawUrl
 * @returns {string | null}
 */
export function extractYoutubeVideoId(rawUrl) {
  if (typeof rawUrl !== "string" || !rawUrl) return null;
  let u;
  try {
    u = new URL(rawUrl);
  } catch {
    return null;
  }
  if (!YT_HOST_RE.test(u.hostname)) return null;
  const fromQuery = u.searchParams.get("v");
  if (fromQuery) return fromQuery;
  const m = u.pathname.match(/^\/(?:embed|shorts|v|live)\/([\w-]{6,})/);
  if (m) return m[1];
  if (/^youtu\.be$/i.test(u.hostname)) {
    const tail = u.pathname.replace(/^\//, "").split("/")[0];
    if (tail) return tail;
  }
  return null;
}

/**
 * Return a canonical form of the URL for matching/storage.
 * Returns the input unchanged if it's not parseable as a URL.
 *
 * @param {string} rawUrl
 * @returns {string}
 */
export function normalizeUrl(rawUrl) {
  if (typeof rawUrl !== "string" || !rawUrl) return rawUrl;
  let u;
  try {
    u = new URL(rawUrl);
  } catch {
    return rawUrl;
  }

  u.hash = "";
  u.hostname = u.hostname.toLowerCase();

  if (YT_HOST_RE.test(u.hostname)) {
    // YouTube: identity is the video id. Anything else (t=, list=, index=,
    // ab_channel=, pp=, etc.) is noise.
    const videoId = extractYoutubeVideoId(u.toString());
    if (videoId) {
      u.hostname = "www.youtube.com";
      u.pathname = "/watch";
      u.search = `?v=${videoId}`;
      return u.toString();
    }
    // Not a recognisable video URL (e.g. /feed/subscriptions). Leave it.
    return u.toString();
  }

  for (const key of [...u.searchParams.keys()]) {
    if (TRACKING_PARAMS.has(key.toLowerCase())) {
      u.searchParams.delete(key);
    }
  }
  return u.toString();
}
