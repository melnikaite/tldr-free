// Unit tests for extension/src/lib/page-videos.js — run with
// `task test:extension` (plain `node --test`, no dependencies).
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  buildChips,
  chipKeyForJob,
  embedKey,
  formatDuration,
  isEmbedUrl,
  isVideoKey,
  scopeSniffed,
  videoKey,
} from "../src/lib/page-videos.js";

const PAGE = "https://kino.example/film/123?utm_source=x#comments";

/** @param {number} frameId @param {object} r */
const frame = (frameId, r) => ({
  frameId,
  result: { frameUrl: "https://kino.example/film/123", isTop: frameId === 0, media: [], embeds: [], ...r },
});
const media = (o) => ({ kind: "video", src: null, blob: false, duration: 0, paused: true, label: "", ...o });

test("embed whitelist includes YouTube embeds", () => {
  assert.ok(isEmbedUrl("https://www.youtube-nocookie.com/embed/Hz7FVOu40_Y"));
  assert.ok(isEmbedUrl("https://www.youtube.com/embed/Hz7FVOu40_Y?autoplay=1"));
  assert.ok(isEmbedUrl("https://player.vimeo.com/video/123"));
  assert.ok(!isEmbedUrl("https://ads.example/embed/xyz"));
  assert.equal(
    embedKey("https://www.youtube-nocookie.com/embed/Hz7FVOu40_Y?rel=0"),
    "https://www.youtube.com/watch?v=Hz7FVOu40_Y",
  );
});

test("video key keeps the fragment, normalizes the page part", () => {
  const k = videoKey(PAGE, 5400.4);
  assert.equal(k, "https://kino.example/film/123#tldr-video=5400");
  assert.ok(isVideoKey(k));
  assert.ok(!isVideoKey("https://kino.example/film/123"));
});

test("YouTube embed → youtube chip; its own frame enriches it, no second chip", () => {
  const chips = buildChips({
    pageUrl: PAGE,
    frames: [
      frame(0, { embeds: [{ src: "https://www.youtube-nocookie.com/embed/Hz7FVOu40_Y", label: "Trailer" }] }),
      frame(5, {
        frameUrl: "https://www.youtube-nocookie.com/embed/Hz7FVOu40_Y",
        media: [media({ blob: true, duration: 150, paused: false })],
      }),
    ],
  });
  assert.equal(chips.length, 1);
  assert.equal(chips[0].type, "youtube");
  assert.equal(chips[0].key, "https://www.youtube.com/watch?v=Hz7FVOu40_Y");
  assert.equal(chips[0].mediaUrl, null);
  assert.equal(chips[0].frameId, 5);
  assert.equal(chips[0].duration, 150);
  assert.equal(chips[0].playing, true);
  assert.equal(chips[0].label, "Trailer");
});

test("embed with unknown duration is kept; known short one is dropped", () => {
  const chips = buildChips({
    pageUrl: PAGE,
    frames: [
      frame(0, { embeds: [
        { src: "https://player.vimeo.com/video/1", label: "" },
        { src: "https://player.vimeo.com/video/2", label: "" },
      ] }),
      frame(3, { frameUrl: "https://player.vimeo.com/video/2", media: [media({ duration: 20 })] }),
    ],
  });
  assert.deepEqual(chips.map((c) => c.key), ["https://player.vimeo.com/video/1"]);
  assert.equal(chips[0].type, "embed");
  assert.equal(chips[0].mediaUrl, "https://player.vimeo.com/video/1");
});

test("native media: needs http src and known duration >= 60, dedups by key", () => {
  const chips = buildChips({
    pageUrl: PAGE,
    frames: [
      frame(0, { media: [
        media({ src: "https://cdn.example/film.mp4", duration: 5400 }),
        media({ src: "https://cdn.example/film-hd.mp4", duration: 5400.2, paused: false }),
        media({ src: "https://cdn.example/ad.mp4", duration: 30 }),
        media({ src: "https://cdn.example/unknown.mp4", duration: 0 }),
        media({ kind: "audio", src: "https://cdn.example/pod.mp3", duration: 1800, label: "Episode 1" }),
      ] }),
    ],
  });
  assert.deepEqual(chips.map((c) => c.key), [
    "https://kino.example/film/123#tldr-video=5400",
    "https://kino.example/film/123#tldr-video=1800",
  ]);
  assert.equal(chips[0].mediaUrl, "https://cdn.example/film.mp4");
  assert.equal(chips[0].playing, true);
  assert.equal(chips[1].label, "Episode 1");
});

test("blob player becomes a stream chip only with a manifest from its frame's origin", () => {
  const player = frame(7, {
    frameUrl: "https://player.cdn.example/e/abc?token=1",
    media: [media({ blob: true, duration: 6000 })],
  });
  const none = buildChips({ pageUrl: PAGE, frames: [frame(0, {}), player] });
  assert.equal(none.length, 0);
  const other = buildChips({
    pageUrl: PAGE,
    frames: [player],
    sniffed: [{ kind: "hls", frameUrl: "https://elsewhere.example/x" }],
  });
  assert.equal(other.length, 0);
  const chips = buildChips({
    pageUrl: PAGE,
    frames: [player],
    sniffed: [
      { kind: "subtitle", frameUrl: "https://player.cdn.example/e/abc" },
      { kind: "hls", frameUrl: "https://player.cdn.example" },
    ],
  });
  assert.equal(chips.length, 1);
  assert.equal(chips[0].type, "stream");
  assert.equal(chips[0].key, "https://kino.example/film/123#tldr-video=6000");
  assert.equal(chips[0].frameId, 7);
  assert.equal(chips[0].mediaUrl, null);
  // Unknown duration → no chip yet.
  const early = buildChips({
    pageUrl: PAGE,
    frames: [frame(7, { frameUrl: "https://player.cdn.example/e/abc", media: [media({ blob: true })] })],
    sniffed: [{ kind: "hls", frameUrl: "https://player.cdn.example" }],
  });
  assert.equal(early.length, 0);
});

test("scopeSniffed matches by origin", () => {
  const list = [
    { url: "a", frameUrl: "https://p.example/e/1" },
    { url: "b", frameUrl: "https://p.example" },
    { url: "c", frameUrl: "https://q.example/" },
    { url: "d", frameUrl: null },
  ];
  assert.deepEqual(scopeSniffed(list, "https://p.example/e/2?x=1").map((e) => e.url), ["a", "b"]);
  assert.deepEqual(scopeSniffed(list, null), []);
});

test("labels are disambiguated", () => {
  const chips = buildChips({
    pageUrl: PAGE,
    frames: [frame(0, { media: [
      media({ src: "https://c/a.mp4", duration: 100 }),
      media({ src: "https://c/b.mp4", duration: 200 }),
    ] })],
  });
  assert.deepEqual(chips.map((c) => c.label), ["Video", "Video 2"]);
});

test("chipKeyForJob: page job, video job, foreign job", () => {
  const page = "https://kino.example/film/123";
  const chips = [{ key: `${page}#tldr-video=60` }];
  assert.equal(chipKeyForJob(page, page, /** @type {any} */ (chips)), "page");
  assert.equal(chipKeyForJob(`${page}#tldr-video=60`, page, /** @type {any} */ (chips)), `${page}#tldr-video=60`);
  assert.equal(chipKeyForJob("https://other/", page, /** @type {any} */ (chips)), null);
  assert.equal(chipKeyForJob(null, page, /** @type {any} */ (chips)), null);
});

test("formatDuration", () => {
  assert.equal(formatDuration(65), "1:05");
  assert.equal(formatDuration(5400), "1:30:00");
  assert.equal(formatDuration(null), "");
  assert.equal(formatDuration(NaN), "");
});
