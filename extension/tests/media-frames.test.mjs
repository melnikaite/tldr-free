// Unit tests for extension/src/lib/media-frames.js — run with
// `task test:extension` (plain `node --test`, no dependencies).
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  compareMedia,
  cuesFromText,
  canAutoRebind,
  durationMatches,
  expectedRange,
  pickBestFrame,
} from "../src/lib/media-frames.js";

const m = (duration, area = 0, paused = true) => ({ duration, area, paused });

test("longest finite duration wins over a bigger ad", () => {
  const best = pickBestFrame([
    { frameId: 0, result: m(30, 1e6, false) },
    { frameId: 7, result: m(5400, 1e5, true) },
  ]);
  assert.equal(best?.frameId, 7);
});

test("NaN / Infinity / 0 durations rank as unknown", () => {
  assert.ok(compareMedia(m(10), m(NaN)) < 0);
  assert.ok(compareMedia(m(10), m(Infinity)) < 0);
  assert.ok(compareMedia(m(0, 50), m(NaN, 10)) < 0); // falls through to area
});

test("ties break on area, then playing", () => {
  assert.ok(compareMedia(m(100, 20), m(100, 10)) < 0);
  assert.ok(compareMedia(m(100, 10, false), m(100, 10, true)) < 0);
  assert.equal(compareMedia(m(100, 10), m(100, 10)), 0);
});

test("frames without media are ignored; earlier frame keeps ties", () => {
  assert.equal(pickBestFrame([{ frameId: 0, result: null }]), null);
  assert.equal(pickBestFrame([]), null);
  const best = pickBestFrame([
    { frameId: 0, result: null },
    { frameId: 3, result: m(60) },
    { frameId: 4, result: m(60) },
  ]);
  assert.equal(best?.frameId, 3);
});

test("cuesFromText spans marker to marker, last +60s", () => {
  assert.deepEqual(cuesFromText("[00:01] a\n[1:02:03] b\nnoise"), [
    { start: 1, end: 3723, text: "a" },
    { start: 3723, end: 3783, text: "b" },
  ]);
  assert.deepEqual(cuesFromText(""), []);
});

test("expectedRange: job duration is ±3 s, last marker a loose lower bound", () => {
  assert.deepEqual(expectedRange(100, 50), { min: 97, max: 103 });
  assert.deepEqual(expectedRange(null, 50), { min: 40, max: 0 });
  assert.equal(expectedRange(null, 0), null);
  assert.equal(expectedRange(0, undefined), null);
});

test("durationMatches: unknown never matches, max 0 is unbounded", () => {
  const r = expectedRange(5400, 0);
  assert.ok(durationMatches(5402.5, r));
  assert.ok(!durationMatches(30, r));
  assert.ok(!durationMatches(5410, r)); // another rip of the same film
  assert.ok(!durationMatches(NaN, r));
  assert.ok(!durationMatches(Infinity, r));
  assert.ok(durationMatches(1e5, { min: 40, max: 0 }));
  assert.ok(!durationMatches(100, null));
});

test("known range: matching element beats a longer non-matching one", () => {
  const range = expectedRange(5400, 0);
  const best = pickBestFrame(
    [
      { frameId: 0, result: m(30, 1e6, false) }, // pre-roll ad
      { frameId: 2, result: m(36000, 1e6, false) }, // unrelated long stream
      { frameId: 5, result: m(5401, 1e5, true) }, // the film
      { frameId: 6, result: m(5420, 1e6, false) }, // another rip
    ],
    range,
  );
  assert.equal(best?.frameId, 5);
});

test("known range, nothing matches: falls back to plain ranking", () => {
  const best = pickBestFrame(
    [
      { frameId: 0, result: m(30, 1e6, false) },
      { frameId: 1, result: m(NaN, 1e6, true) },
    ],
    expectedRange(5400, 0),
  );
  assert.equal(best?.frameId, 0);
});

test("canAutoRebind: known duration, matching, same frame only", () => {
  const range = expectedRange(5400, 0);
  const info = { isVideo: true, captionsBound: false, matches: true, frameUrl: "https://p/a" };
  assert.ok(canAutoRebind(info, range, "https://p/a"));
  assert.ok(canAutoRebind(info, range, null)); // nothing bound yet since the explicit pick
  assert.ok(!canAutoRebind(info, range, "https://p/b")); // other player iframe
  assert.ok(!canAutoRebind(info, null, "https://p/a")); // unknown duration
  assert.ok(!canAutoRebind({ ...info, matches: false }, range, null));
  assert.ok(!canAutoRebind({ ...info, captionsBound: true }, range, null));
  assert.ok(!canAutoRebind({ ...info, isVideo: false }, range, null));
});
