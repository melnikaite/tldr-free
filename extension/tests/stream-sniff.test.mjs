// Unit tests for extension/src/lib/stream-sniff.js — run with
// `task test:extension` (plain `node --test`, no dependencies).
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  classifyStream,
  headerValue,
  looksLikeMaster,
  toSniffedStreams,
  mediaHeadersFor,
  pickSniffedMedia,
  upsertStream,
  recordFrame,
  lookupFrameTabs,
  pruneFrameTab,
  debugEntry,
  pushDebug,
  summarizeDebug,
} from "../src/lib/stream-sniff.js";

/** @param {Partial<import("../src/lib/stream-sniff.js").SniffedStream>} o */
function entry(o) {
  return {
    url: "https://cdn.example/x.m3u8",
    kind: "hls",
    frameUrl: null,
    referer: null,
    origin: null,
    userAgent: null,
    firstSeen: 1,
    lastSeen: 1,
    ...o,
  };
}

test("classifies by extension first", () => {
  assert.equal(classifyStream("https://a.b/v/master.m3u8?t=1"), "hls");
  assert.equal(classifyStream("https://a.b/manifest.mpd"), "dash");
  assert.equal(classifyStream("https://a.b/subs/ru.vtt"), "subtitle");
  assert.equal(classifyStream("https://a.b/subs/en.SRT"), "subtitle");
  // Segments are never recorded, even with a manifest-ish content type.
  assert.equal(classifyStream("https://a.b/seg-1.ts", "application/vnd.apple.mpegurl"), null);
  assert.equal(classifyStream("https://a.b/chunk.m4s"), null);
});

test("falls back to Content-Type for extension-less URLs", () => {
  assert.equal(classifyStream("https://a.b/play/abc", "application/x-mpegURL; charset=utf-8"), "hls");
  assert.equal(classifyStream("https://a.b/play/abc", "audio/mpegurl"), "hls");
  assert.equal(classifyStream("https://a.b/play/abc", "application/dash+xml"), "dash");
  assert.equal(classifyStream("https://a.b/play/abc", "text/vtt"), "subtitle");
  assert.equal(classifyStream("https://a.b/play/abc", "video/mp2t"), null);
  assert.equal(classifyStream("https://a.b/play/abc", null), null);
  assert.equal(classifyStream("blob:https://a.b/123", "application/x-mpegurl"), null);
});

test("headerValue is case-insensitive", () => {
  const h = [{ name: "referer", value: "https://p/" }, { name: "Origin", value: "https://p" }];
  assert.equal(headerValue(h, "Referer"), "https://p/");
  assert.equal(headerValue(h, "ORIGIN"), "https://p");
  assert.equal(headerValue(h, "User-Agent"), null);
  assert.equal(headerValue(undefined, "Referer"), null);
});

test("upsertStream dedupes by URL, keeps order, caps oldest", () => {
  let list = [];
  list = upsertStream(list, entry({ url: "u1", firstSeen: 1, lastSeen: 1 }));
  list = upsertStream(list, entry({ url: "u2", firstSeen: 2, lastSeen: 2 }));
  list = upsertStream(list, entry({ url: "u1", firstSeen: 3, lastSeen: 3, referer: "r" }));
  assert.deepEqual(list.map((e) => e.url), ["u1", "u2"]);
  assert.equal(list[0].firstSeen, 1);
  assert.equal(list[0].lastSeen, 3);
  assert.equal(list[0].referer, "r");
  for (let i = 0; i < 5; i++) list = upsertStream(list, entry({ url: `n${i}` }), 3);
  assert.deepEqual(list.map((e) => e.url), ["n2", "n3", "n4"]);
});

test("mediaHeadersFor uses sent headers, falls back to the frame origin", () => {
  assert.deepEqual(
    mediaHeadersFor(entry({ referer: "https://p.x/embed/1", origin: "https://p.x", userAgent: "UA" })),
    { Referer: "https://p.x/embed/1", Origin: "https://p.x", "User-Agent": "UA" },
  );
  assert.deepEqual(mediaHeadersFor(entry({ frameUrl: "https://p.x/embed/1?a=b" })), {
    Referer: "https://p.x/",
    Origin: "https://p.x",
  });
  assert.equal(mediaHeadersFor(entry({})), null);
});

test("pickSniffedMedia: latest master-looking manifest wins, subtitles excluded", () => {
  assert.equal(pickSniffedMedia([]), null);
  assert.equal(pickSniffedMedia([entry({ kind: "subtitle", url: "s.vtt" })]), null);
  const picked = pickSniffedMedia([
    entry({ url: "https://c/ru/master.m3u8", firstSeen: 1, lastSeen: 1 }),
    entry({ url: "https://c/ru/360/index.m3u8", firstSeen: 2, lastSeen: 3 }),
    entry({ url: "https://c/subs.vtt", kind: "subtitle", firstSeen: 1 }),
    entry({ url: "https://c/de/master.m3u8", firstSeen: 5, lastSeen: 5, frameUrl: "https://player.y/e/1" }),
  ]);
  assert.equal(picked?.mediaUrl, "https://c/de/master.m3u8");
  assert.deepEqual(picked?.mediaHeaders, { Referer: "https://player.y/", Origin: "https://player.y" });
  assert.deepEqual(picked?.altCandidates, [
    { mediaUrl: "https://c/ru/master.m3u8", kind: "video", label: "Stream 2 (HLS)" },
    { mediaUrl: "https://c/ru/360/index.m3u8", kind: "video", label: "Stream 3 (HLS)" },
  ]);
});

test("pickSniffedMedia: no master-looking name falls back to the earliest", () => {
  const picked = pickSniffedMedia([
    entry({ url: "https://c/b/x.m3u8", firstSeen: 5, lastSeen: 9 }),
    entry({ url: "https://c/a/y.m3u8", firstSeen: 2, lastSeen: 2 }),
  ]);
  assert.equal(picked?.mediaUrl, "https://c/a/y.m3u8");
  assert.equal(looksLikeMaster("https://c/v/playlist.m3u8?t=1"), true);
  assert.equal(looksLikeMaster("https://c/v/m.mpd"), true);
  assert.equal(looksLikeMaster("https://c/v/index-a1.m3u8"), false);
});

test("subtitle segments don't evict manifests", () => {
  let list = [entry({ url: "https://c/master.m3u8" })];
  for (let i = 0; i < 10; i++) {
    list = upsertStream(list, entry({ url: `https://c/s${i}.vtt`, kind: "subtitle" }), 3);
  }
  assert.deepEqual(list.map((e) => e.url), [
    "https://c/master.m3u8", "https://c/s7.vtt", "https://c/s8.vtt", "https://c/s9.vtt",
  ]);
});

test("toSniffedStreams: snake_case, http(s) only, most recent kept", () => {
  const out = toSniffedStreams([
    entry({ url: "https://c/a.m3u8", firstSeen: 1, lastSeen: 4 }),
    entry({ url: "blob:https://c/x", firstSeen: 1, lastSeen: 5 }),
    entry({ url: "https://c/b.vtt", kind: "subtitle", firstSeen: 2, lastSeen: 3 }),
  ], 1);
  assert.deepEqual(out, [{ url: "https://c/a.m3u8", kind: "hls", first_seen: 1, last_seen: 4 }]);
});

test("recordFrame + lookupFrameTabs: by origin, latest tab first", () => {
  let m = {};
  m = recordFrame(m, "https://player.example/embed/1?t=x", 5, 1);
  m = recordFrame(m, "https://player.example/embed/2", 7, 2);
  assert.deepEqual(lookupFrameTabs(m, "https://player.example"), [7, 5]);
  // any URL on the origin works as the key
  assert.deepEqual(lookupFrameTabs(m, "https://player.example/sw.js"), [7, 5]);
  // re-seeing tab 5 moves it to the front without duplicating
  m = recordFrame(m, "https://player.example/embed/3", 5, 3);
  assert.deepEqual(lookupFrameTabs(m, "https://player.example"), [5, 7]);
  assert.deepEqual(lookupFrameTabs(m, "https://other.example"), []);
  assert.deepEqual(lookupFrameTabs(m, null), []);
});

test("recordFrame ignores non-http frames and negative tab ids", () => {
  const m = {};
  assert.equal(recordFrame(m, "about:blank", 1, 1), m);
  assert.equal(recordFrame(m, "chrome-extension://abc/x.html", 1, 1), m);
  assert.equal(recordFrame(m, "https://a.example/", -1, 1), m);
});

test("pruneFrameTab drops only that tab and empty origins", () => {
  let m = {};
  m = recordFrame(m, "https://a.example/", 1, 1);
  m = recordFrame(m, "https://a.example/", 2, 2);
  m = recordFrame(m, "https://b.example/", 1, 3);
  m = pruneFrameTab(m, 1);
  assert.deepEqual(lookupFrameTabs(m, "https://a.example"), [2]);
  assert.deepEqual(Object.keys(m), ["https://a.example"]);
});

test("debugEntry strips the query and skips noise", () => {
  const e = debugEntry(
    "https://cdn.example/a/very/long/path/playlist?token=SECRET",
    "text/plain; charset=utf-8",
    200,
    true,
    9,
  );
  assert.deepEqual(e, {
    host: "cdn.example",
    pathTail: "/a/very/long/path/playlist",
    ext: "",
    contentType: "text/plain",
    status: 200,
    viaSW: true,
    ts: 9,
  });
  assert.ok(!JSON.stringify(e).includes("SECRET"));
  const long = debugEntry(`https://x.example/${"a".repeat(100)}.txt`, null, 404, false, 1);
  assert.equal(long?.pathTail.length, 60);
  assert.equal(debugEntry("https://x.example/a.png", null, 200, false, 1), null);
  assert.equal(debugEntry("https://x.example/app.js?v=1", null, 200, false, 1), null);
  assert.equal(debugEntry("https://x.example/f", "font/woff2", 200, false, 1), null);
  assert.equal(debugEntry("https://x.example/s", "text/css", 200, false, 1), null);
  assert.equal(debugEntry("data:text/plain,x", null, 200, false, 1), null);
});

test("pushDebug keeps the newest cap entries; summarizeDebug renders lines", () => {
  let ring = [];
  for (let i = 0; i < 5; i++) {
    ring = pushDebug(ring, /** @type {any} */ ({ host: "h", pathTail: `/${i}`, ext: "", contentType: null, status: 200, viaSW: i === 4, ts: i }), 3);
  }
  assert.deepEqual(ring.map((e) => e.ts), [2, 3, 4]);
  const lines = summarizeDebug(ring);
  assert.equal(lines.length, 3);
  assert.match(lines[2], /\[sw\] h\/4/);
});
