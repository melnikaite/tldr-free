// Unit tests for extension/src/lib/url.js — run with
// `task test:extension` (plain `node --test`, no dependencies).
import { test } from "node:test";
import assert from "node:assert/strict";

import { normalizeUrl, pickTabForUrl, sameMediaPage } from "../src/lib/url.js";

test("normalizeUrl drops plain fragments, keeps route-like ones", () => {
  assert.equal(normalizeUrl("https://a.example/p#section"), "https://a.example/p");
  assert.equal(normalizeUrl("https://a.example/p#t=754"), "https://a.example/p");
  assert.equal(normalizeUrl("https://app.example/#/inbox/42"), "https://app.example/#/inbox/42");
  assert.equal(normalizeUrl("https://app.example/#!/item/1"), "https://app.example/#!/item/1");
});

test("normalizeUrl keeps ref generally, strips it on listed hosts", () => {
  assert.equal(
    normalizeUrl("https://github.com/o/r/blob/x.md?ref=main&utm_source=tw"),
    "https://github.com/o/r/blob/x.md?ref=main",
  );
  assert.equal(
    normalizeUrl("https://www.producthunt.com/posts/foo?ref=header_nav"),
    "https://www.producthunt.com/posts/foo",
  );
  assert.equal(normalizeUrl("https://a.example/p?ref_src=twsrc&id=1"), "https://a.example/p?id=1");
});

test("normalizeUrl YouTube unchanged: video id only, fragment dropped", () => {
  const want = "https://www.youtube.com/watch?v=abcdefghijk";
  assert.equal(normalizeUrl("https://m.youtube.com/watch?v=abcdefghijk&t=5#/x"), want);
  assert.equal(normalizeUrl("https://youtu.be/abcdefghijk?t=3"), want);
  assert.equal(normalizeUrl("https://www.youtube.com/shorts/abcdefghijk"), want);
});

test("sameMediaPage matches YouTube across host/path forms", () => {
  const job = "https://www.youtube.com/watch?v=abcdefghijk";
  for (const t of [
    "https://m.youtube.com/watch?v=abcdefghijk&t=10s",
    "https://music.youtube.com/watch?v=abcdefghijk",
    "https://www.youtube.com/live/abcdefghijk",
    "https://youtu.be/abcdefghijk",
  ]) assert.ok(sameMediaPage(t, job), t);
  assert.ok(!sameMediaPage("https://www.youtube.com/watch?v=zzzzzzzzzzz", job));
  assert.ok(!sameMediaPage("https://www.youtube.com/feed/subscriptions", job));
});

test("sameMediaPage compares other pages via normalizeUrl", () => {
  assert.ok(sameMediaPage("https://a.example/p?utm_source=x#c", "https://a.example/p"));
  assert.ok(!sameMediaPage("https://a.example/other", "https://a.example/p"));
  assert.ok(!sameMediaPage(undefined, "https://a.example/p"));
});

test("pickTabForUrl prefers the panel's own window", () => {
  const tabs = [
    { id: 1, windowId: 10, url: "https://a.example/p" },
    { id: 2, windowId: 20, url: "https://a.example/p?utm_source=x" },
    { id: 3, windowId: 20, url: "https://b.example/" },
  ];
  assert.equal(pickTabForUrl(tabs, "https://a.example/p", 20)?.id, 2);
  assert.equal(pickTabForUrl(tabs, "https://a.example/p", 99)?.id, 1);
  assert.equal(pickTabForUrl(tabs, "https://a.example/p", null)?.id, 1);
  assert.equal(pickTabForUrl(tabs, "https://c.example/", 20), undefined);
});
