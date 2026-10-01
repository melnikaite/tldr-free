// Unit tests for extension/src/lib/text-search.js — run with
// `task test:extension` (plain `node --test`, no dependencies).
import { test } from "node:test";
import assert from "node:assert/strict";

import { findMatches, foldQuery, foldWithMap } from "../src/lib/text-search.js";

/**
 * @param {string} text
 * @param {string} query
 */
function hits(text, query) {
  return findMatches(foldWithMap(text), foldQuery(query)).map(([s, e]) =>
    text.slice(s, e),
  );
}

test("folds case and accents", () => {
  assert.equal(foldWithMap("Ärger Öl Über").folded, "arger ol uber");
  assert.deepEqual(hits("Ärger und Über", "arger"), ["Ärger"]);
  assert.deepEqual(hits("Ärger und Über", "ÜBER"), ["Über"]);
});

test("ß folds to ss and maps back onto the single original char", () => {
  const f = foldWithMap("Straße");
  assert.equal(f.folded, "strasse");
  assert.equal(f.folded.length, 7);
  assert.deepEqual(hits("Die Straße hier", "strasse"), ["Straße"]);
  assert.deepEqual(hits("Die Straße hier", "straße"), ["Straße"]);
  // A match that ends inside the expanded ß still covers the whole char.
  assert.deepEqual(hits("Straße", "stras"), ["Straß"]);
});

test("decomposed input (base + combining mark) maps to both code units", () => {
  const text = "café noir";
  assert.deepEqual(hits(text, "café"), ["café"]);
});

test("offsets after a length-changing char stay correct", () => {
  const text = "ßß abc";
  const [[s, e]] = findMatches(foldWithMap(text), foldQuery("abc"));
  assert.equal(text.slice(s, e), "abc");
  assert.equal(s, 3);
});

test("multiple, non-overlapping matches; empty query matches nothing", () => {
  assert.deepEqual(hits("aaaa", "aa"), ["aa", "aa"]);
  assert.deepEqual(hits("Foo foo FOO", "foo"), ["Foo", "foo", "FOO"]);
  assert.deepEqual(hits("anything", "   "), []);
  assert.deepEqual(hits("anything", "zzz"), []);
});

test("astral characters keep offsets in UTF-16 units", () => {
  const text = "😀 hello";
  const [[s, e]] = findMatches(foldWithMap(text), foldQuery("hello"));
  assert.equal(text.slice(s, e), "hello");
});
