// Pure helpers for the side panel's in-transcript search (sidepanel/
// transcript.js). Kept DOM-free so they can be unit-tested with plain
// `node --test` (extension/tests/).
//
// Matching is case-insensitive and accent-folding: every character is
// NFD-decomposed, combining marks are stripped, it is lower-cased, and
// ß folds to "ss". Because folding can change the length (ß → 2 chars,
// a precomposed char + mark → 1 char), `foldWithMap` keeps, for every
// folded character, the [start, end) range of the original character it
// came from — so a match found in the folded string maps back onto the
// exact original characters to highlight.

/**
 * @typedef {object} FoldedText
 * @property {string} folded  - the folded string that is searched
 * @property {number[]} starts  - starts[i] = original offset of the char
 *   that produced folded[i]
 * @property {number[]} ends  - ends[i] = original end offset (exclusive)
 *   of that same char
 */

/**
 * Fold one original character (a full code point).
 * @param {string} ch
 * @returns {string}
 */
function foldChar(ch) {
  return ch
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .replace(/ß/g, "ss");
}

/**
 * Fold `text` for searching and keep the folded → original offset map.
 * @param {string} text
 * @returns {FoldedText}
 */
export function foldWithMap(text) {
  let folded = "";
  /** @type {number[]} */
  const starts = [];
  /** @type {number[]} */
  const ends = [];
  let i = 0;
  for (const ch of text) {
    const end = i + ch.length;
    const f = foldChar(ch);
    if (!f) {
      // A char that folds to nothing (a standalone combining mark in
      // already-decomposed text) belongs to the char before it — extend
      // that char's end so a highlight doesn't cut the mark off.
      for (let k = ends.length - 1; k >= 0 && ends[k] === i; k--) ends[k] = end;
    }
    for (let k = 0; k < f.length; k++) {
      starts.push(i);
      ends.push(end);
    }
    folded += f;
    i = end;
  }
  return { folded, starts, ends };
}

/**
 * Fold a query string (no offset map needed). Surrounding whitespace is
 * trimmed so a stray space doesn't turn "foo " into a no-match.
 * @param {string} query
 * @returns {string}
 */
export function foldQuery(query) {
  return foldWithMap(query.trim()).folded;
}

/**
 * Find all non-overlapping occurrences of `foldedQuery` in `haystack`,
 * returned as [start, end) ranges in the ORIGINAL string's offsets.
 * An empty query yields no matches.
 * @param {FoldedText} haystack
 * @param {string} foldedQuery  - already folded via `foldQuery`
 * @returns {Array<[number, number]>}
 */
export function findMatches(haystack, foldedQuery) {
  /** @type {Array<[number, number]>} */
  const out = [];
  if (!foldedQuery) return out;
  const { folded, starts, ends } = haystack;
  let from = 0;
  for (;;) {
    const idx = folded.indexOf(foldedQuery, from);
    if (idx < 0) break;
    const last = idx + foldedQuery.length - 1;
    const start = starts[idx];
    const end = ends[last];
    // Two folded chars from the same original char (ß → "ss") can make a
    // match start mid-character; the mapped range then overlaps the
    // previous one — drop it rather than double-highlight.
    if (out.length === 0 || start >= out[out.length - 1][1]) {
      out.push([start, end]);
    }
    from = idx + foldedQuery.length;
  }
  return out;
}
