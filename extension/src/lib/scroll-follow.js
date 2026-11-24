// "Follow" mode for the side panel's page-level scroll (summary stream + chat).
//
// Auto-scroll to the bottom only while the reader is AT the bottom. Any
// user-initiated upward scroll (wheel, touch drag, keyboard, scrollbar drag)
// turns following off immediately; scrolling back down to the bottom turns
// it on again. Our own scrolls are told apart from the user's by remembering
// the scrollTop we set: a scroll event landing exactly there is ours.
//
// Wheel / key / touch handlers flip following off synchronously so a fast
// token stream can't snap the view back down between the user's input and
// the resulting scroll event. Not used by the transcript tab (own logic).

const AT_BOTTOM_PX = 8;

let _following = true;
/** scrollTop we last set ourselves (null = none pending). @type {number | null} */
let _progTop = null;
let _lastTop = 0;

function _root() {
  return document.scrollingElement || document.documentElement;
}

function _atBottom() {
  const r = _root();
  return r.scrollHeight - r.scrollTop - window.innerHeight <= AT_BOTTOM_PX;
}

/** Whether auto-scroll currently follows the end of the page. */
export function isFollowing() {
  return _following;
}

/** Force follow mode on/off (submit → on; genuine job switch → off). */
export function setFollowing(/** @type {boolean} */ on) {
  _following = on;
}

/** Programmatic scroll that is never mistaken for a user scroll. */
export function programmaticScrollTo(/** @type {number} */ top) {
  const r = _root();
  r.scrollTop = top;
  _progTop = r.scrollTop;
  _lastTop = r.scrollTop;
}

/** Scroll to the end — only while following. */
export function followScrollToEnd() {
  if (!_following) return;
  const r = _root();
  if (r.scrollHeight - r.scrollTop - window.innerHeight <= 0) return;
  programmaticScrollTo(r.scrollHeight);
}

window.addEventListener("wheel", (ev) => {
  if (ev.deltaY < 0) _following = false;
}, { passive: true });

const _UP_KEYS = new Set(["ArrowUp", "PageUp", "Home"]);
window.addEventListener("keydown", (ev) => {
  const t = /** @type {HTMLElement | null} */ (ev.target);
  // Keys typed into the composer / search fields don't scroll the page.
  if (t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT" || t.isContentEditable)) return;
  if (_UP_KEYS.has(ev.key) || (ev.key === " " && ev.shiftKey)) _following = false;
});

let _touchY = 0;
window.addEventListener("touchstart", (ev) => {
  _touchY = ev.touches[0]?.clientY ?? 0;
}, { passive: true });
window.addEventListener("touchmove", (ev) => {
  // Finger moving down = content scrolls up.
  if ((ev.touches[0]?.clientY ?? 0) > _touchY) _following = false;
}, { passive: true });

window.addEventListener("scroll", () => {
  const top = _root().scrollTop;
  if (_progTop !== null && top === _progTop) {
    _progTop = null;
    _lastTop = top;
    return;
  }
  _progTop = null;
  if (_atBottom()) _following = true;
  else if (top < _lastTop) _following = false;  // scrollbar drag / anything else upward
  _lastTop = top;
}, { passive: true });
