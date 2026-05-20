// Read browser cookies for a given domain or URL via chrome.cookies API and
// convert them to the Cookie shape expected by the daemon (api-types.js).

/** @import { Cookie } from "./api-types.js" */

/** @param {chrome.cookies.Cookie} c @returns {Cookie} */
function _toDaemonCookie(c) {
  return {
    name: c.name,
    value: c.value,
    domain: c.domain,
    path: c.path,
    secure: c.secure,
    http_only: c.httpOnly,
    expires: c.expirationDate ?? null,
  };
}

/**
 * Domain-scoped: returns every cookie whose ``domain`` matches the input or
 * is a subdomain of it (chrome.cookies.getAll's semantics). Used by the
 * YouTube path where we want everything for ``.youtube.com``.
 *
 * @param {string} domain
 * @returns {Promise<Cookie[]>}
 */
export async function getCookiesForDomain(domain) {
  const browserCookies = await chrome.cookies.getAll({ domain });
  return browserCookies.map(_toDaemonCookie);
}

/**
 * URL-scoped: returns exactly the cookies a real HTTP request to ``url``
 * would send — host match, path match, Secure/HttpOnly/SameSite all
 * factored in by Chrome. Used by the generic media path: the extension
 * found a media URL (a CDN .mp4, a Vimeo player iframe, …) and wants to
 * forward the auth tokens that the browser itself would attach.
 *
 * Prefer this over ``getCookiesForDomain`` when you know the concrete
 * target URL — it's a smaller, more accurate cookie set and avoids
 * leaking unrelated cookies from sibling subdomains.
 *
 * @param {string} url
 * @returns {Promise<Cookie[]>}
 */
export async function getCookiesForUrl(url) {
  const browserCookies = await chrome.cookies.getAll({ url });
  return browserCookies.map(_toDaemonCookie);
}
