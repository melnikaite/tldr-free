// Read browser cookies for a given domain via chrome.cookies API and convert
// them to the Cookie shape expected by the daemon (api-types.js).

/** @import { Cookie } from "./api-types.js" */

/**
 * @param {string} domain
 * @returns {Promise<Cookie[]>}
 */
export async function getCookiesForDomain(domain) {
  const browserCookies = await chrome.cookies.getAll({ domain });
  return browserCookies.map((c) => ({
    name: c.name,
    value: c.value,
    domain: c.domain,
    path: c.path,
    secure: c.secure,
    http_only: c.httpOnly,
    expires: c.expirationDate ?? null,
  }));
}
