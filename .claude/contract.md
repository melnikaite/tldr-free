# Contracts between layers

The agreements that survive any internal refactor. Break one, the other
side stops working in a way the type checker can't catch.

## API contract is mirrored, not generated

`daemon/src/api/schemas.py` ↔ `extension/src/lib/api-types.js`. Manual sync
— when you change one, change the other in the same commit. Bump
`DAEMON_API_VERSION` in `daemon/src/config.py` for breaking shape changes
so old extensions detect the mismatch instead of mis-parsing payloads.

`extension/src/lib/daemon-client.js` is the only place that issues HTTP to
the daemon. New endpoints go there with a JSDoc return-type annotation
against the api-types alias.

## URL normalization

The extension normalizes every URL through `lib/url.js#normalizeUrl` before
sending to the daemon (both create and lookup). Implications:

- Same article via `?utm_source=tw` and direct link map to the same canonical URL.
- Clicking `[12:34]` (opens `?v=X&t=754s`) stays the same job as the original `?v=X`.
- For YouTube, identity is the video id alone — `/shorts/`, `/embed/`,
  `youtu.be/...`, `&list=...` all collapse to
  `https://www.youtube.com/watch?v=<id>`.
- Daemon stores whatever it gets and looks up on the same canonical form.

If you add a new URL family, extend `normalizeUrl` — never special-case on
the daemon side.
