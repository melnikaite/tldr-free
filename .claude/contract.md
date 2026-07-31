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

## Settings API writes to an overrides file, never to the template

`GET /config` / `PATCH /config` / `POST /config/test`
(`daemon/src/api/config.py`) let the extension edit backend/model/API
key/output language without hand-editing YAML. `tldr.yaml` is a hand-edited,
comment-heavy template — writing to it with `yaml.safe_dump` would destroy
those comments. `PATCH` instead writes `tldr.local.yaml` (a sibling file,
`src/config.py#overrides_path`), which `get_config()` deep-merges on top of
the template before env-var overrides are applied (env still wins over
both). Every `PATCH` is validated (`config.validate_full_config`) before
anything is written. API keys are never echoed back by `GET`/`PATCH` — only
`api_key_set` / `api_key_hint` (last 4 chars) / `api_key_source`.
