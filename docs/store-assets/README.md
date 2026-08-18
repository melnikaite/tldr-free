# Chrome Web Store assets

Regenerated, not hand-drawn — every image here comes out of the extension's
own code or the repository's own palette, so a UI change can be re-shot
instead of re-illustrated.

| File | Size | What it is |
|---|---|---|
| `screenshot-welcome-1280x800.png` | 1280×800 | Listing screenshot. The real side-panel document in an iframe at its real 400px width, on a brand canvas. Store screenshots must be 1280×800 or 640×400, which a bare 400px panel capture is not — hence the canvas. |
| `promo-small-440x280.png` | 440×280 | Small promo tile (the store's card image). |
| `promo-marquee-1400x560.png` | 1400×560 | Marquee tile, used only if the extension is featured. |

Brand-only tiles by design: no fabricated UI, no screenshot collage. Palette
(`#1a73e8` + white wordmark) matches `docs/logo-banner.svg`.

## Regenerating

`welcome-shot.html` renders the real `sidepanel/welcome.js` against the real
`sidepanel/style.css`; `listing.html` frames it; `tile.html` draws both tiles.
From this directory:

```bash
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CH" --headless=new --disable-gpu --allow-file-access-from-files \
  --window-size=1280,800 --hide-scrollbars --virtual-time-budget=2500 \
  --screenshot=screenshot-welcome-1280x800.png "file://$PWD/listing.html"
"$CH" --headless=new --disable-gpu --window-size=440,280 --hide-scrollbars \
  --screenshot=promo-small-440x280.png "file://$PWD/tile.html?size=small"
"$CH" --headless=new --disable-gpu --window-size=1400,560 --hide-scrollbars \
  --screenshot=promo-marquee-1400x560.png "file://$PWD/tile.html?size=large"
```

`welcome-shot.html?step=model` renders the second onboarding state (daemon up,
no model yet) — that capture lives in `../screenshots/sidepanel-welcome-model.png`.

Still to produce by hand: nothing for the listing itself. The store also asks
for a developer account (one-time $5 fee) and a reviewer API key, neither of
which belongs in a repository.
