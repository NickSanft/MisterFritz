# admin_static

Files served publicly at `/static/...` by the admin panel (see the
`Mount("/static", StaticFiles(...))` in `admin_panel.py`). The path is exempt
from the admin Basic-auth gate — everything here is fonts and CSS, and gating
them buys nothing while it would render `/chat/login` unstyled behind a
password prompt.

## Fonts (`fonts/`)

The dark-academia chat theme (`_theme_chat.html`) self-hosts three families so a
127.0.0.1 panel never beacons to a third party on page load, renders correctly
on an egress-less host or in Docker, and does not have to widen its CSP to allow
`fonts.googleapis.com` / `fonts.gstatic.com`.

The `@font-face` rules in `_theme_chat.html` reference these exact filenames.
**Until they are present the theme falls back to system serifs — the page is
fully usable, just not in the intended faces.** Drop in the latin-subset `woff2`
for each face below (all three families are SIL OFL 1.1; get them from Google
Fonts or the foundry, subset to latin, convert to woff2):

| File | Family | Weight / style |
|---|---|---|
| `cormorant-garamond-400.woff2` | Cormorant Garamond | 400 |
| `cormorant-garamond-600.woff2` | Cormorant Garamond | 600 |
| `cormorant-garamond-700.woff2` | Cormorant Garamond | 700 |
| `eb-garamond-400.woff2` | EB Garamond | 400 |
| `eb-garamond-400-italic.woff2` | EB Garamond | 400 italic |
| `eb-garamond-600.woff2` | EB Garamond | 600 (`.fritz-md strong` needs it) |
| `jetbrains-mono-400.woff2` | JetBrains Mono | 400 |

Six faces, latin subset, ~150–170 KB total, cached forever. Ship the families'
`OFL.txt` alongside them here when you add them — the licence requires the text
to travel with the fonts.
