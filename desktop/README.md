# Mister Fritz — the desktop body

Fritz's body on the Windows desktop: a Tauri 2 app that talks to the brain
over the `/companion` WebSocket ([plan 14](../plans/14-desktop-companion.md),
[build plan](../plans/14-desktop-companion-build.md)). Windows only.

Today it is one persistent, transparent, never-focused avatar window with a
WebGL VRM in it, a control window (tray > Settings…), a tray, three hotkeys,
the prompt (*Ask Fritz*: Ctrl+Alt+F, a double-click on him, or his menu),
and a mouse hook that makes his body his to click. That is Phase 1 of the
build plan up to PR 6; its CI job (PR 7) is next, and voice (Phase 2), the
wallpaper and the widgets come after. Spike 0 proved the window
([docs/SPIKE0.md](docs/SPIKE0.md)), and its table is PR 2's acceptance.

## What it keeps, and where

Everything is in `%APPDATA%\com.divora.fritz\` (the tray's *Open Fritz's logs*
opens it):

| File | What |
|---|---|
| `settings.json` | The settings ([`settings.rs`](src-tauri/src/settings.rs)), saved as they change. A file that doesn't parse is moved to `settings.corrupt-<time>.json` and the defaults used; one that can't be read is left alone and not saved over that run |
| `companion.token` | The brain's `COMPANION_TOKEN`, saved by Settings › Brain ([`companion.rs`](src-tauri/src/companion.rs)). Never in the settings or a log, and given only to his own window |
| `startup.log`, `crash.log` | A line per start, stop and resume, and every panic. Kept across runs (the previous one as `.old` past 64 KiB): read these after a start that died |
| `logs\fritz.log` | This run's detailed log, started afresh each run |
| `diagnostics\foreground.log` | Spike 0's foreground logger, **off** unless `diagnostics.foregroundLog` is `true` (read at start): it writes other apps' window titles |
| `avatar\fritz.vrm` | The owner's model, if there is one (or `avatar.vrmPath`) |

The hotkeys are `hotkeys.arrange` (Ctrl+Alt+A), `hotkeys.park`
(Ctrl+Alt+H) and `hotkeys.prompt` (Ctrl+Alt+F), registered at start and
set in Settings › Hotkeys: click a key's field and press the new chord (Esc
cancels). It is bound the moment it is saved: checked (Ctrl and Alt
together, or Win, or F13 to F24 alone; no other action's key, nor
push-to-talk's Ctrl+Alt+Space), taken from Windows, saved, and only then is
the old key let go, so a key another app holds is refused on its row with
the old one still working. A chord Fritz or another app already holds does
its job instead of reaching the field. Each row says when the key set isn't
the one working (it fell back to its default at start, or Windows wouldn't
give it), and *Default* puts the default back. A key edited into
`settings.json` by hand takes effect at the next start. His eyes follow the cursor while
`avatar.lookatCursor` is `true` (the default): on a VRoid export the eyes
turn, and on any model his head turns part of the way, which is what shows
on the placeholder.

His page also tells the shell where his body is on his window: a few boxes
round him, measured from the model itself, so a click on him can be told
from one beside him or in the room over his head. It is the shape only,
never anything about the pointer.

While he is on screen, a mouse hook (`src-tauri/src/hook.rs`) watches the
mouse buttons across the desktop to tell a press on his body from one
beside him. His body is his to the left and right buttons: a click there
never reaches the window behind him, while one beside him or in the room
over his head goes through as before.

Double-click him to ask him something: the prompt opens over him. Windows
usually keeps the keyboard where it was after a click on him, so the box
asks you to click in it ("Click here to type…"); Ctrl+Alt+F gives it the
keyboard at once. A click anywhere but in the box or on him, the
double-click again or 20 s with no click closes it, and the keyboard stays
where you left it. On his menu, what you choose decides: *Ask Fritz…*
gives the box the keyboard, and anything else closes it.

Right-click him for his menu, the tray's: it works by mouse, so the app
you were in keeps the keyboard, and a click elsewhere (or 20 s) closes it.
Its first item, *Ask Fritz…*, opens the box with the keyboard in it (or
gives a waiting box the keyboard).
The hook is in only while he can be clicked: never while he
is hidden or arranged, under a fullscreen app or presentation mode, on the
lock screen or with the display asleep. It keeps no record of where you
click (the log names gestures, never a point), ignores clicks a script or a
remote tool injects, and lets the middle button, the wheel and moves pass
untouched. Settings › Fritz › *Clicks on him* (`avatar.clicks`, on by
default) turns it off, and Settings › Diagnostics › *The mouse hook* says
whether it is in, why not, and the slowest answer it gave.

Both windows wear `settings.theme`: the mood (`ink`, `dusk`, `midnight`),
the accent (`brand`, `abyssal`, `ember`), dark or light, and the motion
(0 to 1: functional, ambient, rich), set in Settings › Appearance or the
tray's *Appearance* menu, and changed in both windows at once. He takes a
tint from it, his outlines leaning toward the mood and a rim light in the
accent. With no motion, or while Windows asks for less (its *Animation
effects* switch) and *Keep his motion when Windows asks for less*
(`theme.motionUser`) is off, he only blinks and breathes: no sway, and his
eyes stay ahead. The parchment grain (`theme.grain`) is the control
window's alone.

The tray's *Settings…* opens the control window: his brain (where it is,
the token, and whether he is connected), show or hide him, arrange him, his
eyes following the cursor, his hotkeys, the theme, which model he wears
(with its folder, and loading it again after you drop in a new export), his
logs, and the foreground log, and a question to ask him. Closing it only
hides it. The rest (`avatar.vrmPath`, `brain.clientId` and
`brain.clientName`) is still set by editing `settings.json` with Fritz quit,
until the sections for them land.

## His brain

He connects to the brain's `/companion` WebSocket at `brain.url`
(`ws://127.0.0.1:8001/companion`, the admin panel's port), with the brain's
`COMPANION_TOKEN`: start the brain with `COMPANION_ENABLED=true` and a
`COMPANION_TOKEN` of 32 characters or more in its `.env`, then paste the
token into Settings › Brain and press *Connect*. Until there is a token,
Settings shows at every start with the first run in place of its sections:
four steps (connect his brain, choose his look, learn his keys, summon him),
with *Skip setup* at every step and *Summon Fritz* at the last, which brings
him onto the desktop and puts Settings away. It waits to show while the lock
screen, a fullscreen app or a presentation is in front, and never opens over
them. Nothing about it is saved; About › *Replay the first run* shows it
again.

His own window holds the connection (the control window only hears of it):
it says hello, is online once the brain welcomes it, and answers the brain's
pings. A wrong token, a protocol version the brain doesn't speak, or a hello
it can't read is a refusal, which he never retries by himself (*Reconnect*,
or a new token, tries again). Anything else (the brain down, full or quiet)
he retries after 1, 2, 4, 8, 16, then every 30 seconds, and Settings says
why and when. The protocol's TypeScript (`src/shared/protocol.ts` and its
fixtures) is generated from the brain's `companion_protocol.py` by
`python scripts/gen_protocol_ts.py` at the repository's root; the brain's
pytest fails while they differ.

### Who he is to the brain

His hello names the desk with `brain.clientId` and `brain.clientName`, and
the brain calls the desk `desktop-<brain.clientId>`. Both are made from the
PC's name (`%COMPUTERNAME%`) when `settings.json` is first written, and
kept: the id in lower case, only a-z and the digits kept, each run of
anything else one dash, none at either end (`desk` if nothing is left), so
`NICK_PC` is `desktop-nick-pc`; the name as Windows has it (`this PC` if it
has none). Renaming the PC later changes neither, as long as `settings.json`
keeps them. The default, though, is made from the PC's name as it is at
each start: an id edited into `settings.json` that isn't one (lower-case
letters, digits and dashes, starting with a letter or digit, at most 32) is
put back to it at the next start and saved, and the log says so, and a
`settings.json` deleted or set aside starts from it too. On a renamed PC,
either gives the desk the id of its new name: a new desk to the brain.

While he is connected, Settings › Brain's *Who he is to it* says who he is
to the brain, whom he is talking with, and in which thread, for example
`desktop-nick-pc, talking with NICK_PC, in the thread desktop-nick-pc#desktop.`
Unlinked, the desk is a person of its own to him, called by the PC's name.
To be yourself at the desk, copy the identity from that row into the brain's
`IDENTITY_LINKS` (`desktop-nick-pc=discord-<your id>`) and restart the
brain: the row then names your identity. The root README's
[Desktop companion](../README.md#desktop-companion) section says what a link
carries and what it doesn't (his memories of you do; the conversation, files
and reminders don't), and what an unlinked desk leaves behind, so link the
desk before you talk to him there.

A new `brain.clientId` is a new desk to the brain, and an unlinked one.
What an unlinked desk learned stays under its old id. What a linked one
learned is under your identity: put the new id on the desk's side of its
`IDENTITY_LINKS` link and restart the brain, and the new desk has it all
again, the desk thread too. Until then he is talking with a stranger, and
the brain's log says so as the desk connects (`desktop-<new id> connected
as itself: IDENTITY_LINKS links desktop-<old id>, not this desk`). Or quit
Fritz and write the old id back into `brain.clientId`.

## His speech bubble

His replies show in a bubble over his head, on its own panel in the mood's
`--surface-2` (it floats over whatever is on the desktop, so it never
trusts the desktop behind it). The prompt (Ctrl+Alt+F, below) or Settings
› Brain's *Ask him something* asks him: he glances up while
he thinks, the reply streams in, a tool's notice shows as a smaller line,
and the brain's last word on the reply replaces what streamed. It holds 6
seconds, or longer for a long reply (50 ms a character, at most 30
seconds), and longer while the cursor is over his window (only while his
eyes follow the cursor: otherwise the app hears nothing of it), then
fades. A turn that fails shows his apology; asking again while he answers
stops the first reply. Pictures aren't shown in the bubble yet: it says
how many came. With no motion, the bubble only fades in and out, and he
doesn't glance or tilt. While he is hidden (the tray, the hotkey, or a
fullscreen app on his monitor) the bubble waits, so a reply that came
meanwhile is there to read when he is back.

His window is 340×640 the first time (shorter on a screen that hasn't the
room), the top 28% kept clear for the bubble (he stands as tall as he did in
300×460). A window you arranged before keeps its size, so he looks smaller
in it until you arrange him taller.

## The prompt

*Ask Fritz* is a small window, hidden until Ctrl+Alt+F (`hotkeys.prompt`),
*Ask Fritz…* in his menu (the tray icon's, or a right-click on him) or a
double-click on him opens it, just above Fritz on his monitor (below
him if there's no room above; where he would stand while he is hidden).
The key and the menu open it with the keyboard in it; after a double-click Windows
usually keeps the keyboard where it was, and the box waits for you to click
in it (see *Clicks on him* above). The key again closes it, or, while it
waits, gives it the keyboard; the double-click again closes it. It doesn't open over the lock screen, another
app's fullscreen window (a game, a video, a presentation), or while you are
arranging him: the key then does nothing, and takes nothing. Enter asks him. The prompt closes
once his window has sent the question on to the brain; a refusal (he isn't
connected, his window is closed) leaves your words in it with the reason.
Shift+Enter is a new line, ↑ in an empty field brings back the last
question, and Esc closes it and keeps what you'd typed. Closed by Enter, Esc
or Alt+F4, it gives the keyboard back to the window you were in; clicking
away leaves it where you clicked. Its line says whether he can be asked now,
in Settings' words. Nothing in it moves, and it uses only the text colours
that read well on its panel in every mood.

## Build

Node ≥ 20.19, pnpm, Rust stable (MSRV 1.89, MSVC toolchain) and the Visual
Studio C++ Build Tools. From `desktop/`:

```powershell
pnpm install
pnpm build                                     # checks, type-checks and bundles the pages into dist/
cargo build --release --locked --features custom-protocol --manifest-path src-tauri/Cargo.toml
```

The app is `src-tauri/target/release/mister-fritz-desktop.exe`.

- **Always pass `--features custom-protocol`** to any `cargo build` or
  `cargo test` you run yourself. Without it the exe is rebuilt as a dev build
  that loads `http://localhost:1431`, and the window stays empty. To
  type-check only, `cargo check --manifest-path src-tauri/Cargo.toml` is safe.
- **`--locked`, and the exact Tauri pins in `Cargo.toml`, matter**: tauri
  2.11's own version ranges also accept the 2.12 runtime crates, and that mix
  does not compile. Move tauri, tauri-build, tauri-runtime, tauri-runtime-wry
  and tauri-utils together, never one at a time.
- **The pages are embedded in the exe at compile time**: after any change to
  a page (`avatar.html`, `control.html`) or under `src/` or `public/`, run
  `pnpm build` *and* the cargo build again.
- **Quit Fritz before you build.** Windows won't let cargo replace an exe
  that is running, so a build while he is up ends with `error: failed to
  remove file` naming `mister-fritz-desktop.exe`, then `Access is denied.
  (os error 5)`. Quit him from the tray (*Quit Mister Fritz*) and build
  again.
- **Smart App Control** (Windows 11) blocks the Tauri CLI's unsigned native
  module on some machines, so `pnpm tauri build` fails with "An Application
  Control policy has blocked this file". The two commands above are what it
  runs, minus the installer, and need no CLI.
- **Microsoft Defender may take a fresh build for malware.** The exe is
  unsigned, and Defender's machine-learning detection (a name ending in
  `!ml`) can quarantine one. If a build vanishes, or won't start, look in
  Windows Security › Virus & threat protection › *Protection history*.
  *Restore* puts the exe back, but Defender may find it again; *Allow on
  device*, on the detection it then raises, is what keeps it. It can also
  be submitted to Microsoft as a false positive, through Microsoft's file
  submission page (microsoft.com/wdsi/filesubmission). A submission
  Microsoft clears covers that one file, by its SHA-256 (`Get-FileHash`
  shows it), so every rebuild is a new file to Defender and may be flagged
  again. Signing the exe might help, though Microsoft doesn't promise it
  ends this; plan 14's risk table leaves it to consider in Phase 5.

`pnpm build` refuses an avatar page that could animate outside its paced
loop. `scripts/check-paced.mjs` reads our own modules `avatar.html` reaches
(`scripts/reach.mjs` follows the imports; packages are listed, not read) for
requestAnimationFrame, setAnimationLoop, the Web Animations API and motion
set from code. `scripts/check-no-css-animation.mjs` reads every stylesheet
the page reaches for @keyframes, animation, transition and smooth
scrolling, then, after `vite build`, the built page's stylesheets, packages'
included; a built page it refuses is removed, so the cargo build that
follows stops rather than embed it. `pnpm sync:divora` compares
`src/divora/tokens.css` with DivoraVoice's tokens at the commit its header
pins (it reads GitHub, so it isn't part of the build; `--sha <sha> --write`
takes another commit's tokens and pins it).

## Test

```powershell
pnpm test:run                                                        # the pages ("node" and "dom" projects)
cargo test --locked --features custom-protocol --lib --manifest-path src-tauri/Cargo.toml   # the shell
```

`FRITZ_SHOW_SETTINGS` set to anything shows the control window at start, for
a run nobody clicks; set to a section (`brain`, `hotkeys`, `appearance`,
`model`, …) it opens there, and set to `wizard`, at the first run. Without a
token it opens at the first run anyway. In a debug build,
`FRITZ_PRETEND_SAY` asks the brain its text once he first connects, as Ask
would (a release build never reads it).

## The placeholder model

`public/placeholder.vrm` is generated, CC0, and committed so a fresh checkout
always has a body: `python scripts/make_placeholder_vrm.py` rewrites it,
byte for byte. The owner's real model (plan 14 P6, a VRoid export) is never
committed; drop it at `%APPDATA%\com.divora.fritz\avatar\fritz.vrm` (or point
`avatar.vrmPath` at it) and it is used instead.

## Where things come from

[docs/LIFTED.md](docs/LIFTED.md) lists every file lifted from DivoraVoiceMod
(MIT, see [LICENSES/DivoraVoiceMod.txt](LICENSES/DivoraVoiceMod.txt)) or
LorraineAtelier, with its source and commit; each lifted file says so at the
top.
