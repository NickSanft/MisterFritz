# 7. Web chat: mobile, bug fixes, and the butler restyle

[← back to index](README.md)

**Effort:** L (1-3 days)  
**Depends on:** nothing

## Goal
Today the browser chat at :8001/chat is a desktop-only admin page wearing a chat costume: no viewport meta so phones render it zoomed out, a JavaScript SyntaxError that silently disables the confirm on the destructive "New conversation" button, sub-AA muted text everywhere, a transcript invisible to screen readers, no Enter-to-send, no way to stop a runaway reply, and browser alert() boxes for upload failures. When this lands, /chat is a surface Fritz would not be ashamed to answer the door in: it lays out correctly on a phone, the destructive action actually asks first, every piece of text clears WCAG AA in both light and dark, Fritz's prose reads in a serif with monospace reserved for code, the composer sticks to the bottom of the viewport, streaming follows the scroll only when you're already at the bottom, Enter sends and Stop stops, code blocks are syntax-highlighted with copy buttons, errors appear inline, and the chat no longer inherits the admin panel's "Mister Fritz · admin" chrome with six nav links that lead to Basic-auth walls.

## Definition of done

- [ ] `grep -c 'name="viewport"' admin_templates/base.html admin_templates/chat_base.html` returns 1 for each; /chat and every admin page render without horizontal body scroll at 375px viewport width.
- [ ] `grep -rn 'onsubmit' admin_templates/` returns nothing. Clicking 'New conversation' and pressing Cancel leaves the LangGraph checkpoint rows intact.
- [ ] Every text/background pair in both light and dark palettes measures >= 4.5:1 (normal text) or >= 3:1 (>=18.66px bold / >=24px text and non-text UI). --brass is used only for focus rings and rules, never for text.
- [ ] `grep -rn '#[0-9a-fA-F]\{3,6\}' admin_templates/chat.html admin_templates/base.html` returns only lines inside _theme.html's :root and dark-mode blocks — no colour literals survive in component rules.
- [ ] `#chat-messages` carries role="log"; a `role="status" aria-live="polite"` region exists; the streaming bubble carries aria-busy="true" for exactly the duration of the stream.
- [ ] `grep -c 'alert(' admin_templates/chat.html` returns 0. Upload failures surface in the inline #chat-notice region.
- [ ] Scrolling up mid-stream is not overridden; scrolling back to the bottom resumes auto-follow.
- [ ] Enter sends on desktop, Shift+Enter newlines, Enter newlines on coarse pointers, and IME composition Enter does not send.
- [ ] A Stop button aborts the client fetch and its tooltip/label makes clear that Fritz finishes composing server-side.
- [ ] Fenced code blocks render with a `.codehilite` wrapper and readable token colours in both themes, and each carries a working Copy button that degrades gracefully in a non-secure context.
- [ ] admin_templates/chat_base.html exists, chat.html and chat_login.html extend it, and it contains no href to /, /users, /schedules, /documents, or /health.
- [ ] `prefers-reduced-motion: reduce` stops the caret blink and all transitions.
- [ ] `ruff check .` is clean and `pytest tests/ --cov=. --cov-fail-under=60` passes with the new test classes added and no existing assertion silently deleted.

## Current state (verified against the working tree)
VERIFIED AGAINST THE REPO. All line numbers below were read this session.

**Mobile baseline — confirmed.** `admin_templates/base.html:3-5` is `<html lang="en">` / `<head>` / `<meta charset="utf-8">`. There is no viewport meta tag. `grep -rn "@media\|viewport" admin_templates/` returns exit 1 (zero matches) across all eight templates. Every page therefore renders at a ~980px assumed viewport and is pinch-zoomed out on phones. Two additional mobile faults the audit did not name: `chat.html:58` sets `.chat-input-row textarea { font-size: 0.95rem }` (15.2px) and `chat_login.html:25` sets the username input to `0.95rem` — iOS Safari force-zooms the viewport on focus for any input under 16px. And `chat.html:396` calls `textarea.focus()` in `.finally`, which re-pops the soft keyboard after every send.

**Broken confirm — confirmed empirically, not by inspection.** `chat.html:105-106`:
```
<form class="inline" method="post" action="/chat/forget"
      onsubmit="return confirm('Start a fresh conversation? Your memories and schedules stay; only this thread\\'s context is reset.');">
```
`cat -A` confirms two literal backslashes. Jinja passes them through (no `{{ }}`/`{% %}`); HTML attribute parsing does not process backslashes; so the JS engine receives `...this thread\\'s context...` where `\\` is an escaped backslash and the following `'` terminates the string. I extracted the exact attribute bytes and fed them to `new Function()` under node: **`SyntaxError - missing ) after argument list`**. A handler that fails to compile is null, so `POST /chat/forget` fires with no confirmation whatsoever. `chat_forget` (admin_panel.py:636-646) calls `privacy.forget_conversation` (privacy.py:61), which `DELETE`s from `checkpoints` and `writes` — unrecoverable.

**Contrast — audit numbers confirmed by computation, and the audit undercounts the blast radius.** `--muted: #738291` (base.html:9), `--bg: #f7f7f5` (:10), `--card: #ffffff` (:11). Relative luminances: muted 0.21656, bg 0.92889, card 1.0 → **3.672:1** on bg and **3.939:1** on card. Both fail AA 4.5:1. Consumers the audit named: `.role` (chat.html:25-28), `.chat-progress` (:47-50), `.muted`/`.pill` (base.html:51-55), `th` (base.html:44). **Consumers the audit missed:** `h2 { color: var(--muted) }` (base.html:33) — every section heading on every admin page; `.chat-pending-attach` (chat.html:70); `.chat-doc-upload label` (chat.html:89); `.chat-empty` (chat.html:95-98); and the inline `color: var(--muted)` on the Username label (chat_login.html:19).

**Screen readers — confirmed.** `chat.html:116` is a bare `<div id="chat-messages">`. No `role`, no `aria-live`, no `aria-busy`. Streamed replies are announced never.

**Scroll — confirmed.** The only scroll call is `window.scrollTo(0, document.body.scrollHeight)` at `chat.html:397`, inside `.finally`. It runs once per turn, unconditionally, so it yanks you to the bottom even if you had scrolled up to read, and never follows during streaming.

**Enter-to-send / Stop — confirmed absent.** No `keydown` listener anywhere in chat.html. No `AbortController`; the fetch at `chat.html:326` passes no signal.

**Reflow jump — confirmed, and the root cause is narrower than "textContent then innerHTML."** `chat.html:354` does `body.textContent = data` with no `white-space` rule in effect, so newlines collapse and the streaming bubble is shorter than its final height; `chat.html:365` then assigns `body.innerHTML = payload.html` (block `<p>`/`<pre>`) and the bubble jumps. The fix is a single CSS declaration, not a JS rewrite.

**alert() — confirmed** at `chat.html:189` (non-image file), `:199` (upload failed), `:206` (network catch).

**Motion — confirmed.** The `blink` keyframe caret (`chat.html:20-24`) runs `1s steps(2,start) infinite` with no `prefers-reduced-motion` guard.

**Admin chrome bleed — confirmed.** `chat.html:1` is `{% extends "base.html" %}`, so a non-admin chat user gets the `<h1>Mister Fritz · admin</h1>` at base.html:73 and the six nav links at base.html:75-80. Five of those six (`/`, `/users`, `/schedules`, `/documents`, `/health`) are gated by `_BasicAuthMiddleware` (admin_panel.py:77-78 exempts only `/chat*`), so a chat user clicking them hits a browser Basic-auth prompt for a password they don't have.

**Hardcoded hexes — confirmed at every cited line, plus five more in base.html.** chat.html: `#e8e5dc`:12, `#fff`:17, `#ecebe5`:32 and :36, `#fff`:64, `#f5f3ed`:66 and :70, `#f9f8f3`:86. base.html (audit didn't list these): `#fff`:23, `#fff`:31, `#ecebe5`:47, `#ecebe5`:54, `#ecebe5`:60, `#e0dfd9`:63, `#fff`:65, `#fff`:67. None can respond to a dark-mode media query.

**Typography — confirmed.** `base.html:18` sets `ui-monospace, "SF Mono", Menlo, Consolas, monospace` on `body`; nothing overrides it, so Fritz's prose, table cells, buttons and nav are all monospace.

**Streaming wire format — relevant to sequencing.** `admin_panel.py:492` documents `event=token data=<accumulated text so far>`, and `mister_fritz.py:433` and `:494` both call `streaming_callback(accumulated_text)`. So token frames are cumulative today, which is why `chat.html:354` assigns rather than appends. `tests/test_admin_panel.py:516` asserts exactly `["Very", "Very well", "Very well, sir."]` — that assertion is the wire contract.

**Server-side cancellation does not exist.** `chat_stream` spawns a plain daemon `threading.Thread` (admin_panel.py:558) and `_event_generator` (:560-577) never checks `request.is_disconnected()`. Aborting the client fetch cannot stop `ask_stuff`.

**Two small server-side bugs found while reading.** `chat_send` renders `chat.html` at admin_panel.py:462-468 and :473-480 without an `is_admin` key, so after a no-JS synchronous send the admin document-upload panel silently vanishes; and the error-path message dicts at :465-466 omit `html`, so an error reply skips the `bubble-body` markdown branch at chat.html:121.

**Deps.** `Pygments==2.19.2` is already resolved in requirements.txt:192 but only as a transitive pin; the explicit admin-panel block (requirements.txt:286-292) lists only `starlette>=0.46`, `python-multipart>=0.0.22`, `Markdown>=3.5`.

## Change sites

### `admin_templates/base.html:4-5`

Add the viewport meta tag. This one line is the entire mobile baseline fix for all eight templates.

BEFORE (base.html:3-6):
<head>
    <meta charset="utf-8">
    <title>{% block title %}Mister Fritz Admin{% endblock %}</title>
    <style>

AFTER:
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}Mister Fritz Admin{% endblock %}</title>
    {% include "_theme.html" %}

### `admin_templates/chat.html:105-108`

Delete the SyntaxError-producing inline onsubmit attribute entirely; give the form an id and attach the confirm from JS. The listener must be registered ABOVE the `if (!form || !list) return;` bail at chat.html:168, or a missing #chat-form silently re-disables the confirm.

BEFORE (chat.html:105-108):
<form class="inline" method="post" action="/chat/forget"
      onsubmit="return confirm('Start a fresh conversation? Your memories and schedules stay; only this thread\\'s context is reset.');">
    <button type="submit">New conversation</button>
</form>

AFTER (markup):
<form class="inline" method="post" action="/chat/forget" id="chat-forget-form">
    <button type="submit">New conversation</button>
</form>

AFTER (script, inserted at chat.html:166 — BEFORE the `if (!form || !list) return;` guard on :168):
    const forgetForm = document.getElementById("chat-forget-form");
    if (forgetForm) {
        forgetForm.addEventListener("submit", function (ev) {
            const ok = window.confirm(
                "Start a fresh conversation? Your memories and schedules " +
                "stay; only this thread's context is reset."
            );
            if (!ok) ev.preventDefault();
        });
    }

### `admin_templates/_theme.html:new file`

NEW. A single <style> partial holding the design tokens, reset, typography, responsive rules and dark-mode overrides. base.html and (step 4) chat_base.html both {% include %} it, so there is exactly one palette and two chromes. Jinja2Templates(directory=admin_templates) at admin_panel.py:58 resolves the include with no config change.

<style>
:root {
    /* Butler's study, by daylight. Contrast ratios computed against --bg
       (L=0.8658) and --card (L=0.9400). */
    --fg:        #2b2118;  /* 13.7:1 on bg, 14.9:1 on card */
    --muted:     #6f6357;  /*  5.1:1 on bg,  5.5:1 on card — was 3.67/3.94 */
    --bg:        #f4efe4;  /* parchment */
    --card:      #fbf8f1;  /* foolscap */
    --surface-2: #efe8da;
    --surface-3: #e6dcc9;
    --code-bg:   #ece4d4;
    --user-bubble: #e7dcc6;
    --border:    #ddd4c2;
    --accent:    #5b3f30;  /* mahogany, unchanged — 8.3:1 on bg */
    --on-accent: #fdfaf3;  /* 9.6:1 on accent */
    --brass:     #9a7b3f;  /* 3.5:1 on bg — DECORATION AND FOCUS RINGS ONLY,
                              never text. Passes 1.4.11 non-text 3:1. */
    --danger:    #b3261e;  /* 5.7:1 on bg */

    --font-prose: "Iowan Old Style", "Palatino Linotype", Palatino,
                  "Book Antiqua", Georgia, "Times New Roman", serif;
    --font-ui:    system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    --font-mono:  ui-monospace, "SF Mono", Menlo, Consolas, monospace;

    --radius: 8px;
    --col: 780px;
}

@media (prefers-color-scheme: dark) {
    :root {
        /* The study, by lamplight. */
        --fg:        #ece3d4;  /* 14.5:1 on bg */
        --muted:     #a99a86;  /*  6.7:1 on bg,  6.3:1 on card */
        --bg:        #17130f;
        --card:      #1f1a15;
        --surface-2: #262019;
        --surface-3: #2f2820;
        --code-bg:   #12100d;
        --user-bubble: #2c241b;
        --border:    #37302a;
        --accent:    #c98f6b;  /* 6.7:1 on bg — links stay legible */
        --on-accent: #17130f;
        --brass:     #c9a75f;
        --danger:    #ef8a80;  /* 7.1:1 on card */
    }
}

* { box-sizing: border-box; }
body {
    font-family: var(--font-ui);
    background: var(--bg); color: var(--fg);
    margin: 0; padding: 0; line-height: 1.55;
    -webkit-text-size-adjust: 100%;
}
/* Monospace is now opt-in, reserved for code. */
code, pre, kbd, samp { font-family: var(--font-mono); }

:where(a, button, input, textarea, select, [tabindex]):focus-visible {
    outline: 2px solid var(--brass);
    outline-offset: 2px;
}

/* ...existing base.html rules for header/main/h2/.card/table/.muted/.pill/
   button, with every hardcoded hex swapped for a var (see next change site) */

@media (max-width: 640px) {
    header { flex-direction: column; align-items: flex-start;
             gap: 0.5rem; padding: 0.75rem 1rem; }
    header nav { display: flex; flex-wrap: wrap; gap: 0.25rem 1rem; }
    header nav a { margin-right: 0; }
    main { padding: 1rem 0.85rem 0; }
    /* Tables are the one thing that genuinely cannot reflow. */
    table { display: block; overflow-x: auto; }
    button, .icon-btn { min-height: 44px; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}
</style>

### `admin_templates/base.html:6-69, 23, 31, 33, 47, 54, 60, 63, 65, 67`

Move the whole <style> block into _theme.html and replace it with the include. While moving, swap the eight hardcoded hexes the audit did not list: #fff:23 and :31 → var(--on-accent); #ecebe5:47 → var(--code-bg); #ecebe5:54 → var(--code-bg); #ecebe5:60 → var(--surface-2); #e0dfd9:63 → var(--surface-3); #fff:65 → var(--card); #fff:67 → var(--on-accent). h2 at :33 keeps `color: var(--muted)` — it now passes because --muted changed.

BEFORE (base.html:57-67):
        button {
            font-family: inherit; font-size: 0.85rem;
            padding: 0.25rem 0.6rem; border-radius: 3px;
            border: 1px solid var(--border); background: #ecebe5;
            color: var(--fg); cursor: pointer;
        }
        button:hover { background: #e0dfd9; }
        button.danger {
            border-color: var(--danger); color: var(--danger); background: #fff;
        }
        button.danger:hover { background: var(--danger); color: #fff; }

AFTER (in _theme.html):
button {
    font-family: var(--font-ui); font-size: 0.875rem;
    padding: 0.35rem 0.7rem; border-radius: 4px;
    border: 1px solid var(--border); background: var(--surface-2);
    color: var(--fg); cursor: pointer;
}
button:hover { background: var(--surface-3); }
button.danger {
    border-color: var(--danger); color: var(--danger); background: var(--card);
}
button.danger:hover { background: var(--danger); color: var(--on-accent); }

### `admin_templates/chat.html:5-99`

Restyle the chat-local <style> block: serif prose for Fritz, all eight hardcoded hexes → vars, sticky composer, reduced-motion guard on the caret, streaming pre-wrap to kill the reflow jump, 16px composer font to stop iOS zoom, mobile bubble gutters.

/* Fritz speaks in a serif; only his code is monospace. */
.chat-bubble .bubble-body {
    font-family: var(--font-prose);
    font-size: 1.02rem;
    line-height: 1.65;
}
.chat-bubble .bubble-body code,
.chat-bubble .bubble-body pre { font-family: var(--font-mono); }

.chat-bubble.user {
    background: var(--user-bubble);   /* was #e8e5dc (:12) */
    color: var(--fg); margin-left: 4rem; white-space: pre-wrap;
}
.chat-bubble.fritz {
    background: var(--card);          /* was #fff (:17) */
    border: 1px solid var(--border); color: var(--fg); margin-right: 4rem;
}
.chat-bubble .bubble-body pre  { background: var(--code-bg); }  /* was #ecebe5 (:32) */
.chat-bubble .bubble-body code { background: var(--code-bg); }  /* was #ecebe5 (:36) */
.chat-input-row .icon-btn        { background: var(--card); }      /* was #fff (:64) */
.chat-input-row .icon-btn:hover  { background: var(--surface-2); } /* was #f5f3ed (:66) */
.chat-pending-attach             { background: var(--surface-2); } /* was #f5f3ed (:70) */
.chat-doc-upload                 { background: var(--surface-2); } /* was #f9f8f3 (:86) */

/* THE reflow-jump fix. During streaming the bubble is a single collapsed
   text node; pre-wrap gives it roughly its final line count so the swap to
   innerHTML on `done` doesn't jump. The .streaming class is already removed
   at :375/:380/:387/:391, so this un-applies itself at exactly the right
   moment — no new bookkeeping. */
.chat-bubble.fritz.streaming .bubble-body {
    white-space: pre-wrap;
    min-height: 1.6em;
}

@media (prefers-reduced-motion: reduce) {
    .chat-bubble.fritz.streaming::after { animation: none; opacity: 0.6; }
}

/* Sticky composer. Works today inside base.html's document flow AND
   unchanged once step 4's chat_base.html makes the column a flex parent. */
.chat-composer {
    position: sticky; bottom: 0; z-index: 2;
    background: var(--card);
    border-top: 1px solid var(--border);
    padding: 0.75rem 0 max(0.75rem, env(safe-area-inset-bottom));
    margin-top: 1rem;
}
.chat-input-row { display: flex; gap: 0.5rem; align-items: stretch; margin: 0; }
.chat-input-row textarea {
    flex: 1; padding: 0.6rem 0.75rem;
    border: 1px solid var(--border); border-radius: 6px;
    font-family: var(--font-ui);
    font-size: 16px;   /* MUST stay >= 16px: iOS Safari force-zooms below it.
                          Was 0.95rem = 15.2px (:58). */
    resize: vertical; min-height: 3rem;
    background: var(--card); color: var(--fg);
}

.chat-notice {
    margin: 0.5rem 0; padding: 0.5rem 0.75rem;
    border-radius: 6px; font-size: 0.9rem;
    border: 1px solid var(--border); background: var(--surface-2);
}
.chat-notice.error { border-color: var(--danger); color: var(--danger); }

.code-block { position: relative; }
.code-block .copy-btn {
    position: absolute; top: 0.4rem; right: 0.4rem;
    font-size: 0.72rem; padding: 0.15rem 0.45rem; opacity: 0;
    transition: opacity 0.12s;
}
.code-block:hover .copy-btn,
.code-block .copy-btn:focus-visible { opacity: 1; }
@media (pointer: coarse) { .code-block .copy-btn { opacity: 1; } }

.visually-hidden {
    position: absolute; width: 1px; height: 1px; overflow: hidden;
    clip: rect(0 0 0 0); clip-path: inset(50%); white-space: nowrap;
}

@media (max-width: 640px) {
    .chat-bubble.user   { margin-left: 1.25rem; }
    .chat-bubble.fritz  { margin-right: 1.25rem; }
    .chat-progress      { margin-right: 1.25rem; }
    .chat-header        { flex-wrap: wrap; }
}

### `admin_templates/chat.html:115-116, 135-149`

Add the live-region wiring to the transcript, a status region, an inline notice element, and the Stop button. Wrap the composer in .chat-composer so it can go sticky.

BEFORE (chat.html:116):
        <div id="chat-messages">

AFTER:
        <div id="chat-messages" role="log"
             aria-label="Conversation with Mister Fritz"></div>
<!-- role="log" gives an implicit polite live region scoped to additions.
     We deliberately do NOT put aria-live on it directly: token frames rewrite
     the bubble ~30x/sec and a raw polite region would re-announce the whole
     reply on every frame. Instead the streaming bubble carries aria-busy=
     "true" until `done`, and coarse status goes to #chat-status below. -->

NEW (immediately after the transcript div):
        <p id="chat-status" class="visually-hidden" role="status"
           aria-live="polite"></p>
        <div id="chat-notice" class="chat-notice" role="alert" hidden></div>

BEFORE (chat.html:141-149):
        <form class="chat-input-row" id="chat-form" method="post" action="/chat/send">
            ...
            <button type="submit">Send</button>
        </form>

AFTER:
        <div class="chat-composer">
          <form class="chat-input-row" id="chat-form" method="post" action="/chat/send">
            <button type="button" class="icon-btn" id="chat-attach-btn"
                    title="Attach an image" aria-label="Attach an image">📎</button>
            <input type="file" id="chat-image-input" accept="image/*" style="display: none;">
            <label for="chat-message-input" class="visually-hidden">Message</label>
            <textarea id="chat-message-input" name="message"
                      placeholder="Type a message... (or drop an image anywhere)"
                      required autofocus rows="2"></textarea>
            <button type="submit" id="chat-send-btn">Send</button>
            <button type="button" id="chat-stop-btn" class="danger" hidden
                    title="Stop displaying this reply. Fritz finishes composing in the background.">Stop</button>
          </form>
        </div>

<!-- KEEP the placeholder substring "Type a message" verbatim: it is asserted
     by tests/test_admin_panel.py:392. Changing it fails the suite. -->

### `admin_templates/chat.html:326-405`

Rework the fetch/SSE render loop: AbortController + Stop, follow-scroll that respects the user's scroll position, an applyToken() seam that isolates the cumulative-vs-delta wire assumption, aria-busy toggling, copy buttons on done, and scroll/focus behaviour that doesn't fight mobile.

    const NEAR_BOTTOM_PX = 120;
    const statusEl = document.getElementById("chat-status");
    const noticeEl = document.getElementById("chat-notice");
    const sendBtn  = document.getElementById("chat-send-btn");
    const stopBtn  = document.getElementById("chat-stop-btn");
    let activeController = null;
    let noticeTimer = null;

    function isPinnedToBottom() {
        const d = document.documentElement;
        return (d.scrollHeight - window.innerHeight - window.scrollY) < NEAR_BOTTOM_PX;
    }
    function followScroll(wasPinned) {
        if (wasPinned) window.scrollTo(0, document.documentElement.scrollHeight);
    }

    function showNotice(text, kind) {
        if (!noticeEl) return;
        noticeEl.textContent = text;
        noticeEl.className = "chat-notice" + (kind ? " " + kind : "");
        noticeEl.hidden = false;
        clearTimeout(noticeTimer);
        noticeTimer = setTimeout(function () { noticeEl.hidden = true; }, 8000);
    }
    function setStatus(text) { if (statusEl) statusEl.textContent = text; }

    // THE ONE PLACE that knows the token wire format. admin_panel.chat_stream's
    // docstring (:492) and mister_fritz.py:433/:494 both send the FULL
    // accumulated text in every `token` frame, so we assign. If the
    // token-streaming item switches to per-chunk deltas, change this function
    // to `el.textContent += data` and nothing else in the loop moves.
    function applyToken(el, data) { el.textContent = data; }

    function addCopyButtons(scope) {
        const pres = scope.querySelectorAll("pre");
        for (let i = 0; i < pres.length; i++) {
            const pre = pres[i];
            if (pre.parentElement && pre.parentElement.classList.contains("code-block")) continue;
            const wrap = document.createElement("div");
            wrap.className = "code-block";
            pre.parentNode.insertBefore(wrap, pre);
            wrap.appendChild(pre);
            const btn = document.createElement("button");
            btn.type = "button"; btn.className = "copy-btn"; btn.textContent = "Copy";
            btn.addEventListener("click", function () {
                const code = pre.querySelector("code") || pre;
                // navigator.clipboard is undefined in non-secure contexts.
                // http://127.0.0.1:8001 IS secure (localhost exemption), but a
                // LAN-IP visit is not — hence the select-for-manual-copy fallback.
                const done = function (label) {
                    btn.textContent = label;
                    setTimeout(function () { btn.textContent = "Copy"; }, 1800);
                };
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(code.innerText)
                        .then(function () { done("Copied"); })
                        .catch(function () { selectNode(code); done("Ctrl+C"); });
                } else { selectNode(code); done("Ctrl+C"); }
            });
            wrap.appendChild(btn);
        }
    }
    function selectNode(node) {
        const sel = window.getSelection(); const r = document.createRange();
        r.selectNodeContents(node); sel.removeAllRanges(); sel.addRange(r);
    }
    addCopyButtons(list);  // server-rendered history from _load_chat_history

    // Enter sends; Shift+Enter newlines. Skipped on touch (no reliable
    // modifier on a soft keyboard) and during IME composition.
    const textarea = form.querySelector("textarea[name=message]");
    textarea.addEventListener("keydown", function (ev) {
        if (ev.key !== "Enter" || ev.shiftKey || ev.isComposing || ev.keyCode === 229) return;
        if (window.matchMedia("(pointer: coarse)").matches) return;
        ev.preventDefault();
        if (typeof form.requestSubmit === "function") form.requestSubmit();
        else form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    });

    stopBtn.addEventListener("click", function () {
        if (activeController) activeController.abort();
    });

    // --- inside the existing submit handler ---
    fritzBubble.setAttribute("aria-busy", "true");
    setStatus("Mister Fritz is composing a reply.");
    sendBtn.hidden = true; stopBtn.hidden = false;
    activeController = new AbortController();

    fetch("/chat/stream", {
        method: "POST", body: fd, credentials: "same-origin",
        signal: activeController.signal,
    })
    // ... in the frame loop, replace chat.html:352-383 branch bodies:
        const pinned = isPinnedToBottom();
        if (eventName === "token") {
            applyToken(body, data);
        } else if (eventName === "progress") {
            const lineEl = makeProgressLine(data);
            list.insertBefore(lineEl, fritzBubble);
            progressLines.push(lineEl);
            setStatus(data);
        } else if (eventName === "done") {
            try {
                const payload = JSON.parse(data);
                body.innerHTML = payload.html || (payload.text ? escapeText(payload.text) : "(no response)");
                for (const url of (payload.images || [])) { /* unchanged */ }
            } catch (e) { body.textContent = data || "(no response)"; }
            fritzBubble.classList.remove("streaming");
            fritzBubble.removeAttribute("aria-busy");
            addCopyButtons(body);
            setStatus("Reply received.");
            for (const l of progressLines) l.remove();
            progressLines.length = 0;
        } else if (eventName === "error") { /* + removeAttribute("aria-busy") */ }
        followScroll(pinned);

    .catch(function (err) {
        if (err.name === "AbortError") {
            fritzBubble.classList.add("stopped");
            setStatus("Stopped.");
        } else {
            body.textContent = "Network error: " + err.message;
        }
        fritzBubble.classList.remove("streaming");
        fritzBubble.removeAttribute("aria-busy");
    })
    .finally(function () {
        activeController = null;
        sendBtn.hidden = false; stopBtn.hidden = true;
        textarea.disabled = false; sendBtn.disabled = false;
        // Refocusing re-pops the soft keyboard on every send; desktop only.
        if (window.matchMedia("(pointer: fine)").matches) textarea.focus();
        followScroll(true);   // replaces the unconditional scrollTo at :397
    });

### `admin_templates/chat.html:186-207`

Replace the three alert() calls with the inline #chat-notice region.

BEFORE (:188-190, :198-201, :206):
        if (!file.type.startsWith("image/")) { alert("That doesn't look like an image."); return; }
        ...
        if (!ok) { alert("Upload failed: " + (body.error || "unknown")); return; }
        ...
        .catch(err => alert("Upload failed: " + err.message));

AFTER:
        if (!file.type.startsWith("image/")) {
            showNotice("That doesn't look like an image, sir.", "error"); return;
        }
        ...
        if (!ok) {
            showNotice("Upload failed: " + (body.error || "unknown"), "error"); return;
        }
        ...
        .catch(function (err) { showNotice("Upload failed: " + err.message, "error"); });

### `admin_templates/chat_base.html:new file`

NEW. A standalone chat chrome that does NOT extend base.html, so chat users stop seeing "Mister Fritz · admin" (base.html:73) and the six nav links (base.html:75-80), five of which lead to Basic-auth walls. Shares _theme.html so there is still one palette.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}Mister Fritz{% endblock %}</title>
    {% include "_theme.html" %}
    <style>
        html { height: 100%; }
        body.chat-app {
            display: flex; flex-direction: column;
            min-height: 100vh;
            min-height: 100dvh;   /* dvh keeps the composer above the mobile
                                     URL bar; vh above is the fallback. */
        }
        body.chat-app > header.chat-chrome {
            background: var(--accent); color: var(--on-accent);
            padding: 0.7rem 1rem; display: flex;
            justify-content: space-between; align-items: center; gap: 1rem;
            flex: 0 0 auto;
        }
        body.chat-app > header.chat-chrome h1 {
            margin: 0; font-family: var(--font-prose);
            font-size: 1.05rem; font-weight: 600; letter-spacing: 0.01em;
        }
        body.chat-app > header.chat-chrome .chrome-right {
            display: flex; align-items: center; gap: 0.5rem;
            font-size: 0.85rem; color: var(--on-accent);
        }
        body.chat-app > main {
            flex: 1 1 auto; width: 100%;
            max-width: var(--col); margin: 0 auto;
            padding: 1rem 1rem 0;
        }
    </style>
</head>
<body class="chat-app">
    <header class="chat-chrome">
        <h1>Mister Fritz</h1>
        <div class="chrome-right">{% block chrome_actions %}{% endblock %}</div>
    </header>
    <main>{% block body %}{% endblock %}</main>
</body>
</html>

### `admin_templates/chat.html:1, 101-113`

Switch the parent template and move the two header forms up into the chrome. The .chat-header row then disappears from the transcript column.

BEFORE (:1):
{% extends "base.html" %}
AFTER:
{% extends "chat_base.html" %}

NEW block (replaces the .chat-header div at :102-113):
{% block chrome_actions %}
    <span>{{ username }}</span>
    <form class="inline" method="post" action="/chat/forget" id="chat-forget-form">
        <button type="submit">New conversation</button>
    </form>
    <form class="inline" method="post" action="/chat/logout">
        <button type="submit">Switch user</button>
    </form>
{% endblock %}

<!-- tests/test_admin_panel.py:391 asserts "alice" appears in the /chat body;
     {{ username }} above satisfies it. -->

### `admin_templates/chat_login.html:1, 6, 25`

Switch parent to chat_base.html and bump the username input to 16px. The <h2>Sign in to chat</h2> at :6 MUST stay verbatim — it is asserted three times in the suite.

BEFORE (:1):  {% extends "base.html" %}
AFTER  (:1):  {% extends "chat_base.html" %}

UNCHANGED (:6) — asserted by tests/test_admin_panel.py:333, :382, :398:
    <h2 style="margin-top: 0;">Sign in to chat</h2>

BEFORE (:25): border-radius: 3px; font-family: inherit; font-size: 0.95rem; margin-bottom: 1rem;
AFTER  (:25): border-radius: 4px; font-family: var(--font-ui); font-size: 16px; margin-bottom: 1rem;

### `admin_panel.py:313-323`

Enable the codehilite markdown extension so fenced code blocks get Pygments token classes, gated by a new CHAT_CODE_HIGHLIGHT knob. guess_lang=False is load-bearing: without it Pygments guesses on unlabelled fences and syntax-colours plain prose.

BEFORE (admin_panel.py:315):
_MARKDOWN_EXTENSIONS = ["fenced_code", "tables", "nl2br"]

AFTER:
_MARKDOWN_EXTENSIONS = ["fenced_code", "tables", "nl2br"]
_MARKDOWN_EXTENSION_CONFIGS: dict = {}
if CHAT_CODE_HIGHLIGHT:
    _MARKDOWN_EXTENSIONS.append("codehilite")
    # guess_lang=False: without it Pygments guesses a lexer for unlabelled
    # fences and syntax-colours English prose. noclasses=False emits token
    # classes we style from _theme.html rather than inline styles that can't
    # respond to prefers-color-scheme.
    _MARKDOWN_EXTENSION_CONFIGS["codehilite"] = {
        "guess_lang": False,
        "noclasses": False,
    }


def _render_markdown(text: str) -> str:
    if not text:
        return ""
    return md_lib.markdown(
        text,
        extensions=_MARKDOWN_EXTENSIONS,
        extension_configs=_MARKDOWN_EXTENSION_CONFIGS,
    )

# and add CHAT_CODE_HIGHLIGHT to the existing `from fritz_utils import (...)`
# block at admin_panel.py:42-52, alphabetically after CHAT_ALLOWED_IMAGE_TYPES.

### `admin_panel.py:462-480`

Two one-line correctness fixes found while reading: chat_send's TemplateResponse calls omit is_admin (so the admin doc-upload panel vanishes after a no-JS send) and the error-path message dicts omit html (so the error reply skips the bubble-body branch at chat.html:121).

BEFORE (:462-468):
        return templates.TemplateResponse(request, "chat.html", {
            "username": user,
            "messages": [
                {"role": "user", "content": message},
                {"role": "fritz", "content": f"❌ An error occurred: {e}"},
            ],
        })

AFTER:
        return templates.TemplateResponse(request, "chat.html", {
            "username": user,
            "is_admin": fritz_utils.is_admin(user),
            "messages": [
                {"role": "user", "content": message, "html": None},
                {"role": "fritz", "content": f"❌ An error occurred: {e}",
                 "html": None},
            ],
        })

# Same "is_admin": fritz_utils.is_admin(user) addition to the success-path
# TemplateResponse at :473-480.

### `fritz_utils.py:164-169`

Add the single new knob, next to the existing CHAT_* block, following the module's env-var convention.

# Syntax-highlight fenced code blocks in Fritz's chat replies via Pygments.
# Set to "false" if the highlight CSS clashes with a customised theme or you
# want to drop the Pygments dependency. Rendering degrades to plain <pre><code>.
CHAT_CODE_HIGHLIGHT: bool = os.environ.get(
    "CHAT_CODE_HIGHLIGHT", "true"
).strip().lower() not in ("0", "false", "no", "off")

### `requirements.txt:291-292`

Pin Pygments explicitly. It already resolves to 2.19.2 at line 192, but only as a transitive dep of something in the LLM stack — codehilite must not depend on that accident.

BEFORE (:291-292):
# Server-side markdown rendering for Fritz's chat replies (Phase web-chat-3).
Markdown>=3.5

AFTER:
# Server-side markdown rendering for Fritz's chat replies (Phase web-chat-3).
Markdown>=3.5
# Syntax highlighting for fenced code blocks in chat replies (web-chat-redesign).
# Already resolved transitively via the LLM stack; pinned here so codehilite
# doesn't silently degrade if that transitive edge goes away.
Pygments>=2.17

### `.env.example:62-70`

Document the new knob AND the three pre-existing CHAT_* knobs that are implemented in fritz_utils.py:153/159/162 but were never documented — the repo convention says every knob appears here.

# ----- Web chat (:8001/chat — cookie identity, no password) -----
# HMAC key for the chat identity cookie. If unset, one is generated and
# persisted to .chat_cookie_secret on first boot (gitignored).
# CHAT_COOKIE_SECRET=
# Hard caps on chat uploads, in bytes. Defaults are 10 MiB each.
# CHAT_IMAGE_UPLOAD_MAX_BYTES=10485760
# CHAT_DOC_UPLOAD_MAX_BYTES=10485760
# Syntax-highlight fenced code blocks in Fritz's replies (needs Pygments).
# Set to false to fall back to plain <pre><code>.
# CHAT_CODE_HIGHLIGHT=true

### `tests/test_admin_panel.py:append after line 731`

New regression tests for exactly the defects being fixed. Deterministic, no Ollama, no network — they read rendered HTML from TestClient or the template files off disk.

class TestChatTemplateAccessibility(unittest.TestCase):
    def _chat_html(self):
        client = _build_client()
        client.post("/chat/login", data={"username": "alice"})
        return client.get("/chat").text

    def test_viewport_meta_present(self):
        self.assertIn('name="viewport"', self._chat_html())

    def test_transcript_is_a_log_region(self):
        html = self._chat_html()
        self.assertIn('id="chat-messages"', html)
        self.assertIn('role="log"', html)

    def test_status_region_present(self):
        self.assertIn('role="status"', self._chat_html())

    def test_forget_form_has_no_inline_onsubmit(self):
        # Regression guard: the old inline onsubmit was a JS SyntaxError, so
        # the handler compiled to null and POST /chat/forget fired with no
        # confirmation at all. Confirm now lives in an addEventListener.
        html = self._chat_html()
        self.assertNotIn("onsubmit", html)
        self.assertIn('id="chat-forget-form"', html)

    def test_login_page_has_viewport_meta(self):
        client = _build_client()
        self.assertIn('name="viewport"', client.get("/chat").text)


class TestChatTemplateSource(unittest.TestCase):
    """Assertions against the template files themselves — cheaper and more
    precise than scraping rendered output for CSS/JS properties."""

    def _read(self, name):
        return (Path(admin_panel.__file__).parent / "admin_templates" / name
                ).read_text(encoding="utf-8")

    def test_no_alert_calls_in_chat_script(self):
        self.assertNotIn("alert(", self._read("chat.html"))

    def test_theme_has_dark_mode_and_mobile_breakpoint(self):
        theme = self._read("_theme.html")
        self.assertIn("prefers-color-scheme: dark", theme)
        self.assertIn("@media (max-width", theme)
        self.assertIn("prefers-reduced-motion", theme)

    def test_chat_extends_chat_base_not_admin_base(self):
        self.assertIn('{% extends "chat_base.html" %}', self._read("chat.html"))

    def test_chat_base_has_no_admin_nav(self):
        chrome = self._read("chat_base.html")
        for href in ('href="/users"', 'href="/schedules"', 'href="/documents"'):
            self.assertNotIn(href, chrome)


class TestRenderMarkdownCodehilite(unittest.TestCase):
    def test_fenced_code_gets_codehilite_wrapper(self):
        html = admin_panel._render_markdown("```python\nprint('hi')\n```")
        self.assertIn("codehilite", html)
        self.assertIn("<pre>", html)   # same contract as the Phase-3 test

## Steps

1. STEP 1 — correctness and a11y quick wins (one commit, shippable alone). (a) Add the viewport meta to admin_templates/base.html after line 4. (b) Bump `--muted` at base.html:9 from `#738291` to `#5f6b76` — computed 5.08:1 on `--bg` and 5.45:1 on `--card`, both clearing AA. This is a stopgap so the a11y fix is not held hostage to step 2's palette; step 2 replaces the value again. (c) Delete the inline `onsubmit` at chat.html:105-106, add `id="chat-forget-form"`, and register the confirm via addEventListener at chat.html:166 — ABOVE the `if (!form || !list) return;` guard on line 168. (d) Add `role="log"` + `aria-label` to `#chat-messages` (chat.html:116), add the visually-hidden `#chat-status` `role="status"` element, and set/remove `aria-busy` on the streaming bubble at chat.html:314 and :375/:380/:387/:391. (e) Add the `prefers-reduced-motion` guard for the caret keyframe at chat.html:20-24. (f) Fix the two server-side gaps in admin_panel.chat_send at :462-468 and :473-480 (missing `is_admin`, missing `html`). Add the TestChatTemplateAccessibility class. Verify: `pytest tests/test_admin_panel.py -v` and `ruff check .`.
2. STEP 2 — extract the theme partial and lay in the butler's-study palette. Create admin_templates/_theme.html holding the `:root` tokens, reset, typography, focus rings, the `@media (max-width: 640px)` block and the `@media (prefers-color-scheme: dark)` override. Move base.html's entire `<style>` (lines 6-69) into it, replacing it with `{% include "_theme.html" %}`. While moving, swap all eight base.html hardcoded hexes (:23, :31, :47, :54, :60, :63, :65, :67) for vars. Land the light + dark values verbatim from the change-site sketch — the contrast ratios there are computed, not guessed. Do NOT touch chat.html yet. Verify: load every admin page (/, /users, /schedules, /documents) and confirm nothing regressed, then toggle OS dark mode and confirm the panel inverts.
3. STEP 3 — restyle chat.html against the tokens. Swap the eight hardcoded hexes at chat.html:12, :17, :32, :36, :64, :66, :70, :86. Point `.chat-bubble .bubble-body` at `--font-prose` and pin `code`/`pre` to `--font-mono`. Add the `.chat-bubble.fritz.streaming .bubble-body { white-space: pre-wrap; min-height: 1.6em }` rule — this is the whole reflow-jump fix. Wrap the form in `.chat-composer` and make it `position: sticky; bottom: 0` with `env(safe-area-inset-bottom)` padding. Raise the textarea to `font-size: 16px` (chat.html:58) and the login input likewise (chat_login.html:25). Add the mobile bubble-gutter block. Verify at 375px width in devtools: no horizontal body scroll, composer pinned, focusing the textarea does not zoom on an iOS simulator or Safari responsive mode.
4. STEP 4 — streaming UX. Introduce `applyToken()`, `isPinnedToBottom()`, `followScroll()`, `showNotice()` and `setStatus()` at the top of the IIFE. Replace the unconditional `window.scrollTo` at chat.html:397 with `followScroll(pinned)` calls inside each SSE branch, capturing `pinned` BEFORE the DOM mutation. Add the textarea `keydown` handler (Enter sends; skip on `shiftKey`, `isComposing`/keyCode 229, and `(pointer: coarse)`). Add the `#chat-stop-btn` + AbortController, swapping Send/Stop visibility. Handle `err.name === "AbortError"` distinctly in the catch. Replace the three `alert()` calls at chat.html:189, :199, :206 with `showNotice(..., "error")`. Gate the `textarea.focus()` at :396 behind `(pointer: fine)`.
5. STEP 5 — code highlighting and copy buttons. Add `CHAT_CODE_HIGHLIGHT` to fritz_utils.py, import it in admin_panel.py's fritz_utils import block (:42-52), and wire `codehilite` with `guess_lang: False` into `_MARKDOWN_EXTENSIONS`/`_render_markdown` (admin_panel.py:315-323). Pin `Pygments>=2.17` in requirements.txt. Generate the highlight CSS with `python -m pygments -S default -f html -a .codehilite` and `-S native` for dark, trim to the token classes markdown actually emits (`.k .kd .kn .s .s1 .s2 .c1 .cm .n .nb .nf .nc .mi .mf .o .p`), and paste both into _theme.html under the light block and the `prefers-color-scheme: dark` block respectively. Add `addCopyButtons()` and call it once on page load over `list` (for server-rendered history) and once per `done` frame over `body`. Keep the non-secure-context fallback — `navigator.clipboard` is undefined when the panel is reached over a LAN IP rather than 127.0.0.1.
6. STEP 6 — split chat_base.html from the admin chrome. Create admin_templates/chat_base.html as a standalone document including _theme.html, with the flex app-column layout and a `{% block chrome_actions %}`. Change chat.html:1 and chat_login.html:1 from `{% extends "base.html" %}` to `{% extends "chat_base.html" %}`. Move the two forms out of `.chat-header` (chat.html:102-113) into `{% block chrome_actions %}` and delete the now-empty `.chat-header` rules at chat.html:90-94. Do NOT change the `<h2>Sign in to chat</h2>` text at chat_login.html:6 or the `placeholder="Type a message..."` string — both are asserted by the suite. Add TestChatTemplateSource. Verify: a chat user sees no link to /users, /schedules, /documents, /health, or /.
7. STEP 7 — docs and changelog. Add the four `CHAT_*` entries to .env.example (one new knob plus three pre-existing undocumented ones). Add a `### Changed` entry to CHANGELOG.md under `## [Unreleased]` in the established phase-narrative voice — name the confirm SyntaxError explicitly since it was a live data-loss bug. Update the README `## Chat UI` section at line 312 to mention mobile support, dark mode, Enter-to-send and Stop. Final gate: `ruff check .` then `pytest tests/ --tb=short --cov=. --cov-fail-under=60`.

## Config and env changes

- NEW: CHAT_CODE_HIGHLIGHT (default "true") — fritz_utils.py, imported by admin_panel.py. Set false to skip the codehilite markdown extension and fall back to plain <pre><code>. Must be added to .env.example.
- Documentation-only backfill in .env.example for three knobs that already exist in fritz_utils.py but were never documented: CHAT_COOKIE_SECRET (fritz_utils.py:153), CHAT_IMAGE_UPLOAD_MAX_BYTES (:159), CHAT_DOC_UPLOAD_MAX_BYTES (:162). No code change; the repo convention is that every knob appears in .env.example.
- requirements.txt: add explicit `Pygments>=2.17` next to `Markdown>=3.5` (line 292). Already resolves to 2.19.2 at line 192 transitively, but codehilite must not rely on that.
- No new ports, secrets, or routes. Every route in admin_panel.create_app (:791-820) is unchanged.

## Tests
### New

- tests/test_admin_panel.py::TestChatTemplateAccessibility::test_viewport_meta_present — GET /chat with a login cookie, assert 'name="viewport"' in the body.
- tests/test_admin_panel.py::TestChatTemplateAccessibility::test_transcript_is_a_log_region — assert both 'id="chat-messages"' and 'role="log"'.
- tests/test_admin_panel.py::TestChatTemplateAccessibility::test_status_region_present — assert 'role="status"'.
- tests/test_admin_panel.py::TestChatTemplateAccessibility::test_forget_form_has_no_inline_onsubmit — assert 'onsubmit' NOT in the rendered page and 'id="chat-forget-form"' is. This is the direct regression guard for the SyntaxError that silently disabled the destructive-action confirm.
- tests/test_admin_panel.py::TestChatTemplateAccessibility::test_login_page_has_viewport_meta — GET /chat unauthenticated (renders chat_login.html).
- tests/test_admin_panel.py::TestChatTemplateSource::test_no_alert_calls_in_chat_script — read admin_templates/chat.html off disk, assert 'alert(' absent.
- tests/test_admin_panel.py::TestChatTemplateSource::test_theme_has_dark_mode_and_mobile_breakpoint — read _theme.html, assert 'prefers-color-scheme: dark', '@media (max-width', 'prefers-reduced-motion'. Guards the exact repo-wide gap the audit found (zero @media queries).
- tests/test_admin_panel.py::TestChatTemplateSource::test_chat_extends_chat_base_not_admin_base
- tests/test_admin_panel.py::TestChatTemplateSource::test_chat_base_has_no_admin_nav — assert href="/users", href="/schedules", href="/documents" absent from chat_base.html.
- tests/test_admin_panel.py::TestRenderMarkdownCodehilite::test_fenced_code_gets_codehilite_wrapper — assert 'codehilite' and '<pre>' in the rendered HTML.
- tests/test_admin_panel.py: extend TestChatSend with test_sync_send_passes_is_admin — patch fritz_utils.is_admin to return True, POST /chat/send, assert the admin doc-upload label ('Add to shared docs') appears. Guards the admin_panel.py:473-480 fix.

### Existing tests affected

- tests/test_admin_panel.py::TestChatPageWithCookie::test_authed_user_sees_chat_ui (lines 386-392) — asserts `self.assertIn("Type a message", r.text)` and `self.assertIn("alice", r.text)`. WILL BREAK if the composer placeholder is reworded or if {{ username }} is dropped when the header forms move into chrome_actions. Mitigation: keep the placeholder string containing the literal substring 'Type a message' and keep `<span>{{ username }}</span>` in the chrome. If you prefer a better placeholder, update this assertion to a stable hook such as `assertIn('id="chat-form"', r.text)` in the same commit — do not leave it to a follow-up.
- tests/test_admin_panel.py::TestChatBypassesAdminAuth::test_chat_landing_does_not_require_basic_auth (line 333), TestChatLogout::test_logout_clears_cookie (line 382), TestChatPageWithCookie::test_tampered_cookie_renders_login (line 398) — all three assert the literal 'Sign in to chat'. That string is the <h2> at chat_login.html:6. Do NOT reword it when reparenting chat_login.html to chat_base.html. All three pass unchanged if you leave line 6 alone.
- tests/test_admin_panel.py::TestRenderMarkdown::test_code_fence_renders_pre_code (lines 585-588) — asserts '<pre>' and 'print'. codehilite wraps output as <div class="codehilite"><pre><span></span><code>...<span class="nb">print</span>..., so both substrings survive and the test should still pass. It is AT RISK rather than certain-to-break; run it explicitly after step 5 and, if the wrapper shape changed upstream in a newer Markdown release, tighten the assertion to '<pre' rather than '<pre>'.
- tests/test_admin_panel.py::TestChatStreamSuccess::test_streams_token_events_then_done (lines 493-521) — asserts tokens == ['Very', 'Very well', 'Very well, sir.'], i.e. the cumulative wire contract. This plan does NOT change it. It is named here because it is the assertion the token-streaming item must edit, and because the client's applyToken() seam exists precisely so that edit stays a one-function change.
- tests/test_admin_panel.py::TestOverviewPage::test_renders_version_and_uptime (line 98) — asserts 'Overview'. overview.html still extends base.html, whose nav retains the link, so this is unaffected by the chat_base split. Listed to record that it was checked, not assumed.
- tests/test_admin_panel.py::TestChatSend::test_authed_send_invokes_ask_stuff_with_username (line 424) — asserts 'Very well.' appears in the rendered page. Adding the is_admin key to the TemplateResponse context does not affect it.
- No test in tests/ currently reads admin_templates/*.html from disk, so TestChatTemplateSource introduces that pattern. Use Path(admin_panel.__file__).parent — the suite reloads fritz_utils/admin_panel under a patched environment (test_admin_panel.py:38-54) and relative cwd is not guaranteed.

### Manual verification

- iPhone-width check: devtools at 375x812, GET /chat. Body must not scroll horizontally. The composer must be pinned to the bottom of the viewport, above the home-indicator inset. Bubbles must have ~1.25rem gutters, not 4rem.
- iOS zoom check: on a real iPhone or Safari responsive mode, tap the composer textarea and the username input on /chat login. The viewport must NOT zoom. This is the whole reason both are pinned at 16px.
- Confirm dialog: click 'New conversation' and press Cancel. The conversation must survive. Before this change the dialog never appeared and the checkpoint was deleted. Verify the checkpoint row count in fritz.db (`SELECT count(*) FROM checkpoints WHERE thread_id='<user>'`) is unchanged after a Cancel.
- Dark mode: toggle the OS theme with /chat open. Palette must invert with no white flash and no unreadable text. Check the code block, the user bubble, the pill, and the pending-attachment row specifically — those were the hardcoded hexes.
- Follow-scroll: send a long prompt, scroll up mid-stream, and confirm the page does NOT yank you back down. Scroll back to the bottom and confirm it resumes following.
- Enter-to-send: Enter sends on desktop; Shift+Enter inserts a newline; on a touch device Enter inserts a newline. With a Japanese/Chinese IME, pressing Enter to commit a candidate must NOT send the message.
- Stop: send a prompt, hit Stop mid-stream. The bubble must stop updating and the status must read 'Stopped.' THEN reload the page — the full reply WILL be present in history, because the server thread ran to completion. Confirm this is what happens and that it does not look like a bug to a user; if it does, retitle the button.
- Copy buttons: hover a code block, click Copy, paste elsewhere. Then reach the panel over a LAN IP instead of 127.0.0.1 (non-secure context) and confirm the button selects the code and says 'Ctrl+C' instead of throwing.
- Screen reader: with NVDA (Windows) or VoiceOver, send a message. The status region should announce 'Mister Fritz is composing a reply', then progress lines, then 'Reply received' — and must NOT read the partial reply out on every token frame.
- Reduced motion: enable the OS 'reduce motion' setting and confirm the streaming caret stops blinking.
- No-JS fallback: disable JavaScript entirely and POST a message. The synchronous /chat/send path must still render a full page with both bubbles, and (with is_admin true) the document-upload panel must still be present — that is the admin_panel.py:473-480 fix.
- Admin chrome: as a non-admin chat user, confirm there is no visible link to /, /users, /schedules, /documents, or /health, and no 'admin' in the header.

## Risks

- The confirm fix is the only change here with data-loss stakes, and it is easy to reintroduce the bug: if the addEventListener registration is placed AFTER `if (!form || !list) return;` at chat.html:168, any future change that removes #chat-form silently disables the confirm again. The test_forget_form_has_no_inline_onsubmit assertion catches the attribute coming back, but not misplacement — so also verify the Cancel path manually once. Detection: click New conversation, press Cancel, query fritz.db checkpoint rows.
- `min-height: 100dvh` in chat_base.html is unsupported below Safari 15.4 / Chrome 108. The `min-height: 100vh` declared immediately before it is the fallback, and dropping it makes the composer sit under the mobile URL bar. Detection: composer partially hidden behind browser chrome on an older phone.
- `navigator.clipboard` is undefined outside secure contexts. http://127.0.0.1:8001 qualifies via the localhost exemption, but the README documents SSH-forwarding and a user who instead browses a LAN IP gets a non-secure context. Without the guard the Copy button throws a TypeError into the console and does nothing. Detection: open the panel by LAN IP and click Copy.
- codehilite with `guess_lang` left at its default True will syntax-colour unlabelled fences — including prose Fritz emits inside triple backticks. Detection: ask Fritz for a plain quoted block and check for stray <span class="k"> around English words.
- The chat_base split removes chat.html's dependency on base.html. If a future admin-panel style is added only to base.html's block, /chat will silently miss it. Mitigated by putting everything shared in _theme.html; the risk is a developer adding a rule to the wrong file. Detection: TestChatTemplateSource::test_theme_has_dark_mode_and_mobile_breakpoint only guards the tokens, not future additions — code review is the real control.
- SEQUENCING (not a blocker, a rework risk): the token-streaming item changes SSE `token` frames from cumulative to delta. This plan does not depend on it, but it does touch the same render loop. The `applyToken()` seam confines the collision to one two-line function plus the assertion at tests/test_admin_panel.py:516. Land this item FIRST; doing token-streaming first means redoing the loop.
- SEQUENCING: the web-auth item will change what belongs in the chat chrome (real sessions, sign-out, possibly per-user badges). chat_base.html's `{% block chrome_actions %}` is the seam that makes that a template-block change rather than a restructure. Landing this item first makes web-auth cheaper; landing web-auth first means it builds chrome inside base.html that this item then has to move.
- Aggressive aria-live on the transcript would make the chat unusable with a screen reader — the whole reply would be re-read on every token frame. That is why role="log" plus a separate coarse #chat-status region is specified rather than aria-live on #chat-messages. Getting this wrong is worse than the current silence. Detection: the NVDA/VoiceOver check in the manual plan.
- Stop is cosmetic on the server side. admin_panel.py:558 spawns a detached daemon thread and _event_generator (:560-577) never checks request.is_disconnected(), so aborting the client does not stop Ollama, does not stop the checkpoint write, and does not stop the audit_log 'result: ok' entry. If the button is labelled in a way that implies cancellation, users will report it as broken. Real cancellation belongs to token-streaming or latency-tax.
- Moving base.html's <style> into an include is the highest-blast-radius mechanical edit here — a dropped brace breaks all eight templates at once. Do it as its own commit with no other changes so a bisect is trivial.
- System serif availability varies. "Iowan Old Style" is macOS/iOS, "Palatino Linotype" is Windows, and many Linux desktops land on the generic `serif`. Fritz's prose will look different across platforms. This is deliberate — the alternative is a webfont, which the offline-first / no-CDN deployment model rules out. Detection: eyeball on Windows, macOS and a Linux VM.

## Rollback
"Every change is confined to admin_templates/*.html (4 edited, 2 new), a ~15-line diff in admin_panel.py, a 6-line addition to fritz_utils.py, two lines in requirements.txt, and additive test classes. There is no schema change, no new route, no new port, and no persisted state, so `git revert` of the step commits restores the prior behaviour completely with no cleanup. Sequence the commits per step so a bisect lands on one concern; step 2 (moving base.html's <style> into _theme.html) must be its own commit with no other edits. A feature flag is not warranted for a template-only change on a localhost-bound panel, with one exception already in the plan: CHAT_CODE_HIGHLIGHT=false disables codehilite at runtime, which is the only change carrying a third-party rendering dependency. If a partial rollback is ever needed, deleting admin_templates/chat_base.html and reverting the two `{% extends %}` lines in chat.html:1 and chat_login.html:1 restores the admin chrome without touching anything else."

## Open questions for you to decide

- Should step 6 (chat_base.html) move to the front? The 'butler's study' layout genuinely wants a flex app column, which fights base.html's `main { padding: 2rem; max-width: 1100px }`. This plan avoids that fight by using `position: sticky; bottom: 0` on the composer, which works in document flow today AND unchanged after the split — so the ordering is safe as written. But if the engineer wants a scroll-container transcript (independent scrollbar rather than document scroll), do step 6 first, because that layout cannot be expressed without the standalone chrome and the follow-scroll helpers would then read the container's scrollTop instead of window.scrollY.
- What should the Stop button actually claim? 'Stop' is honest about the UI but not about the server. Options: (a) label it 'Stop' with the explanatory tooltip specified here; (b) label it 'Hide reply'; (c) defer the button entirely until token-streaming can also cancel the worker thread. This plan picks (a). Owner's call — it is a product-honesty question, not a technical one.
- Enter-to-send on touch devices is disabled here on the theory that a soft keyboard offers no reliable Shift. Some users will want the opposite. If so, the fix is a per-user preference, which implies persistence and therefore belongs with web-auth rather than here.
- Should the admin panel adopt the parchment palette too, or stay neutral? This plan applies _theme.html to both because sharing one token set is the smallest change and it fixes the h2/th/muted contrast failures on the admin pages for free. If the owner wants the admin panel to stay visually distinct from the chat, _theme.html should carry only the reset + responsive + a11y rules, and the palette should split into two `:root` blocks. That is a strictly larger change.
- CHAT_CODE_HIGHLIGHT may be one knob more than this repo needs. It is included because the plan adds Pygments as an explicit dependency and the repo convention is an env var per behavioural switch. If the owner would rather not grow the config surface, drop the flag and always enable codehilite — Markdown's codehilite already degrades to a plain <pre><code> when Pygments is unimportable, so the failure mode is benign either way.
- The exact Pygments token classes that need styling cannot be determined statically — it depends on which lexers Fritz's replies actually trigger. The step-5 class list is the common subset. The experiment that settles it: run the panel, ask Fritz for code in Python, JS, bash and JSON, and inspect the emitted <span class> values; add any that render as unstyled default-colour text.
- Whether TestRenderMarkdown::test_code_fence_renders_pre_code survives codehilite is asserted here from the known wrapper shape (<div class="codehilite"><pre>), not from a run — the test environment for this plan did not execute the suite. Running `pytest tests/test_admin_panel.py::TestRenderMarkdown -v` immediately after step 5 settles it in one command.
