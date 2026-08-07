"""HMAC-signed cookies for the local chat UI.

The chat surface (`/chat/*`) has its own identity model separate from the
admin panel's HTTP Basic auth. A user picks a username on first visit; we
store it in a tamper-resistant cookie so they don't have to type it again.

The cookie is issued only after `POST /chat/login` verifies the shared
`CHAT_PASSWORD`, so holding a valid cookie does mean the bearer cleared the
perimeter. What the cookie does *not* establish is which person cleared it:
anyone who knows the password may type any username and get that person's
memories, schedules and thread. Set `CHAT_ALLOWED_USERS` to narrow the set of
claimable names; per-user invite tokens are the real fix and are not built yet.

So: the signature makes the username unforgeable *after* issuance, and the
password makes issuance non-anonymous. Neither makes the username a proven
identity.

Cookie payload format:
    <username>:<unix_ts>:<hex_hmac>

The HMAC covers `<username>:<unix_ts>` so neither field can be tampered
without re-signing.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Optional

# Cookie name used for the chat identity cookie.
COOKIE_NAME = "fritz_chat_id"

# Cookie lifetime in seconds. Rolled forward on every request.
COOKIE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60  # 30 days


def make_cookie(username: str, secret: str, *, now: Optional[float] = None) -> str:
    """Return a signed cookie value for the given username.

    `now` is exposed for tests; production callers don't pass it.
    """
    if not username:
        raise ValueError("username is required")
    if not secret:
        raise ValueError("secret is required")
    ts = int(now if now is not None else time.time())
    payload = f"{username}:{ts}".encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"{username}:{ts}:{sig}"


def verify_cookie(
    token: Optional[str],
    secret: str,
    *,
    max_age_seconds: int = COOKIE_MAX_AGE_SECONDS,
    now: Optional[float] = None,
) -> Optional[str]:
    """Verify a cookie value and return the username if valid, else None.

    Checks signature first (constant-time) and only then checks age.
    """
    if not token or not secret:
        return None
    parts = token.split(":")
    if len(parts) != 3:
        return None
    username, ts_str, sig = parts
    if not username or not ts_str or not sig:
        return None
    try:
        ts = int(ts_str)
    except ValueError:
        return None

    expected = hmac.new(
        secret.encode("utf-8"),
        f"{username}:{ts}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None

    current = int(now if now is not None else time.time())
    if current - ts > max_age_seconds:
        return None
    if current + 60 < ts:
        # Cookie minted in the future by more than a minute — clock skew or tampering.
        return None
    return username
