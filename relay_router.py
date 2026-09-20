"""Routing for replies to relayed DMs.

When Fritz carries a message for someone (see relay_store and /tell), the
recipient can answer it by replying to that DM in Discord. This module decides
whether an incoming DM is such an answer, and carries it back.

INERT IN THIS COMMIT. try_route_reply returns False for every message, so
main_discord's DM path behaves exactly as it does today. The hook lands on its
own so the ordering change above the bot's most load-bearing line is reviewable
— and revertible — by itself; PR 4 fills the function in.

**This module must never import mister_fritz, directly or through anything
else.** A relayed message is attacker-controlled text, and the agent it would
otherwise reach binds file tools with read/write/exec whenever the user has a
workspace. Code that cannot reach ask_stuff cannot contaminate the LangGraph
checkpoint or the Chroma memories that are injected into that person's next
prompt. tests/test_relay_router.py asserts the import graph, because a guard
that says "do not import this" is only as good as the thing that checks.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def try_route_reply(client, ctx) -> bool:
    """Was this DM an answer to a relayed message, and has it been dealt with?

    True means on_message must stop: the message was a relay reply and this
    module has already answered it, one way or another. False means the
    message is nothing to do with the relay and belongs to the agent, exactly
    as before — which is what every message gets in this commit.
    """
    return False
