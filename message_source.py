from enum import Enum

class MessageSource(Enum):
    DISCORD_TEXT = 0,
    DISCORD_VOICE = 1,
    LOCAL = 2
