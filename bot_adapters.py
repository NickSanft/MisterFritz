"""Shared helpers used by both the Discord and Telegram entrypoints.

Keep this module deliberately small. Anything platform-specific (streaming,
attachments, voice) belongs in the respective main_*.py adapter, not here.
"""


def split_into_chunks(s: str, chunk_size: int = 2000) -> list[str]:
    """Split a string into chunks of at most chunk_size characters.

    chunk_size defaults to Discord's 2000-char message cap. Pass 4096 for
    Telegram's reply limit.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]
