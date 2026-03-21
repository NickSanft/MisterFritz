"""main_telegram.py — Telegram adapter for MisterFritz."""

import asyncio
import logging
import os
import tempfile

from telegram import Update
from telegram.ext import Application, MessageHandler, filters

from fritz_utils import TELEGRAM_BOT_TOKEN, MessageSource, validate_config
from mister_fritz import ask_stuff
from observability import init_logging, start_metrics_server
from stt import transcribe

init_logging()
logger = logging.getLogger(__name__)


async def handle_text(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    text = update.message.text or ""
    await update.message.reply_text("✍️ *Mister Fritz is thinking...*", parse_mode="Markdown")

    loop = asyncio.get_running_loop()
    response_data = await loop.run_in_executor(
        None,
        lambda: ask_stuff(text, MessageSource.TELEGRAM_TEXT, user_id),
    )
    reply = response_data.get("text") or "I appear to have misplaced my thoughts."
    # Telegram max message length is 4096 characters
    await update.message.reply_text(reply[:4096])


async def handle_voice(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    voice = update.message.voice
    tg_file = await context.bot.get_file(voice.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        ogg_path = tmp.name
    try:
        await tg_file.download_to_drive(ogg_path)
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, transcribe, ogg_path)
        if not text:
            await update.message.reply_text(
                "I'm afraid I couldn't make heads or tails of that audio."
            )
            return

        response_data = await loop.run_in_executor(
            None,
            lambda: ask_stuff(text, MessageSource.TELEGRAM_VOICE, user_id),
        )
        reply = response_data.get("text") or "Most peculiar — I have no response."
        await update.message.reply_text(reply[:4096])
    finally:
        if os.path.exists(ogg_path):
            os.remove(ogg_path)


def main():
    validate_config()
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Add it to your .env file or environment."
        )
    start_metrics_server()
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logger.info("Telegram bot starting")
    app.run_polling()


if __name__ == "__main__":
    main()
