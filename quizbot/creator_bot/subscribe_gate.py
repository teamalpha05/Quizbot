"""
Advance Quiz Bot — Open Source Project
This project was originally developed by Gagan (github.com/devgaganin).
Reference: https://t.me/advance_quiz_bot
The codebase has been reviewed and verified with the assistance of Claude AI.
"""

from __future__ import annotations

import logging

from pyrogram import Client
from pyrogram.errors import UserNotParticipant
from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from quizbot.shared import config

logger = logging.getLogger(__name__)

_JOIN_PROMPT_PHOTO = "https://graph.org/file/d44f024a08ded19452152.jpg"


async def subscribe_gate(app: Client, m: Message) -> bool:
    """Block commands until the user joins REQUIRED_SUB_CHANNEL.

    Also keeps the existing LOG_GROUP banned-user check.
    Returns True when the command must be blocked.
    """
    # 1. Required channel subscription check
    if config.REQUIRED_SUB_CHANNEL:
        channel = config.REQUIRED_SUB_CHANNEL.lstrip("@")

        try:
            member = await app.get_chat_member(channel, m.from_user.id)
            status = str(member.status).upper()

            # User is not a member / has been removed or banned.
            if status in (
                "CHATMEMBERSTATUS.LEFT",
                "CHATMEMBERSTATUS.BANNED",
            ):
                raise UserNotParticipant

        except UserNotParticipant:
            await m.reply_photo(
                _JOIN_PROMPT_PHOTO,
                caption="📢 Please join our channel to continue.",
                reply_markup=InlineKeyboardMarkup(
                    [[
                        InlineKeyboardButton(
                            "🔗 Join Channel",
                            url=f"https://t.me/{channel}",
                        )
                    ]]
                ),
            )
            return True

        except Exception as exc:
            # Do not silently allow the command when the required
            # subscription check itself fails.
            logger.error(
                "Required channel subscription check failed for %s: %s",
                channel,
                exc,
            )
            return True

    # 2. Keep the existing LOG_GROUP banned-user check
    if config.LOG_GROUP:
        try:
            member = await app.get_chat_member(
                config.LOG_GROUP,
                m.from_user.id,
            )
            status = str(member.status).upper()

            if status == "CHATMEMBERSTATUS.BANNED":
                await m.reply_text("🚫 Banned")
                return True

        except UserNotParticipant:
            # Not being a member of LOG_GROUP is not itself a ban.
            pass

        except Exception as exc:
            # Preserve the old behavior for LOG_GROUP failures:
            # do not block a user just because this optional check failed.
            logger.debug(
                "LOG_GROUP check failed (allowing through): %s",
                exc,
            )

    return False
