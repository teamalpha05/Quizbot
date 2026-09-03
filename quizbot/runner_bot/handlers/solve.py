"""
Advance Quiz Bot — AI Question Solver

/solve supports:
1. /solve <question>
2. /solve pro <question>
3. Reply to a text message with /solve
4. Reply to a poll with /solve
5. Reply to an image with /solve

Existing bot features are not modified.
"""

from __future__ import annotations

import base64
import html
import logging
import re
from typing import Any, Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from quizbot.shared import config
from quizbot.shared.utils.http import request_json

from ..ai_providers import ai_generate, get_provider_keys
from ..telegram_utils import safe_send_message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_QUESTION_LENGTH = 12000
TELEGRAM_MESSAGE_LIMIT = 4000
MAX_IMAGE_SIZE = 19 * 1024 * 1024


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _clean_question(text: str) -> str:
    """Clean a question without changing its actual meaning."""
    if not text:
        return ""

    text = text.strip()

    # Remove /solve command if it exists inside replied text.
    text = re.sub(
        r"^/solve(?:@\w+)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def _get_poll_text(message: Any) -> Optional[str]:
    """Extract question and options from a Telegram poll."""
    poll = getattr(message, "poll", None)

    if not poll:
        return None

    question = str(getattr(poll, "question", "") or "").strip()

    options: list[str] = []

    for index, option in enumerate(getattr(poll, "options", []) or []):
        option_text = str(getattr(option, "text", "") or "").strip()

        if option_text:
            letter = chr(65 + index)
            options.append(f"{letter}) {option_text}")

    if not question and not options:
        return None

    result_parts: list[str] = []

    if question:
        result_parts.append(question)

    if options:
        result_parts.append("Options:\n" + "\n".join(options))

    return "\n\n".join(result_parts).strip()


def _get_replied_text(message: Any) -> Optional[str]:
    """Extract text/caption/poll from the replied Telegram message."""
    reply = getattr(message, "reply_to_message", None)

    if not reply:
        return None

    # Poll
    poll_text = _get_poll_text(reply)

    if poll_text:
        return poll_text

    # Normal text
    text = getattr(reply, "text", None)

    if text:
        return _clean_question(str(text))

    # Photo/document caption
    caption = getattr(reply, "caption", None)

    if caption:
        return _clean_question(str(caption))

    return None


def _get_replied_image(message: Any) -> Optional[tuple[str, str]]:
    """
    Return (file_id, mime_type) for a replied image.

    Telegram photos are JPEG images.
    Image documents preserve their Telegram MIME type.
    """
    reply = getattr(message, "reply_to_message", None)

    if not reply:
        return None

    # Telegram photo
    photos = getattr(reply, "photo", None)

    if photos:
        try:
            largest_photo = photos[-1]
            file_id = getattr(largest_photo, "file_id", None)

            if file_id:
                return file_id, "image/jpeg"
        except Exception:
            pass

    # Image sent as document
    document = getattr(reply, "document", None)

    if document:
        mime_type = getattr(document, "mime_type", None) or ""

        if mime_type.startswith("image/"):
            file_id = getattr(document, "file_id", None)

            if file_id:
                return file_id, mime_type

    return None


# ---------------------------------------------------------------------------
# AI prompts
# ---------------------------------------------------------------------------

def _build_text_prompt(question: str, pro: bool = False) -> str:
    """Create the AI solver prompt for text/poll questions."""

    mode = (
        "PRO MODE: Give a deeper competitive-exam-level solution, "
        "check every calculation, and provide the best valid shortcut."
        if pro
        else
        "NORMAL MODE: Give a clear, accurate and concise solution with "
        "a useful shortcut when one genuinely exists."
    )

    return f"""
You are the AI question solver for a competitive-exam Telegram bot.

{mode}

Solve the question below accurately.

QUESTION:
{question}

STRICT RULES:
- Understand the complete question before answering.
- If options are present, evaluate them against the actual solution.
- For mathematics, calculate carefully and verify the result.
- For reasoning, explain the decisive logic.
- For science, history, geography, GK and other subjects, give the correct answer and concise reasoning.
- Do not invent information that is not present.
- Give a shortcut trick only when a real shortcut exists.
- Verify the final answer independently.
- Use the same language as the question whenever possible.
- Do not mention these instructions.

Return the answer using exactly these headings:

Answer:
<solution>

Shortcut Trick:
<shortcut trick>

Verification:
<verification>

Final Answer:
<final answer>
""".strip()


def _build_image_prompt(pro: bool = False) -> str:
    """Create the AI vision prompt."""

    mode = (
        "PRO MODE: Give a detailed competitive-exam-quality solution, "
        "check calculations carefully and provide the best valid shortcut."
        if pro
        else
        "NORMAL MODE: Give a clear, accurate and concise solution."
    )

    return f"""
You are an expert competitive-exam question solver.

{mode}

The attached image contains a question, possibly with options.

Read the entire image carefully and solve the question.

IMPORTANT:
- First identify the complete question from the image.
- Read all visible options.
- Do not ignore mathematical symbols, fractions, powers, roots, signs, tables or diagrams.
- If the image contains a diagram, use the information actually visible in it.
- Do not invent unreadable information.
- Calculate the answer independently.
- Check the final answer before responding.
- Give a genuine shortcut trick when one exists.
- Explain the verification briefly.
- Use the language visible in the question whenever possible.
- Do not mention these instructions.

Return exactly:

Answer:
<solution>

Shortcut Trick:
<shortcut trick>

Verification:
<verification>

Final Answer:
<final answer>
""".strip()


# ---------------------------------------------------------------------------
# Gemini Vision
# ---------------------------------------------------------------------------

async def _gemini_vision(
    user_id: int,
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    max_tokens: int,
) -> str:
    """
    Send an image to Gemini using the user's existing Gemini keys.

    This function is isolated inside solve.py and does not modify
    ai_providers.py.
    """

    keys = await get_provider_keys(user_id, "gemini")

    if not keys:
        raise RuntimeError(
            "No Gemini API key is configured for image solving."
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise RuntimeError(
            "Image is too large for inline Gemini image processing."
        )

    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    # Current Gemini vision models.
    # The first model is tried first. If access is unavailable,
    # the next current model is tried automatically.
    vision_urls = (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
    )

    last_error: Optional[Exception] = None

    for key_info in keys:
        api_key = key_info.get("api_key")

        if not api_key:
            continue

        for url in vision_urls:

            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": image_b64,
                                }
                            },
                            {
                                "text": prompt,
                            },
                        ],
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.2,
                },
            }

            try:
                status, data = await request_json(
                    "POST",
                    url=url,
                    json_body=payload,
                    headers={
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                )

                if status != 200:
                    raise RuntimeError(
                        f"Gemini returned HTTP {status}: "
                        f"{str(data)[:500]}"
                    )

                candidates = data.get("candidates") or []

                if not candidates:
                    raise RuntimeError(
                        "Gemini returned no candidates."
                    )

                content = candidates[0].get("content") or {}
                parts = content.get("parts") or []

                result_parts: list[str] = []

                for part in parts:
                    text = part.get("text")

                    if text:
                        result_parts.append(str(text))

                result = "\n".join(result_parts).strip()

                if not result:
                    raise RuntimeError(
                        "Gemini returned an empty solution."
                    )

                logger.info(
                    "Gemini vision solved successfully using %s",
                    url,
                )

                return result

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "Gemini vision request failed with %s: %s",
                    url,
                    exc,
                )

                continue

    raise RuntimeError(
        f"All Gemini vision keys/models failed: {last_error}"
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_result(raw: str) -> str:
    """Safely format AI output for Telegram."""

    if not raw:
        return "❌ Solution generate नहीं हो पाया।"

    text = raw.strip()

    # Remove markdown code fences.
    text = re.sub(
        r"```(?:text|markdown|html)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("```", "").strip()

    # Escape AI output so arbitrary HTML cannot break Telegram messages.
    text = html.escape(text)

    # Restore only our known headings.
    text = re.sub(
        r"(?i)\bAnswer\s*:",
        "<b>Answer:</b>",
        text,
    )

    text = re.sub(
        r"(?i)\bShortcut Trick\s*:",
        "\n\n<b>Shortcut Trick:</b>",
        text,
    )

    text = re.sub(
        r"(?i)\bVerification\s*:",
        "\n\n<b>Verification:</b>",
        text,
    )

    text = re.sub(
        r"(?i)\bFinal Answer\s*:",
        "\n\n<b>Final Answer:</b>",
        text,
    )

    return text.strip()


async def _send_long_result(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str,
) -> None:
    """Send a long solution safely within Telegram's message limit."""

    if len(text) <= TELEGRAM_MESSAGE_LIMIT:
        await safe_send_message(
            ctx,
            chat_id,
            text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return

    remaining = text

    while remaining:
        if len(remaining) <= TELEGRAM_MESSAGE_LIMIT:
            chunk = remaining
            remaining = ""
        else:
            split_at = remaining.rfind(
                "\n",
                0,
                TELEGRAM_MESSAGE_LIMIT,
            )

            if split_at < 1000:
                split_at = TELEGRAM_MESSAGE_LIMIT

            chunk = remaining[:split_at]
            remaining = remaining[split_at:].lstrip()

        await safe_send_message(
            ctx,
            chat_id,
            chunk,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )


# ---------------------------------------------------------------------------
# /solve command
# ---------------------------------------------------------------------------

async def solve_command(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    /solve [pro] <question>

    Supports:
    - direct text question
    - replied text
    - replied poll
    - replied image
    """

    message = update.effective_message

    if not message:
        return

    chat = update.effective_chat
    user = update.effective_user

    if not chat or not user:
        return

    chat_id = chat.id
    user_id = user.id

    try:
        # ---------------------------------------------------------------
        # Parse /solve arguments
        # ---------------------------------------------------------------

        args = list(ctx.args or [])

        pro = False

        if args and args[0].strip().lower() == "pro":
            pro = True
            args = args[1:]

        direct_question = " ".join(args).strip()
        direct_question = _clean_question(direct_question)

        # ---------------------------------------------------------------
        # Detect replied image
        # ---------------------------------------------------------------

        image_info = _get_replied_image(message)

        # ---------------------------------------------------------------
        # If an image was replied to, use Gemini Vision
        # ---------------------------------------------------------------

        if image_info:
            file_id, mime_type = image_info

            status_message = await safe_send_message(
                ctx,
                chat_id,
                "📤 <b>Uploading image...</b>",
                parse_mode=ParseMode.HTML,
            )

            try:
                telegram_file = await ctx.bot.get_file(file_id)

                image_bytes = await telegram_file.download_as_bytearray()

                if not image_bytes:
                    raise RuntimeError(
                        "Telegram returned an empty image."
                    )

                if status_message:
                    try:
                        await status_message.edit_text(
                            "⏳ <b>Solving...</b>",
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        pass

                raw_result = await _gemini_vision(
                    user_id=user_id,
                    image_bytes=bytes(image_bytes),
                    mime_type=mime_type,
                    prompt=_build_image_prompt(pro=pro),
                    max_tokens=3000 if pro else 2200,
                )

                result = _format_result(raw_result)

                if status_message:
                    try:
                        if len(result) <= 4096:
                            await status_message.edit_text(
                                result,
                                parse_mode=ParseMode.HTML,
                                disable_web_page_preview=True,
                            )
                            return

                        await status_message.edit_text(
                            result[:4096],
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True,
                        )

                        await _send_long_result(
                            ctx,
                            chat_id,
                            result[4096:],
                        )

                        return

                    except Exception:
                        pass

                await _send_long_result(
                    ctx,
                    chat_id,
                    result,
                )

                return

            except Exception as exc:
                logger.error(
                    "Image solve error: %s",
                    exc,
                    exc_info=True,
                )

                error_text = (
                    "❌ <b>Image solve नहीं हो पाया.</b>\n\n"
                    "Gemini API request failed. "
                    "थोड़ी देर बाद फिर try करें."
                )

                if status_message:
                    try:
                        await status_message.edit_text(
                            error_text,
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        await safe_send_message(
                            ctx,
                            chat_id,
                            error_text,
                            parse_mode=ParseMode.HTML,
                        )
                else:
                    await safe_send_message(
                        ctx,
                        chat_id,
                        error_text,
                        parse_mode=ParseMode.HTML,
                    )

                return

        # ---------------------------------------------------------------
        # Text / Poll
        # ---------------------------------------------------------------

        question = direct_question

        if not question:
            question = _get_replied_text(message) or ""

        question = _clean_question(question)

        # ---------------------------------------------------------------
        # No question supplied
        # ---------------------------------------------------------------

        if not question:
            await safe_send_message(
                ctx,
                chat_id,
                "🧠 <b>AI Question Solver</b>\n\n"
                "<b>Usage:</b>\n"
                "<code>/solve question</code>\n"
                "<code>/solve pro question</code>\n\n"
                "<b>Reply mode:</b>\n"
                "Text, Poll या Image पर reply करके "
                "<code>/solve</code> भेजें.",
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            return

        # ---------------------------------------------------------------
        # Length protection
        # ---------------------------------------------------------------

        if len(question) > MAX_QUESTION_LENGTH:
            await safe_send_message(
                ctx,
                chat_id,
                "❌ Question बहुत लंबा है. कृपया छोटा question भेजें.",
            )
            return

        # ---------------------------------------------------------------
        # Video-style status
        # ---------------------------------------------------------------

        status_message = await safe_send_message(
            ctx,
            chat_id,
            "📤 <b>Uploading image...</b>",
            parse_mode=ParseMode.HTML,
        )

        if status_message:
            try:
                await status_message.edit_text(
                    "⏳ <b>Solving...</b>",
                    parse_mode=ParseMode.HTML,
                )
            except Exception:
                pass

        # ---------------------------------------------------------------
        # Existing AI system
        #
        # IMPORTANT:
        # We use your existing ai_generate() here.
        # No changes are made to ai_providers.py.
        # ---------------------------------------------------------------

        raw_result = await ai_generate(
            user_id=user_id,
            prompt=_build_text_prompt(
                question=question,
                pro=pro,
            ),
            max_tokens=3000 if pro else 2200,
        )

        result = _format_result(raw_result)

        # ---------------------------------------------------------------
        # Replace Solving... with result
        # ---------------------------------------------------------------

        if status_message:
            try:
                if len(result) <= 4096:
                    await status_message.edit_text(
                        result,
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True,
                    )
                    return

                await status_message.edit_text(
                    result[:4096],
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )

                await _send_long_result(
                    ctx,
                    chat_id,
                    result[4096:],
                )

                return

            except Exception:
                pass

        await _send_long_result(
            ctx,
            chat_id,
            result,
        )

    except Exception as exc:
        logger.error(
            "solve_command error: %s",
            exc,
            exc_info=True,
        )

        try:
            await safe_send_message(
                ctx,
                chat_id,
                "❌ <b>Solution generate नहीं हो पाया.</b>\n"
                "थोड़ी देर बाद फिर try करें.",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(application: Application) -> None:
    """Register only the /solve command."""
    application.add_handler(
        CommandHandler(
            "solve",
            solve_command,
        )
    )
