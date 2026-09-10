"""
Advance Quiz Bot — /google research command.

Independent feature: does not modify /solve or /search.

Usage:
    /google <topic>
    Reply to a text message + /google
    Reply to a poll/quiz + /google

Search provider: Google Programmable Search / Custom Search JSON API.
The research answer is then written by the bot's existing AI provider stack.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any, Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from quizbot.shared import config
from quizbot.shared.utils.http import request_json

from ..ai_providers import ai_generate
from ..telegram_utils import safe_send_message
from ..google_report import build_google_report_html

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 500
MAX_RESULTS = 8
MAX_SNIPPET_LENGTH = 900
MAX_RESULT_TEXT = 9000


def _clean_command_query(text: str) -> str:
    return re.sub(r"^/google(?:@\w+)?\s*", "", text or "", flags=re.I).strip()


def _poll_text(message: Any) -> Optional[str]:
    poll = getattr(message, "poll", None)
    if not poll:
        return None

    question = str(getattr(poll, "question", "") or "").strip()
    options: list[str] = []
    for i, option in enumerate(getattr(poll, "options", []) or []):
        value = str(getattr(option, "text", "") or "").strip()
        if value:
            options.append(f"{chr(65 + i)}) {value}")

    if not question and not options:
        return None

    parts = [question] if question else []
    if options:
        parts.append("Options:\n" + "\n".join(options))
    return "\n\n".join(parts).strip()


def _reply_query(message: Any) -> Optional[str]:
    reply = getattr(message, "reply_to_message", None)
    if not reply:
        return None

    poll = _poll_text(reply)
    if poll:
        return poll

    text = getattr(reply, "text", None) or getattr(reply, "caption", None)
    if text:
        return str(text).strip()
    return None


async def _google_search(query: str) -> list[dict[str, str]]:
    api_key = config.GOOGLE_SEARCH_API_KEY
    cx = config.GOOGLE_SEARCH_CX
    if not api_key or not cx:
        raise RuntimeError(
            "Google Search is not configured. Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX."
        )

    status, data = await request_json(
        "GET",
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": api_key,
            "cx": cx,
            "q": query[:MAX_QUERY_LENGTH],
            "num": MAX_RESULTS,
            "safe": "active",
        },
    )

    if status != 200 or not isinstance(data, dict):
        raise RuntimeError(f"Google Search HTTP {status}: {str(data)[:500]}")

    items = data.get("items") or []
    results: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        link = str(item.get("link") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if not title or not link:
            continue
        results.append({
            "title": title[:300],
            "link": link,
            "snippet": snippet[:MAX_SNIPPET_LENGTH],
        })

    if not results:
        raise RuntimeError("Google Search returned no useful results.")
    return results


def _research_prompt(query: str, results: list[dict[str, str]]) -> str:
    source_block = []
    for i, item in enumerate(results, 1):
        source_block.append(
            f"SOURCE {i}\n"
            f"TITLE: {item['title']}\n"
            f"URL: {item['link']}\n"
            f"SNIPPET: {item['snippet']}"
        )

    return f"""
You are an expert research writer for competitive-exam learners.

USER RESEARCH REQUEST:
{query}

GOOGLE SEARCH RESULTS:
{chr(10).join(source_block)}

TASK:
Create a detailed, accurate research article about the request using ONLY the
facts supported by the supplied Google results. Do not invent facts. If the
sources do not establish a detail, omit it or clearly say it could not be
verified from the supplied results.

IMPORTANT:
- If the request is a Telegram poll/quiz, determine what the question asks,
  assess every option against the supplied evidence, and clearly state the
  best-supported answer. Do not assume an option is correct merely because it
  is written as an option.
- Preserve dates, names, numbers and terminology carefully.
- Use the user's language when practical; Hindi questions should get Hindi.
- Write useful detail, not a one-paragraph answer.
- Use Markdown headings, bold text, bullet lists and Markdown tables when a
  table improves clarity.
- Put [1], [2], etc. immediately after factual claims where useful. These
  numbers refer to the supplied sources in order.
- Do not create a Sources section; the HTML page will add the source cards.
- Do not use raw HTML.
- Do not use LaTeX.

Suggested structure when applicable:
# Title
## Direct answer / conclusion
## Detailed explanation
## Key facts
## Timeline / comparison table
## Exam points

Return only the Markdown article.
""".strip()


async def google_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    try:
        query = _clean_command_query(message.text or "")
        if not query:
            query = _reply_query(message) or ""

        if not query:
            await safe_send_message(
                ctx,
                message.chat_id,
                "🔎 Usage: /google <topic>\n\nOr reply to a text/poll/quiz with /google.",
            )
            return

        status = await safe_send_message(ctx, message.chat_id, "🔎 Google पर search कर रहा हूँ…")

        try:
            results = await _google_search(query)
            prompt = _research_prompt(query, results)
            article = await ai_generate(message.from_user.id if message.from_user else 0, prompt, max_tokens=7000)
            article = article.strip()
            if not article:
                raise RuntimeError("AI returned an empty research article.")

            html_bytes, filename = build_google_report_html(query, article, results)
            await ctx.bot.send_document(
                chat_id=message.chat_id,
                document=html_bytes,
                filename=filename,
                caption="🔎 Google Research Report",
                parse_mode=ParseMode.HTML,
                reply_to_message_id=message.message_id,
            )

        except Exception as exc:
            logger.error("/google failed for %r: %s", query, exc, exc_info=True)
            await safe_send_message(
                ctx,
                message.chat_id,
                "❌ Google research could not be generated.\nPlease check Google Search API settings and try again.",
            )

    except Exception as exc:
        logger.error("google_command error: %s", exc, exc_info=True)


def register(application: Application) -> None:
    application.add_handler(CommandHandler("google", google_command))
