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

# Web verification limits. The Telegram poll answer is treated only as a
# claim/reference; it is never treated as the final truth.
MAX_WEB_QUERY_LENGTH = 500
MAX_WEB_RESULTS = 6
MAX_WEB_SNIPPET_LENGTH = 900


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _clean_question(text: str) -> str:
    """Clean a question without changing its actual meaning."""
    if not text:
        return ""

    text = text.strip()

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

    # Telegram's marked answer is only a reference. It can itself be wrong,
    # so the solver must independently verify it.
    marked_id = getattr(poll, "correct_option_id", None)

    result_parts: list[str] = []

    if question:
        result_parts.append(question)

    if options:
        result_parts.append(
            "Options:\n" + "\n".join(options)
        )

    if isinstance(marked_id, int) and 0 <= marked_id < len(options):
        result_parts.append(
            "Telegram marked answer (REFERENCE ONLY — may be wrong): "
            + options[marked_id]
        )

    return "\n\n".join(result_parts).strip()


def _get_replied_text(message: Any) -> Optional[str]:
    """Extract text, caption, or poll from the replied message."""
    reply = getattr(message, "reply_to_message", None)

    if not reply:
        return None

    poll_text = _get_poll_text(reply)

    if poll_text:
        return poll_text

    text = getattr(reply, "text", None)

    if text:
        return _clean_question(str(text))

    caption = getattr(reply, "caption", None)

    if caption:
        return _clean_question(str(caption))

    return None


def _get_replied_image(message: Any) -> Optional[tuple[str, str]]:
    """
    Return (file_id, mime_type) for a replied image.

    Telegram photos are treated as JPEG.
    Image documents use their Telegram MIME type.
    """
    reply = getattr(message, "reply_to_message", None)

    if not reply:
        return None

    photos = getattr(reply, "photo", None)

    if photos:
        try:
            largest_photo = photos[-1]
            file_id = getattr(largest_photo, "file_id", None)

            if file_id:
                return file_id, "image/jpeg"
        except Exception:
            pass

    document = getattr(reply, "document", None)

    if document:
        mime_type = getattr(document, "mime_type", None) or ""

        if mime_type.startswith("image/"):
            file_id = getattr(document, "file_id", None)

            if file_id:
                return file_id, mime_type

    return None


# ---------------------------------------------------------------------------
# Web verification
# ---------------------------------------------------------------------------

async def _web_verify(question: str) -> list[dict[str, str]]:
    """Search Google and return independent evidence for the question.

    This is deliberately separate from the existing AI provider system.
    The quiz's marked option is not sent as an authoritative answer.
    """
    api_key = getattr(config, "GOOGLE_SEARCH_API_KEY", None)
    cx = getattr(config, "GOOGLE_SEARCH_CX", None)

    if not api_key or not cx:
        logger.warning("/solve web verification skipped: Google Search is not configured")
        return []

    try:
        status, data = await request_json(
            "GET",
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": api_key,
                "cx": cx,
                "q": question[:MAX_WEB_QUERY_LENGTH],
                "num": MAX_WEB_RESULTS,
                "safe": "active",
            },
        )
    except Exception as exc:
        # Web verification must never break the existing /solve command.
        logger.warning("/solve Google verification request failed: %s", exc)
        return []

    if status != 200 or not isinstance(data, dict):
        logger.warning("/solve Google verification HTTP %s: %s", status, str(data)[:500])
        return []

    evidence: list[dict[str, str]] = []
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        link = str(item.get("link") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if title and link:
            evidence.append({
                "title": title[:300],
                "link": link,
                "snippet": snippet[:MAX_WEB_SNIPPET_LENGTH],
            })

    return evidence


def _build_search_query(question_block: str) -> str:
    """Extract just the core question text for a Google search query.

    The full question block (used inside the AI prompt) may also contain an
    'Options:' list, a 'Telegram marked answer' line, or a 'MARKED:' line
    (from image extraction). Those only add noise to a Google search and
    hurt result relevance, so the query uses the question text only while
    the AI prompt still gets the full block with options.
    """
    if not question_block:
        return ""

    core = question_block
    core = re.split(r"\n\s*options\s*:", core, maxsplit=1, flags=re.IGNORECASE)[0]
    core = re.split(r"\n\s*marked\s*:", core, maxsplit=1, flags=re.IGNORECASE)[0]
    core = re.split(r"telegram marked answer", core, maxsplit=1, flags=re.IGNORECASE)[0]

    return core.strip()


def _build_verified_text_prompt(
    question: str,
    evidence: list[dict[str, str]],
    pro: bool = False,
) -> str:
    """Build a solver prompt that independently verifies every option."""
    source_block = []
    for i, item in enumerate(evidence, 1):
        source_block.append(
            f"SOURCE {i}\n"
            f"TITLE: {item['title']}\n"
            f"URL: {item['link']}\n"
            f"SNIPPET: {item['snippet']}"
        )

    mode = (
        "Give a detailed competitive-exam solution and cross-check conflicting facts."
        if pro else
        "Give a concise but evidence-based competitive-exam solution."
    )

    return f"""
You are an expert competitive-exam fact checker and question solver.

{mode}

QUESTION AND OPTIONS:
{question}

WEB SEARCH EVIDENCE:
{chr(10).join(source_block)}

CRITICAL VERIFICATION RULES:
1. The Telegram quiz's marked/correct option is NOT authoritative. It may be wrong.
2. Do NOT assume any option is correct merely because it was marked in Telegram.
3. Determine the actual answer independently from the question and all options.
4. Verify EVERY option that is factually checkable, especially names, ranks, dates,
   years, numbers, percentages, places and official titles.
5. Prefer primary/official sources and strong authoritative sources when the supplied
   evidence supports them. Do not invent facts or sources.
6. If sources conflict, explicitly identify the conflict and decide only when the
   evidence is strong enough. Otherwise say that the answer could not be verified.
7. The final answer MUST be one of the supplied options when options are present.
8. Never manufacture an answer that is not among the supplied options.
9. Go through EVERY option one by one and state, for each, whether the evidence
   supports it, contradicts it, or says nothing about it. Do not shortcut this by
   only comparing the two options that "look" most likely.
10. If, after checking every option, the evidence does not clearly single out one
    option, do NOT present a confident final answer. Instead say plainly that it
    could not be fully verified from the search results, and only then name the
    option that is most likely with a clear "not fully confirmed" caveat.
11. Do not use the Telegram marked answer as evidence for correctness.
12. For current/recent questions, pay close attention to the year in the question.
    Do not use an older year's ranking/report to answer a newer-year question.
13. For rankings/reports, verify the exact edition/year before choosing an option.
14. If a previous-year value differs from the asked-year value, explain that clearly.
15. Use the user's language when practical; Hindi questions should get Hindi.
16. Use exactly these four sections:
Answer
Shortcut Trick
Verification
Final Answer
17. In Verification, go through each option's evidence status, then explicitly state
    whether the Telegram-marked answer (if one is visible in the question text) is
    supported, contradicted, or not determinable.
18. Do not pretend a source says something that is only inferred.
19. No LaTeX, no $ or $$.

Return only the four requested sections.
""".strip()

# ---------------------------------------------------------------------------
# AI prompts
# ---------------------------------------------------------------------------

def _build_text_prompt(question: str, pro: bool = False) -> str:
    """Create the AI solver prompt for text and poll questions."""

    mode = (
        "PRO MODE: Give a more detailed competitive-exam solution, "
        "verify every calculation, and provide the most useful shortcut."
        if pro
        else
        "NORMAL MODE: Give a concise, accurate and exam-oriented solution."
    )

    return f"""
You are an expert competitive-exam question solver.

{mode}

Solve the following question accurately.

QUESTION:
{question}

STRICT OUTPUT RULES:

1. Understand the complete question before solving.
2. If options are present, check them against the actual solution.
3. Verify all calculations before giving the final answer.
4. Do not rewrite the question.
5. Do not rewrite the options.
6. Do not repeat the same solution.
7. Do not add unnecessary introduction or conclusion.
8. Use exactly these four sections:
   Answer
   Shortcut Trick
   Verification
   Final Answer
9. Use the same language as the question whenever possible.
10. If the question is in Hindi, answer in Hindi.
11. If the question is in English, answer in English.
12. Give a shortcut only when a genuine shortcut exists.
13. Keep Verification short and decisive.
14. Final Answer must contain only the correct option and answer.
15. Never use LaTeX.
16. Never use $ or $$.
17. Never use LaTeX commands such as \\sqrt, \\times, \\div, \\approx or \\frac.
18. Use Unicode mathematical symbols instead.
19. Use √ for square root.
20. Use × for multiplication.
21. Use ÷ for division.
22. Use ≈ for approximation.
23. Use − for subtraction.
24. Use ≤, ≥ and ≠ when required.
25. Use Unicode superscripts such as ², ³ and ⁴ for powers.
26. Write fractions in normal form such as 3/5.
27. Put important calculations on separate lines.
28. Keep mathematical expressions clean and readable.
29. Example formatting:
   √7387 ≈ 85.91
   83 × 89 = 7387
   Difference = 89 − 83 = 6
30. Do not output any extra sections.
31. Do not repeat Final Answer elsewhere.

Return exactly:

Answer

[correct option and concise solution]

Shortcut Trick

[short useful trick]

Verification

[short verification]

Final Answer

[correct option and final answer]
""".strip()


def _build_image_prompt(pro: bool = False) -> str:
    """Create the AI vision prompt."""

    mode = (
        "PRO MODE: Give a more detailed competitive-exam solution, "
        "verify every calculation, and provide the most useful shortcut."
        if pro
        else
        "NORMAL MODE: Give a concise, accurate and exam-oriented solution."
    )

    return f"""
You are an expert competitive-exam question solver.

{mode}

The attached image contains a question, possibly with options.

Read the complete image carefully and solve the question accurately.

STRICT RULES:

1. Read the complete question.
2. Read all visible options.
3. Carefully read mathematical symbols, fractions, powers, roots, signs,
   tables and diagrams.
4. Use only information actually visible in the image.
5. Do not invent unreadable information.
6. Verify all calculations before giving the final answer.
7. Do not rewrite the question.
8. Do not rewrite the options.
9. Do not repeat the solution.
10. Use exactly these four sections:
    Answer
    Shortcut Trick
    Verification
    Final Answer
11. Use the language visible in the question whenever possible.
12. If the question is in Hindi, answer in Hindi.
13. If the question is in English, answer in English.
14. Give a shortcut only when a genuine shortcut exists.
15. Keep Verification short and decisive.
16. Final Answer must contain only the correct option and answer.
17. Never use LaTeX.
18. Never use $ or $$.
19. Never use LaTeX commands such as \\sqrt, \\times, \\div, \\approx or \\frac.
20. Use Unicode mathematical symbols instead.
21. Use √ for square root.
22. Use × for multiplication.
23. Use ÷ for division.
24. Use ≈ for approximation.
25. Use − for subtraction.
26. Use ≤, ≥ and ≠ when required.
27. Use Unicode superscripts such as ², ³ and ⁴ for powers.
28. Write fractions in normal form such as 3/5.
29. Put important calculations on separate lines.
30. Keep mathematical expressions clean and readable.
31. Example formatting:
    √7387 ≈ 85.91
    83 × 89 = 7387
    Difference = 89 − 83 = 6
32. Do not output any extra sections.
33. Do not repeat Final Answer elsewhere.

Return exactly:

Answer

[correct option and concise solution]

Shortcut Trick

[short useful trick]

Verification

[short verification]

Final Answer

[correct option and final answer]
""".strip()


def _build_verified_image_prompt(
    extracted_text: str,
    evidence: list[dict[str, str]],
    pro: bool = False,
) -> str:
    """Build an image-solver prompt that independently verifies every option
    using web search evidence, the same way the text/poll path does.

    The original image is still attached to this call so the model can
    double-check exact wording, math, diagrams, etc. that a plain-text
    extraction might miss — but the ANSWER must come from the evidence
    below, not from the model's own memory or from anything marked in the
    image.
    """
    source_block = []
    for i, item in enumerate(evidence, 1):
        source_block.append(
            f"SOURCE {i}\n"
            f"TITLE: {item['title']}\n"
            f"URL: {item['link']}\n"
            f"SNIPPET: {item['snippet']}"
        )

    mode = (
        "Give a detailed competitive-exam solution and cross-check conflicting facts."
        if pro else
        "Give a concise but evidence-based competitive-exam solution."
    )

    return f"""
You are an expert competitive-exam fact checker and question solver.

{mode}

The attached image contains the original question (and possibly options).
Use the image to confirm exact wording, numbers, math, diagrams, or anything
the text extraction below might have missed.

QUESTION AND OPTIONS EXTRACTED FROM THE IMAGE (for reference only — re-check
against the image itself if anything looks incomplete or garbled):
{extracted_text}

WEB SEARCH EVIDENCE:
{chr(10).join(source_block)}

CRITICAL VERIFICATION RULES:
1. Anything marked/highlighted/ticked as correct in the image (a "MARKED:" line
   above, or a visible tick/highlight) is NOT authoritative. It may be wrong.
2. Do NOT assume any option is correct merely because it appears marked in the image.
3. Determine the actual answer independently from the question and all options.
4. Verify EVERY option that is factually checkable, especially names, ranks, dates,
   years, numbers, percentages, places and official titles.
5. Prefer primary/official sources and strong authoritative sources when the supplied
   evidence supports them. Do not invent facts or sources.
6. If sources conflict, explicitly identify the conflict and decide only when the
   evidence is strong enough. Otherwise say that the answer could not be verified.
7. The final answer MUST be one of the options visible in the image when options
   are present. Never manufacture an option that isn't there.
8. Go through EVERY option one by one and state, for each, whether the evidence
   supports it, contradicts it, or says nothing about it. Do not shortcut this by
   only comparing the two options that "look" most likely.
9. If, after checking every option, the evidence does not clearly single out one
   option, do NOT present a confident final answer. Instead say plainly that it
   could not be fully verified from the search results, and only then name the
   option that is most likely with a clear "not fully confirmed" caveat.
10. For current/recent questions, pay close attention to the year in the question.
    Do not use an older year's ranking/report to answer a newer-year question.
11. For rankings/reports, verify the exact edition/year before choosing an option.
12. If a previous-year value differs from the asked-year value, explain that clearly.
13. Read all mathematical symbols, fractions, powers, roots, signs, tables and
    diagrams in the image carefully, and verify all calculations before answering.
14. Use the language visible in the question whenever possible; Hindi questions
    should get a Hindi answer, English questions an English answer.
15. Use exactly these four sections:
Answer
Shortcut Trick
Verification
Final Answer
16. In Verification, go through each option's evidence status, then explicitly state
    whether anything marked/highlighted in the image is supported, contradicted, or
    not determinable.
17. Do not pretend a source says something that is only inferred.
18. Never use LaTeX or $ / $$. Use Unicode symbols instead: √ × ÷ ≈ − ≤ ≥ ≠ and
    superscripts such as ² ³ ⁴. Write fractions in normal form such as 3/5.

Return only the four requested sections.
""".strip()


# ---------------------------------------------------------------------------
# Gemini Vision
# ---------------------------------------------------------------------------

async def _gemini_vision(
    user_id: int,
    prompt: str,
    max_tokens: int,
    image_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None,
) -> str:
    """
    Send a prompt to Gemini using the user's existing Gemini keys.

    If image_bytes/mime_type are provided, the image is attached
    (vision mode). If not, this is a plain text-only Gemini call —
    used as a fallback when the primary ai_generate() provider
    (Pollinations) is unavailable.

    This function is isolated inside solve.py and does not modify
    ai_providers.py.
    """

    keys = await get_provider_keys(user_id, "gemini")

    if not keys:
        raise RuntimeError(
            "No Gemini API key is configured."
        )

    image_b64: Optional[str] = None

    if image_bytes is not None:
        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise RuntimeError(
                "Image is too large for inline Gemini image processing."
            )

        image_b64 = base64.b64encode(image_bytes).decode("ascii")

    # Current Gemini vision models.
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

            parts: list[dict[str, Any]] = []

            if image_b64 is not None:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type or "image/jpeg",
                            "data": image_b64,
                        }
                    }
                )

            parts.append({"text": prompt})

            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts,
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
                    "Gemini request solved successfully using %s",
                    url,
                )

                return result

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "Gemini request failed with %s: %s",
                    url,
                    exc,
                )

                continue

    raise RuntimeError(
        f"All Gemini keys/models failed: {last_error}"
    )


async def _gemini_text_generate(
    user_id: int,
    prompt: str,
    max_tokens: int,
) -> str:
    """Generate text with Gemini, no image attached.

    Used as a fallback for the text/poll /solve path when the primary
    ai_generate() provider (Pollinations) is unavailable — e.g. returns
    HTTP 429 (queue full) or 402 (payment required).
    """
    return await _gemini_vision(
        user_id=user_id,
        prompt=prompt,
        max_tokens=max_tokens,
    )


_IMAGE_EXTRACTION_PROMPT = """
Read this image carefully and output ONLY the following — nothing else:

Line 1 onward: the exact question text, exactly as written in the image.
Then a line "Options:" followed by each visible option on its own line.
If one option is visibly highlighted, ticked, circled, or otherwise marked as
correct in the image, add one final line: "MARKED: <that option's exact text>".
If nothing is marked, omit that line entirely.

Do not solve the question. Do not explain anything. Do not add commentary.
""".strip()


async def _gemini_extract_text(
    user_id: int,
    image_bytes: bytes,
    mime_type: str,
) -> Optional[str]:
    """Extract the question/options from an image as plain text.

    This is a separate, lightweight call used ONLY to get clean text for an
    independent Google search — it never generates the final answer. If it
    fails for any reason, image solving falls back to the original
    non-verified single-call flow rather than breaking /solve.
    """
    try:
        raw = await _gemini_vision(
            user_id=user_id,
            image_bytes=image_bytes,
            mime_type=mime_type,
            prompt=_IMAGE_EXTRACTION_PROMPT,
            max_tokens=600,
        )
    except Exception as exc:
        logger.warning("/solve image text extraction failed: %s", exc)
        return None

    text = (raw or "").strip()

    return text or None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _convert_math_to_unicode(text: str) -> str:
    """Convert common LaTeX math commands into readable Unicode math."""

    if not text:
        return ""

    # Remove math delimiters.
    text = text.replace("$$", "")
    text = text.replace("$", "")

    # Common LaTeX operators.
    replacements = {
        r"\times": "×",
        r"\cdot": "×",
        r"\div": "÷",
        r"\approx": "≈",
        r"\pm": "±",
        r"\leq": "≤",
        r"\le": "≤",
        r"\geq": "≥",
        r"\ge": "≥",
        r"\neq": "≠",
        r"\ne": "≠",
        r"\infty": "∞",
        r"\degree": "°",
        r"\circ": "°",
        r"\rightarrow": "→",
        r"\left": "",
        r"\right": "",
        r"\," : " ",
        r"\;" : " ",
        r"\!" : "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Convert simple square-root expressions:
    # \sqrt{7387} -> √7387
    text = re.sub(
        r"\\sqrt\s*\{([^{}]*)\}",
        r"√\1",
        text,
        flags=re.IGNORECASE,
    )

    # Convert simple fractions:
    # \frac{3}{5} -> 3/5
    text = re.sub(
        r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        r"\1/\2",
        text,
        flags=re.IGNORECASE,
    )

    # Remove remaining LaTeX commands while preserving their content.
    text = re.sub(
        r"\\[a-zA-Z]+",
        "",
        text,
    )

    # Remove simple LaTeX braces.
    text = text.replace("{", "")
    text = text.replace("}", "")

    return text


def _format_result(raw: str) -> str:
    """Safely format AI output for Telegram."""

    if not raw:
        return "❌ Solution could not be generated."

    text = raw.strip()

    # Remove markdown code fences.
    text = re.sub(
        r"```(?:text|markdown|html)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace("```", "").strip()

    # Convert mathematical notation before HTML escaping.
    text = _convert_math_to_unicode(text)

    # Escape arbitrary HTML from AI output.
    text = html.escape(text)

    # Normalize section headings.
    text = re.sub(
        r"(?im)^\s*Answer\s*:?\s*$",
        "<b>Answer</b>",
        text,
    )

    text = re.sub(
        r"(?im)^\s*Shortcut\s*Trick\s*:?\s*$",
        "\n\n<b>Shortcut Trick</b>",
        text,
    )

    text = re.sub(
        r"(?im)^\s*Verification\s*:?\s*$",
        "\n\n<b>Verification</b>",
        text,
    )

    text = re.sub(
        r"(?im)^\s*Final\s*Answer\s*:?\s*$",
        "\n\n<b>Final Answer</b>",
        text,
    )

    # Clean excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
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
        # Parse command arguments
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
        # Image solving
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
                            "🔎 <b>Verifying...</b>",
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        pass

                # Extract plain question/options text from the image so it can
                # be independently checked against Google, the same way
                # text and poll questions are. If extraction or the search
                # fails for any reason, fall back to the original single-call
                # image solver rather than breaking /solve.
                extracted_text = await _gemini_extract_text(
                    user_id=user_id,
                    image_bytes=bytes(image_bytes),
                    mime_type=mime_type,
                )

                image_web_evidence: list[dict[str, str]] = []

                if extracted_text:
                    image_web_evidence = await _web_verify(
                        _build_search_query(extracted_text)
                    )

                if status_message:
                    try:
                        await status_message.edit_text(
                            "⏳ <b>Solving...</b>",
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        pass

                if image_web_evidence:
                    image_prompt = _build_verified_image_prompt(
                        extracted_text=extracted_text,
                        evidence=image_web_evidence,
                        pro=pro,
                    )
                else:
                    image_prompt = _build_image_prompt(pro=pro)

                try:
                    raw_result = await _gemini_vision(
                        user_id=user_id,
                        image_bytes=bytes(image_bytes),
                        mime_type=mime_type,
                        prompt=image_prompt,
                        max_tokens=3000 if pro else 2200,
                    )
                except Exception as verified_image_exc:
                    if image_web_evidence:
                        logger.warning(
                            "/solve verified image prompt failed, using normal image solver: %s",
                            verified_image_exc,
                            exc_info=True,
                        )
                        raw_result = await _gemini_vision(
                            user_id=user_id,
                            image_bytes=bytes(image_bytes),
                            mime_type=mime_type,
                            prompt=_build_image_prompt(pro=pro),
                            max_tokens=3000 if pro else 2200,
                        )
                    else:
                        raise

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
                    "❌ <b>Image solve failed.</b>\n\n"
                    "Gemini Vision request failed. "
                    "Please try again."
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
                "Reply to a Text, Poll or Image and send "
                "<code>/solve</code>.",
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
                "❌ Question is too long. Please send a shorter question.",
            )
            return

        # ---------------------------------------------------------------
        # Status message
        # ---------------------------------------------------------------

        status_message = await safe_send_message(
            ctx,
            chat_id,
            "⏳ <b>Solving...</b>",
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
        # This keeps the existing ai_generate() provider system intact.
        # ---------------------------------------------------------------

        # First perform independent web verification when Google Search is configured.
        # If web verification is unavailable, retain the existing AI solver as a
        # fallback rather than breaking /solve. The search query uses only the
        # core question text (not the options/marked-answer block) for relevance,
        # but the AI prompt below still receives the full question+options.
        web_evidence = await _web_verify(_build_search_query(question))

        if web_evidence:
            solve_prompt = _build_verified_text_prompt(
                question=question,
                evidence=web_evidence,
                pro=pro,
            )
        else:
            solve_prompt = _build_text_prompt(
                question=question,
                pro=pro,
            )

        try:
            raw_result = await ai_generate(
                user_id=user_id,
                prompt=solve_prompt,
                max_tokens=3500 if pro else 2800,
            )
        except Exception as verified_exc:
            logger.warning(
                "/solve verified prompt failed, using normal solver: %s",
                verified_exc,
                exc_info=True,
            )
            try:
                raw_result = await ai_generate(
                    user_id=user_id,
                    prompt=_build_text_prompt(question=question, pro=pro),
                    max_tokens=3000 if pro else 2200,
                )
            except Exception as normal_exc:
                # The primary provider (Pollinations) is down/rate-limited
                # (e.g. HTTP 429/402). Fall back to Gemini text generation
                # so /solve keeps working instead of failing outright.
                logger.warning(
                    "/solve normal solver also failed, falling back to Gemini text: %s",
                    normal_exc,
                    exc_info=True,
                )
                raw_result = await _gemini_text_generate(
                    user_id=user_id,
                    prompt=solve_prompt,
                    max_tokens=3500 if pro else 2800,
                )

        result = _format_result(raw_result)

        # ---------------------------------------------------------------
        # Replace status message with result
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
                "❌ <b>Solution could not be generated.</b>\n"
                "Please try again.",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(application: Application) -> None:
    """Register the /solve command."""
    application.add_handler(
        CommandHandler(
            "solve",
            solve_command,
        )
    )
