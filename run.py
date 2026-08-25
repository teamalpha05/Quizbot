#!/usr/bin/env python3
"""
Advance Quiz Bot — Open Source Project
Render-compatible launcher with Telegram bots + HTTP health server.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from aiohttp import web

from quizbot.database import init_db, close_db
from quizbot.shared import config
from quizbot.shared.utils.http import close_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger("launcher")

# Silence noisy third-party debug logs by default.
for noisy in ("httpx", "httpcore", "apscheduler", "pymongo"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


async def health(request):
    return web.Response(
        text="Quiz Bot is running!",
        status=200,
        content_type="text/plain",
    )


async def start_health_server():
    """
    Small HTTP server required by Render Web Service.
    Render provides the PORT environment variable.
    """
    port = int(os.environ.get("PORT", "8080"))

    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(
        runner,
        host="0.0.0.0",
        port=port,
    )

    await site.start()

    logger.info("Render health server started on port %s", port)

    return runner


async def _run_creator_bot() -> None:
    from quizbot.creator_bot.bot import run_creator_bot

    logger.info("Starting Creator Bot (Pyrogram)...")
    await run_creator_bot()


async def _run_runner_bot() -> None:
    from quizbot.runner_bot.bot import run_runner_bot

    logger.info("Starting Runner Bot (python-telegram-bot)...")
    await run_runner_bot()


async def _run_mini_app() -> None:
    from quizbot.mini_app.server import run_mini_app_server

    logger.info("Starting Mini App server (FastAPI)...")
    await run_mini_app_server()


async def main(only: str | None) -> None:

    problems = config.validate(bot=only or "both")

    if problems:
        for p in problems:
            logger.error("Config problem: %s", p)

        logger.error(
            "Fix the above in your environment variables "
            "(see .env.example)."
        )

        sys.exit(1)

    logger.info(
        "Connecting to MongoDB (db=%s) ...",
        config.MONGODB_DB_NAME,
    )

    await init_db(
        config.MONGODB_URI,
        config.MONGODB_DB_NAME,
    )

    logger.info("Database ready.")

    tasks: list[asyncio.Task] = []

    # Start Render HTTP server.
    health_runner = await start_health_server()

    # Start Creator Bot.
    if only in (None, "creator"):
        tasks.append(
            asyncio.create_task(
                _run_creator_bot(),
                name="creator_bot",
            )
        )

    # Start Runner Bot.
    if only in (None, "runner"):
        tasks.append(
            asyncio.create_task(
                _run_runner_bot(),
                name="runner_bot",
            )
        )

    # Mini App.
    if only == "miniapp":
        tasks.append(
            asyncio.create_task(
                _run_mini_app(),
                name="mini_app",
            )
        )

    elif only is None and config.MINI_APP_DOMAIN:
        tasks.append(
            asyncio.create_task(
                _run_mini_app(),
                name="mini_app",
            )
        )

    stop_event = asyncio.Event()

    def _handle_signal(*_args):
        logger.info(
            "Shutdown signal received, stopping bots..."
        )
        stop_event.set()

    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig,
                _handle_signal,
            )
        except NotImplementedError:
            pass

    try:

        # Keep all services alive until Render sends a shutdown signal.
        await stop_event.wait()

    except asyncio.CancelledError:
        logger.info("Main task cancelled.")

    finally:

        logger.info("Shutting down...")

        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        await health_runner.cleanup()

        await close_session()
        await close_db()

        logger.info("Shutdown complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the Advance Quiz Bot platform."
    )

    parser.add_argument(
        "--only",
        choices=[
            "creator",
            "runner",
            "miniapp",
        ],
        default=None,
        help=(
            "Run only one component "
            "(default: run both bots)."
        ),
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            main(args.only)
        )

    except KeyboardInterrupt:
        pass
