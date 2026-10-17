#!/usr/bin/env python3
"""
Entry point for the one-time story video renderer.

Delegates everything to telegram.bot. All setup (logging, secrets,
intake config) happens inside telegram.bot on import, so this file
only needs to import it and call main().

Run with:
    python -u main.py
"""

from __future__ import annotations

import sys


def _fail(message: str) -> int:
    sys.stderr.write(f"fatal: {message}\n")
    return 1


def main() -> int:
    try:
        from telegram.bot import main as bot_main
    except Exception as exc:
        return _fail(
            f"cannot import telegram.bot "
            f"({type(exc).__name__}: {exc})"
        )
    try:
        bot_main()
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130
    except Exception as exc:
        return _fail(f"telegram.bot.main() raised {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
