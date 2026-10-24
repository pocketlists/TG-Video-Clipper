#!/usr/bin/env python3
from __future__ import annotations

import faulthandler
import sys
import traceback
from pathlib import Path

faulthandler.enable()

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _fail(message: str) -> int:
    sys.stderr.write(f"fatal: {message}\n")
    sys.stderr.flush()
    return 1


def main() -> int:
    try:
        from telegram.bot import main as bot_main
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return _fail(f"cannot import telegram.bot ({type(exc).__name__}: {exc})")

    try:
        bot_main()
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return _fail(f"telegram.bot.main() raised {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
