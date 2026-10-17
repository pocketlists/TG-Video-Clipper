"""
utils/logger.py
Structured logging with secret redaction. Call `setup_logging()` once
at process start; import `get_logger(name)` everywhere else.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Optional


_SECRET_KEY_HINTS = ("TOKEN", "KEY", "HASH", "SECRET", "PASSWORD", "PASSWD")

_SECRET_PATTERNS = (
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b"),          # bot tokens
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),                 # google api keys
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),                   # openai-like keys
    re.compile(r"(?i)(authorization:\s*)\S+"),
)


def _redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("<redacted>", text)
    for name, value in os.environ.items():
        if not value or len(value) < 6:
            continue
        if any(hint in name.upper() for hint in _SECRET_KEY_HINTS):
            text = text.replace(value, f"<redacted:{name}>")
    return text


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _redact(super().format(record))


_configured = False


def setup_logging(level: Optional[str] = None) -> None:
    global _configured
    if _configured:
        return
    lvl = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_RedactingFormatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(lvl)
    logging.getLogger("pyrogram").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
