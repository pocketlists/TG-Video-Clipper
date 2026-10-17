"""
utils/error_handler.py
PipelineError with stage/module/hint context, plus helpers for building
user-facing error messages and classifying errors as retryable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


STAGE_INTAKE     = "intake"
STAGE_EXTRACT    = "extract"
STAGE_VALIDATE   = "validate"
STAGE_PLAN       = "plan"
STAGE_TRANSCRIBE = "transcribe"
STAGE_SYNC       = "sync"
STAGE_RENDER     = "render"
STAGE_UPLOAD     = "upload"


_RETRYABLE_HINTS = (
    "429", "rate limit", "resource_exhausted", "quota",
    "timeout", "timed out", "connection", "temporarily",
)


class PipelineError(Exception):
    """Single error type surfaced to telegram/bot.py for user display."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        hint: Optional[str] = None,
        cause: Optional[BaseException] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.hint = hint
        self.cause = cause
        if retryable is None:
            retryable = _looks_retryable(message, cause)
        self.retryable = retryable

    def user_text(self) -> str:
        lines = [f"❌ [{self.stage}] {self.message}"]
        if self.hint:
            lines.append(f"💡 {self.hint}")
        if self.cause is not None:
            lines.append(f"🔍 {type(self.cause).__name__}: {self.cause}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"[{self.stage}] {self.message}"


def _looks_retryable(message: str, cause: Optional[BaseException]) -> bool:
    haystacks = [message.lower()]
    if cause is not None:
        haystacks.append(str(cause).lower())
        haystacks.append(type(cause).__name__.lower())
    return any(token in h for h in haystacks for token in _RETRYABLE_HINTS)


def wrap(
    stage: str,
    exc: BaseException,
    *,
    hint: Optional[str] = None,
) -> PipelineError:
    """Convert any exception into a PipelineError preserving context."""
    if isinstance(exc, PipelineError):
        return exc
    return PipelineError(stage, str(exc) or type(exc).__name__, hint=hint, cause=exc)


def aggregate(
    stage: str,
    errors: Iterable[BaseException],
    *,
    hint: Optional[str] = None,
) -> PipelineError:
    """Collapse several errors into one PipelineError with a joined message."""
    items = list(errors)
    if not items:
        return PipelineError(stage, "unknown error", hint=hint)
    if len(items) == 1:
        return wrap(stage, items[0], hint=hint)
    joined = "; ".join(f"{type(e).__name__}: {e}" for e in items)
    return PipelineError(stage, joined, hint=hint, cause=items[0])
