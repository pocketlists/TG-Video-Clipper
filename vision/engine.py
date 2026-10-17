"""
vision/engine.py
Abstract VisionEngine interface. No implementation, no model imports.
Implementations live in sibling modules (rapidocr_engine.py, etc.) and
register themselves in vision/__init__.py's factory. Keeping the
interface here means media/image_cleaner.py and telegram/bot.py can be
written against it today, and a real engine can drop in later with no
caller changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TextBlock:
    text: str
    bbox: List[float] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class VisionResult:
    text: str
    blocks: List[TextBlock] = field(default_factory=list)
    engine: str = ""
    error: Optional[str] = None

    def is_empty(self) -> bool:
        return not self.text.strip() and not self.blocks


class VisionEngine(ABC):
    """Base class for any OCR / text-extraction engine."""

    name: str = "abstract"

    @abstractmethod
    def extract_text(self, image_path: str) -> VisionResult:
        """Return extracted text and blocks for a single image file."""

    def close(self) -> None:
        """Optional teardown; called once at process shutdown."""
