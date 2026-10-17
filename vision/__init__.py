"""
vision/__init__.py
Factory for VisionEngine implementations.

To add an engine:
  1. drop vision/<name>_engine.py with class <Name>Engine(VisionEngine)
  2. add the name to _ENGINES below
  3. set "engine": "<name>" in config/defaults.json (or VISION_ENGINE env)

No engine registered means get_engine() returns None and every caller
must treat OCR as unavailable — same behaviour as today.
"""

from __future__ import annotations

from typing import Optional

from vision.engine import VisionEngine, VisionResult, TextBlock  # noqa: F401


_ENGINES = {
    # "rapidocr": "vision.rapidocr_engine:RapidOCREngine",
    # "tesseract": "vision.tesseract_engine:TesseractEngine",
}


def register(name: str, dotted_path: str) -> None:
    _ENGINES[name] = dotted_path


def available_engines() -> list[str]:
    return sorted(_ENGINES.keys())


def get_engine(name: Optional[str]) -> Optional[VisionEngine]:
    """Return an engine instance, or None if the name is unknown/empty."""
    if not name or name == "none":
        return None
    if name not in _ENGINES:
        return None
    module_path, _, class_name = _ENGINES[name].partition(":")
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls()
    except (ImportError, AttributeError):
        return None
