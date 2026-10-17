"""
media/image_cleaner.py
Validates images before they reach the renderer:
  - decodes with Pillow to reject corrupt / truncated / bomb files
  - records dimensions, format and sha256 for duplicate detection
  - warns on animated files (only the first frame will render)
  - optionally runs a registered VisionEngine to attach OCR text
  - reports rejections and duplicates

Never deletes, renames or rewrites the original file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from media.detector import classify
from media.formats import CATEGORY_IMAGE
from utils.logger import get_logger
from vision import get_engine as get_vision_engine
from vision.engine import VisionEngine


logger = get_logger("media.image_cleaner")

_HASH_CHUNK = 64 * 1024


@dataclass
class CleanImage:
    path: Path
    original_name: str
    safe_name: str
    width: int
    height: int
    format: str
    sha256: str
    frames: int = 1
    is_animated: bool = False
    vision_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class RejectedImage:
    path: Path
    reason: str


@dataclass
class CleanupReport:
    clean: List[CleanImage] = field(default_factory=list)
    rejected: List[RejectedImage] = field(default_factory=list)
    duplicates: List[Tuple[CleanImage, CleanImage]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def names(self) -> List[str]:
        return [c.safe_name for c in self.clean]

    def has_rejections(self) -> bool:
        return bool(self.rejected)

    def user_summary(self) -> str:
        lines: List[str] = [f"✅ {len(self.clean)} image(s) validated"]
        if self.rejected:
            lines.append(f"❌ {len(self.rejected)} rejected:")
            for r in self.rejected[:10]:
                lines.append(f"  • {r.path.name}: {r.reason}")
            if len(self.rejected) > 10:
                lines.append(f"  … and {len(self.rejected) - 10} more")
        if self.duplicates:
            lines.append(f"🔁 {len(self.duplicates)} duplicate(s):")
            for dup, first in self.duplicates[:5]:
                lines.append(f"  • {dup.path.name} == {first.path.name}")
        if self.warnings:
            lines.append(f"⚠️ {len(self.warnings)} warning(s):")
            for w in self.warnings[:5]:
                lines.append(f"  • {w}")
            if len(self.warnings) > 5:
                lines.append(f"  … and {len(self.warnings) - 5} more")
        return "\n".join(lines)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def inspect_image(
    path: Path,
    vision_engine: Optional[VisionEngine] = None,
) -> CleanImage | RejectedImage:
    """Validate one file. Never raises; returns CleanImage or RejectedImage."""
    path = Path(path)
    if not path.is_file():
        return RejectedImage(path, "not a file")

    try:
        detection = classify(path)
    except Exception as exc:
        return RejectedImage(path, f"detection failed: {exc}")
    if detection.category != CATEGORY_IMAGE:
        return RejectedImage(
            path, f"detected as {detection.category}, not an image"
        )

    try:
        from PIL import Image
    except ImportError as exc:
        return RejectedImage(path, f"Pillow not installed: {exc}")

    try:
        with Image.open(path) as img:
            width, height = img.size
            fmt = (img.format or "").upper()
            frames = int(getattr(img, "n_frames", 1) or 1)
            img.load()
    except Exception as exc:
        return RejectedImage(path, f"Pillow cannot decode: {exc}")

    if width <= 0 or height <= 0:
        return RejectedImage(path, f"invalid dimensions {width}x{height}")

    warnings: List[str] = []
    is_animated = frames > 1
    if is_animated:
        warnings.append(f"animated ({frames} frames); only first frame renders")

    try:
        sha = _sha256(path)
    except OSError as exc:
        return RejectedImage(path, f"cannot hash: {exc}")

    vision_text: Optional[str] = None
    if vision_engine is not None:
        try:
            result = vision_engine.extract_text(str(path))
            if result is not None:
                if result.text:
                    vision_text = result.text
                if result.error:
                    warnings.append(f"vision: {result.error}")
        except Exception as exc:
            warnings.append(f"vision failed: {exc}")

    return CleanImage(
        path=path,
        original_name=path.name,
        safe_name=path.name,
        width=width,
        height=height,
        format=fmt,
        sha256=sha,
        frames=frames,
        is_animated=is_animated,
        vision_text=vision_text,
        warnings=list(warnings),
    )


def clean_images(
    files: Iterable[Path],
    vision_engine_name: Optional[str] = None,
) -> CleanupReport:
    """Validate a set of image files. Never raises on individual files;
    problems are recorded in the report."""
    report = CleanupReport()
    engine = get_vision_engine(vision_engine_name) if vision_engine_name else None

    seen_hashes: dict[str, CleanImage] = {}
    for path in files:
        path = Path(path)
        result = inspect_image(path, engine)

        if isinstance(result, RejectedImage):
            report.rejected.append(result)
            logger.warning(f"rejected {path.name}: {result.reason}")
            continue

        if result.sha256 in seen_hashes:
            first = seen_hashes[result.sha256]
            report.duplicates.append((result, first))
            result.warnings.append(f"duplicate of {first.path.name}")
            logger.info(f"duplicate: {path.name} == {first.path.name}")
        else:
            seen_hashes[result.sha256] = result

        report.clean.append(result)
        for w in result.warnings:
            report.warnings.append(f"{path.name}: {w}")

    if engine is not None:
        try:
            engine.close()
        except Exception as exc:
            logger.warning(f"vision engine close failed: {exc}")

    logger.info(
        f"image cleanup: {len(report.clean)} clean, "
        f"{len(report.rejected)} rejected, "
        f"{len(report.duplicates)} duplicate(s)"
    )
    return report


def collect_image_files(directory: Path) -> List[Path]:
    """Return every file under `directory` that classifies as an image.
    Sorted by name (natural order is applied later by the caller)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: List[Path] = []
    for child in directory.rglob("*"):
        if not child.is_file():
            continue
        try:
            if classify(child).category == CATEGORY_IMAGE:
                found.append(child)
        except Exception:
            continue
    found.sort(key=lambda p: p.name.lower())
    return found
