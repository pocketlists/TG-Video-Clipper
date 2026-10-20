"""
planning/plan_parser.py

edit_plan.json schema v1 parser.
Supports both standard v1 plans and extended plans with explicit
start_time, end_time, duration, and crop coordinates.
Also includes an auto-repair pre-processor for unescaped inner quotes
inside story_line strings.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.error_handler import PipelineError, STAGE_PLAN


SUPPORTED_VERSIONS = (1,)

KNOWN_TOP_KEYS = {
    "version",
    "images",
    "meta",
    "metadata",
    "notes",
}

KNOWN_IMAGE_KEYS = {
    "file",
    "description",
    "story_line",
    "keep",
    "order",
    "effects",
    "narration_id",
    "transition",
    "notes",
    "start_time",
    "end_time",
    "duration",
    "crop",
}


class PlanParseError(PipelineError):
    def __init__(
        self,
        message: str,
        *,
        hint: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            STAGE_PLAN,
            message,
            hint=hint,
            cause=cause,
        )


@dataclass
class PlanImage:
    file: str
    story_line: str = ""
    keep: bool = True
    order: Optional[int] = None
    description: str = ""
    effects: Dict[str, Any] = field(default_factory=dict)
    narration_id: Optional[str] = None
    transition: Optional[str] = None
    notes: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    crop: Optional[Tuple[int, int, int, int]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EditPlan:
    version: int
    images: List[PlanImage]
    warnings: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def kept(self) -> List[PlanImage]:
        return [img for img in self.images if img.keep]

    def rejected(self) -> List[PlanImage]:
        return [img for img in self.images if not img.keep]

    def ordered_kept(self) -> List[PlanImage]:
        kept = self.kept()
        if not kept:
            return []

        if all(img.order is not None for img in kept):
            index_of = {id(img): i for i, img in enumerate(self.images)}
            return sorted(
                kept,
                key=lambda img: (img.order, index_of[id(img)]),
            )

        return kept


def _parse_optional_float(
    val: Any,
    field_name: str,
    idx: int,
    file_value: str,
) -> Optional[float]:
    if val is None or val == "":
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) '{field_name}' must be a number"
        )
    return float(val)


def _parse_crop(
    crop_raw: Any,
    idx: int,
    file_value: str,
) -> Optional[Tuple[int, int, int, int]]:
    if crop_raw is None or crop_raw == {}:
        return None

    if isinstance(crop_raw, (list, tuple)) and len(crop_raw) == 4:
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in crop_raw)
        except (TypeError, ValueError) as exc:
            raise PlanParseError(
                f"images[{idx}] ({file_value}) 'crop' array values must be numeric"
            ) from exc
        if x2 > x1 and y2 > y1:
            return (max(0, x1), max(0, y1), x2, y2)
        return None

    if not isinstance(crop_raw, dict):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) 'crop' must be an object or 4-element array"
        )

    try:
        if all(k in crop_raw for k in ("x1", "y1", "x2", "y2")):
            x1 = int(round(float(crop_raw["x1"])))
            y1 = int(round(float(crop_raw["y1"])))
            x2 = int(round(float(crop_raw["x2"])))
            y2 = int(round(float(crop_raw["y2"])))
        elif all(k in crop_raw for k in ("x", "y", "width", "height")):
            x1 = int(round(float(crop_raw["x"])))
            y1 = int(round(float(crop_raw["y"])))
            w = int(round(float(crop_raw["width"])))
            h = int(round(float(crop_raw["height"])))
            x2 = x1 + w
            y2 = y1 + h
        else:
            return None
    except (TypeError, ValueError) as exc:
        raise PlanParseError(
            f"images[{idx}] ({file_value}) 'crop' coordinates must be numbers"
        ) from exc

    if x2 <= x1 or y2 <= y1:
        return None

    return (max(0, x1), max(0, y1), x2, y2)


def _parse_entry(
    entry: Any,
    idx: int,
) -> Tuple[PlanImage, List[str]]:
    warnings: List[str] = []

    if not isinstance(entry, dict):
        raise PlanParseError(
            f"images[{idx}] must be an object, got {type(entry).__name__}"
        )

    unknown = set(entry.keys()) - KNOWN_IMAGE_KEYS
    if unknown:
        warnings.append(
            f"images[{idx}] unknown keys ignored: {sorted(unknown)}"
        )

    file_value = entry.get("file")
    if not isinstance(file_value, str) or not file_value.strip():
        raise PlanParseError(
            f"images[{idx}] missing required non-empty 'file' string"
        )
    file_value = file_value.strip()

    keep = entry.get("keep", True)
    if not isinstance(keep, bool):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) 'keep' must be boolean, got {type(keep).__name__}"
        )

    story_line = entry.get("story_line", "")
    if not isinstance(story_line, str):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) 'story_line' must be a string"
        )

    if keep and not story_line.strip():
        warnings.append(
            f"images[{idx}] ({file_value}) keep=true but 'story_line' is empty; "
            "sync will rely on sibling narration or explicit timestamps"
        )

    order = entry.get("order")
    if order is not None and (not isinstance(order, int) or isinstance(order, bool)):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) 'order' must be an integer"
        )

    description = entry.get("description", "")
    if not isinstance(description, str):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) 'description' must be a string"
        )

    effects_raw = entry.get("effects")
    if effects_raw is None or effects_raw == "":
        effects: Dict[str, Any] = {}
    elif isinstance(effects_raw, str):
        effect_name = effects_raw.strip()
        effects = {"effect": effect_name} if effect_name else {}
    elif isinstance(effects_raw, dict):
        effects = dict(effects_raw)
    else:
        raise PlanParseError(
            f"images[{idx}] ({file_value}) 'effects' must be a string or an object"
        )

    if effects:
        explicit_effect = effects.get("effect")
        if explicit_effect is not None and not isinstance(explicit_effect, str):
            raise PlanParseError(
                f"images[{idx}] ({file_value}) 'effects.effect' must be a string"
            )

    narration_id = entry.get("narration_id")
    if narration_id is not None and not isinstance(narration_id, str):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) 'narration_id' must be a string"
        )

    transition = entry.get("transition")
    if transition is not None and not isinstance(transition, str):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) 'transition' must be a string"
        )

    notes = entry.get("notes", "")
    if not isinstance(notes, str):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) 'notes' must be a string"
        )

    start_time = _parse_optional_float(entry.get("start_time"), "start_time", idx, file_value)
    end_time = _parse_optional_float(entry.get("end_time"), "end_time", idx, file_value)
    duration = _parse_optional_float(entry.get("duration"), "duration", idx, file_value)

    if start_time is not None and end_time is None and duration is not None and duration > 0:
        end_time = round(start_time + duration, 6)
    elif start_time is not None and end_time is not None and duration is None and end_time > start_time:
        duration = round(end_time - start_time, 6)

    crop = _parse_crop(entry.get("crop"), idx, file_value)

    parsed = PlanImage(
        file=file_value,
        story_line=story_line,
        keep=keep,
        order=order,
        description=description,
        effects=effects,
        narration_id=narration_id,
        transition=transition,
        notes=notes,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        crop=crop,
        raw=entry,
    )

    return parsed, warnings


def parse_plan(data: Any) -> EditPlan:
    if not isinstance(data, dict):
        raise PlanParseError(
            f"edit_plan must be a JSON object, got {type(data).__name__}"
        )

    warnings: List[str] = []

    version = data.get("version")
    if isinstance(version, str) and version.strip().isdigit():
        warnings.append(f"'version' was string {version!r}; coerced to int")
        version = int(version.strip())

    if not isinstance(version, int) or isinstance(version, bool):
        raise PlanParseError(
            "missing or non-integer 'version' (expected 1)",
            hint='add "version": 1 at the top of edit_plan.json',
        )

    if version not in SUPPORTED_VERSIONS:
        raise PlanParseError(
            f"unsupported plan version {version} (supported: {list(SUPPORTED_VERSIONS)})"
        )

    unknown_top = set(data.keys()) - KNOWN_TOP_KEYS
    if unknown_top:
        warnings.append(f"unknown top-level keys ignored: {sorted(unknown_top)}")

    images_raw = data.get("images")
    if not isinstance(images_raw, list):
        raise PlanParseError("'images' must be an array")
    if not images_raw:
        raise PlanParseError("'images' array is empty — nothing to render")

    images: List[PlanImage] = []
    for idx, entry in enumerate(images_raw):
        parsed, entry_warnings = _parse_entry(entry, idx)
        images.append(parsed)
        warnings.extend(entry_warnings)

    if not any(img.keep for img in images):
        raise PlanParseError("every image in the plan has keep=false — nothing to render")

    kept_imgs = [img for img in images if img.keep]
    with_order = sum(1 for img in kept_imgs if img.order is not None)
    without_order = len(kept_imgs) - with_order

    if with_order and without_order:
        warnings.append(
            f"mixed 'order' presence: {with_order} entries have order, "
            f"{without_order} do not; file order is used for unordered ones"
        )

    if with_order > 1:
        orders = [img.order for img in kept_imgs if img.order is not None]
        duplicates = [o for o, count in Counter(orders).items() if count > 1]
        if duplicates:
            warnings.append(f"duplicate 'order' values: {sorted(duplicates)}")

    file_counts = Counter(img.file for img in images)
    duplicate_files = [name for name, count in file_counts.items() if count > 1]
    if duplicate_files:
        warnings.append(f"duplicate 'file' entries: {duplicate_files[:5]}")

    return EditPlan(
        version=version,
        images=images,
        warnings=warnings,
        raw=data,
    )


def _repair_unescaped_quotes(text: str) -> str:
    """
    Repairs unescaped double-quotes inside JSON string values like
    "story_line": "Wo sochta hai, "Ye kya hai?!"" -> escapes inner quotes.
    """
    repaired_lines: List[str] = []
    pattern = re.compile(r'^(\s*"(?:story_line|description|notes|text)"\s*:\s*")(.*)("(\s*,?)\s*)$')

    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            prefix, inner, suffix, _ = m.groups()
            fixed_inner = re.sub(r'(?<!\\)"', r'\"', inner)
            repaired_lines.append(f"{prefix}{fixed_inner}{suffix}")
        else:
            repaired_lines.append(line)

    return "\n".join(repaired_lines)


def parse_plan_text(text: str) -> EditPlan:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = _repair_unescaped_quotes(text)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise PlanParseError(
                f"edit_plan.json is not valid JSON: {exc.msg} at line {exc.lineno}, col {exc.colno}",
                cause=exc,
            ) from exc

    return parse_plan(data)


def parse_plan_file(path: Path) -> EditPlan:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PlanParseError(
            f"cannot read plan file {path}: {exc}",
            cause=exc,
        ) from exc

    return parse_plan_text(text)


def looks_like_edit_plan(data: Any) -> bool:
    if not isinstance(data, dict):
        return False

    images = data.get("images")
    if not isinstance(images, list) or not images:
        return False

    return all(
        isinstance(entry, dict)
        and isinstance(entry.get("file"), str)
        and bool(entry["file"])
        and "keep" in entry
        for entry in images
    )
