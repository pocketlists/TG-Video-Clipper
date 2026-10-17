
"""
planning/plan_parser.py

edit_plan.json schema v1 parser.
Converts JSON data into dataclasses and validates its structure.
"""

from __future__ import annotations

import json
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
            index_of = {
                id(img): i
                for i, img in enumerate(self.images)
            }

            return sorted(
                kept,
                key=lambda img: (
                    img.order,
                    index_of[id(img)],
                ),
            )

        return kept


def _parse_entry(
    entry: Any,
    idx: int,
) -> Tuple[PlanImage, List[str]]:
    warnings: List[str] = []

    if not isinstance(entry, dict):
        raise PlanParseError(
            f"images[{idx}] must be an object, "
            f"got {type(entry).__name__}"
        )

    unknown = set(entry.keys()) - KNOWN_IMAGE_KEYS

    if unknown:
        warnings.append(
            f"images[{idx}] unknown keys ignored: "
            f"{sorted(unknown)}"
        )

    file_value = entry.get("file")

    if not isinstance(file_value, str) or not file_value.strip():
        raise PlanParseError(
            f"images[{idx}] missing required "
            "non-empty 'file' string"
        )

    file_value = file_value.strip()

    keep = entry.get("keep", True)

    if not isinstance(keep, bool):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) "
            f"'keep' must be boolean, got {type(keep).__name__}"
        )

    story_line = entry.get("story_line", "")

    if not isinstance(story_line, str):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) "
            "'story_line' must be a string"
        )

    if keep and not story_line.strip():
        warnings.append(
            f"images[{idx}] ({file_value}) keep=true but "
            "'story_line' is empty; sync will rely on "
            "sibling narration"
        )

    order = entry.get("order")

    if order is not None and (
        not isinstance(order, int) or isinstance(order, bool)
    ):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) "
            "'order' must be an integer"
        )

    description = entry.get("description", "")

    if not isinstance(description, str):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) "
            "'description' must be a string"
        )

    # Support both:
    # "effects": "zoom_in"
    # "effects": {"effect": "zoom_in"}
    effects_raw = entry.get("effects")

    if effects_raw is None or effects_raw == "":
        effects: Dict[str, Any] = {}

    elif isinstance(effects_raw, str):
        effect_name = effects_raw.strip()

        effects = (
            {"effect": effect_name}
            if effect_name
            else {}
        )

    elif isinstance(effects_raw, dict):
        effects = dict(effects_raw)

    else:
        raise PlanParseError(
            f"images[{idx}] ({file_value}) "
            "'effects' must be a string or an object"
        )

    # Warn about effect names the timeline may not support.
    if effects:
        explicit_effect = effects.get("effect")

        if explicit_effect is not None and not isinstance(
            explicit_effect, str
        ):
            raise PlanParseError(
                f"images[{idx}] ({file_value}) "
                "'effects.effect' must be a string"
            )

    narration_id = entry.get("narration_id")

    if narration_id is not None and not isinstance(
        narration_id, str
    ):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) "
            "'narration_id' must be a string"
        )

    transition = entry.get("transition")

    if transition is not None and not isinstance(
        transition, str
    ):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) "
            "'transition' must be a string"
        )

    notes = entry.get("notes", "")

    if not isinstance(notes, str):
        raise PlanParseError(
            f"images[{idx}] ({file_value}) "
            "'notes' must be a string"
        )

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
        raw=entry,
    )

    return parsed, warnings


def parse_plan(data: Any) -> EditPlan:
    if not isinstance(data, dict):
        raise PlanParseError(
            "edit_plan must be a JSON object, "
            f"got {type(data).__name__}"
        )

    warnings: List[str] = []

    version = data.get("version")

    if isinstance(version, str) and version.strip().isdigit():
        warnings.append(
            f"'version' was string {version!r}; coerced to int"
        )
        version = int(version.strip())

    if not isinstance(version, int) or isinstance(version, bool):
        raise PlanParseError(
            "missing or non-integer 'version' (expected 1)",
            hint='add "version": 1 at the top of edit_plan.json',
        )

    if version not in SUPPORTED_VERSIONS:
        raise PlanParseError(
            f"unsupported plan version {version} "
            f"(supported: {list(SUPPORTED_VERSIONS)})"
        )

    unknown_top = set(data.keys()) - KNOWN_TOP_KEYS

    if unknown_top:
        warnings.append(
            f"unknown top-level keys ignored: {sorted(unknown_top)}"
        )

    images_raw = data.get("images")

    if not isinstance(images_raw, list):
        raise PlanParseError("'images' must be an array")

    if not images_raw:
        raise PlanParseError(
            "'images' array is empty — nothing to render"
        )

    images: List[PlanImage] = []

    for idx, entry in enumerate(images_raw):
        parsed, entry_warnings = _parse_entry(entry, idx)
        images.append(parsed)
        warnings.extend(entry_warnings)

    if not any(img.keep for img in images):
        raise PlanParseError(
            "every image in the plan has keep=false — "
            "nothing to render"
        )

    with_order = sum(
        1 for img in images if img.order is not None
    )
    without_order = len(images) - with_order

    if with_order and without_order:
        warnings.append(
            f"mixed 'order' presence: {with_order} entries have "
            f"order, {without_order} do not; "
            "file order is used for unordered ones"
        )

    if with_order > 1:
        orders = [
            img.order
            for img in images
            if img.order is not None
        ]

        duplicates = [
            order
            for order, count in Counter(orders).items()
            if count > 1
        ]

        if duplicates:
            warnings.append(
                f"duplicate 'order' values: {sorted(duplicates)}"
            )

    file_counts = Counter(img.file for img in images)

    duplicate_files = [
        name
        for name, count in file_counts.items()
        if count > 1
    ]

    if duplicate_files:
        warnings.append(
            f"duplicate 'file' entries: {duplicate_files[:5]}"
        )

    return EditPlan(
        version=version,
        images=images,
        warnings=warnings,
        raw=data,
    )


def parse_plan_text(text: str) -> EditPlan:
    try:
        data = json.loads(text)

    except json.JSONDecodeError as exc:
        raise PlanParseError(
            f"edit_plan.json is not valid JSON: {exc.msg} "
            f"at line {exc.lineno}, col {exc.colno}",
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
    """Quick structural check; full validation happens in parse_plan()."""
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
