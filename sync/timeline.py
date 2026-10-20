from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from planning.plan_parser import EditPlan, PlanImage
from utils.error_handler import PipelineError, STAGE_SYNC
from utils.logger import get_logger

if TYPE_CHECKING:
    from audio.sentences import Sentence
    from sync.story_matcher import MatchResult, MatchedScene

logger = get_logger("sync.timeline")

EFFECTS_POOL = [
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "pan_up",
    "pan_down",
    "zoomin_pan_left",
    "zoomin_pan_right",
    "zoomout_pan_left",
    "zoomout_pan_right",
    "diagonal_zoom_in",
    "diagonal_zoom_out",
]


@dataclass
class RenderScene:
    image_filename: str
    start: float
    end: float
    effect: str
    text: str = ""
    source: str = "plan"
    crop: Optional[Tuple[int, int, int, int]] = None
    transition: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def duration(self) -> float:
        return self.end - self.start

    def as_dict(self) -> Dict[str, Any]:
        return {
            "image_filename": self.image_filename,
            "start": self.start,
            "end": self.end,
            "effect": self.effect,
            "text": self.text,
            "crop": self.crop,
            "transition": self.transition,
        }


def _effect_from_plan(effects: Any) -> Optional[str]:
    if isinstance(effects, str):
        clean = effects.strip()
        return clean if clean in EFFECTS_POOL else None

    if not isinstance(effects, dict):
        return None

    explicit = effects.get("effect")
    if isinstance(explicit, str) and explicit.strip() in EFFECTS_POOL:
        return explicit.strip()

    pan = effects.get("pan")
    if isinstance(pan, str):
        pan = pan.lower().strip()
    else:
        pan = None

    if pan not in {"left", "right", "up", "down"}:
        pan = None

    zoom = effects.get("zoom")
    if zoom is True:
        combined = {
            "left": "zoomin_pan_left",
            "right": "zoomin_pan_right",
        }.get(pan)
        if combined in EFFECTS_POOL:
            return combined
        return "zoom_in"

    if zoom is False and pan:
        return f"pan_{pan}"

    if pan:
        return f"pan_{pan}"

    return None


def _assign_effects(
    images: Sequence[Any],
    explicit_effects: Optional[Sequence[Optional[str]]] = None,
) -> List[str]:
    rng = random.Random(len(images))
    result: List[str] = []
    previous: Optional[str] = None

    for index, image in enumerate(images):
        effect: Optional[str] = None

        if explicit_effects is not None and index < len(explicit_effects):
            candidate = explicit_effects[index]
            if isinstance(candidate, str) and candidate in EFFECTS_POOL:
                effect = candidate

        if effect is None and image is not None:
            candidate = _effect_from_plan(getattr(image, "effects", None))
            if candidate in EFFECTS_POOL:
                effect = candidate

        if effect is None:
            choices = [item for item in EFFECTS_POOL if item != previous]
            if not choices:
                choices = list(EFFECTS_POOL)
            effect = rng.choice(choices)

        result.append(effect)
        previous = effect

    return result


def _scene_text(scene: MatchedScene, sentences: List[Sentence]) -> str:
    plan_item = getattr(scene, "plan_item", None) or getattr(scene, "image", None)
    story_line = getattr(plan_item, "story_line", "") or getattr(plan_item, "story", "")
    if isinstance(story_line, str) and story_line.strip():
        return story_line.strip()

    pieces: List[str] = []
    for index in getattr(scene, "sentence_indices", []):
        if 0 <= index < len(sentences):
            text = sentences[index].text.strip()
            if text:
                pieces.append(text)

    return " ".join(pieces)


def _plan_image(matched: Any) -> Any:
    return getattr(matched, "plan_item", None) or getattr(matched, "image", None)


def _scene_file(matched: Any) -> str:
    img = _plan_image(matched)
    name = (
        getattr(img, "file", None)
        or getattr(img, "filename", None)
        or getattr(matched, "media_path", None)
        or getattr(matched, "file", None)
    )
    return str(name).strip() if name else ""


def _scene_crop(matched: Any) -> Optional[Tuple[int, int, int, int]]:
    img = _plan_image(matched)
    crop = getattr(img, "crop", None) or getattr(matched, "crop", None)
    if isinstance(crop, (list, tuple)) and len(crop) == 4:
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in crop)
            if x2 > x1 and y2 > y1:
                return (max(0, x1), max(0, y1), x2, y2)
        except (TypeError, ValueError):
            return None
    return None


def _merge_adjacent(
    scenes: List[RenderScene],
    min_duration: float = 0.25,
    preserve_distinct_entries: bool = False,
) -> List[RenderScene]:
    merged: List[RenderScene] = []
    for scene in scenes:
        if (
            merged
            and not preserve_distinct_entries
            and merged[-1].image_filename == scene.image_filename
            and merged[-1].crop == scene.crop
            and merged[-1].effect == scene.effect
            and scene.crop is None
        ):
            prev = merged[-1]
            prev.end = scene.end
            if scene.text:
                prev.text = f"{prev.text} {scene.text}".strip()
            prev.warnings.extend(scene.warnings)
        elif merged and (scene.end - scene.start) < min_duration:
            merged[-1].end = scene.end
        else:
            merged.append(scene)
    return merged


def build_scenes_from_match(
    plan: EditPlan,
    match: MatchResult,
    sentences: List[Sentence],
) -> List[RenderScene]:
    scenes_source = (
        getattr(match, "scenes", None)
        or getattr(match, "matches", None)
        or list(match)
    )
    if not scenes_source:
        raise PipelineError(STAGE_SYNC, "matcher produced no scenes")

    images = [_plan_image(item) for item in scenes_source]
    effects = _assign_effects(images)
    scenes: List[RenderScene] = []

    has_explicit_plan_data = (
        str(getattr(match, "method", "")) == "explicit"
        or any(_scene_crop(m) is not None for m in scenes_source)
        or any(getattr(_plan_image(m), "start_time", None) is not None for m in scenes_source)
    )

    for matched, effect in zip(scenes_source, effects):
        filename = _scene_file(matched)
        if not filename:
            raise PipelineError(STAGE_SYNC, "matched scene has no image file")

        try:
            start = float(getattr(matched, "start", getattr(matched, "start_time", 0.0)))
            end = float(getattr(matched, "end", getattr(matched, "end_time", 3.0)))
        except (TypeError, ValueError) as exc:
            raise PipelineError(
                STAGE_SYNC,
                f"invalid timestamps for image {filename}",
            ) from exc

        if start < 0 or end <= start:
            raise PipelineError(
                STAGE_SYNC,
                f"invalid scene span for {filename}: {start:.3f}–{end:.3f}",
            )

        plan_img = _plan_image(matched)
        transition = getattr(plan_img, "transition", None) if plan_img else None

        scenes.append(
            RenderScene(
                image_filename=filename,
                start=start,
                end=end,
                effect=effect,
                text=_scene_text(matched, sentences),
                source="plan",
                crop=_scene_crop(matched),
                transition=transition,
            )
        )

    scenes = _merge_adjacent(
        scenes,
        min_duration=0.25 if has_explicit_plan_data else 0.8,
        preserve_distinct_entries=has_explicit_plan_data,
    )
    _validate_spans(scenes)

    logger.info(
        "built %d render scenes from plan (method=%s)",
        len(scenes),
        str(getattr(match, "method", "plan")),
    )
    return scenes


def _legacy_entry_to_scene(
    entry: Dict[str, Any],
    index: int,
) -> Optional[RenderScene]:
    image = entry.get("image") or entry.get("file")
    if not isinstance(image, str) or not image.strip():
        return None

    try:
        start = float(entry["start"])
        end = float(entry["end"])
    except (KeyError, TypeError, ValueError):
        logger.warning("ignoring legacy entry %d: invalid timestamps", index)
        return None

    if start < 0:
        start = 0.0
    if end <= start:
        logger.warning("ignoring legacy entry %d: invalid duration", index)
        return None

    text = entry.get("reason") or entry.get("text") or ""

    return RenderScene(
        image_filename=image.strip(),
        start=start,
        end=end,
        effect="zoom_in",
        text=str(text),
        source="legacy_timeline",
    )


def build_scenes_from_legacy_timeline(
    timeline: Any,
    transcript_duration: float,
) -> List[RenderScene]:
    if not isinstance(timeline, list) or not timeline:
        raise PipelineError(
            STAGE_SYNC,
            "legacy timeline.json is empty or is not an array",
        )

    scenes: List[RenderScene] = []
    for index, entry in enumerate(timeline):
        if isinstance(entry, dict):
            scene = _legacy_entry_to_scene(entry, index)
            if scene is not None:
                scenes.append(scene)

    if not scenes:
        raise PipelineError(STAGE_SYNC, "no usable entries in legacy timeline.json")

    scenes.sort(key=lambda s: (s.start, s.end))

    try:
        duration = float(transcript_duration)
    except (TypeError, ValueError):
        duration = 0.0

    if duration < 0:
        duration = 0.0

    if duration > 0:
        clipped: List[RenderScene] = []
        for scene in scenes:
            start = min(max(scene.start, 0.0), duration)
            end = min(max(scene.end, 0.0), duration)
            if end <= start:
                continue
            scene.start = start
            scene.end = end
            clipped.append(scene)
        scenes = clipped

    if not scenes:
        raise PipelineError(STAGE_SYNC, "no timeline entries fall within audio duration")

    scenes[0].start = 0.0

    for index in range(1, len(scenes)):
        previous = scenes[index - 1]
        current = scenes[index]
        if current.start < previous.end:
            orig_dur = current.end - current.start
            current.start = previous.end
            current.end = current.start + orig_dur
            if duration > 0:
                current.end = min(current.end, duration)

    scenes = [s for s in scenes if s.end > s.start]
    if not scenes:
        raise PipelineError(STAGE_SYNC, "timeline contains no valid positive-duration scenes")

    if duration > 0:
        scenes[-1].end = duration

    fake_images = [
        PlanImage(file=scene.image_filename, effects={}, raw={})
        for scene in scenes
    ]
    for scene, effect in zip(scenes, _assign_effects(fake_images)):
        scene.effect = effect

    _validate_spans(scenes)
    logger.info("adapted %d scenes from legacy timeline.json", len(scenes))
    return scenes


def _validate_spans(scenes: List[RenderScene]) -> None:
    previous_end = 0.0
    for index, scene in enumerate(scenes):
        if scene.start < 0:
            raise PipelineError(STAGE_SYNC, f"scene {index} has negative start timestamp")
        if scene.end <= scene.start:
            raise PipelineError(
                STAGE_SYNC,
                f"scene {index} ({scene.image_filename}) has zero/negative duration",
            )
        if index > 0 and scene.start < previous_end - 1e-6:
            raise PipelineError(STAGE_SYNC, f"scene {index} overlaps previous scene")
        previous_end = scene.end
