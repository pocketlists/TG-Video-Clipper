
"""
sync/timeline.py

Converts matcher output or legacy timeline entries into RenderScene objects.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

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
    """One image and its assigned time span in the rendered video."""

    image_filename: str
    start: float
    end: float
    effect: str
    text: str = ""
    source: str = "plan"
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
        }


def _effect_from_plan(effects: Dict[str, Any]) -> Optional[str]:
    """Resolve a supported effect from an image's plan settings."""
    if not isinstance(effects, dict):
        return None

    explicit = effects.get("effect")
    if isinstance(explicit, str) and explicit in EFFECTS_POOL:
        return explicit

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
    images: Sequence[PlanImage],
    explicit_effects: Optional[Sequence[Optional[str]]] = None,
) -> List[str]:
    """Assign deterministic automatic effects while preserving explicit ones."""
    rng = random.Random(len(images))
    result: List[str] = []
    previous: Optional[str] = None

    for index, image in enumerate(images):
        effect: Optional[str] = None

        if explicit_effects is not None and index < len(explicit_effects):
            candidate = explicit_effects[index]
            if isinstance(candidate, str) and candidate in EFFECTS_POOL:
                effect = candidate

        if effect is None:
            candidate = _effect_from_plan(
                getattr(image, "effects", {})
            )
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
    """Join the transcript sentences associated with a matched scene."""
    pieces: List[str] = []

    for index in scene.sentence_indices:
        if 0 <= index < len(sentences):
            text = sentences[index].text.strip()
            if text:
                pieces.append(text)

    if pieces:
        return " ".join(pieces)

    story_line = getattr(scene.image, "story_line", "")
    return story_line.strip() if isinstance(story_line, str) else ""


def build_scenes_from_match(
    plan: EditPlan,
    match: MatchResult,
    sentences: List[Sentence],
) -> List[RenderScene]:
    """Convert matcher output into scenes for the renderer."""
    if not match.scenes:
        raise PipelineError(STAGE_SYNC, "matcher produced no scenes")

    images = [item.image for item in match.scenes]
    effects = _assign_effects(images)
    scenes: List[RenderScene] = []

    for matched, effect in zip(match.scenes, effects):
        try:
            start = float(matched.start)
            end = float(matched.end)
        except (TypeError, ValueError) as exc:
            raise PipelineError(
                STAGE_SYNC,
                f"invalid timestamps for image {matched.image.file}",
            ) from exc

        if start < 0 or end <= start:
            raise PipelineError(
                STAGE_SYNC,
                f"invalid scene span for {matched.image.file}: "
                f"{start:.3f}–{end:.3f}",
            )

        scene = RenderScene(
            image_filename=matched.image.file,
            start=start,
            end=end,
            effect=effect,
            text=_scene_text(matched, sentences),
            source="plan",
        )

        transition = getattr(matched.image, "transition", None)
        if transition:
            scene.warnings.append(
                f"transition '{transition}' is not supported and was ignored"
            )

        scenes.append(scene)

    _validate_spans(scenes)

    logger.info(
        "built %d render scenes from plan (method=%s)",
        len(scenes),
        match.method,
    )
    return scenes


def _legacy_entry_to_scene(
    entry: Dict[str, Any],
    index: int,
) -> Optional[RenderScene]:
    """Parse one legacy timeline entry."""
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
    """Convert legacy timeline JSON entries into validated render scenes."""
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
        raise PipelineError(
            STAGE_SYNC,
            "no usable entries in legacy timeline.json",
        )

    scenes.sort(key=lambda scene: (scene.start, scene.end))

    try:
        duration = float(transcript_duration)
    except (TypeError, ValueError):
        duration = 0.0

    if duration < 0:
        duration = 0.0

    # Clamp to the audio duration and remove overlaps.
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
        raise PipelineError(
            STAGE_SYNC,
            "no timeline entries fall within the audio duration",
        )

    scenes[0].start = 0.0

    for index in range(1, len(scenes)):
        previous = scenes[index - 1]
        current = scenes[index]

        if current.start < previous.end:
            original_duration = current.end - current.start
            current.start = previous.end
            current.end = current.start + original_duration

            if duration > 0:
                current.end = min(current.end, duration)

    # Remove any scenes made invalid by overlap correction.
    scenes = [
        scene for scene in scenes
        if scene.end > scene.start
    ]

    if not scenes:
        raise PipelineError(
            STAGE_SYNC,
            "timeline contains no valid positive-duration scenes",
        )

    if duration > 0:
        scenes[-1].end = duration

    fake_images = [
        PlanImage(file=scene.image_filename, effects={}, raw={})
        for scene in scenes
    ]

    for scene, effect in zip(scenes, _assign_effects(fake_images)):
        scene.effect = effect

    _validate_spans(scenes)

    logger.info(
        "adapted %d scenes from legacy timeline.json",
        len(scenes),
    )
    return scenes


def _validate_spans(scenes: List[RenderScene]) -> None:
    """Reject negative, zero-duration, or overlapping scene spans."""
    previous_end = 0.0

    for index, scene in enumerate(scenes):
        if scene.start < 0:
            raise PipelineError(
                STAGE_SYNC,
                f"scene {index} has a negative start timestamp",
            )

        if scene.end <= scene.start:
            raise PipelineError(
                STAGE_SYNC,
                f"scene {index} ({scene.image_filename}) "
                "has zero or negative duration",
            )

        if index > 0 and scene.start < previous_end - 1e-6:
            raise PipelineError(
                STAGE_SYNC,
                f"scene {index} overlaps the previous scene",
            )

        previous_end = scene.end
