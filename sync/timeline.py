
"""
sync/timeline.py

Turns a MatchResult + EditPlan into the flat list of render scenes.
Also adapts legacy timeline.json files.

Explicit effects from edit_plan.json are preserved. Missing effects
are assigned automatically while avoiding consecutive repetitions
where possible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from audio.sentences import Sentence
from planning.plan_parser import EditPlan, PlanImage
from sync.story_matcher import MatchResult, MatchedScene
from utils.error_handler import PipelineError, STAGE_SYNC
from utils.logger import get_logger


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

_PAN_TO_ZOOMIN = {
    "left": "zoomin_pan_left",
    "right": "zoomin_pan_right",
    "up": "zoomin_pan_up",
    "down": "zoomin_pan_down",
}

_PAN_PLAIN = {"left", "right", "up", "down"}


@dataclass
class RenderScene:
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


# ---------------------------------------------------------------------------
# Effect resolution
# ---------------------------------------------------------------------------

def _effect_from_plan(effects: Dict[str, Any]) -> Optional[str]:
    """Resolve an effect specified in the edit plan."""
    if not isinstance(effects, dict) or not effects:
        return None

    explicit = effects.get("effect")

    if isinstance(explicit, str) and explicit in EFFECTS_POOL:
        return explicit

    pan = effects.get("pan")
    pan = pan.lower().strip() if isinstance(pan, str) else None

    if pan not in _PAN_PLAIN:
        pan = None

    zoom = effects.get("zoom")

    if zoom is True:
        combined = _PAN_TO_ZOOMIN.get(pan) if pan else None

        if combined in EFFECTS_POOL:
            return combined

        return "zoom_in"

    if zoom is False:
        return f"pan_{pan}" if pan else None

    if pan:
        return f"pan_{pan}"

    return None


def _assign_effects(
    images: Sequence[PlanImage],
    explicit_effects: Optional[Sequence[Optional[str]]] = None,
) -> List[str]:
    """
    Assign one effect to every image entry.

    Explicitly requested effects are preserved, including consecutive
    repeated effects. Only automatically assigned effects try to avoid
    repeating the preceding scene's effect.
    """
    rng = random.Random(len(images))

    result: List[str] = []
    last: Optional[str] = None

    for index, image in enumerate(images):
        chosen: Optional[str] = None

        # Optional explicit override supplied by the caller.
        if explicit_effects is not None and index < len(explicit_effects):
            candidate = explicit_effects[index]

            if isinstance(candidate, str) and candidate in EFFECTS_POOL:
                chosen = candidate

        # Otherwise use the effect stored on the plan image.
        if chosen is None:
            chosen = _effect_from_plan(image.effects)

        # Explicit effects must not be changed just to avoid repetition.
        if chosen is not None:
            result.append(chosen)
            last = chosen
            continue

        # Only auto-generated effects are selected from the pool.
        available = [effect for effect in EFFECTS_POOL if effect != last]

        if not available:
            available = list(EFFECTS_POOL)

        chosen = rng.choice(available)

        result.append(chosen)
        last = chosen

    return result


# ---------------------------------------------------------------------------
# Text attachment
# ---------------------------------------------------------------------------

def _scene_text(scene: MatchedScene, sentences: List[Sentence]) -> str:
    pieces: List[str] = []

    for index in scene.sentence_indices:
        if 0 <= index < len(sentences):
            text = sentences[index].text.strip()

            if text:
                pieces.append(text)

    if pieces:
        return " ".join(pieces)

    if scene.image.story_line:
        return scene.image.story_line.strip()

    return ""


# ---------------------------------------------------------------------------
# From MatchResult
# ---------------------------------------------------------------------------

def build_scenes_from_match(
    plan: EditPlan,
    match: MatchResult,
    sentences: List[Sentence],
) -> List[RenderScene]:
    if not match.scenes:
        raise PipelineError(STAGE_SYNC, "matcher produced no scenes")

    images = [scene.image for scene in match.scenes]
    effects = _assign_effects(images)

    scenes: List[RenderScene] = []

    for scene, effect in zip(match.scenes, effects):
        if scene.end <= scene.start:
            raise PipelineError(
                STAGE_SYNC,
                f"invalid span for '{scene.image.file}': "
                f"{scene.start:.3f}–{scene.end:.3f}",
            )

        render_scene = RenderScene(
            image_filename=scene.image.file,
            start=scene.start,
            end=scene.end,
            effect=effect,
            text=_scene_text(scene, sentences),
            source="plan",
        )

        if scene.image.transition:
            render_scene.warnings.append(
                f"transition '{scene.image.transition}' "
                "not yet supported; ignored"
            )

        scenes.append(render_scene)

    _validate_spans(scenes)

    logger.info(
        f"built {len(scenes)} render scenes from plan "
        f"(method={match.method})"
    )

    return scenes


# ---------------------------------------------------------------------------
# Legacy timeline.json adapter
# ---------------------------------------------------------------------------

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
        return None

    if end <= start:
        return None

    return RenderScene(
        image_filename=image.strip(),
        start=start,
        end=end,
        effect="zoom_in",
        text=str(entry.get("reason") or entry.get("text") or ""),
        source="legacy_timeline",
    )


def build_scenes_from_legacy_timeline(
    timeline: Any,
    transcript_duration: float,
) -> List[RenderScene]:
    """
    Adapt legacy timeline.json entries to RenderScene objects.

    Entries are sorted, overlaps are clamped, and the final scene is
    extended to cover the transcript duration when that duration is
    available.
    """
    if not isinstance(timeline, list) or not timeline:
        raise PipelineError(
            STAGE_SYNC,
            "legacy timeline.json is empty or not an array",
        )

    scenes: List[RenderScene] = []

    for index, entry in enumerate(timeline):
        if not isinstance(entry, dict):
            continue

        scene = _legacy_entry_to_scene(entry, index)

        if scene is not None:
            scenes.append(scene)

    if not scenes:
        raise PipelineError(
            STAGE_SYNC,
            "no usable entries in legacy timeline.json",
        )

    scenes.sort(key=lambda scene: scene.start)
    scenes[0].start = 0.0

    for index in range(1, len(scenes)):
        previous = scenes[index - 1]
        current = scenes[index]

        if current.start < previous.end:
            current.start = previous.end

            if current.end <= current.start:
                current.end = current.start + 0.05

    if transcript_duration > 0:
        scenes[-1].end = max(
            transcript_duration,
            scenes[-1].start + 0.05,
        )

    fake_images = [
        PlanImage(
            file=scene.image_filename,
            effects={},
            raw={},
        )
        for scene in scenes
    ]

    assigned_effects = _assign_effects(fake_images)

    for scene, effect in zip(scenes, assigned_effects):
        scene.effect = effect

    _validate_spans(scenes)

    logger.info(
        f"adapted {len(scenes)} scenes from legacy timeline.json"
    )

    return scenes


# ---------------------------------------------------------------------------
# Span validation
# ---------------------------------------------------------------------------

def _validate_spans(scenes: List[RenderScene]) -> None:
    previous_end = -1.0

    for index, scene in enumerate(scenes):
        if scene.start < 0:
            raise PipelineError(
                STAGE_SYNC,
                f"scene {index} has negative start: "
                f"{scene.start:.3f}",
            )

        if scene.end <= scene.start:
            raise PipelineError(
                STAGE_SYNC,
                f"scene {index} ({scene.image_filename}) "
                "has zero or negative duration",
            )

        if scene.start < previous_end - 1e-6:
            logger.warning(
                f"scene {index} starts at {scene.start:.3f} "
                f"before previous end {previous_end:.3f} — clamping"
            )

            scene.start = previous_end

            if scene.end <= scene.start:
                raise PipelineError(
                    STAGE_SYNC,
                    f"scene {index} ({scene.image_filename}) "
                    "became invalid after overlap correction",
                )

        previous_end = scene.end
