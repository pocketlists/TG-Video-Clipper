"""
sync/timeline.py

Convert MatchResult + EditPlan into the RenderScene list consumed by the
renderer. Also adapt legacy timeline.json arrays containing {image,start,end}.

Responsibilities:
  - resolve per-image visual effects, respecting explicit plan effects;
  - attach sentence/story text to scenes;
  - adapt legacy timeline entries;
  - validate scene spans before rendering.

Timing alignment belongs to sync.story_matcher. This module validates the
result and repairs legacy timeline entries where it can do so safely.
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


# Keep these effect names aligned with render/effects.py.
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
    # Vertical combined effects are not supported by render/effects.py.
    "up": "zoomin_pan_up",
    "down": "zoomin_pan_down",
}
_PAN_PLAIN = {"left", "right", "up", "down"}


@dataclass
class RenderScene:
    """A single image interval prepared for the video renderer."""

    image_filename: str
    start: float
    end: float
    effect: str
    text: str = ""
    source: str = "plan"
    warnings: List[str] = field(default_factory=list)

    def duration(self) -> float:
        """Return this scene's duration in seconds."""
        return self.end - self.start

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the renderer-relevant fields."""
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
    """Resolve a supported effect from one image's plan settings.

    Returns None when the plan does not specify a usable effect, allowing the
    caller to choose an automatic effect.
    """
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
    """Return one supported effect per image.

    Explicit effects are preserved, even if two adjacent scenes intentionally
    use the same effect. Only automatically selected effects avoid repeating
    the previous effect where another option is available.

    A local Random instance keeps the output deterministic without changing
    Python's process-wide random state.
    """
    rng = random.Random(len(images))
    result: List[str] = []
    last: Optional[str] = None

    for i, image in enumerate(images):
        chosen: Optional[str] = None
        is_explicit = False

        if explicit_effects is not None and i < len(explicit_effects):
            candidate = explicit_effects[i]
            if isinstance(candidate, str) and candidate in EFFECTS_POOL:
                chosen = candidate
                is_explicit = True

        if chosen is None:
            candidate = _effect_from_plan(getattr(image, "effects", {}))
            if candidate in EFFECTS_POOL:
                chosen = candidate
                is_explicit = True

        if chosen is None:
            pool = [effect for effect in EFFECTS_POOL if effect != last]
            if not pool:
                pool = list(EFFECTS_POOL)
            chosen = rng.choice(pool)

        # Do not override an explicit plan choice just to force variation.
        # Variation applies only to auto-generated effects.
        elif not is_explicit and chosen == last:
            pool = [effect for effect in EFFECTS_POOL if effect != last]
            if pool:
                chosen = rng.choice(pool)

        result.append(chosen)
        last = chosen

    return result


# ---------------------------------------------------------------------------
# Text attachment
# ---------------------------------------------------------------------------

def _scene_text(scene: MatchedScene, sentences: List[Sentence]) -> str:
    """Build the scene's display/log text from matched transcript sentences."""
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


# ---------------------------------------------------------------------------
# From MatchResult
# ---------------------------------------------------------------------------

def build_scenes_from_match(
    plan: EditPlan,
    match: MatchResult,
    sentences: List[Sentence],
) -> List[RenderScene]:
    """Convert matcher output to renderer scenes and validate their spans."""
    if not match.scenes:
        raise PipelineError(STAGE_SYNC, "matcher produced no scenes")

    images = [scene.image for scene in match.scenes]
    effects = _assign_effects(images)
    scenes: List[RenderScene] = []

    for matched_scene, effect in zip(match.scenes, effects):
        try:
            start = float(matched_scene.start)
            end = float(matched_scene.end)
        except (TypeError, ValueError) as exc:
            raise PipelineError(
                STAGE_SYNC,
                f"invalid timestamps for '{matched_scene.image.file}'",
            ) from exc

        if end <= start:
            raise PipelineError(
                STAGE_SYNC,
                f"invalid span for '{matched_scene.image.file}': "
                f"{start:.3f}–{end:.3f}",
            )

        render_scene = RenderScene(
            image_filename=matched_scene.image.file,
            start=start,
            end=end,
            effect=effect,
            text=_scene_text(matched_scene, sentences),
            source="plan",
        )

        transition = getattr(matched_scene.image, "transition", None)
        if transition:
            render_scene.warnings.append(
                f"transition '{transition}' not yet supported; ignored"
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
    """Parse one legacy timeline entry; return None if it is unusable."""
    image = entry.get("image") or entry.get("file")
    if not isinstance(image, str) or not image.strip():
        return None

    try:
        start = float(entry["start"])
        end = float(entry["end"])
    except (KeyError, TypeError, ValueError):
        logger.warning(
            f"ignoring legacy timeline entry {index}: invalid times"
        )
        return None

    if end <= start:
        logger.warning(
            f"ignoring legacy timeline entry {index}: "
            "end must be after start"
        )
        return None

    reason = entry.get("reason") or entry.get("text") or ""

    return RenderScene(
        image_filename=image.strip(),
        start=start,
        end=end,
        effect="zoom_in",  # Reassigned after entries are normalized.
        text=str(reason),
        source="legacy_timeline",
    )


def build_scenes_from_legacy_timeline(
    timeline: Any,
    transcript_duration: float,
) -> List[RenderScene]:
    """Adapt legacy timeline JSON into safe, chronologically ordered scenes.

    Entries are sorted by start time, clamped to non-negative timestamps, and
    made non-overlapping. If transcript_duration is positive, the final scene
    is extended or trimmed to the audio duration where possible.
    """
    if not isinstance(timeline, list) or not timeline:
        raise PipelineError(
            STAGE_SYNC, "legacy timeline.json is empty or not an array"
        )

    scenes: List[RenderScene] = []

    for index, entry in enumerate(timeline):
        if isinstance(entry, dict):
            scene = _legacy_entry_to_scene(entry, index)
            if scene is not None:
                scenes.append(scene)

    if not scenes:
        raise PipelineError(
            STAGE_SYNC, "no usable entries in legacy timeline.json"
        )

    # Clamp negative starts before sorting. Preserve each valid duration.
    for scene in scenes:
        if scene.start < 0:
            duration = scene.end - scene.start
            scene.start = 0.0
            scene.end = max(duration, 0.05)

    scenes.sort(key=lambda scene: (scene.start, scene.end))
    scenes[0].start = 0.0

    if scenes[0].end <= scenes[0].start:
        scenes[0].end = 0.05

    for index in range(1, len(scenes)):
        previous = scenes[index - 1]
        current = scenes[index]

        if current.start < previous.end:
            # Move the current scene to the previous scene's end. If it no
            # longer has positive duration, give it a small safe interval.
            old_duration = current.end - current.start
            current.start = previous.end
            current.end = max(
                current.start + old_duration,
                current.start + 0.05,
            )

    try:
        audio_duration = float(transcript_duration)
    except (TypeError, ValueError):
        audio_duration = 0.0

    if audio_duration > 0:
        # If the last scene starts beyond the audio end, move it back to a
        # short interval ending at the audio duration.
        if scenes[-1].start >= audio_duration:
            scenes[-1].end = audio_duration
            scenes[-1].start = max(0.0, audio_duration - 0.05)
        else:
            scenes[-1].end = audio_duration

        # Keep earlier scenes inside the audio duration and non-overlapping.
        for index, scene in enumerate(scenes[:-1]):
            scene.start = min(max(scene.start, 0.0), audio_duration)
            scene.end = min(
                max(scene.end, scene.start + 0.05),
                audio_duration,
            )

            if index > 0 and scene.start < scenes[index - 1].end:
                scene.start = scenes[index - 1].end
                if scene.end <= scene.start:
                    scene.end = min(audio_duration, scene.start + 0.05)

    fake_images = [
        PlanImage(file=scene.image_filename, effects={}, raw={})
        for scene in scenes
    ]

    for scene, effect in zip(scenes, _assign_effects(fake_images)):
        scene.effect = effect

    _validate_spans(scenes)
    logger.info(f"adapted {len(scenes)} scenes from legacy timeline.json")
    return scenes


# ---------------------------------------------------------------------------
# Span safety
# ---------------------------------------------------------------------------

def _validate_spans(scenes: List[RenderScene]) -> None:
    """Raise if any scene has invalid duration or overlaps the previous one."""
    previous_end = 0.0

    for index, scene in enumerate(scenes):
        if scene.start < 0:
            raise PipelineError(
                STAGE_SYNC,
                f"scene {index} has negative start: {scene.start:.3f}",
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
                f"scene {index} overlaps previous scene: starts at "
                f"{scene.start:.3f}, previous ends at {previous_end:.3f}",
            )

        previous_end = scene.end
