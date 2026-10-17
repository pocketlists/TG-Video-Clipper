"""
sync/timeline.py
Turns a MatchResult + EditPlan into the flat list of render scenes the
renderer consumes. Also adapts legacy timeline.json files (array of
{image,start,end}) so old plans keep working.

This module owns no timing decisions — the matcher already produced
monotonic spans. It only:
  - attaches an effect (respecting plan.effects when given)
  - attaches per-scene text (for logging / future caption support)
  - ensures no two consecutive scenes share the same effect
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


# Matches video_editor.EFFECTS in the old code — preserved verbatim so
# rendered clips look identical.
EFFECTS_POOL = [
    "zoom_in", "zoom_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
    "zoomin_pan_left", "zoomin_pan_right",
    "zoomout_pan_left", "zoomout_pan_right",
    "diagonal_zoom_in", "diagonal_zoom_out",
]

_PAN_TO_ZOOMIN = {
    "left":  "zoomin_pan_left",
    "right": "zoomin_pan_right",
    "up":    "zoomin_pan_up",   # not in pool — falls back to zoom_in
    "down":  "zoomin_pan_down", # ditto
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
    """Return an effect name derived from plan.effects, or None if the
    plan did not express an intent we can honour."""
    if not effects:
        return None

    explicit = effects.get("effect")
    if isinstance(explicit, str) and explicit in EFFECTS_POOL:
        return explicit

    pan = effects.get("pan")
    pan = pan.lower() if isinstance(pan, str) else None
    if pan not in _PAN_PLAIN:
        pan = None

    zoom = effects.get("zoom")
    if zoom is True:
        if pan and pan in _PAN_TO_ZOOMIN and _PAN_TO_ZOOMIN[pan] in EFFECTS_POOL:
            return _PAN_TO_ZOOMIN[pan]
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
    """One effect per image; never two consecutive identical effects.
    Explicit plan-driven effects win; unset ones are picked from the pool."""
    random.seed(len(images))  # deterministic per plan, like old code's feel
    result: List[str] = []
    last: Optional[str] = None
    for i, img in enumerate(images):
        chosen: Optional[str] = None
        if explicit_effects is not None and i < len(explicit_effects):
            chosen = explicit_effects[i]
        if chosen is None:
            chosen = _effect_from_plan(img.effects)
        if chosen is None or chosen == last:
            pool = [e for e in EFFECTS_POOL if e != last] or list(EFFECTS_POOL)
            chosen = random.choice(pool)
        result.append(chosen)
        last = chosen
    return result


# ---------------------------------------------------------------------------
# Text attachment
# ---------------------------------------------------------------------------

def _scene_text(scene: MatchedScene, sentences: List[Sentence]) -> str:
    pieces = []
    for idx in scene.sentence_indices:
        if 0 <= idx < len(sentences):
            pieces.append(sentences[idx].text.strip())
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

    images = [s.image for s in match.scenes]
    effects = _assign_effects(images)

    scenes: List[RenderScene] = []
    for scene, effect in zip(match.scenes, effects):
        if scene.end <= scene.start:
            raise PipelineError(
                STAGE_SYNC,
                f"invalid span for '{scene.image.file}': "
                f"{scene.start:.3f}–{scene.end:.3f}",
            )
        rs = RenderScene(
            image_filename=scene.image.file,
            start=scene.start,
            end=scene.end,
            effect=effect,
            text=_scene_text(scene, sentences),
            source="plan",
        )
        if scene.image.transition:
            rs.warnings.append(
                f"transition '{scene.image.transition}' not yet supported; ignored"
            )
        scenes.append(rs)

    _validate_spans(scenes)
    logger.info(f"built {len(scenes)} render scenes from plan (method={match.method})")
    return scenes


# ---------------------------------------------------------------------------
# Legacy timeline.json adapter
# ---------------------------------------------------------------------------

def _legacy_entry_to_scene(
    entry: Dict[str, Any], index: int,
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
        effect="zoom_in",          # placeholder; reassigned below
        text=str(entry.get("reason") or entry.get("text") or ""),
        source="legacy_timeline",
    )


def build_scenes_from_legacy_timeline(
    timeline: Any,
    transcript_duration: float,
) -> List[RenderScene]:
    """Adapt a legacy timeline.json (array of {image,start,end}) to
    RenderScene. Sorting, de-overlap and end-extension are applied so the
    result is safe to render even if the file was hand-edited."""
    if not isinstance(timeline, list) or not timeline:
        raise PipelineError(
            STAGE_SYNC, "legacy timeline.json is empty or not an array"
        )

    scenes: List[RenderScene] = []
    for idx, entry in enumerate(timeline):
        if not isinstance(entry, dict):
            continue
        scene = _legacy_entry_to_scene(entry, idx)
        if scene is not None:
            scenes.append(scene)

    if not scenes:
        raise PipelineError(
            STAGE_SYNC, "no usable entries in legacy timeline.json"
        )

    scenes.sort(key=lambda s: s.start)
    scenes[0].start = 0.0

    for i in range(1, len(scenes)):
        if scenes[i].start < scenes[i - 1].end:
            scenes[i].start = scenes[i - 1].end
            if scenes[i].end <= scenes[i].start:
                scenes[i].end = scenes[i].start + 0.05

    if transcript_duration > 0:
        scenes[-1].end = max(transcript_duration, scenes[-1].start + 0.05)

    fake_images = [
        PlanImage(file=s.image_filename, effects={}, raw={}) for s in scenes
    ]
    for s, eff in zip(scenes, _assign_effects(fake_images)):
        s.effect = eff

    _validate_spans(scenes)
    logger.info(f"adapted {len(scenes)} scenes from legacy timeline.json")
    return scenes


# ---------------------------------------------------------------------------
# Span safety
# ---------------------------------------------------------------------------

def _validate_spans(scenes: List[RenderScene]) -> None:
    prev_end = -1.0
    for i, s in enumerate(scenes):
        if s.start < 0:
            raise PipelineError(
                STAGE_SYNC, f"scene {i} has negative start: {s.start:.3f}"
            )
        if s.end <= s.start:
            raise PipelineError(
                STAGE_SYNC,
                f"scene {i} ({s.image_filename}) has zero or negative duration",
            )
        if s.start < prev_end - 1e-6:
            logger.warning(
                f"scene {i} starts at {s.start:.3f} before previous end "
                f"{prev_end:.3f} — clamping"
            )
            s.start = prev_end
        prev_end = s.end
