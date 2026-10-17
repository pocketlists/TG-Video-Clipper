"""
render/video_renderer.py
Orchestrates the full render: standardize → per-scene clip → concat →
final assembly → verify. Consumes the scene dicts produced by
sync/timeline.py and the quality preset chosen by the user.

Frame-grid quantization is preserved from the old video_editor.py:
each scene's start/end is rounded to the global fps grid (not its own),
so cumulative rounding drift across many short scenes stays zero. The
encoded clip duration is `total_frames / fps`, matching the audio
timeline exactly.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from config.settings import quality_presets, default_quality, QualityPreset
from media.detector import classify
from media.formats import CATEGORY_IMAGE
from render.encoder import (
    standardize_image,
    create_clip,
    concat_clips,
    assemble,
    verify_output,
)
from utils.error_handler import PipelineError, STAGE_RENDER
from utils.logger import get_logger


logger = get_logger("render.video_renderer")
ProgressFn = Optional[Callable[[str], None]]


# Image extensions we will try to resolve for a plan reference
_IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".gif", ".webp", ".bmp",
    ".tif", ".tiff", ".avif", ".heic", ".heif", ".jxl", ".ico",
}


# ---------------------------------------------------------------------------
# Image path resolution (same rules as plan_validator._Index.resolve)
# ---------------------------------------------------------------------------

def _numeric_prefix(name: str) -> Optional[int]:
    i = 0
    while i < len(name) and name[i].isdigit():
        i += 1
    if i == 0:
        return None
    try:
        return int(name[:i])
    except ValueError:
        return None


def _resolve_image(media_dir: Path, filename: str) -> Path:
    media_dir = Path(media_dir)
    exact = media_dir / filename
    if exact.is_file():
        return exact

    lower = filename.lower()
    for child in media_dir.rglob("*"):
        if child.is_file() and child.name.lower() == lower:
            return child

    prefix = _numeric_prefix(filename)
    if prefix is not None:
        candidates: List[Path] = []
        for child in media_dir.rglob("*"):
            if not child.is_file() or child.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            if _numeric_prefix(child.name) == prefix:
                candidates.append(child)
        if candidates:
            candidates.sort(key=lambda p: p.name.lower())
            return candidates[0]

    raise PipelineError(
        STAGE_RENDER,
        f"image '{filename}' not found under {media_dir}",
        hint="plan_validator should have caught this earlier",
    )


def _validate_image(path: Path, filename: str) -> None:
    result = classify(path)
    if result.category != CATEGORY_IMAGE:
        raise PipelineError(
            STAGE_RENDER,
            f"'{filename}' resolved to '{path.name}' but it is {result.category}, not image",
        )


# ---------------------------------------------------------------------------
# Scene coerce
# ---------------------------------------------------------------------------

def _scene_attr(scene, key: str, default=None):
    if isinstance(scene, dict):
        return scene.get(key, default)
    return getattr(scene, key, default)


def _scene_to_dict(scene) -> Dict:
    return {
        "image_filename": str(_scene_attr(scene, "image_filename", "")),
        "start": float(_scene_attr(scene, "start", 0.0)),
        "end": float(_scene_attr(scene, "end", 0.0)),
        "effect": str(_scene_attr(scene, "effect", "zoom_in")),
        "text": str(_scene_attr(scene, "text", "") or ""),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render(
    scenes: Sequence,
    narration_path: Path,
    output_path: Path,
    media_dir: Path,
    temp_dir: Path,
    *,
    quality: str,
    bgm_path: Optional[Path] = None,
    progress: ProgressFn = None,
) -> Path:
    """
    scenes: iterable of RenderScene (or dicts) produced by sync/timeline.py
    narration_path: original clean audio; never modified
    output_path: final .mp4 destination
    media_dir: folder containing the images referenced by scenes
    temp_dir: scratch directory (safe to delete afterwards)
    quality: key of config.settings.quality_presets()
    bgm_path: optional background music (looped, volume 0.08)
    progress: optional str callback for stage updates
    """
    if not scenes:
        raise PipelineError(STAGE_RENDER, "render called with zero scenes")

    presets = quality_presets()
    if quality not in presets:
        raise PipelineError(
            STAGE_RENDER,
            f"unknown quality {quality!r} (valid: {sorted(presets)})",
        )
    preset: QualityPreset = presets[quality]

    narration_path = Path(narration_path)
    output_path = Path(output_path)
    media_dir = Path(media_dir)
    temp_dir = Path(temp_dir)

    if not narration_path.is_file():
        raise PipelineError(
            STAGE_RENDER, f"narration audio not found: {narration_path}"
        )
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- 1) Prepare per-scene render plans ----------------
    scene_list = [_scene_to_dict(s) for s in scenes]
    total = len(scene_list)
    logger.info(f"render start: {total} scenes at quality={quality}")

    if progress:
        progress(f"preparing {total} scenes")

    prepared: List[Dict] = []
    prev_end_frame = 0
    for idx, sc in enumerate(scene_list):
        filename = sc["image_filename"].strip()
        if not filename:
            raise PipelineError(
                STAGE_RENDER, f"scene {idx + 1} has empty image_filename"
            )
        start = max(sc["start"], 0.0)
        end = max(sc["end"], start + 0.05)

        start_frame = round(start * preset.fps)
        end_frame = round(end * preset.fps)
        # monotonic: next scene can't start before the previous frame ended
        start_frame = max(start_frame, prev_end_frame)
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        prev_end_frame = end_frame

        total_frames = end_frame - start_frame
        prepared.append({
            "index": idx,
            "filename": filename,
            "effect": sc["effect"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "total_frames": total_frames,
            "duration": total_frames / preset.fps,
        })

    if prev_end_frame < 1:
        raise PipelineError(STAGE_RENDER, "all scenes collapsed to zero frames")

    # ---------------- 2) Per-scene clip render ----------------
    clip_paths: List[Path] = []
    for idx, sc in enumerate(prepared):
        if progress:
            progress(f"clip {idx + 1}/{total}")

        img_path = _resolve_image(media_dir, sc["filename"])
        _validate_image(img_path, sc["filename"])

        standardized = temp_dir / f"std_{idx:04d}.png"
        clip_path = temp_dir / f"clip_{idx:04d}.mp4"

        try:
            standardize_image(img_path, standardized, preset.width, preset.height)
            create_clip(
                standardized,
                clip_path,
                effect=sc["effect"],
                width=preset.width,
                height=preset.height,
                fps=preset.fps,
                total_frames=sc["total_frames"],
                crf=preset.crf,
                preset=preset.encoder_preset,
            )
        finally:
            standardized.unlink(missing_ok=True)

        if not clip_path.is_file() or clip_path.stat().st_size < 512:
            raise PipelineError(
                STAGE_RENDER,
                f"clip {idx + 1} ({sc['filename']}) produced no output",
            )
        clip_paths.append(clip_path)
        logger.info(
            f"clip {idx + 1}/{total} ok: {sc['filename']} "
            f"({sc['duration']:.3f}s, {sc['total_frames']} frames, {sc['effect']})"
        )

    # ---------------- 3) Concat ----------------
    if progress:
        progress("concatenating clips")
    concat_path = temp_dir / "concat.mp4"
    concat_clips(clip_paths, concat_path)

    # ---------------- 4) Final assembly ----------------
    if progress:
        progress("mixing audio")
    assemble(
        video_path=concat_path,
        narration_path=narration_path,
        output_path=output_path,
        bgm_path=bgm_path,
        crf=preset.crf,
        preset=preset.encoder_preset,
        audio_bitrate="192k",
    )

    # ---------------- 5) Verify ----------------
    duration = verify_output(output_path)
    logger.info(
        f"render done: {output_path.name} ({duration:.2f}s, quality={quality})"
    )

    # ---------------- 6) Cleanup temporary clips ----------------
    for path in clip_paths:
        path.unlink(missing_ok=True)
    concat_path.unlink(missing_ok=True)

    return output_path


def available_qualities() -> List[str]:
    """Ordered low→high, matching config.settings.quality_order()."""
    from config.settings import quality_order
    return quality_order()


def quality_label(quality: str) -> str:
    presets = quality_presets()
    if quality in presets:
        return presets[quality].label
    return presets[default_quality()].label
