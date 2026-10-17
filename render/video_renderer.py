
"""
render/video_renderer.py

Orchestrates the full render:
standardize → per-scene clip → concat → final assembly → verify.

Consumes scene objects produced by sync/timeline.py and the quality
preset chosen by the user.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from config.settings import (
    quality_presets,
    default_quality,
    QualityPreset,
)
from media.detector import classify
from media.formats import CATEGORY_IMAGE
from collections import Counter, defaultdict

from media.panel_cropper import assign_panels, detect_panels, plan_shots
from render.compositor import render_shot
from render.encoder import (
    concat_clips,
    assemble,
    verify_output,
)
from utils.error_handler import PipelineError, STAGE_RENDER
from utils.logger import get_logger


logger = get_logger("render.video_renderer")
ProgressFn = Optional[Callable[[str], None]]


_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
    ".heic",
    ".heif",
    ".jxl",
    ".ico",
}


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

def _numeric_prefix(name: str) -> Optional[int]:
    """Return the leading numeric ID from a filename, if present."""
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
    """
    Resolve an image reference.

    Resolution order:
    1. Exact path.
    2. Case-insensitive filename match.
    3. Matching leading numeric ID.

    Missing images raise an error instead of silently selecting
    an unrelated image.
    """
    media_dir = Path(media_dir)
    filename = str(filename or "").strip()

    if not filename:
        raise PipelineError(
            STAGE_RENDER,
            "empty image filename",
        )

    # Prevent a plan reference from escaping the media directory.
    relative = Path(filename)

    if relative.is_absolute() or ".." in relative.parts:
        raise PipelineError(
            STAGE_RENDER,
            f"unsafe image path: {filename!r}",
        )

    exact = media_dir / relative

    if exact.is_file():
        return exact

    lower = relative.name.lower()

    for child in media_dir.rglob("*"):
        if child.is_file() and child.name.lower() == lower:
            return child

    prefix = _numeric_prefix(relative.name)

    if prefix is not None:
        candidates: List[Path] = []

        for child in media_dir.rglob("*"):
            if not child.is_file():
                continue

            if child.suffix.lower() not in _IMAGE_SUFFIXES:
                continue

            if _numeric_prefix(child.name) == prefix:
                candidates.append(child)

        if candidates:
            candidates.sort(key=lambda path: str(path).lower())
            return candidates[0]

    raise PipelineError(
        STAGE_RENDER,
        f"image {filename!r} not found under {media_dir}",
        hint="Check the image filename and plan validation results.",
    )


def _validate_image(path: Path, filename: str) -> None:
    """Ensure that the resolved media file is an image."""
    result = classify(path)

    if result.category != CATEGORY_IMAGE:
        raise PipelineError(
            STAGE_RENDER,
            (
                f"{filename!r} resolved to {path.name!r}, "
                f"but its category is {result.category!r}, not image"
            ),
        )


# ---------------------------------------------------------------------------
# Scene conversion
# ---------------------------------------------------------------------------

def _scene_attr(scene, key: str, default=None):
    """Read a field from either a dictionary or a scene object."""
    if isinstance(scene, dict):
        return scene.get(key, default)

    return getattr(scene, key, default)


def _scene_to_dict(scene) -> Dict:
    """Normalize a scene object into the renderer's internal format."""
    return {
        "image_filename": str(
            _scene_attr(scene, "image_filename", "") or ""
        ),
        "start": float(_scene_attr(scene, "start", 0.0)),
        "end": float(_scene_attr(scene, "end", 0.0)),
        "effect": str(
            _scene_attr(scene, "effect", "zoom_in") or "zoom_in"
        ),
        "text": str(_scene_attr(scene, "text", "") or ""),
    }


# ---------------------------------------------------------------------------
# Main renderer
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
    Render scenes into the final video.

    Args:
        scenes:
            RenderScene objects or dictionaries from sync/timeline.py.
        narration_path:
            Original narration audio.
        output_path:
            Destination MP4 path.
        media_dir:
            Directory containing source images.
        temp_dir:
            Temporary workspace for standardized images and clips.
        quality:
            Key returned by config.settings.quality_presets().
        bgm_path:
            Optional background music.
        progress:
            Optional callback receiving progress messages.

    Returns:
        Path to the verified output video.

    Raises:
        PipelineError: If validation or rendering fails.
    """
    if not scenes:
        raise PipelineError(
            STAGE_RENDER,
            "render called with zero scenes",
        )

    presets = quality_presets()

    if quality not in presets:
        raise PipelineError(
            STAGE_RENDER,
            f"unknown quality {quality!r}; valid: {sorted(presets)}",
        )

    preset: QualityPreset = presets[quality]

    narration_path = Path(narration_path)
    output_path = Path(output_path)
    media_dir = Path(media_dir)
    temp_dir = Path(temp_dir)

    if not narration_path.is_file():
        raise PipelineError(
            STAGE_RENDER,
            f"narration audio not found: {narration_path}",
        )

    if bgm_path is not None:
        bgm_path = Path(bgm_path)

        if not bgm_path.is_file():
            logger.warning(
                "Background music was requested but not found: %s",
                bgm_path,
            )
            bgm_path = None

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # 1. Prepare scene timing
    # -------------------------------------------------------

    scene_list = [_scene_to_dict(scene) for scene in scenes]
    total = len(scene_list)

    logger.info(
        "Render starting: %s scenes, quality=%s",
        total,
        quality,
    )

    if progress:
        progress(f"Preparing {total} scenes")

    prepared: List[Dict] = []
    previous_end_frame = 0

    for idx, scene in enumerate(scene_list):
        filename = scene["image_filename"].strip()

        if not filename:
            raise PipelineError(
                STAGE_RENDER,
                f"scene {idx + 1} has an empty image_filename",
            )

        start = scene["start"]
        end = scene["end"]

        if start < 0:
            logger.warning(
                "Scene %s has a negative start time; clamping to zero",
                idx + 1,
            )

        start = max(start, 0.0)

        if end <= start:
            raise PipelineError(
                STAGE_RENDER,
                (
                    f"scene {idx + 1} has invalid timing: "
                    f"start={start}, end={end}"
                ),
            )

        # Quantize scene boundaries to the output FPS grid.
        start_frame = round(start * preset.fps)
        end_frame = round(end * preset.fps)

        # Prevent frame overlap caused by rounding.
        start_frame = max(start_frame, previous_end_frame)

        if end_frame <= start_frame:
            end_frame = start_frame + 1

        previous_end_frame = end_frame

        total_frames = end_frame - start_frame

        prepared.append(
            {
                "index": idx,
                "filename": filename,
                "effect": scene["effect"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "total_frames": total_frames,
                "duration": total_frames / preset.fps,
            }
        )

    if previous_end_frame < 1:
        raise PipelineError(
            STAGE_RENDER,
            "all scenes collapsed to zero frames",
        )

    # -------------------------------------------------------
    # 2. Render each scene
    # -------------------------------------------------------

    clip_paths: List[Path] = []
    concat_path = temp_dir / "concat.mp4"

    resolved: List[Path] = []
    checked = set()

    for scene in prepared:
        path = _resolve_image(media_dir, scene["filename"])

        if path not in checked:
            _validate_image(path, scene["filename"])
            checked.add(path)

        resolved.append(path)

    uses = Counter(resolved)
    used: Dict[Path, int] = defaultdict(int)
    panel_cache: Dict[Path, List] = {}

    try:
        for idx, scene in enumerate(prepared):
            if progress:
                progress(f"Rendering clip {idx + 1}/{total}")

            image_path = resolved[idx]

            if image_path not in panel_cache:
                panel_cache[image_path] = detect_panels(image_path)

            group = assign_panels(
                panel_cache[image_path],
                uses[image_path],
                used[image_path],
            )
            used[image_path] += 1

            shots = plan_shots(group, scene["total_frames"], preset.fps)

            for sub, (box, frames) in enumerate(shots):
                clip_path = temp_dir / f"clip_{idx:04d}_{sub:02d}.mp4"
                clip_paths.append(clip_path)

                render_shot(
                    image_path,
                    box,
                    scene["effect"],
                    clip_path,
                    width=preset.width,
                    height=preset.height,
                    fps=preset.fps,
                    total_frames=frames,
                    crf=preset.crf,
                    preset=preset.encoder_preset,
                )

                if (
                    not clip_path.is_file()
                    or clip_path.stat().st_size < 512
                ):
                    raise PipelineError(
                        STAGE_RENDER,
                        (
                            f"clip {idx + 1}.{sub} for "
                            f"{scene['filename']!r} produced no valid output"
                        ),
                    )

            logger.info(
                "Clip %s/%s: %s | %.3fs | %s frames | %s shot(s) | effect=%s",
                idx + 1,
                total,
                scene["filename"],
                scene["duration"],
                scene["total_frames"],
                len(shots),
                scene["effect"],
            )

        # ---------------------------------------------------
        # 3. Concatenate scene clips
        # ---------------------------------------------------

        if progress:
            progress("Concatenating clips")

        concat_clips(
            clip_paths,
            concat_path,
        )

        if not concat_path.is_file() or concat_path.stat().st_size < 512:
            raise PipelineError(
                STAGE_RENDER,
                "FFmpeg concatenation did not produce a valid video",
            )

        # ---------------------------------------------------
        # 4. Assemble narration and optional background music
        # ---------------------------------------------------

        if progress:
            progress("Assembling audio and video")

        assemble(
            video_path=concat_path,
            narration_path=narration_path,
            output_path=output_path,
            bgm_path=bgm_path,
            crf=preset.crf,
            preset=preset.encoder_preset,
            audio_bitrate="192k",
        )

        # ---------------------------------------------------
        # 5. Verify final output
        # ---------------------------------------------------

        if not output_path.is_file() or output_path.stat().st_size < 512:
            raise PipelineError(
                STAGE_RENDER,
                "final render did not produce a valid output file",
            )

        duration = verify_output(output_path)

        logger.info(
            "Render complete: %s | %.2fs | quality=%s",
            output_path.name,
            duration,
            quality,
        )

        if progress:
            progress("Render completed successfully")

        return output_path

    finally:
        # Clean up intermediate files even when a later stage fails.
        for path in clip_paths:
            path.unlink(missing_ok=True)

        concat_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------

def available_qualities() -> List[str]:
    """Return quality options in configured low-to-high order."""
    from config.settings import quality_order

    return quality_order()


def quality_label(quality: str) -> str:
    """Return a human-readable label for a quality preset."""
    presets = quality_presets()

    if quality in presets:
        return presets[quality].label

    return presets[default_quality()].label
