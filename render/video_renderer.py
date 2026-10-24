from __future__ import annotations

import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.settings import QualityPreset, default_quality, quality_presets
from media.detector import classify
from media.formats import CATEGORY_IMAGE
from media.panel_cropper import (
    Box,
    PanelRegion,
    assign_panels,
    build_sequence_panel_plan,
    detect_panels_detailed,
    plan_shots,
    select_smart_effect,
)
from render.compositor import render_shot
from render.encoder import assemble, concat_clips, verify_output
from utils.error_handler import PipelineError, STAGE_RENDER
from utils.logger import get_logger

logger = get_logger("render.video_renderer")
ProgressFn = Optional[Callable[[str], None]]
_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

_IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".gif", ".webp", ".bmp",
    ".tif", ".tiff", ".avif", ".heic", ".heif", ".jxl", ".ico",
}


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
    filename = str(filename or "").strip()

    if not filename:
        raise PipelineError(STAGE_RENDER, "empty image filename")

    relative = Path(filename.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise PipelineError(STAGE_RENDER, f"unsafe image path: {filename!r}")

    exact = media_dir / relative
    if exact.is_file():
        return exact

    lower_name = relative.name.lower()
    for child in media_dir.rglob("*"):
        if child.is_file() and child.name.lower() == lower_name:
            return child

    prefix = _numeric_prefix(relative.name)
    if prefix is not None:
        candidates: List[Path] = []
        for child in media_dir.rglob("*"):
            if not child.is_file() or child.suffix.lower() not in _IMAGE_SUFFIXES:
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
    result = classify(path)
    if result.category != CATEGORY_IMAGE:
        raise PipelineError(
            STAGE_RENDER,
            f"{filename!r} resolved to {path.name!r}, but its category is {result.category!r}",
        )


def _scene_attr(scene: Any, key: str, default: Any = None) -> Any:
    if isinstance(scene, dict):
        return scene.get(key, default)
    return getattr(scene, key, default)


def _coerce_crop(crop_val: Any) -> Optional[Box]:
    if crop_val is None:
        return None
    if isinstance(crop_val, (list, tuple)) and len(crop_val) == 4:
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in crop_val)
            if x2 > x1 and y2 > y1:
                return (max(0, x1), max(0, y1), x2, y2)
        except (TypeError, ValueError):
            return None
    if isinstance(crop_val, dict):
        try:
            if all(k in crop_val for k in ("x1", "y1", "x2", "y2")):
                x1 = int(round(float(crop_val["x1"])))
                y1 = int(round(float(crop_val["y1"])))
                x2 = int(round(float(crop_val["x2"])))
                y2 = int(round(float(crop_val["y2"])))
            elif all(k in crop_val for k in ("x", "y", "width", "height")):
                x1 = int(round(float(crop_val["x"])))
                y1 = int(round(float(crop_val["y"])))
                x2 = x1 + int(round(float(crop_val["width"])))
                y2 = y1 + int(round(float(crop_val["height"])))
            else:
                return None
            if x2 > x1 and y2 > y1:
                return (max(0, x1), max(0, y1), x2, y2)
        except (TypeError, ValueError):
            pass
    return None


def _scene_to_dict(scene: Any) -> Dict[str, Any]:
    out = {
        "image_filename": str(_scene_attr(scene, "image_filename", "") or ""),
        "start": float(_scene_attr(scene, "start", 0.0)),
        "end": float(_scene_attr(scene, "end", 0.0)),
        "effect": str(_scene_attr(scene, "effect", "zoom_in") or "zoom_in"),
        "text": str(_scene_attr(scene, "text", "") or ""),
    }
    crop = _coerce_crop(_scene_attr(scene, "crop", None))
    if crop is not None:
        out["crop"] = crop
    transition = _scene_attr(scene, "transition", None)
    if transition:
        out["transition"] = str(transition).strip().lower()
    return out


def _assign_panel_regions(
    regions: List[PanelRegion],
    uses: int,
    use_index: int,
) -> List[PanelRegion]:
    n = len(regions)
    if n == 0:
        return []
    if uses <= 1:
        return list(regions)
    if n <= uses:
        return [regions[min(n - 1, (use_index * n) // max(1, uses))]]

    heights = np.array([max(1, r.height) for r in regions], dtype=np.float64)
    total_h = float(heights.sum())
    cum = np.cumsum(heights) / total_h
    centers = cum - (heights / total_h) * 0.5
    lo = use_index / float(uses)
    hi = (use_index + 1) / float(uses)
    group = [regions[i] for i in range(n) if lo <= centers[i] < hi]
    return group or [regions[min(n - 1, (use_index * n) // max(1, uses))]]


def _plan_region_shots(
    regions: List[PanelRegion],
    fallback_path: Path,
    total_frames: int,
    fps: int,
    min_shot: float = 1.45,
) -> List[Tuple[Path, Box, int, str]]:
    if not regions:
        return [(fallback_path, (0, 0, 720, 1280), max(1, total_frames), "zoom_in")]

    work: List[PanelRegion] = list(regions)
    while len(work) > 1 and (total_frames / float(max(1, fps)) / len(work)) < min_shot:
        merge_idx: Optional[int] = None
        best_span: Optional[int] = None
        for i in range(len(work) - 1):
            a, b = work[i], work[i + 1]
            if a.stitched_image_path or b.stitched_image_path or a.source_paths != b.source_paths:
                continue
            span_h = max(a.box[3], b.box[3]) - min(a.box[1], b.box[1])
            if best_span is None or span_h < best_span:
                best_span = span_h
                merge_idx = i

        if merge_idx is None:
            break

        a, b = work[merge_idx], work[merge_idx + 1]
        merged_box: Box = (
            min(a.box[0], b.box[0]), min(a.box[1], b.box[1]),
            max(a.box[2], b.box[2]), max(a.box[3], b.box[3]),
        )
        mw, mh = max(1, merged_box[2] - merged_box[0]), max(1, merged_box[3] - merged_box[1])
        work[merge_idx : merge_idx + 2] = [
            PanelRegion(
                source_files=a.source_files,
                source_paths=a.source_paths,
                box=merged_box,
                width=mw,
                height=mh,
                has_black_border=a.has_black_border or b.has_black_border,
                is_landscape_native=(mw / float(mh)) >= 1.28,
                is_short_strip=(mw / float(mh)) >= 1.85,
                is_small_panel=False,
                stitched_image_path=None,
                recommended_effect=select_smart_effect(merged_box, mw, mh, shot_index=merge_idx),
            )
        ]

    weights = np.array(
        [max(1.0, float(r.height * 0.7 + r.width * 0.3)) ** 0.8 for r in work],
        dtype=np.float64,
    )
    cum = np.cumsum(weights) / float(weights.sum())
    edges = [0] + [int(round(c * total_frames)) for c in cum]
    edges[-1] = max(len(work), int(total_frames))

    shots: List[Tuple[Path, Box, int, str]] = []
    for i, r in enumerate(work):
        frames = max(1, edges[i + 1] - edges[i])
        render_path = r.stitched_image_path if r.stitched_image_path is not None else fallback_path
        render_box: Box = (0, 0, r.width, r.height) if r.stitched_image_path is not None else r.box
        shots.append((render_path, render_box, frames, r.recommended_effect))

    diff = int(total_frames) - sum(f for _, _, f, _ in shots)
    if diff != 0 and shots:
        p, b, f, eff = shots[-1]
        shots[-1] = (p, b, max(1, f + diff), eff)

    return shots


def _apply_clip_transition(
    clip_path: Path,
    transition: str,
    duration: float,
    temp_dir: Path,
    crf: int,
    preset: str,
) -> Path:
    """Applies fade in/out inline using FFmpeg without altering absolute timeline bounds."""
    if not transition or transition == "cut":
        return clip_path

    t_dur = min(0.35, duration / 3.0)
    if t_dur < 0.1:
        return clip_path

    out_path = temp_dir / f"trans_{clip_path.name}"
    filters = []

    if transition in ("fade", "dissolve"):
        filters = [
            f"fade=t=in:st=0:d={t_dur}:color=black",
            f"fade=t=out:st={duration - t_dur}:d={t_dur}:color=black"
        ]
    elif transition == "flash":
        filters = [
            f"fade=t=in:st=0:d={t_dur}:color=white",
            f"fade=t=out:st={duration - t_dur}:d={t_dur}:color=white"
        ]
    elif transition == "slide_up":
        # Simplified representation for inline filters (push effects typically need xfade, 
        # using a fast fade-in/out to preserve strict sync without complex multi-input maps).
        filters = [
            f"fade=t=in:st=0:d={t_dur}:color=black",
            f"fade=t=out:st={duration - t_dur}:d={t_dur}:color=black"
        ]
    else:
        return clip_path

    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-i", str(clip_path),
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-an",
        str(out_path)
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and out_path.exists():
        return out_path
    return clip_path


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
    if not scenes:
        raise PipelineError(STAGE_RENDER, "render called with zero scenes")

    presets = quality_presets()
    if quality not in presets:
        raise PipelineError(STAGE_RENDER, f"unknown quality {quality!r}")

    preset: QualityPreset = presets[quality]
    narration_path, output_path, media_dir, temp_dir = map(Path, (narration_path, output_path, media_dir, temp_dir))

    if not narration_path.is_file():
        raise PipelineError(STAGE_RENDER, f"narration audio not found: {narration_path}")

    if bgm_path is not None:
        bgm_path = Path(bgm_path)
        if not bgm_path.is_file():
            logger.warning("Background music requested but not found: %s", bgm_path)
            bgm_path = None

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene_list = [_scene_to_dict(scene) for scene in scenes]
    total = len(scene_list)

    logger.info("Render starting: %s scenes, quality=%s", total, quality)
    if progress:
        progress(f"Preparing {total} scenes")

    prepared: List[Dict[str, Any]] = []
    previous_end_frame = 0

    for idx, scene in enumerate(scene_list):
        filename = scene["image_filename"].strip()
        if not filename:
            raise PipelineError(STAGE_RENDER, f"scene {idx + 1} has an empty image_filename")

        start = max(float(scene["start"]), 0.0)
        end = float(scene["end"])
        if end <= start:
            raise PipelineError(STAGE_RENDER, f"scene {idx + 1} has invalid timing: {start} -> {end}")

        start_frame = max(round(start * preset.fps), previous_end_frame)
        end_frame = round(end * preset.fps)
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        previous_end_frame = end_frame
        total_frames = end_frame - start_frame

        prepared.append(
            {
                "index": idx,
                "filename": filename,
                "effect": scene["effect"],
                "crop": scene.get("crop"),
                "transition": scene.get("transition"),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "total_frames": total_frames,
                "duration": total_frames / preset.fps,
            }
        )

    resolved: List[Path] = []
    checked: set[Path] = set()
    for scene in prepared:
        path = _resolve_image(media_dir, scene["filename"])
        if path not in checked:
            _validate_image(path, scene["filename"])
            checked.add(path)
        resolved.append(path)

    has_any_auto_crop = any(s.get("crop") is None for s in prepared)
    panel_plan: Dict[Path, List[PanelRegion]] = {}
    if has_any_auto_crop:
        panel_plan = build_sequence_panel_plan(resolved, temp_dir)

    auto_uses = Counter(path for path, sc in zip(resolved, prepared) if sc.get("crop") is None)
    auto_used: Dict[Path, int] = defaultdict(int)
    
    clip_paths: List[Path] = []
    concat_path = temp_dir / "concat.mp4"
    global_shot_counter = 0
    last_effect: Optional[str] = None

    try:
        for idx, scene in enumerate(prepared):
            if progress:
                progress(f"Rendering clip {idx + 1}/{total}")

            image_path = resolved[idx]
            explicit_crop: Optional[Box] = scene.get("crop")

            if explicit_crop is not None:
                shots: List[Tuple[Path, Box, int, str]] = [
                    (image_path, explicit_crop, scene["total_frames"], scene["effect"])
                ]
            else:
                regions = panel_plan.get(image_path) or []
                group = _assign_panel_regions(regions, auto_uses[image_path], auto_used[image_path])
                auto_used[image_path] += 1
                shots = _plan_region_shots(group, image_path, scene["total_frames"], preset.fps)

            # Sub-clip generation
            sub_clips: List[Path] = []
            for sub, (shot_img_path, box, frames, rec_effect) in enumerate(shots):
                raw_clip_path = temp_dir / f"raw_clip_{idx:04d}_{sub:02d}.mp4"

                shot_w = max(1, box[2] - box[0])
                shot_h = max(1, box[3] - box[1])

                if explicit_crop is not None and scene.get("effect"):
                    effect_to_use = scene["effect"]
                else:
                    smart_eff = select_smart_effect(box, shot_w, shot_h, shot_index=global_shot_counter, prev_effect=last_effect)
                    effect_to_use = smart_eff or rec_effect or scene["effect"]

                last_effect = effect_to_use
                global_shot_counter += 1

                render_shot(
                    shot_img_path, box, effect_to_use, raw_clip_path,
                    width=preset.width, height=preset.height,
                    fps=preset.fps, total_frames=frames,
                    crf=preset.crf, preset=preset.encoder_preset,
                    shot_index=global_shot_counter,
                )

                if not raw_clip_path.is_file() or raw_clip_path.stat().st_size < 512:
                    raise PipelineError(STAGE_RENDER, f"clip {idx + 1}.{sub} produced no valid output")
                    
                sub_clips.append(raw_clip_path)

            # Merge sub-shots to a single scene clip if multiple panels were detected inside one scene timing
            scene_clip_path = temp_dir / f"scene_{idx:04d}.mp4"
            if len(sub_clips) > 1:
                concat_clips(sub_clips, scene_clip_path)
                for sc in sub_clips:
                    sc.unlink(missing_ok=True)
            else:
                shutil.move(str(sub_clips[0]), str(scene_clip_path))

            # Apply Sync-Safe Transitions (Fade/Flash on top of clip without altering duration)
            if scene.get("transition"):
                final_clip_path = _apply_clip_transition(
                    scene_clip_path, 
                    scene["transition"], 
                    scene["duration"], 
                    temp_dir,
                    preset.crf,
                    preset.encoder_preset
                )
            else:
                final_clip_path = scene_clip_path
                
            clip_paths.append(final_clip_path)

            logger.info(
                "Clip %s/%s: %s | %.3fs | %s frames | effect=%s | trans=%s",
                idx + 1, total, scene["filename"], scene["duration"],
                scene["total_frames"], last_effect, scene.get("transition", "cut")
            )

        if progress:
            progress("Concatenating clips")
        concat_clips(clip_paths, concat_path)

        if not concat_path.is_file() or concat_path.stat().st_size < 512:
            raise PipelineError(STAGE_RENDER, "FFmpeg concatenation did not produce a valid video")

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

        if not output_path.is_file() or output_path.stat().st_size < 512:
            raise PipelineError(STAGE_RENDER, "final render did not produce a valid output file")

        duration = verify_output(output_path)
        logger.info("Render complete: %s | %.2fs | quality=%s", output_path.name, duration, quality)

        if progress:
            progress("Render completed successfully")

        return output_path

    finally:
        for path in clip_paths:
            path.unlink(missing_ok=True)
        for path in temp_dir.glob("raw_clip_*.mp4"):
            path.unlink(missing_ok=True)
        for path in temp_dir.glob("scene_*.mp4"):
            path.unlink(missing_ok=True)
        concat_path.unlink(missing_ok=True)


def available_qualities() -> List[str]:
    from config.settings import quality_order
    return quality_order()


def quality_label(quality: str) -> str:
    presets = quality_presets()
    if quality in presets:
        return presets[quality].label
    return presets[default_quality()].label
