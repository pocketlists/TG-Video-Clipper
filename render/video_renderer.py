from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.settings import (
    QualityPreset,
    default_quality,
    quality_presets,
)
from media.detector import classify
from media.formats import CATEGORY_IMAGE
from media.panel_cropper import (
    Box,
    PanelRegion,
    assign_panels,
    build_sequence_panel_plan,
    detect_panels,
    detect_panels_detailed,
    plan_shots,
    select_smart_effect,
)
from render.compositor import render_shot
from render.encoder import (
    assemble,
    concat_clips,
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

    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise PipelineError(STAGE_RENDER, f"unsafe image path: {filename!r}")

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
    result = classify(path)
    if result.category != CATEGORY_IMAGE:
        raise PipelineError(
            STAGE_RENDER,
            f"{filename!r} resolved to {path.name!r}, "
            f"but its category is {result.category!r}, not image",
        )


def _scene_attr(scene, key: str, default=None):
    if isinstance(scene, dict):
        return scene.get(key, default)
    return getattr(scene, key, default)


def _coerce_crop(crop_val) -> Optional[Box]:
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
            return None
    return None


def _scene_to_dict(scene) -> Dict:
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
        out["transition"] = str(transition)
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
    while (
        len(work) > 1
        and (total_frames / float(max(1, fps)) / len(work)) < min_shot
    ):
        merge_idx: Optional[int] = None
        best_span: Optional[int] = None
        for i in range(len(work) - 1):
            a, b = work[i], work[i + 1]
            if a.stitched_image_path or b.stitched_image_path:
                continue
            if a.source_paths != b.source_paths:
                continue
            span_h = max(a.box[3], b.box[3]) - min(a.box[1], b.box[1])
            if best_span is None or span_h < best_span:
                best_span = span_h
                merge_idx = i

        if merge_idx is None:
            break

        a, b = work[merge_idx], work[merge_idx + 1]
        merged_box: Box = (
            min(a.box[0], b.box[0]),
            min(a.box[1], b.box[1]),
            max(a.box[2], b.box[2]),
            max(a.box[3], b.box[3]),
        )
        mw = max(1, merged_box[2] - merged_box[0])
        mh = max(1, merged_box[3] - merged_box[1])
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
        raise PipelineError(STAGE_RENDER, f"narration audio not found: {narration_path}")

    if bgm_path is not None:
        bgm_path = Path(bgm_path)
        if not bgm_path.is_file():
            logger.warning("Background music was requested but not found: %s", bgm_path)
            bgm_path = None

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene_list = [_scene_to_dict(scene) for scene in scenes]
    total = len(scene_list)

    logger.info("Render starting: %s scenes, quality=%s", total, quality)
    if progress:
        progress(f"Preparing {total} scenes")

    prepared: List[Dict] = []
    previous_end_frame = 0

    for idx, scene in enumerate(scene_list):
        filename = scene["image_filename"].strip()
        if not filename:
            raise PipelineError(STAGE_RENDER, f"scene {idx + 1} has an empty image_filename")

        start = max(float(scene["start"]), 0.0)
        end = float(scene["end"])
        if end <= start:
            raise PipelineError(
                STAGE_RENDER,
                f"scene {idx + 1} has invalid timing: start={start}, end={end}",
            )

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

    if previous_end_frame < 1:
        raise PipelineError(STAGE_RENDER, "all scenes collapsed to zero frames")

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
        raw_panel_counts: Dict[Path, int] = {
            p: len(detect_panels_detailed(p).panels) for p in set(resolved)
        }

        merged_prepared: List[Dict] = []
        merged_resolved: List[Path] = []
        skip_next = False

        for i in range(len(prepared)):
            if skip_next:
                skip_next = False
                continue

            curr_scene = dict(prepared[i])
            curr_path = resolved[i]

            if (
                i + 1 < len(prepared)
                and curr_scene.get("crop") is None
                and prepared[i + 1].get("crop") is None
            ):
                next_path = resolved[i + 1]
                curr_regions = panel_plan.get(curr_path, [])
                stitched_next = any(
                    r.stitched_image_path is not None and next_path in r.source_paths
                    for r in curr_regions
                )
                if stitched_next and raw_panel_counts.get(next_path, 0) <= 1 and curr_path != next_path:
                    next_scene = prepared[i + 1]
                    curr_scene["end_frame"] = next_scene["end_frame"]
                    curr_scene["total_frames"] += next_scene["total_frames"]
                    curr_scene["duration"] = curr_scene["total_frames"] / preset.fps
                    skip_next = True

            merged_prepared.append(curr_scene)
            merged_resolved.append(curr_path)

        prepared = merged_prepared
        resolved = merged_resolved
        total = len(prepared)

    auto_uses = Counter(
        path for path, sc in zip(resolved, prepared) if sc.get("crop") is None
    )
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
                    (
                        image_path,
                        explicit_crop,
                        scene["total_frames"],
                        scene["effect"],
                    )
                ]
            else:
                regions = panel_plan.get(image_path) or []
                group = _assign_panel_regions(
                    regions,
                    auto_uses[image_path],
                    auto_used[image_path],
                )
                auto_used[image_path] += 1
                shots = _plan_region_shots(
                    group,
                    image_path,
                    scene["total_frames"],
                    preset.fps,
                )

            for sub, (shot_img_path, box, frames, rec_effect) in enumerate(shots):
                clip_path = temp_dir / f"clip_{idx:04d}_{sub:02d}.mp4"
                clip_paths.append(clip_path)

                shot_w = max(1, box[2] - box[0])
                shot_h = max(1, box[3] - box[1])

                if explicit_crop is not None and scene.get("effect"):
                    effect_to_use = scene["effect"]
                else:
                    smart_eff = select_smart_effect(
                        box,
                        shot_w,
                        shot_h,
                        shot_index=global_shot_counter,
                        prev_effect=last_effect,
                    )
                    effect_to_use = smart_eff or rec_effect or scene["effect"]

                last_effect = effect_to_use
                global_shot_counter += 1

                render_shot(
                    shot_img_path,
                    box,
                    effect_to_use,
                    clip_path,
                    width=preset.width,
                    height=preset.height,
                    fps=preset.fps,
                    total_frames=frames,
                    crf=preset.crf,
                    preset=preset.encoder_preset,
                    shot_index=global_shot_counter,
                )

                if not clip_path.is_file() or clip_path.stat().st_size < 512:
                    raise PipelineError(
                        STAGE_RENDER,
                        f"clip {idx + 1}.{sub} for {scene['filename']!r} produced no valid output",
                    )

            logger.info(
                "Clip %s/%s: %s | %.3fs | %s frames | %s shot(s) | effect=%s",
                idx + 1,
                total,
                scene["filename"],
                scene["duration"],
                scene["total_frames"],
                len(shots),
                last_effect,
            )

        if progress:
            progress("Concatenating clips")

        concat_clips(clip_paths, concat_path)

        if not concat_path.is_file() or concat_path.stat().st_size < 512:
            raise PipelineError(
                STAGE_RENDER,
                "FFmpeg concatenation did not produce a valid video",
            )

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
        for path in clip_paths:
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
p
