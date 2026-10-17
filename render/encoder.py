"""
render/encoder.py
FFmpeg command builders for the four rendering stages:
  - image standardization (blurred background + fit-contain)
  - single scene clip (zoompan)
  - clip concat
  - final assembly (video + narration [+ optional BGM])

Every subprocess.run() here checks returncode and raises
PipelineError(STAGE_RENDER, ...) with the tail of stderr on failure.
No SFX, no subtitles, no silent failures.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from render.effects import build_zoompan_filter
from utils.error_handler import PipelineError, STAGE_RENDER
from utils.logger import get_logger


logger = get_logger("render.encoder")

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE = shutil.which("ffprobe") or "ffprobe"
_STDERR_TAIL = 800


def _tail(text: Optional[str]) -> str:
    if not text:
        return ""
    return text[-_STDERR_TAIL:].strip()


def _run(cmd: List[str], what: str) -> None:
    logger.debug(f"ffmpeg[{what}]: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise PipelineError(
            STAGE_RENDER, f"ffmpeg binary not found ({_FFMPEG})", cause=exc
        ) from exc
    except OSError as exc:
        raise PipelineError(
            STAGE_RENDER, f"failed to invoke ffmpeg: {exc}", cause=exc
        ) from exc
    if proc.returncode != 0:
        raise PipelineError(
            STAGE_RENDER,
            f"{what} failed (exit {proc.returncode})",
            cause=Exception(_tail(proc.stderr)),
        )


# ---------------------------------------------------------------------------
# 1) Image standardization: blurred cover + fit-contain overlay
# ---------------------------------------------------------------------------

def standardize_image(
    image_path: Path,
    output_path: Path,
    width: int,
    height: int,
) -> None:
    """Force any aspect ratio into exact width×height using a blurred
    cover background plus a fit-contain foreground. No stretching, no
    cropping of subject matter."""
    filter_complex = (
        "[0:v]split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase:"
        f"flags=lanczos,crop={width}:{height},boxblur=10:5[bgblur];"
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease:"
        f"flags=lanczos,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2[fgpad];"
        "[bgblur][fgpad]overlay=(W-w)/2:(H-h)/2:format=auto[out]"
    )
    cmd = [
        _FFMPEG, "-y", "-i", str(image_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-frames:v", "1",
        str(output_path),
    ]
    _run(cmd, f"standardize({Path(image_path).name})")


# ---------------------------------------------------------------------------
# 2) Single scene clip
# ---------------------------------------------------------------------------

def create_clip(
    standardized_image: Path,
    output_path: Path,
    effect: str,
    width: int,
    height: int,
    fps: int,
    total_frames: int,
    crf: int,
    preset: str,
) -> None:
    """Render exactly total_frames frames from a still using zoompan."""
    total_frames = max(int(total_frames), 1)
    filter_str = build_zoompan_filter(effect, width, height, fps, total_frames)
    cmd = [
        _FFMPEG, "-y",
        "-loop", "1", "-i", str(standardized_image),
        "-filter_complex", f"[0:v]{filter_str}[v]",
        "-map", "[v]",
        "-frames:v", str(total_frames),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
    ]
    _run(cmd, f"clip({Path(standardized_image).name})")


# ---------------------------------------------------------------------------
# 3) Clip concat
# ---------------------------------------------------------------------------

def concat_clips(clip_paths: List[Path], output_path: Path) -> None:
    if not clip_paths:
        raise PipelineError(STAGE_RENDER, "no clips to concatenate")

    concat_txt = Path(output_path).with_suffix(".concat.txt")
    try:
        with open(concat_txt, "w", encoding="utf-8") as f:
            for clip in clip_paths:
                resolved = Path(clip).resolve().as_posix().replace("'", r"'\''")
                f.write(f"file '{resolved}'\n")
        cmd = [
            _FFMPEG, "-y",
            "-fflags", "+genpts",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_txt),
            "-an",
            "-c:v", "copy",
            str(output_path),
        ]
        _run(cmd, "concat")
    finally:
        concat_txt.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 4) Final assembly: video + narration (+ optional BGM)
# ---------------------------------------------------------------------------

def assemble(
    video_path: Path,
    narration_path: Path,
    output_path: Path,
    *,
    bgm_path: Optional[Path] = None,
    bgm_volume: float = 0.08,
    crf: int = 21,
    preset: str = "medium",
    audio_bitrate: str = "192k",
) -> None:
    """Mux narration (+ optional looping BGM) onto the concat'd video.

    BGM is applied with `-stream_loop -1` and mixed at `bgm_volume`,
    using `amix` with `normalize=0` so narration loudness does not drop
    when BGM is added. Video is re-encoded to preserve timeline exactness
    (concat uses copy, so the video stream is already H.264 — re-encoding
    here just applies the target CRF for the final file).
    """
    cmd = [
        _FFMPEG, "-y",
        "-i", str(video_path),
        "-i", str(narration_path),
    ]
    filter_parts = ["[0:v]setpts=PTS-STARTPTS[v_out]"]

    if bgm_path is not None and Path(bgm_path).is_file():
        cmd.extend(["-stream_loop", "-1", "-i", str(bgm_path)])
        filter_parts.append(f"[2:a]volume={bgm_volume}[bgm]")
        filter_parts.append(
            "[1:a][bgm]amix=inputs=2:duration=first:"
            "dropout_transition=2:normalize=0[a_mix]"
        )
        audio_map = "[a_mix]"
    else:
        audio_map = "1:a"

    cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", "[v_out]",
        "-map", audio_map,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-shortest",
        str(output_path),
    ])
    _run(cmd, "final assembly")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_output(path: Path) -> float:
    """Return duration (seconds) of the produced file via ffprobe.
    Raises PipelineError if the file is missing, empty, or unreadable."""
    path = Path(path)
    if not path.is_file():
        raise PipelineError(STAGE_RENDER, f"output not written: {path}")
    if path.stat().st_size < 1024:
        raise PipelineError(
            STAGE_RENDER,
            f"output is suspiciously small ({path.stat().st_size} bytes): {path}",
        )
    try:
        proc = subprocess.run(
            [_FFPROBE, "-v", "quiet", "-print_format", "json",
             "-show_format", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise PipelineError(
            STAGE_RENDER, f"ffprobe failed on {path.name}: {exc}", cause=exc
        ) from exc
    if proc.returncode != 0:
        raise PipelineError(
            STAGE_RENDER,
            f"ffprobe rejected output {path.name}",
            cause=Exception(_tail(proc.stderr)),
        )
    import json as _json
    try:
        data = _json.loads(proc.stdout or "{}")
        duration = float(data.get("format", {}).get("duration", 0) or 0)
    except (ValueError, _json.JSONDecodeError):
        duration = 0.0
    if duration <= 0:
        raise PipelineError(
            STAGE_RENDER, f"output has zero duration: {path.name}"
        )
    return duration
