from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from config.settings import render_config
from utils.error_handler import PipelineError, STAGE_RENDER

Box = Tuple[int, int, int, int]

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_cfg = render_config()
SS = max(1, min(int(_cfg.supersample), 2))
ZOOM = min(float(_cfg.zoom_amount), 0.12)
PAN_ZOOM = min(float(_cfg.pan_zoom_level), 1.10)
MAX_UPSCALE = 1.7
BG_BRIGHTNESS = 0.55


def _ease(t: float) -> float:
    s = t * t * (3.0 - 2.0 * t)
    return 0.35 * t + 0.65 * s


def _motion(effect: str, e: float) -> Tuple[float, float, float]:
    zi = 1.0 + ZOOM * e
    zo = 1.0 + ZOOM * (1.0 - e)
    table = {
        "zoom_in": (zi, 0.5, 0.5),
        "zoom_out": (zo, 0.5, 0.5),
        "pan_left": (PAN_ZOOM, e, 0.5),
        "pan_right": (PAN_ZOOM, 1 - e, 0.5),
        "pan_up": (PAN_ZOOM, 0.5, 1 - e),
        "pan_down": (PAN_ZOOM, 0.5, e),
        "zoomin_pan_left": (zi, e, 0.5),
        "zoomin_pan_right": (zi, 1 - e, 0.5),
        "zoomout_pan_left": (zo, e, 0.5),
        "zoomout_pan_right": (zo, 1 - e, 0.5),
        "diagonal_zoom_in": (zi, e, e),
        "diagonal_zoom_out": (zo, 1 - e, 1 - e),
    }
    return table.get(effect, table["zoom_in"])


def _canvas(image_path: Path, box: Box, width: int, height: int) -> Image.Image:
    with Image.open(image_path) as im:
        try:
            im.seek(0)
        except EOFError:
            pass
        src = im.convert("RGB")
        full_w = src.width
        crop = src.crop(box)
    pw, ph = crop.size
    cw, chh = width * SS, height * SS
    scale = min(cw / pw, MAX_UPSCALE * cw / full_w)
    fw = max(int(round(pw * scale)), 1)
    fh = max(int(round(ph * scale)), 1)
    fg = crop.resize((fw, fh), Image.LANCZOS)
    total_h = max(fh, chh)
    if fw >= cw and fh >= total_h:
        return fg
    small = ImageOps.fit(
        fg.resize((max(fw // 8, 8), max(fh // 8, 8)), Image.BILINEAR),
        (max(cw // 16, 8), max(total_h // 16, 8)),
        Image.BILINEAR,
    ).filter(ImageFilter.GaussianBlur(3))
    bg = small.resize((cw, total_h), Image.BICUBIC)
    bg = ImageEnhance.Brightness(bg).enhance(BG_BRIGHTNESS)
    bg.paste(fg, ((cw - fw) // 2, (total_h - fh) // 2))
    return bg


def render_shot(
    image_path: Path,
    box: Box,
    effect: str,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    total_frames: int,
    crf: int,
    preset: str,
) -> None:
    canvas = _canvas(image_path, box, width, height)
    cw, chh = canvas.size
    w_ss, h_ss = width * SS, height * SS
    tall = chh - h_ss > 6 * SS

    err = tempfile.TemporaryFile()
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-an", str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
    n = max(int(total_frames), 1)
    try:
        for i in range(n):
            e = _ease(i / (n - 1)) if n > 1 else 0.0
            if tall:
                zoom, fx, fy = 1.0 + 0.04 * e, 0.5, e
            else:
                zoom, fx, fy = _motion(effect, e)
            ww, wh = w_ss / zoom, h_ss / zoom
            x0 = (cw - ww) * fx
            y0 = (chh - wh) * fy
            frame = canvas.resize(
                (width, height),
                Image.BICUBIC,
                box=(x0, y0, x0 + ww, y0 + wh),
            )
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        code = proc.wait()
    except (BrokenPipeError, OSError):
        code = proc.wait()
    if code != 0:
        err.seek(0)
        raise PipelineError(
            STAGE_RENDER,
            f"shot render failed (exit {code})",
            cause=Exception(err.read().decode(errors="ignore")[-800:]),
        )
    err.close()
