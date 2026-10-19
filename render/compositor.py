from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from config.settings import render_config
from media.panel_cropper import select_smart_effect
from utils.error_handler import PipelineError, STAGE_RENDER

Box = Tuple[int, int, int, int]

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_cfg = render_config()
SS = max(1, min(int(_cfg.supersample), 2))
ZOOM = min(float(_cfg.zoom_amount), 0.12)
PAN_ZOOM = min(float(_cfg.pan_zoom_level), 1.12)
MAX_UPSCALE_REGULAR = 1.85
MAX_UPSCALE_SMALL = 2.65
BG_BRIGHTNESS = 0.48
WHITE_TRIM_T = 244.0


def _ease(t: float) -> float:
    s = t * t * (3.0 - 2.0 * t)
    return 0.30 * t + 0.70 * s


def _trim_excess_white(crop: Image.Image) -> Image.Image:
    w, h = crop.size
    if w < 32 or h < 32:
        return crop

    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    rgb = np.asarray(crop, dtype=np.float32)

    row_mean = gray.mean(axis=1)
    row_std = gray.std(axis=1)
    row_cstd = rgb.std(axis=(1, 2))
    active_rows = (row_mean < WHITE_TRIM_T) | (row_std > 5.5) | (row_cstd > 6.0)

    col_mean = gray.mean(axis=0)
    col_std = gray.std(axis=0)
    col_cstd = rgb.std(axis=(0, 2))
    active_cols = (col_mean < WHITE_TRIM_T) | (col_std > 5.5) | (col_cstd > 6.0)

    ys = np.flatnonzero(active_rows)
    xs = np.flatnonzero(active_cols)
    if len(ys) == 0 or len(xs) == 0:
        return crop

    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    cw, ch = x1 - x0, y1 - y0

    area_ratio = (cw * ch) / float(max(1, w * h))
    # Large images keep tight margin; smaller panels keep a thin breathing border
    pad_x = max(2, int(round(cw * (0.005 if area_ratio > 0.65 else 0.012))))
    pad_y = max(2, int(round(ch * (0.005 if area_ratio > 0.65 else 0.012))))

    tx0 = max(0, x0 - pad_x)
    ty0 = max(0, y0 - pad_y)
    tx1 = min(w, x1 + pad_x)
    ty1 = min(h, y1 + pad_y)

    if (tx1 - tx0) < 24 or (ty1 - ty0) < 24:
        return crop
    return crop.crop((tx0, ty0, tx1, ty1))


def _build_ambient_bg(source: Image.Image, cw: int, ch: int) -> Image.Image:
    small_w = max(cw // 12, 16)
    small_h = max(ch // 12, 16)
    thumb = ImageOps.fit(
        source.resize((max(source.width // 4, 32), max(source.height // 4, 32)), Image.BILINEAR),
        (small_w, small_h),
        Image.BILINEAR,
    ).filter(ImageFilter.GaussianBlur(radius=4))
    bg = thumb.resize((cw, ch), Image.BICUBIC)
    bg = ImageEnhance.Brightness(bg).enhance(BG_BRIGHTNESS)
    bg = ImageEnhance.Color(bg).enhance(0.82)
    return bg


def _canvas(image_path: Path, box: Box, width: int, height: int) -> Tuple[Image.Image, str]:
    with Image.open(image_path) as im:
        try:
            im.seek(0)
        except EOFError:
            pass
        src = im.convert("RGB")
        full_w, full_h = src.size
        bx0 = max(0, min(full_w - 1, int(box[0])))
        by0 = max(0, min(full_h - 1, int(box[1])))
        bx1 = max(bx0 + 1, min(full_w, int(box[2])))
        by1 = max(by0 + 1, min(full_h, int(box[3])))
        raw_crop = src.crop((bx0, by0, bx1, by1))

    crop = _trim_excess_white(raw_crop)
    pw, ph = crop.size
    cw, chh = width * SS, height * SS
    target_aspect = cw / float(chh)
    panel_aspect = pw / float(max(1, ph))

    is_small = (pw < 460 and ph < 460) or ((pw * ph) < (full_w * full_h * 0.18))
    is_short_strip = (panel_aspect >= 1.85) or (panel_aspect >= 1.32 and ph < full_w * 0.50)
    is_landscape_native = 1.25 <= panel_aspect < 1.85
    is_tall_strip = (ph / float(max(1, pw))) >= 1.22

    # 1. Native landscape panels: fill or fit the 16:9 frame cleanly in landscape
    if is_landscape_native and not is_small:
        cover_scale = max(cw / float(pw), chh / float(ph))
        contain_scale = min(cw / float(pw), chh / float(ph))
        if cover_scale / max(contain_scale, 1e-6) <= 1.16:
            fw = max(cw, int(round(pw * cover_scale)))
            fh = max(chh, int(round(ph * cover_scale)))
            fg = crop.resize((fw, fh), Image.LANCZOS)
            return fg, "landscape"

    # 2. Short-height wide strips (e.g. 7cm x 3cm): prioritize width & center visibility
    if is_short_strip:
        target_w_fill = cw * 0.94
        min_readable_h = chh * 0.54
        scale_w = target_w_fill / float(pw)
        scale_h = min_readable_h / float(ph)
        scale = max(scale_w, min(scale_h, scale_w * 1.35))
        scale = min(scale, MAX_UPSCALE_SMALL * SS)

        fw = max(1, int(round(pw * scale)))
        fh = max(1, int(round(ph * scale)))
        fg = crop.resize((fw, fh), Image.LANCZOS)
        if scale > 1.15:
            fg = fg.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=3))

        canvas_w = max(cw, fw)
        canvas_h = max(chh, fh)
        bg = _build_ambient_bg(crop, canvas_w, canvas_h)
        bg.paste(fg, ((canvas_w - fw) // 2, (canvas_h - fh) // 2))
        return bg, "wide_strip"

    # 3. Very small panels: boost scale so artwork/dialogue is clear and centered
    if is_small:
        fit_scale = min((cw * 0.82) / float(pw), (chh * 0.86) / float(ph))
        max_cap = (MAX_UPSCALE_SMALL * cw) / float(max(pw, int(full_w * 0.55)))
        scale = min(fit_scale, max(1.25 * SS, max_cap))
        fw = max(1, int(round(pw * scale)))
        fh = max(1, int(round(ph * scale)))
        fg = crop.resize((fw, fh), Image.LANCZOS)
        fg = fg.filter(ImageFilter.UnsharpMask(radius=1.3, percent=120, threshold=3))

        total_w = max(cw, fw)
        total_h = max(chh, fh)
        bg = _build_ambient_bg(crop, total_w, total_h)
        bg.paste(fg, ((total_w - fw) // 2, (total_h - fh) // 2))
        return bg, "small_centered"

    # 4. Tall vertical panels or standard portrait panels
    if is_tall_strip:
        width_fill_ratio = 0.78 if pw < full_w * 0.85 else 0.86
        scale = min(
            (cw * width_fill_ratio) / float(pw),
            (MAX_UPSCALE_REGULAR * cw) / float(max(pw, full_w)),
        )
    else:
        scale = min(
            (cw * 0.88) / float(pw),
            (chh * 0.96) / float(ph),
            (MAX_UPSCALE_REGULAR * cw) / float(max(pw, full_w)),
        )

    fw = max(1, int(round(pw * scale)))
    fh = max(1, int(round(ph * scale)))
    fg = crop.resize((fw, fh), Image.LANCZOS)

    total_w = max(cw, fw)
    total_h = max(chh, fh)
    if fw >= total_w and fh >= total_h:
        return fg, ("tall" if fh > chh + 8 * SS else "standard")

    bg = _build_ambient_bg(crop, total_w, total_h)
    bg.paste(fg, ((total_w - fw) // 2, (total_h - fh) // 2))
    mode = "tall" if total_h - chh > 6 * SS else "standard"
    return bg, mode


def _motion(effect: str, e: float, mode: str) -> Tuple[float, float, float]:
    zi = 1.0 + ZOOM * e
    zo = 1.0 + ZOOM * (1.0 - e)
    pz = PAN_ZOOM

    if mode == "tall":
        if effect in ("pan_up", "zoomout_pan_left", "diagonal_zoom_out"):
            return (1.0 + 0.04 * (1.0 - e), 0.5, 1.0 - e)
        return (1.0 + 0.04 * e, 0.5, e)

    if mode in ("wide_strip", "landscape"):
        wide_table = {
            "pan_left": (pz, e, 0.5),
            "pan_right": (pz, 1.0 - e, 0.5),
            "zoomin_pan_left": (zi, e, 0.5),
            "zoomin_pan_right": (zi, 1.0 - e, 0.5),
            "zoomout_pan_left": (zo, e, 0.5),
            "zoomout_pan_right": (zo, 1.0 - e, 0.5),
            "zoom_in": (zi, 0.5, 0.5),
            "zoom_out": (zo, 0.5, 0.5),
        }
        return wide_table.get(effect, (pz, 1.0 - e, 0.5))

    table = {
        "zoom_in": (zi, 0.5, 0.5),
        "zoom_out": (zo, 0.5, 0.5),
        "pan_left": (pz, e, 0.5),
        "pan_right": (pz, 1.0 - e, 0.5),
        "pan_up": (pz, 0.5, 1.0 - e),
        "pan_down": (pz, 0.5, e),
        "zoomin_pan_left": (zi, e, 0.5),
        "zoomin_pan_right": (zi, 1.0 - e, 0.5),
        "zoomout_pan_left": (zo, e, 0.5),
        "zoomout_pan_right": (zo, 1.0 - e, 0.5),
        "diagonal_zoom_in": (zi, e, e),
        "diagonal_zoom_out": (zo, 1.0 - e, 1.0 - e),
    }
    return table.get(effect, table["zoom_in"])


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
    shot_index: int = 0,
) -> None:
    canvas, mode = _canvas(image_path, box, width, height)
    cw, chh = canvas.size
    w_ss, h_ss = width * SS, height * SS

    chosen_effect = effect
    if not chosen_effect or chosen_effect == "auto":
        chosen_effect = select_smart_effect(box, cw, chh, shot_index=shot_index)
    elif mode == "tall" and chosen_effect not in ("pan_down", "pan_up"):
        chosen_effect = "pan_down" if (shot_index % 2 == 0) else "pan_up"
    elif mode == "wide_strip" and chosen_effect in ("pan_up", "pan_down"):
        chosen_effect = "pan_right" if (shot_index % 2 == 0) else "pan_left"

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
            zoom, fx, fy = _motion(chosen_effect, e, mode)
            ww = min(float(cw), w_ss / zoom)
            wh = min(float(chh), h_ss / zoom)
            x0 = max(0.0, (cw - ww) * fx)
            y0 = max(0.0, (chh - wh) * fy)
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
