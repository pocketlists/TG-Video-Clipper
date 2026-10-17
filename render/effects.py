"""
render/effects.py
FFmpeg zoompan filter builders. Byte-identical output vs. the old
video_editor.ImageProcessor.generate_zoompan_filter so rendered visuals
do not change.

Config (supersample, zoom amount, pan zoom level) comes from
config.settings.render_config() so tuning is a JSON edit, not a code
edit. Values are captured at module import — one process, one render,
one set of knobs.
"""

from __future__ import annotations

from typing import List

from config.settings import render_config
from utils.logger import get_logger


logger = get_logger("render.effects")

_cfg = render_config()

SUPERSAMPLE = _cfg.supersample
ZOOM_AMOUNT = _cfg.zoom_amount
PAN_ZOOM_LEVEL = _cfg.pan_zoom_level


EFFECTS: List[str] = [
    "zoom_in", "zoom_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
    "zoomin_pan_left", "zoomin_pan_right",
    "zoomout_pan_left", "zoomout_pan_right",
    "diagonal_zoom_in", "diagonal_zoom_out",
]


def is_known_effect(effect: str) -> bool:
    return effect in EFFECTS


def _expressions(effect: str, total_frames: int) -> tuple[str, str, str]:
    """Return (zoom_expr, x_expr, y_expr) for the given effect.

    `on` is the ffmpeg zoompan frame counter; total_frames is the clip's
    own frame count so motion is normalized to [0,1] over exactly this
    clip. Smoothstep easing gives slow-start / slow-end so pans don't
    start/stop abruptly.
    """
    t = f"(on/{total_frames})"
    et = f"({t}*{t}*(3-2*{t}))"   # smoothstep
    z_max = 1.0 + ZOOM_AMOUNT
    pz = PAN_ZOOM_LEVEL

    if effect == "zoom_in":
        return (f"(1.0+{ZOOM_AMOUNT}*{et})", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2")
    if effect == "zoom_out":
        return (f"({z_max}-{ZOOM_AMOUNT}*{et})", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2")
    if effect == "pan_left":
        return (f"{pz}", f"((iw-iw/zoom)*{et})", "(ih-ih/zoom)/2")
    if effect == "pan_right":
        return (f"{pz}", f"((iw-iw/zoom)*(1-{et}))", "(ih-ih/zoom)/2")
    if effect == "pan_up":
        return (f"{pz}", "(iw-iw/zoom)/2", f"((ih-ih/zoom)*(1-{et}))")
    if effect == "pan_down":
        return (f"{pz}", "(iw-iw/zoom)/2", f"((ih-ih/zoom)*{et})")
    if effect == "zoomin_pan_left":
        return (f"(1.0+{ZOOM_AMOUNT}*{et})", f"((iw-iw/zoom)*{et})", "(ih-ih/zoom)/2")
    if effect == "zoomin_pan_right":
        return (f"(1.0+{ZOOM_AMOUNT}*{et})", f"((iw-iw/zoom)*(1-{et}))", "(ih-ih/zoom)/2")
    if effect == "zoomout_pan_left":
        return (f"({z_max}-{ZOOM_AMOUNT}*{et})", f"((iw-iw/zoom)*{et})", "(ih-ih/zoom)/2")
    if effect == "zoomout_pan_right":
        return (f"({z_max}-{ZOOM_AMOUNT}*{et})", f"((iw-iw/zoom)*(1-{et}))", "(ih-ih/zoom)/2")
    if effect == "diagonal_zoom_in":
        return (f"(1.0+{ZOOM_AMOUNT}*{et})", f"((iw-iw/zoom)*{et})", f"((ih-ih/zoom)*{et})")
    if effect == "diagonal_zoom_out":
        return (f"({z_max}-{ZOOM_AMOUNT}*{et})", f"((iw-iw/zoom)*(1-{et}))", f"((ih-ih/zoom)*(1-{et}))")

    # unknown effect — safest fallback, same as old code
    logger.warning(f"unknown effect '{effect}', falling back to zoom_in")
    return (f"(1.0+{ZOOM_AMOUNT}*{et})", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2")


def build_zoompan_filter(
    effect: str,
    width: int,
    height: int,
    fps: int,
    total_frames: int,
) -> str:
    """Return the filter chain that produces `total_frames` frames of
    `width`×`height` at `fps` with the given effect.

    The output string is:

        scale=W*S:H*S:flags=lanczos,
        zoompan=z='...':x='...':y='...':d=total_frames:s=WxH:fps=fps

    where S = SUPERSAMPLE. The pre-scale step is what fixes the "zoompan
    pixel rounding jitter" issue in the old code — see the comment there.
    """
    total_frames = max(int(total_frames), 1)
    z, x, y = _expressions(effect, total_frames)

    pre_w = width * SUPERSAMPLE
    pre_h = height * SUPERSAMPLE
    prescale = f"scale={pre_w}:{pre_h}:flags=lanczos"
    d = f"d={total_frames}"
    s = f"s={width}x{height}"
    fps_str = f"fps={fps}"

    return (
        f"{prescale},"
        f"zoompan=z='{z}':x='{x}':y='{y}':{d}:{s}:{fps_str}"
    )
