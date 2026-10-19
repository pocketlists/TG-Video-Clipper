from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

Box = Tuple[int, int, int, int]

WORK_WIDTH = 720
WHITE_THRESHOLD = 242.0
BLACK_BORDER_MAX = 65.0
FLAT_ROW_STD = 6.5
MIN_GUTTER_PX = 5
MIN_PANEL_HEIGHT = 44
MAX_ASPECT_TALL = 1.65
MIN_CHUNK_RATIO = 0.42


@dataclass
class PanelRegion:
    source_files: List[str]
    source_paths: List[Path]
    box: Box
    width: int
    height: int
    open_top: bool = False
    open_bottom: bool = False
    has_black_border: bool = False
    is_landscape_native: bool = False
    is_short_strip: bool = False
    is_small_panel: bool = False
    stitched_image_path: Optional[Path] = None
    recommended_effect: str = "zoom_in"
    panel_id: str = ""


@dataclass
class ImagePanelMap:
    filename: str
    path: Path
    orig_width: int
    orig_height: int
    panels: List[PanelRegion] = field(default_factory=list)
    consumed_panel_indices: set[int] = field(default_factory=set)


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        try:
            im.seek(0)
        except EOFError:
            pass
        return im.convert("RGB")


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _bridge_gaps(content: np.ndarray, min_gap: int) -> np.ndarray:
    res = content.copy()
    for a, b in _runs(~content):
        if a > 0 and b < len(content) and (b - a) < min_gap:
            res[a:b] = True
    return res


def _is_white_or_solid_row(gray: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    row_mean = gray.mean(axis=1)
    row_std = gray.std(axis=1)
    color_std = rgb.std(axis=(1, 2))
    nearly_white = (row_mean >= WHITE_THRESHOLD) & (row_std <= FLAT_ROW_STD * 1.3)
    flat_gutter = (row_std <= FLAT_ROW_STD * 0.75) & (color_std <= FLAT_ROW_STD * 0.85) & (
        (row_mean >= 228.0) | (row_mean <= 18.0)
    )
    return ~(nearly_white | flat_gutter)


def _trim_white_columns(
    gray_band: np.ndarray,
    rgb_band: np.ndarray,
    full_w: int,
    full_h: int,
) -> Tuple[int, int]:
    bh, bw = gray_band.shape
    col_mean = gray_band.mean(axis=0)
    col_std = gray_band.std(axis=0)
    col_cstd = rgb_band.std(axis=(0, 2))

    non_white_col = (col_mean < WHITE_THRESHOLD - 3.0) | (col_std > FLAT_ROW_STD * 0.8) | (
        col_cstd > FLAT_ROW_STD
    )
    xs = np.flatnonzero(non_white_col)
    if len(xs) == 0:
        return 0, bw

    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    content_w = x1 - x0

    # Large images get minimal white margin; smaller/narrower panels keep a balanced pad
    area_ratio = (content_w * bh) / max(1.0, float(full_w * full_h))
    if area_ratio >= 0.55:
        pad = max(2, int(bw * 0.006))
    elif bh / max(1, content_w) < 0.48:
        pad = max(4, int(bw * 0.015))
    else:
        pad = max(3, int(bw * 0.01))

    x0 = max(0, x0 - pad)
    x1 = min(bw, x1 + pad)

    min_allowed_w = int(bw * 0.38)
    if (x1 - x0) < min_allowed_w:
        center = (x0 + x1) // 2
        x0 = max(0, min(bw - min_allowed_w, center - min_allowed_w // 2))
        x1 = min(bw, x0 + min_allowed_w)

    return x0, x1


def _detect_border_state(
    gray: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> Tuple[bool, bool, bool]:
    """
    Returns (has_black_border, open_top, open_bottom).
    Checks if a panel has a black frame line and whether the top or bottom border is missing/cut.
    """
    h, w = gray.shape
    sub = gray[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
    sh, sw = sub.shape
    if sh < 16 or sw < 16:
        return False, False, False

    edge_band = max(3, min(8, sw // 40))
    left_strip = sub[:, :edge_band]
    right_strip = sub[:, -edge_band:]
    top_strip = sub[:edge_band, :]
    bot_strip = sub[-edge_band:, :]

    left_dark = float((left_strip < BLACK_BORDER_MAX).mean())
    right_dark = float((right_strip < BLACK_BORDER_MAX).mean())
    top_dark = float((top_strip < BLACK_BORDER_MAX).mean())
    bot_dark = float((bot_strip < BLACK_BORDER_MAX).mean())

    side_bordered = (left_dark >= 0.42) or (right_dark >= 0.42)
    has_border = side_bordered or (top_dark >= 0.48) or (bot_dark >= 0.48)

    # Top is open if panel touches top of image and top horizontal border line is absent
    touches_top = y0 <= max(6, int(h * 0.015))
    top_row_active = float(sub[:edge_band, :].std()) > FLAT_ROW_STD * 1.15 or (
        float(sub[:edge_band, :].mean()) < WHITE_THRESHOLD - 10.0
    )
    open_top = touches_top and (top_dark < 0.45) and (side_bordered or top_row_active)

    # Bottom is open if panel touches bottom of image and bottom horizontal border line is absent
    touches_bottom = y1 >= (h - max(6, int(h * 0.015)))
    bot_row_active = float(sub[-edge_band:, :].std()) > FLAT_ROW_STD * 1.15 or (
        float(sub[-edge_band:, :].mean()) < WHITE_THRESHOLD - 10.0
    )
    open_bottom = touches_bottom and (bot_dark < 0.45) and (side_bordered or bot_row_active)

    return has_border, open_top, open_bottom


def _split_tall_band(
    energy: np.ndarray,
    gray: np.ndarray,
    y0: int,
    y1: int,
    panel_w: int,
) -> List[Tuple[int, int]]:
    h = y1 - y0
    max_h = max(int(panel_w * MAX_ASPECT_TALL), 1)
    if h <= max_h * 1.18:
        return [(y0, y1)]

    n = max(2, math.ceil(h / max_h))
    cuts = [y0]
    for k in range(1, n):
        ideal = y0 + (h * k) / n
        lo = int(max(cuts[-1] + max_h * MIN_CHUNK_RATIO, ideal - 0.25 * max_h))
        hi = int(min(y1 - max_h * MIN_CHUNK_RATIO, ideal + 0.25 * max_h))
        if hi <= lo:
            cuts.append(int(ideal))
            continue

        # Prefer cutting on white gutters or horizontal black panel borders
        window_energy = energy[lo:hi].copy()
        row_means = gray[lo:hi].mean(axis=1)
        gutter_bonus = np.where((row_means >= WHITE_THRESHOLD) | (row_means <= 40.0), -25.0, 0.0)
        best_cut = lo + int(np.argmin(window_energy + gutter_bonus))
        cuts.append(best_cut)

    cuts.append(y1)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if (cuts[i + 1] - cuts[i]) >= MIN_PANEL_HEIGHT]


def _classify_panel_geometry(box: Box, orig_w: int, orig_h: int) -> Tuple[bool, bool, bool]:
    x0, y0, x1, y1 = box
    pw = max(1, x1 - x0)
    ph = max(1, y1 - y0)
    aspect = pw / float(ph)

    # Native landscape panel (wider than 1.28:1)
    is_landscape = aspect >= 1.28
    # Short-height wide strip (e.g. 7cm x 3cm => aspect >= 1.85 or small height ratio)
    is_short_strip = (aspect >= 1.85) or (aspect >= 1.35 and ph < orig_w * 0.52)
    # Very small panel that needs smart centering + upscale cap
    is_small = (pw < 420 and ph < 420) or ((pw * ph) < (orig_w * orig_h * 0.16) and ph < orig_w * 0.45)
    return is_landscape, is_short_strip, is_small


def select_smart_effect(
    box: Box,
    orig_w: int,
    orig_h: int,
    shot_index: int = 0,
    prev_effect: Optional[str] = None,
) -> str:
    """
    Editor-style deterministic motion selector based on panel aspect ratio and sequence position.
    - Tall vertical strips -> pan_down / pan_up
    - Landscape / short wide strips -> pan_left / pan_right / zoomin_pan_*
    - Square / portrait panels -> alternating zoom_in / zoom_out / diagonal_*
    """
    x0, y0, x1, y1 = box
    pw = max(1, x1 - x0)
    ph = max(1, y1 - y0)
    aspect = pw / float(ph)
    height_ratio = ph / float(max(1, pw))

    if height_ratio >= 1.22:
        # Tall vertical panel: prioritize top-to-bottom reading flow
        candidates = ["pan_down", "pan_up", "pan_down", "diagonal_zoom_in"]
    elif aspect >= 1.80:
        # Short wide strip (e.g. 7x3 ratio): smooth horizontal pan or subtle center zoom
        candidates = ["pan_right", "pan_left", "zoom_in", "zoomout_pan_right", "zoomin_pan_left"]
    elif aspect >= 1.28:
        # Standard landscape panel: left-to-right / right-to-left cinematic pan
        candidates = ["pan_left", "pan_right", "zoomin_pan_right", "zoomin_pan_left", "zoom_out"]
    else:
        # Balanced portrait or square panel: clean depth motion
        candidates = ["zoom_in", "zoom_out", "diagonal_zoom_in", "zoomin_pan_left", "diagonal_zoom_out"]

    ordered = [c for c in candidates if c != prev_effect] or candidates
    return ordered[shot_index % len(ordered)]


def detect_panels_detailed(path: Path) -> ImagePanelMap:
    path = Path(path)
    rgb_img = _open_rgb(path)
    ow, oh = rgb_img.size

    scale = 1.0
    if ow > WORK_WIDTH:
        scale = ow / float(WORK_WIDTH)
        work_h = max(1, int(round(oh / scale)))
        work_rgb = rgb_img.resize((WORK_WIDTH, work_h), Image.BILINEAR)
    else:
        work_rgb = rgb_img

    rgb_arr = np.asarray(work_rgb, dtype=np.float32)
    gray = np.asarray(work_rgb.convert("L"), dtype=np.float32)
    h, w = gray.shape

    row_active = _is_white_or_solid_row(gray, rgb_arr)
    content_mask = _bridge_gaps(row_active, MIN_GUTTER_PX)

    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1])).mean(axis=1)
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1])).mean(axis=1)
    energy = np.convolve(dy + dx, np.ones(9) / 9.0, mode="same")

    panel_map = ImagePanelMap(
        filename=path.name,
        path=path,
        orig_width=ow,
        orig_height=oh,
    )

    raw_bands = _runs(content_mask)
    for y0, y1 in raw_bands:
        bh = y1 - y0
        if bh < MIN_PANEL_HEIGHT:
            continue

        band_gray = gray[y0:y1]
        band_rgb = rgb_arr[y0:y1]

        bg_val = float(np.median(band_gray))
        ink_ratio = float((np.abs(band_gray - bg_val) > 28.0).mean())
        if bh < 110 and ink_ratio < 0.004:
            continue

        x0, x1 = _trim_white_columns(band_gray, band_rgb, w, h)
        has_border, open_top, open_bottom = _detect_border_state(gray, x0, y0, x1, y1)

        sub_rows = _split_tall_band(energy, gray, y0, y1, max(1, x1 - x0))
        for idx_s, (sy0, sy1) in enumerate(sub_rows):
            if (sy1 - sy0) < MIN_PANEL_HEIGHT:
                continue

            sx0, sx1 = _trim_white_columns(gray[sy0:sy1], rgb_arr[sy0:sy1], w, h)
            box_orig: Box = (
                max(0, int(round(sx0 * scale))),
                max(0, int(round(sy0 * scale))),
                min(ow, int(math.ceil(sx1 * scale))),
                min(oh, int(math.ceil(sy1 * scale))),
            )
            if box_orig[2] <= box_orig[0] or box_orig[3] <= box_orig[1]:
                continue

            is_land, is_short, is_small = _classify_panel_geometry(box_orig, ow, oh)
            panel_map.panels.append(
                PanelRegion(
                    source_files=[path.name],
                    source_paths=[path],
                    box=box_orig,
                    width=box_orig[2] - box_orig[0],
                    height=box_orig[3] - box_orig[1],
                    open_top=(open_top if idx_s == 0 else False),
                    open_bottom=(open_bottom if idx_s == len(sub_rows) - 1 else False),
                    has_black_border=has_border,
                    is_landscape_native=is_land,
                    is_short_strip=is_short,
                    is_small_panel=is_small,
                )
            )

    if not panel_map.panels:
        full_box: Box = (0, 0, ow, oh)
        is_land, is_short, is_small = _classify_panel_geometry(full_box, ow, oh)
        panel_map.panels.append(
            PanelRegion(
                source_files=[path.name],
                source_paths=[path],
                box=full_box,
                width=ow,
                height=oh,
                is_landscape_native=is_land,
                is_short_strip=is_short,
                is_small_panel=is_small,
            )
        )

    return panel_map


def _can_stitch_panels(bot_panel: PanelRegion, top_panel: PanelRegion) -> bool:
    """
    True only when `bot_panel` at the bottom of image N and `top_panel` at the top of image N+1
    are two halves of a cut panel whose borders are open at the seam.
    """
    if not (bot_panel.open_bottom and top_panel.open_top):
        return False

    w_a, w_b = bot_panel.width, top_panel.width
    width_ratio = min(w_a, w_b) / float(max(w_a, w_b, 1))
    if width_ratio < 0.72:
        return False

    combined_h = bot_panel.height + top_panel.height
    avg_w = (w_a + w_b) / 2.0
    # Avoid creating an excessively long strip unless it's a single cut box
    if combined_h > avg_w * 2.35:
        return False

    return True


def _write_stitched_panel(
    panel_a: PanelRegion,
    panel_b: PanelRegion,
    cache_dir: Path,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{panel_a.source_paths[-1].name}_{panel_a.box}_{panel_b.source_paths[0].name}_{panel_b.box}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    out_path = cache_dir / f"stitched_{digest}.png"
    if out_path.is_file():
        return out_path

    im_a = _open_rgb(panel_a.source_paths[-1]).crop(panel_a.box)
    im_b = _open_rgb(panel_b.source_paths[0]).crop(panel_b.box)

    target_w = max(im_a.width, im_b.width)
    if im_a.width != target_w:
        im_a = im_a.resize((target_w, max(1, int(round(im_a.height * target_w / im_a.width)))), Image.LANCZOS)
    if im_b.width != target_w:
        im_b = im_b.resize((target_w, max(1, int(round(im_b.height * target_w / im_b.width)))), Image.LANCZOS)

    stitched = Image.new("RGB", (target_w, im_a.height + im_b.height), (255, 255, 255))
    stitched.paste(im_a, (0, 0))
    stitched.paste(im_b, (0, im_a.height))
    stitched.save(out_path, "PNG")
    return out_path


def build_sequence_panel_plan(
    image_paths: Sequence[Path],
    temp_dir: Path,
) -> Dict[Path, List[PanelRegion]]:
    """
    Maps all images in sequence order, joins split panels across consecutive images
    when a panel border is cut across files, and marks the joined second half as consumed
    so it is never repeated when the next image is rendered.
    """
    ordered_unique: List[Path] = []
    seen: set[Path] = set()
    for p in image_paths:
        rp = Path(p).resolve()
        if rp not in seen:
            seen.add(rp)
            ordered_unique.append(Path(p))

    maps: Dict[Path, ImagePanelMap] = {p: detect_panels_detailed(p) for p in ordered_unique}
    stitch_dir = Path(temp_dir) / "stitched_panels"

    for idx in range(len(ordered_unique) - 1):
        curr_path = ordered_unique[idx]
        next_path = ordered_unique[idx + 1]
        if curr_path == next_path:
            continue

        curr_map = maps[curr_path]
        next_map = maps[next_path]
        if not curr_map.panels or not next_map.panels:
            continue

        last_idx = len(curr_map.panels) - 1
        if last_idx in curr_map.consumed_panel_indices:
            continue

        bot_panel = curr_map.panels[last_idx]
        top_panel = next_map.panels[0]

        if 0 not in next_map.consumed_panel_indices and _can_stitch_panels(bot_panel, top_panel):
            stitched_file = _write_stitched_panel(bot_panel, top_panel, stitch_dir)
            with Image.open(stitched_file) as sim:
                sw, sh = sim.size
            s_box: Box = (0, 0, sw, sh)
            is_land, is_short, is_small = _classify_panel_geometry(s_box, sw, sh)

            curr_map.panels[last_idx] = PanelRegion(
                source_files=bot_panel.source_files + top_panel.source_files,
                source_paths=bot_panel.source_paths + top_panel.source_paths,
                box=s_box,
                width=sw,
                height=sh,
                open_top=bot_panel.open_top,
                open_bottom=top_panel.open_bottom,
                has_black_border=True,
                is_landscape_native=is_land,
                is_short_strip=is_short,
                is_small_panel=is_small,
                stitched_image_path=stitched_file,
            )
            # Mark top panel of next image as consumed so it is not reused
            next_map.consumed_panel_indices.add(0)

    result: Dict[Path, List[PanelRegion]] = {}
    prev_effect: Optional[str] = None
    global_shot = 0

    for idx, p in enumerate(ordered_unique):
        pmap = maps[p]
        active = [
            panel
            for i, panel in enumerate(pmap.panels)
            if i not in pmap.consumed_panel_indices
        ]
        # If all panels of image N+1 were stitched into image N, fallback cleanly to the last panel
        # only if image N+1 is explicitly scheduled in the timeline
        if not active and pmap.panels:
            active = [pmap.panels[-1]]

        for panel in active:
            eff = select_smart_effect(
                panel.box,
                panel.width,
                panel.height,
                shot_index=global_shot,
                prev_effect=prev_effect,
            )
            panel.recommended_effect = eff
            prev_effect = eff
            global_shot += 1

        result[p] = active

    return result


def detect_panels(path: Path) -> List[Box]:
    pmap = detect_panels_detailed(path)
    return [p.box for p in pmap.panels]


def assign_panels(panels: List[Box], uses: int, index: int) -> List[Box]:
    n = len(panels)
    if n == 0:
        return [(0, 0, 720, 1280)]
    if uses <= 1:
        return list(panels)
    if n <= uses:
        return [panels[min(n - 1, (index * n) // max(1, uses))]]

    heights = np.array([max(1, b[3] - b[1]) for b in panels], dtype=np.float64)
    total_h = float(heights.sum())
    cum = np.cumsum(heights) / total_h
    centers = cum - (heights / total_h) * 0.5
    lo = index / float(uses)
    hi = (index + 1) / float(uses)
    group = [panels[i] for i in range(n) if lo <= centers[i] < hi]
    return group or [panels[min(n - 1, (index * n) // max(1, uses))]]


def plan_shots(
    panels: List[Box],
    total_frames: int,
    fps: int,
    min_shot: float = 1.45,
) -> List[Tuple[Box, int]]:
    if not panels:
        return [((0, 0, 720, 1280), max(1, total_frames))]

    shots = list(panels)
    while len(shots) > 1 and (total_frames / float(max(1, fps)) / len(shots)) < min_shot:
        best_idx = 0
        best_span: Optional[int] = None
        for i in range(len(shots) - 1):
            a, b = shots[i], shots[i + 1]
            # Only merge panels if they belong to the same coordinate space
            merged_h = max(a[3], b[3]) - min(a[1], b[1])
            if best_span is None or merged_h < best_span:
                best_span = merged_h
                best_idx = i
        a, b = shots[best_idx], shots[best_idx + 1]
        shots[best_idx : best_idx + 2] = [
            (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        ]

    weights = np.array(
        [max(1.0, float((s[3] - s[1]) * 0.7 + (s[2] - s[0]) * 0.3)) ** 0.8 for s in shots],
        dtype=np.float64,
    )
    cum = np.cumsum(weights) / float(weights.sum())
    edges = [0] + [int(round(c * total_frames)) for c in cum]
    edges[-1] = max(len(shots), int(total_frames))

    out: List[Tuple[Box, int]] = []
    for i, s in enumerate(shots):
        frames = max(1, edges[i + 1] - edges[i])
        out.append((s, frames))

    diff = int(total_frames) - sum(f for _, f in out)
    if diff != 0 and out:
        last_box, last_f = out[-1]
        out[-1] = (last_box, max(1, last_f + diff))

    return out
