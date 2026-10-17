from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

Box = Tuple[int, int, int, int]

FLAT_STD = 7.0
MIN_GAP = 6
MIN_BAND = 48
MAX_TALL = 0.95
MIN_CHUNK = 0.45
PAD = 4
WORK_WIDTH = 720


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _bridge(content: np.ndarray, gap: int) -> np.ndarray:
    result = content.copy()
    for a, b in _runs(~content):
        if a > 0 and b < len(content) and b - a < gap:
            result[a:b] = True
    return result


def _cut_rows(energy: np.ndarray, y0: int, y1: int, w: int) -> List[Tuple[int, int]]:
    h = y1 - y0
    max_h = max(int(w * MAX_TALL), 1)
    if h <= max_h * 1.15:
        return [(y0, y1)]
    n = math.ceil(h / max_h)
    cuts = [y0]
    for k in range(1, n):
        ideal = y0 + h * k / n
        lo = int(max(cuts[-1] + max_h * MIN_CHUNK, ideal - 0.2 * max_h))
        hi = int(min(y1 - max_h * MIN_CHUNK, ideal + 0.2 * max_h))
        if hi <= lo:
            cuts.append(int(ideal))
        else:
            cuts.append(lo + int(np.argmin(energy[lo:hi])))
    cuts.append(y1)
    return [(cuts[i], cuts[i + 1]) for i in range(n)]


def detect_panels(path: Path) -> List[Box]:
    with Image.open(path) as im:
        try:
            im.seek(0)
        except EOFError:
            pass
        rgb = im.convert("RGB")
    ow, oh = rgb.size
    f = 1.0
    if ow > WORK_WIDTH:
        f = ow / WORK_WIDTH
        rgb = rgb.resize((WORK_WIDTH, max(int(oh / f), 1)), Image.BILINEAR)
    g = np.asarray(rgb.convert("L"), dtype=np.float32)
    h, w = g.shape

    row_std = g.std(axis=1)
    content = _bridge(row_std >= FLAT_STD, MIN_GAP)

    dy = np.abs(np.diff(g, axis=0, prepend=g[:1])).mean(axis=1)
    dx = np.abs(np.diff(g, axis=1)).mean(axis=1)
    energy = np.convolve(dy + dx, np.ones(9) / 9, mode="same")

    boxes: List[Box] = []
    for y0, y1 in _runs(content):
        bh = y1 - y0
        if bh < MIN_BAND:
            continue
        band = g[y0:y1]
        col = band.std(axis=0) >= FLAT_STD * 0.8
        xs = np.flatnonzero(col)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        bg = float(np.median(band))
        ink = float((np.abs(band - bg) > 40).mean())
        if bh < 140 and ink < 0.003:
            continue
        min_w = int(w * 0.5)
        if x1 - x0 < min_w:
            c = (x0 + x1) // 2
            x0 = max(0, min(w - min_w, c - min_w // 2))
            x1 = x0 + min_w
        x0 = max(0, x0 - PAD)
        x1 = min(w, x1 + PAD)
        for a, b in _cut_rows(energy, y0, y1, x1 - x0):
            if b - a >= MIN_BAND:
                boxes.append(
                    (
                        int(x0 * f),
                        int(a * f),
                        min(ow, int(math.ceil(x1 * f))),
                        min(oh, int(math.ceil(b * f))),
                    )
                )
    if not boxes:
        boxes = [(0, 0, ow, oh)]
    return boxes


def assign_panels(panels: List[Box], uses: int, index: int) -> List[Box]:
    n = len(panels)
    if uses <= 1:
        return list(panels)
    if n <= uses:
        return [panels[min(n - 1, index * n // uses)]]
    heights = np.array([b[3] - b[1] for b in panels], dtype=np.float64)
    cum = np.cumsum(heights) / heights.sum()
    lo = index / uses
    hi = (index + 1) / uses
    group = [
        panels[i]
        for i in range(n)
        if (cum[i] - heights[i] / heights.sum() / 2) >= lo
        and (cum[i] - heights[i] / heights.sum() / 2) < hi
    ]
    return group or [panels[min(n - 1, index * n // uses)]]


def plan_shots(
    panels: List[Box], total_frames: int, fps: int, min_shot: float = 1.6
) -> List[Tuple[Box, int]]:
    shots = list(panels)
    while len(shots) > 1 and total_frames / fps / len(shots) < min_shot:
        best, best_h = 0, None
        for i in range(len(shots) - 1):
            a, b = shots[i], shots[i + 1]
            hh = max(a[3], b[3]) - min(a[1], b[1])
            if best_h is None or hh < best_h:
                best, best_h = i, hh
        a, b = shots[best], shots[best + 1]
        shots[best : best + 2] = [
            (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        ]
    weights = np.array([(s[3] - s[1]) ** 0.8 for s in shots], dtype=np.float64)
    cum = np.cumsum(weights) / weights.sum()
    edges = [0] + [int(round(c * total_frames)) for c in cum]
    edges[-1] = total_frames
    out: List[Tuple[Box, int]] = []
    for i, s in enumerate(shots):
        n = edges[i + 1] - edges[i]
        if n < 1:
            n = 1
        out.append((s, n))
    diff = total_frames - sum(n for _, n in out)
    if diff:
        s, n = out[-1]
        out[-1] = (s, max(1, n + diff))
    return out
