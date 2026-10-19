from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from audio.sentences import Sentence, normalize_for_match
from config.settings import MatcherConfig, load_secrets, matcher_config
from planning.plan_parser import EditPlan, PlanImage
from utils.error_handler import PipelineError, STAGE_SYNC
from utils.logger import get_logger

logger = get_logger("sync.story_matcher")
ProgressFn = Optional[Callable[[str], None]]


@dataclass
class MatchedScene:
    image: PlanImage
    start: float
    end: float
    sentence_indices: List[int] = field(default_factory=list)
    method: str = "gemini"
    confidence: float = 1.0

    @property
    def plan_item(self) -> PlanImage:
        return self.image

    @property
    def media_path(self) -> str:
        return self.image.file

    @property
    def file(self) -> str:
        return self.image.file

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class MatchResult:
    scenes: List[MatchedScene]
    method: str
    warnings: List[str] = field(default_factory=list)
    unmatched: List[int] = field(default_factory=list)

    def __iter__(self):
        return iter(self.scenes)

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, item):
        return self.scenes[item]


def _is_monotonic(
    assignment: Dict[int, List[int]],
    num_images: int,
) -> Tuple[bool, str]:
    last_max = -1
    for idx in range(num_images):
        sent_idxs = assignment.get(idx)
        if not sent_idxs:
            continue
        clean = sorted(set(int(i) for i in sent_idxs))
        if clean[0] <= last_max:
            return (
                False,
                f"plan[{idx}] sentence {clean[0]} overlaps/precedes {last_max}",
            )
        last_max = clean[-1]
    return True, ""


def _similarity(a: str, b: str) -> float:
    na = normalize_for_match(a)
    nb = normalize_for_match(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    tokens_a = set(na.split())
    tokens_b = set(nb.split())
    overlap = len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))
    ratio = SequenceMatcher(None, na, nb).ratio()
    if na in nb or nb in na:
        return max(0.85, (overlap + ratio) * 0.5)
    return 0.55 * ratio + 0.45 * overlap


def _fuzzy_match(
    images: Sequence[PlanImage],
    sentences: Sequence[Sentence],
    threshold: float = 0.5,
) -> Dict[int, List[int]]:
    assignment: Dict[int, List[int]] = {}
    if not images or not sentences:
        return assignment

    cursor = 0
    num_sent = len(sentences)
    num_img = len(images)

    for img_idx, img in enumerate(images):
        query = (img.story_line or img.description or "").strip()
        if not query:
            continue

        max_lookahead = min(
            num_sent,
            max(cursor + 12, int(round((img_idx + 1) * num_sent / max(1, num_img))) + 8),
        )
        best_score = -1.0
        best_sent_idx: Optional[int] = None

        for s_idx in range(cursor, max_lookahead):
            score = _similarity(query, sentences[s_idx].text)
            if score > best_score:
                best_score = score
                best_sent_idx = s_idx

        if best_sent_idx is not None and best_score >= threshold:
            assignment[img_idx] = [best_sent_idx]
            cursor = min(num_sent - 1, best_sent_idx + 1)

    return assignment


def _gemini_match(
    images: Sequence[PlanImage],
    sentences: Sequence[Sentence],
    cfg: MatcherConfig,
) -> Dict[int, List[int]]:
    secrets = load_secrets()
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=secrets.gemini_api_key)
    models_to_try = [cfg.model] + [
        m for m in cfg.fallback_models if m != cfg.model
    ]

    sentence_payload = [
        {"s": i, "text": s.text} for i, s in enumerate(sentences)
    ]
    image_payload = [
        {
            "p": i,
            "file": img.file,
            "story_line": (img.story_line or img.description or "").strip(),
        }
        for i, img in enumerate(images)
    ]

    prompt = (
        "Align each plan entry 'p' to one or more narration sentence indices 's' "
        "in strict chronological order.\n"
        "Return ONLY valid JSON in the format: "
        '{"matches": [{"p": 0, "s": [0]}, ...]}\n\n'
        f"Sentences:\n{json.dumps(sentence_payload, ensure_ascii=False)}\n\n"
        f"Plan:\n{json.dumps(image_payload, ensure_ascii=False)}"
    )

    last_err: Optional[Exception] = None
    for model_name in models_to_try:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            raw_text = (resp.text or "").strip()
            if not raw_text:
                continue
            cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text, flags=re.MULTILINE).strip()
            data = json.loads(cleaned)
            items = data.get("matches", data) if isinstance(data, dict) else data
            if not isinstance(items, list):
                continue

            out: Dict[int, List[int]] = {}
            for row in items:
                if not isinstance(row, dict) or "p" not in row or "s" not in row:
                    continue
                p_idx = int(row["p"])
                s_val = row["s"]
                s_list = [int(x) for x in s_val] if isinstance(s_val, list) else [int(s_val)]
                s_list = [
                    x for x in sorted(set(s_list)) if 0 <= x < len(sentences)
                ]
                if 0 <= p_idx < len(images) and s_list:
                    out[p_idx] = s_list
            if out:
                return out
        except Exception as exc:
            last_err = exc
            continue

    if last_err is not None:
        raise last_err
    return {}


def _assign_raw_spans(
    n: int,
    assignment: Dict[int, List[int]],
    sentences: Sequence[Sentence],
    duration: float,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    spans: List[Optional[Tuple[float, float]]] = [None] * n
    unmatched: List[int] = []

    for i in range(n):
        s_idxs = assignment.get(i)
        if s_idxs:
            valid = [idx for idx in s_idxs if 0 <= idx < len(sentences)]
            if valid:
                start = float(sentences[valid[0]].start)
                end = float(sentences[valid[-1]].end)
                spans[i] = (start, max(end, start + 0.05))
                continue
        unmatched.append(i)

    idx = 0
    while idx < n:
        if spans[idx] is not None:
            idx += 1
            continue
        run_start = idx
        while idx < n and spans[idx] is None:
            idx += 1
        run_end = idx
        count = run_end - run_start

        left_t = spans[run_start - 1][1] if run_start > 0 and spans[run_start - 1] else 0.0
        right_t = (
            spans[run_end][0]
            if run_end < n and spans[run_end] is not None
            else max(duration, left_t + 0.1 * count)
        )

        if right_t <= left_t:
            right_t = left_t + 0.1 * count

        step = (right_t - left_t) / count
        for offset in range(count):
            s_t = left_t + offset * step
            e_t = left_t + (offset + 1) * step
            spans[run_start + offset] = (s_t, e_t)

    return [sp if sp is not None else (0.0, 0.1) for sp in spans], unmatched


def _split_equal_spans(
    scenes: List[MatchedScene],
    raw: List[Tuple[float, float]],
) -> None:
    n = len(raw)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(raw[j][0] - raw[i][0]) < 1e-6 and abs(raw[j][1] - raw[i][1]) < 1e-6:
            j += 1
        count = j - i
        if count > 1:
            start, end = raw[i]
            step = max(0.05, (end - start) / count)
            for k in range(count):
                s_t = start + k * step
                e_t = start + (k + 1) * step if k < count - 1 else end
                raw[i + k] = (s_t, e_t)
                if i + k < len(scenes):
                    scenes[i + k].start = s_t
                    scenes[i + k].end = e_t
        i = j


def _finalize_spans(
    spans: Sequence[Tuple[float, float]],
    n: int,
    duration: float,
) -> List[Tuple[float, float]]:
    if n <= 0 or not spans:
        return []

    out: List[List[float]] = [[float(s), float(e)] for s, e in spans[:n]]
    out[0][0] = 0.0

    for i in range(n - 1):
        curr_start, curr_end = out[i]
        next_start, next_end = out[i + 1]

        if next_start < curr_end:
            # Trim overlap: advance next_start to curr_end
            next_start = max(curr_end, curr_start + 0.05)
            if next_end <= next_start:
                next_end = next_start + 0.05
            out[i + 1][0] = next_start
            out[i + 1][1] = next_end
        elif next_start > curr_end:
            # Hold model: extend current scene until next scene begins
            out[i][1] = next_start

    total_dur = max(float(duration), out[-1][0] + 0.05)
    out[-1][1] = max(total_dur, out[-1][0] + 0.05)

    return [(round(s, 6), round(e, 6)) for s, e in out]


def match_plan_to_transcript(
    plan: EditPlan,
    sentences: Sequence[Sentence],
    progress: ProgressFn = None,
) -> MatchResult:
    if not sentences:
        raise PipelineError(
            STAGE_SYNC,
            "no sentences available to match against plan",
        )

    images = plan.ordered_kept() if hasattr(plan, "ordered_kept") else list(plan)
    if not images:
        raise PipelineError(
            STAGE_SYNC,
            "plan has no kept images to match",
        )

    cfg = matcher_config()
    warnings: List[str] = []
    duration = max(float(sentences[-1].end), 0.1)
    num_images = len(images)

    assignment: Dict[int, List[int]] = {}
    method = "gemini"

    if progress:
        progress("aligning story lines with narration")

    try:
        assignment = _gemini_match(images, sentences, cfg)
        ok, reason = _is_monotonic(assignment, num_images)
        if not ok:
            warnings.append(f"Gemini alignment non-monotonic ({reason}); using fuzzy fallback")
            assignment = {}
    except Exception as exc:
        logger.debug("Gemini matcher unavailable or failed: %s", exc)
        assignment = {}

    if not assignment:
        if cfg.allow_fuzzy_fallback:
            assignment = _fuzzy_match(images, sentences, threshold=cfg.min_confidence)
            ok, _ = _is_monotonic(assignment, num_images)
            if ok and assignment:
                method = "fuzzy"
            else:
                assignment = {}

    if not assignment:
        method = "even"
        warnings.append(
            "no story lines aligned with transcript; distributing images evenly across audio"
        )
        step = duration / num_images
        even_scenes: List[MatchedScene] = []
        for idx, img in enumerate(images):
            s_t = idx * step
            e_t = (idx + 1) * step if idx < num_images - 1 else duration
            s_idx = min(len(sentences) - 1, (idx * len(sentences)) // num_images)
            even_scenes.append(
                MatchedScene(
                    image=img,
                    start=round(s_t, 6),
                    end=round(e_t, 6),
                    sentence_indices=[s_idx],
                    method="even",
                    confidence=0.5,
                )
            )
        return MatchResult(
            scenes=even_scenes,
            method="even",
            warnings=warnings,
            unmatched=list(range(num_images)),
        )

    raw_spans, unmatched = _assign_raw_spans(
        num_images, assignment, sentences, duration
    )

    scenes: List[MatchedScene] = []
    for idx, img in enumerate(images):
        s_t, e_t = raw_spans[idx]
        s_idxs = assignment.get(idx, [])
        scenes.append(
            MatchedScene(
                image=img,
                start=s_t,
                end=e_t,
                sentence_indices=list(s_idxs),
                method=method if idx not in unmatched else "interpolated",
                confidence=1.0 if idx not in unmatched else 0.6,
            )
        )

    _split_equal_spans(scenes, raw_spans)
    final_spans = _finalize_spans(raw_spans, num_images, duration)

    for idx, (s_t, e_t) in enumerate(final_spans):
        scenes[idx].start = s_t
        scenes[idx].end = e_t

    if unmatched:
        warnings.append(
            f"{len(unmatched)}/{num_images} plan image(s) had no direct text match and were interpolated"
        )

    logger.info(
        "story matching complete: entries=%d, sentences=%d, method=%s, duration=%.2fs",
        len(scenes),
        len(sentences),
        method,
        duration,
    )

    return MatchResult(
        scenes=scenes,
        method=method,
        warnings=warnings,
        unmatched=unmatched,
    )


MatchedItem = MatchedScene
__all__ = [
    "MatchedScene",
    "MatchedItem",
    "MatchResult",
    "match_plan_to_transcript",
    "_is_monotonic",
    "_fuzzy_match",
    "_gemini_match",
    "_assign_raw_spans",
    "_split_equal_spans",
    "_finalize_spans",
]
