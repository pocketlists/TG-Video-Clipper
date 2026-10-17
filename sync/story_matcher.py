"""
sync/story_matcher.py
Aligns each kept plan image to the sentence(s) of the transcript that
carry its narration. Gemini is the primary matcher (semantic, batched,
model-rotating); normalized fuzzy text matching is the fallback.
Returns per-image start/end times that respect the original audio
timeline: no negative durations, no overlaps, no gaps that exceed the
audio.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional, Tuple

from audio.sentences import Sentence, normalize_for_match
from config.settings import MatcherConfig, matcher_config
from planning.plan_parser import EditPlan, PlanImage
from utils.error_handler import PipelineError, STAGE_SYNC
from utils.logger import get_logger


logger = get_logger("sync.story_matcher")
ProgressFn = Optional[Callable[[str], None]]

_RATE_LIMIT_HINTS = ("429", "rate limit", "resource_exhausted", "quota", "exhausted")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```\s*$")


@dataclass
class MatchedScene:
    image: PlanImage
    start: float
    end: float
    sentence_indices: List[int]
    method: str               # "gemini" | "fuzzy" | "slot"
    confidence: float
    warnings: List[str] = field(default_factory=list)

    def duration(self) -> float:
        return self.end - self.start


@dataclass
class MatchResult:
    scenes: List[MatchedScene]
    method: str
    warnings: List[str] = field(default_factory=list)
    unmatched_plan_indices: List[int] = field(default_factory=list)
    unused_sentence_indices: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Monotonic validation
# ---------------------------------------------------------------------------

def _is_monotonic(assignment: Dict[int, List[int]], plan_count: int) -> Tuple[bool, str]:
    last_max = -1
    for i in range(plan_count):
        idx_list = assignment.get(i)
        if not idx_list:
            continue
        low, high = min(idx_list), max(idx_list)
        if low <= last_max:
            return False, f"plan[{i}] maps to {idx_list} but earlier plan ends at {last_max}"
        last_max = high
    return True, ""


# ---------------------------------------------------------------------------
# Gemini matcher
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", (text or "").strip()).strip()


def _is_rate_limit(exc: BaseException) -> bool:
    return any(tok in str(exc).lower() for tok in _RATE_LIMIT_HINTS)


def _build_prompt(plan_lines: List[str], sentence_texts: List[str]) -> str:
    plan_block = "\n".join(
        f"{i}: {line}" for i, line in enumerate(plan_lines)
    )
    sent_block = "\n".join(
        f"{i}: {text}" for i, text in enumerate(sentence_texts)
    )
    return f"""You align an image plan to narration sentences.

PLAN LINES (each corresponds to one image, in order):
{plan_block}

NARRATION SENTENCES (in chronological order):
{sent_block}

For every plan line, list the sentence indices whose words correspond
to that image's narration. Follow these rules strictly:

1. Every plan index must appear exactly once in your output.
2. Sentence indices for plan[i] must all be less than sentence indices
   for plan[j] whenever i < j (monotonic). Do not reorder.
3. A plan line may map to one sentence, several consecutive sentences,
   or an empty array if it is a silent emphasis shot that has no direct
   narration.
4. Several consecutive plan lines may map to the same sentence (that
   sentence will be split among those images).
5. Prefer meaning over literal word overlap. Ignore punctuation and
   casing.

Return ONLY JSON of this shape, no other text:
{{"matches": [{{"plan": 0, "sentences": [0]}}, {{"plan": 1, "sentences": [1, 2]}}]}}
"""


def _gemini_match(
    plan_images: List[PlanImage],
    sentences: List[Sentence],
    cfg: MatcherConfig,
) -> Dict[int, List[int]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise PipelineError(STAGE_SYNC, "GEMINI_API_KEY not set")

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise PipelineError(STAGE_SYNC, f"google-genai not installed ({exc})") from exc

    client = genai.Client(api_key=api_key)
    plan_lines = [(img.story_line or "").strip() for img in plan_images]
    sentence_texts = [s.text for s in sentences]

    matches: Dict[int, List[int]] = {}
    batch = cfg.batch_size if cfg.batch_size > 0 else 40
    sent_cursor = 0

    for start in range(0, len(plan_images), batch):
        end = min(start + batch, len(plan_images))
        slice_lines = plan_lines[start:end]
        # Provide a wider window of sentences so the model can still
        # align late-batch items to later sentences.
        window_start = sent_cursor
        window_end = min(len(sentences), sent_cursor + batch * 4 + 20)
        window = sentence_texts[window_start:window_end]

        prompt = _build_prompt(slice_lines, window)
        response = None
        last_error: BaseException | None = None
        for model in [cfg.model] + list(cfg.fallback_models):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                break
            except Exception as exc:
                last_error = exc
                if _is_rate_limit(exc):
                    logger.warning(f"matcher '{model}' rate-limited, trying next")
                else:
                    logger.warning(f"matcher '{model}' failed: {exc}")
        if response is None:
            raise PipelineError(
                STAGE_SYNC,
                f"all matcher models failed: {last_error}",
                cause=last_error,
            )

        try:
            data = json.loads(_strip_fences(response.text or ""))
        except json.JSONDecodeError as exc:
            raise PipelineError(
                STAGE_SYNC,
                f"matcher returned non-JSON: {(response.text or '')[:200]!r}",
                cause=exc,
            ) from exc

        for entry in data.get("matches") or []:
            try:
                plan_local = int(entry["plan"])
                sent_list = [int(x) for x in entry.get("sentences") or []]
            except (KeyError, TypeError, ValueError):
                continue
            plan_global = start + plan_local
            if plan_global >= len(plan_images):
                continue
            sent_global = sorted({
                window_start + s for s in sent_list
                if 0 <= s < len(window)
            })
            matches[plan_global] = sent_global
            if sent_global:
                sent_cursor = max(sent_cursor, sent_global[-1] + 1)

    return matches


# ---------------------------------------------------------------------------
# Fuzzy fallback
# ---------------------------------------------------------------------------

def _best_sentence(
    target: str,
    sentences: List[Sentence],
    cursor: int,
    lookahead: int = 30,
) -> Tuple[Optional[int], float]:
    if cursor >= len(sentences):
        return None, 0.0
    norm_target = normalize_for_match(target)
    if not norm_target:
        return None, 0.0
    best_idx: Optional[int] = None
    best_ratio = 0.0
    end = min(len(sentences), cursor + lookahead)
    for i in range(cursor, end):
        norm_sent = sentences[i].normalized()
        if not norm_sent:
            continue
        if norm_target in norm_sent or norm_sent in norm_target:
            return i, 1.0
        ratio = SequenceMatcher(None, norm_target, norm_sent).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
    return best_idx, best_ratio


def _fuzzy_match(
    plan_images: List[PlanImage],
    sentences: List[Sentence],
    threshold: float,
) -> Dict[int, List[int]]:
    matches: Dict[int, List[int]] = {}
    cursor = 0
    for idx, img in enumerate(plan_images):
        target = (img.story_line or "").strip()
        if not target:
            continue
        sentence_idx, ratio = _best_sentence(target, sentences, cursor)
        if sentence_idx is None or ratio < threshold:
            continue
        matches[idx] = [sentence_idx]
        cursor = sentence_idx + 1
    return matches


# ---------------------------------------------------------------------------
# Fill unmatched plan indices into gap slots
# ---------------------------------------------------------------------------

def _raw_span_from_sentences(
    sentence_indices: List[int],
    sentences: List[Sentence],
) -> Optional[Tuple[float, float]]:
    if not sentence_indices:
        return None
    valid = [i for i in sentence_indices if 0 <= i < len(sentences)]
    if not valid:
        return None
    return sentences[min(valid)].start, sentences[max(valid)].end


def _assign_raw_spans(
    plan_count: int,
    assignment: Dict[int, List[int]],
    sentences: List[Sentence],
    duration: float,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    For each plan index, produce a raw (start, end) span.

    - matched indices: from sentence boundaries
    - unmatched indices: fraction of the gap between neighboring matched
      indices, or of [0, duration] when no neighbor exists
    """
    raw: List[Optional[Tuple[float, float]]] = [None] * plan_count
    for idx, sent_list in assignment.items():
        if 0 <= idx < plan_count:
            raw[idx] = _raw_span_from_sentences(sent_list, sentences)

    unmatched = [i for i in range(plan_count) if raw[i] is None]
    if not unmatched:
        return [r for r in raw if r is not None], unmatched

    matched = [i for i in range(plan_count) if raw[i] is not None]
    if not matched:
        # Nothing matched at all — defer decision to caller
        return [r for r in raw if r is not None], unmatched

    for u in unmatched:
        prev_m = max((m for m in matched if m < u), default=None)
        next_m = min((m for m in matched if m > u), default=None)

        gap_start = raw[prev_m][1] if prev_m is not None else 0.0      # type: ignore[index]
        gap_end = raw[next_m][0] if next_m is not None else duration   # type: ignore[index]
        gap_end = max(gap_end, gap_start + 0.05)

        siblings = [
            i for i in unmatched
            if (prev_m is None or i > prev_m) and (next_m is None or i < next_m)
        ]
        slot_count = max(len(siblings), 1)
        pos = siblings.index(u)
        slot = (gap_end - gap_start) / slot_count
        raw[u] = (gap_start + pos * slot, gap_start + (pos + 1) * slot)

    return [r for r in raw if r is not None], unmatched


# ---------------------------------------------------------------------------
# Apply hold model and split shared sentences across consecutive images
# ---------------------------------------------------------------------------

def _finalize_spans(
    spans: List[Tuple[float, float]],
    plan_count: int,
    duration: float,
) -> List[Tuple[float, float]]:
    if not spans:
        return []
    spans = list(spans)
    spans[0] = (0.0, spans[0][1])
    for i in range(len(spans) - 1):
        s, e = spans[i]
        ns, ne = spans[i + 1]
        if e < ns:
            # extend the current scene to hold until the next begins
            spans[i] = (s, ns)
        elif e > ns:
            # trim the next scene so no overlap
            spans[i + 1] = (e, max(ne, e + 0.05))
    last_s, last_e = spans[-1]
    if duration > 0 and last_e < duration:
        spans[-1] = (last_s, duration)
    return spans


def _split_equal_spans(
    scenes: List[MatchedScene],
    raw_spans: List[Tuple[float, float]],
) -> None:
    """If consecutive scenes share the exact same raw span (multiple
    images for one narration beat), split that span evenly among them."""
    i = 0
    while i < len(scenes):
        j = i
        while j + 1 < len(scenes) and raw_spans[j + 1] == raw_spans[i]:
            j += 1
        if j > i:
            start, end = raw_spans[i]
            span = max(end - start, 0.05)
            slice_dur = span / (j - i + 1)
            for k, idx in enumerate(range(i, j + 1)):
                raw_spans[idx] = (
                    start + k * slice_dur,
                    start + (k + 1) * slice_dur,
                )
        i = j + 1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def match_plan_to_transcript(
    plan: EditPlan,
    sentences: List[Sentence],
    progress: ProgressFn = None,
) -> MatchResult:
    if not sentences:
        raise PipelineError(STAGE_SYNC, "no sentences to align against")

    kept = plan.ordered_kept()
    if not kept:
        raise PipelineError(STAGE_SYNC, "plan has no kept images")

    duration = sentences[-1].end
    if duration <= 0:
        raise PipelineError(STAGE_SYNC, "transcript duration is zero")

    cfg = matcher_config()
    warnings: List[str] = []

    assignment: Dict[int, List[int]] = {}
    method = "none"

    if progress:
        progress("semantic match (Gemini)")
    try:
        assignment = _gemini_match(kept, sentences, cfg)
        ok, reason = _is_monotonic(assignment, len(kept))
        if ok and assignment:
            method = "gemini"
        else:
            warnings.append(f"Gemini match discarded ({reason or 'empty'})")
            assignment = {}
    except Exception as exc:
        warnings.append(f"Gemini match failed: {exc}")
        assignment = {}

    if not assignment:
        if progress:
            progress("fuzzy text match")
        assignment = _fuzzy_match(kept, sentences, cfg.min_confidence)
        if assignment:
            method = "fuzzy"
        else:
            warnings.append("fuzzy match produced no alignments")

    if not assignment:
        if cfg.allow_fuzzy_fallback:
            # last resort: evenly divide the audio; every scene gets a slot
            warnings.append(
                "no plan line could be aligned to the transcript; "
                "falling back to evenly distributed slots"
            )
            span = duration / len(kept)
            assignment = {i: [] for i in range(len(kept))}
            raw_spans: List[Tuple[float, float]] = [
                (i * span, (i + 1) * span) for i in range(len(kept))
            ]
            method = "even"
            scenes = [
                MatchedScene(
                    image=kept[i],
                    start=raw_spans[i][0],
                    end=raw_spans[i][1],
                    sentence_indices=[],
                    method="even",
                    confidence=0.0,
                )
                for i in range(len(kept))
            ]
        else:
            raise PipelineError(
                STAGE_SYNC,
                "could not align any plan image to the transcript",
                hint="refine story_line text in edit_plan.json to match the narration",
            )
    else:
        spans_matched, unmatched = _assign_raw_spans(
            len(kept), assignment, sentences, duration,
        )
        raw_spans = list(spans_matched)
        scenes = [
            MatchedScene(
                image=kept[i],
                start=raw_spans[i][0],
                end=raw_spans[i][1],
                sentence_indices=assignment.get(i, []),
                method=method if i in assignment else "slot",
                confidence=1.0 if i in assignment else 0.4,
            )
            for i in range(len(kept))
        ]
        if unmatched:
            warnings.append(
                f"{len(unmatched)} plan image(s) had no direct sentence match "
                f"and were placed in slot gaps: {unmatched}"
            )

    _split_equal_spans(scenes, raw_spans)
    final_spans = _finalize_spans(raw_spans, len(kept), duration)

    for scene, (s, e) in zip(scenes, final_spans):
        scene.start = s
        scene.end = e
        if scene.end - scene.start < 0.05:
            scene.warnings.append(
                f"scene too short ({scene.end - scene.start:.3f}s)"
            )

    used = sorted({i for lst in assignment.values() for i in lst})
    unused = [i for i in range(len(sentences)) if i not in used]
    if unused:
        warnings.append(f"{len(unused)} transcript sentence(s) had no image mapped")

    logger.info(
        f"matched {len(scenes)} scenes (method={method}, "
        f"unmatched_plan={len([i for i in range(len(kept)) if i not in assignment])})"
    )

    return MatchResult(
        scenes=scenes,
        method=method,
        warnings=warnings,
        unmatched_plan_indices=[
            i for i in range(len(kept)) if i not in assignment
        ],
        unused_sentence_indices=unused,
  )
