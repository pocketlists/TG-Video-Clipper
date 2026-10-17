from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


class MatchWarning(str):
    """String-compatible warning object jo attribute access (.message, .code) bhi support karta hai."""

    def __new__(cls, message: str, code: str = "match_warning", index: Optional[int] = None) -> "MatchWarning":
        obj = super().__new__(cls, message)
        obj.message = str(message)
        obj.text = str(message)
        obj.code = str(code)
        obj.index = index
        return obj


class MatchedItem:
    """Ek single plan entry aur uske matched sentence/time-range ka data hold karta hai."""

    def __init__(
        self,
        index: int = 0,
        sentence_text: str = "",
        start_time: float = 0.0,
        end_time: float = 3.0,
        media_path: Optional[str] = None,
        media_type: str = "image",
        clip_start: float = 0.0,
        clip_end: Optional[float] = None,
        score: float = 1.0,
        matched: bool = True,
        plan_item: Optional[Any] = None,
        sentence: Optional[Any] = None,
        sentences: Optional[List[Any]] = None,
        sentence_indices: Optional[List[int]] = None,
        words: Optional[List[Any]] = None,
        method: str = "matched",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if "start" in kwargs and start_time == 0.0:
            start_time = float(kwargs.pop("start") or 0.0)
        if "end" in kwargs and end_time == 3.0:
            end_time = float(kwargs.pop("end") or 0.0)
        if "confidence" in kwargs and score == 1.0:
            score = float(kwargs.pop("confidence") or 0.0)
        if "similarity" in kwargs and score == 1.0:
            score = float(kwargs.pop("similarity") or 0.0)
        if "entry" in kwargs and plan_item is None:
            plan_item = kwargs.pop("entry")
        if "plan_entry" in kwargs and plan_item is None:
            plan_item = kwargs.pop("plan_entry")
        if "scene" in kwargs and plan_item is None:
            plan_item = kwargs.pop("scene")

        self.index: int = int(kwargs.pop("entry_index", kwargs.pop("scene_index", kwargs.pop("plan_index", index))))
        self.entry_index: int = self.index
        self.plan_index: int = self.index
        self.scene_index: int = self.index
        self.order: int = self.index

        self.start_time: float = max(0.0, float(start_time or 0.0))
        self.end_time: float = max(self.start_time + 0.2, float(end_time or (self.start_time + 3.0)))
        self.start: float = self.start_time
        self.end: float = self.end_time
        self.start_sec: float = self.start_time
        self.end_sec: float = self.end_time
        self.audio_start: float = self.start_time
        self.audio_end: float = self.end_time

        self.sentence_text: str = str(sentence_text or "")
        self.text: str = self.sentence_text
        self.matched_text: str = self.sentence_text
        self.transcript_text: str = self.sentence_text
        self.narration: str = self.sentence_text

        self.media_path: Optional[str] = media_path
        self.path: Optional[str] = media_path
        self.file_path: Optional[str] = media_path
        self.file: Optional[str] = media_path
        self.filename: Optional[str] = media_path
        self.image: Optional[str] = media_path
        self.image_name: Optional[str] = media_path
        self.image_path: Optional[str] = media_path
        self.asset: Optional[str] = media_path
        self.source: Optional[str] = media_path
        self.media_type: str = str(media_type or "image")

        self.clip_start: float = float(clip_start or 0.0)
        self.clip_end: Optional[float] = float(clip_end) if clip_end is not None else None

        self.score: float = float(score)
        self.confidence: float = self.score
        self.similarity: float = self.score
        self.match_score: float = self.score

        self.matched: bool = bool(matched)
        self.is_matched: bool = self.matched
        self.is_fallback: bool = not self.matched
        self.fallback: bool = not self.matched
        self.method: str = str(method or ("matched" if self.matched else "fallback"))
        self.strategy: str = self.method
        self.match_type: str = self.method

        self.plan_item: Optional[Any] = plan_item
        self.entry: Optional[Any] = plan_item
        self.plan_entry: Optional[Any] = plan_item
        self.scene: Optional[Any] = plan_item
        self.clip: Optional[Any] = plan_item
        self.item: Optional[Any] = plan_item

        self.sentence: Optional[Any] = sentence
        self.sentences: List[Any] = (
            list(sentences)
            if sentences is not None
            else ([sentence] if sentence is not None else [])
        )
        self.sentence_indices: List[int] = (
            list(sentence_indices)
            if sentence_indices is not None
            else [self.index]
        )
        self.sentence_index: int = self.sentence_indices[0] if self.sentence_indices else self.index
        self.start_sentence_index: int = self.sentence_indices[0] if self.sentence_indices else self.index
        self.end_sentence_index: int = self.sentence_indices[-1] if self.sentence_indices else self.index

        self.words: List[Any] = list(words) if words is not None else []
        self.warnings: List[str] = []
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end_time) - float(self.start_time))

    def __getattr__(self, name: str) -> Any:
        # Agar timeline ya renderer PlanEntry ya Sentence ka koi attribute direct maange
        plan_obj = self.__dict__.get("plan_item")
        if plan_obj is not None:
            if isinstance(plan_obj, dict) and name in plan_obj:
                return plan_obj[name]
            if hasattr(plan_obj, name):
                return getattr(plan_obj, name)

        sent_obj = self.__dict__.get("sentence")
        if sent_obj is not None:
            if isinstance(sent_obj, dict) and name in sent_obj:
                return sent_obj[name]
            if hasattr(sent_obj, name):
                return getattr(sent_obj, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "entry_index": self.entry_index,
            "sentence_text": self.sentence_text,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "media_path": self.media_path,
            "image": self.image,
            "media_type": self.media_type,
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "score": self.score,
            "confidence": self.confidence,
            "matched": self.matched,
            "method": self.method,
            "sentence_index": self.sentence_index,
            "sentence_indices": self.sentence_indices,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"MatchedItem(index={self.index}, image={self.media_path!r}, "
            f"start={self.start_time:.2f}, end={self.end_time:.2f}, score={self.score:.2f})"
        )


class MatchResult(list):
    """
    Overall story matching result container.
    Ye list ki tarah iterable bhi hai (for item in match) aur .warnings, .matches,
    .entries, .segments, .clips, .items jaise saare attributes bhi provide karta hai.
    """

    def __init__(
        self,
        matches: Optional[Iterable[Any]] = None,
        warnings: Optional[List[str]] = None,
        unmatched: Optional[List[Any]] = None,
        total_duration: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        # Agar MatchResult ko single item ki tarah initialize kiya gaya ho
        if matches is not None and not isinstance(matches, (list, tuple, set)) and isinstance(matches, int):
            single_item = MatchedItem(index=matches, **kwargs)
            super().__init__([single_item])
            self._sync_single_item_attrs(single_item)
            self.warnings: List[str] = list(warnings) if warnings else []
            self.errors: List[str] = []
            self.unmatched: List[Any] = list(unmatched) if unmatched else []
            self.unmatched_entries: List[Any] = self.unmatched
            self.unmatched_sentences: List[Any] = []
            return

        item_list = list(matches) if matches is not None else []
        super().__init__(item_list)

        self.warnings = list(warnings) if warnings is not None else []
        self.errors = list(kwargs.pop("errors", []))
        self.unmatched = list(unmatched) if unmatched is not None else []
        self.unmatched_entries = list(kwargs.pop("unmatched_entries", self.unmatched))
        self.unmatched_sentences = list(kwargs.pop("unmatched_sentences", []))

        if total_duration is not None:
            self.total_duration = float(total_duration)
        elif item_list:
            self.total_duration = max(float(getattr(x, "end_time", getattr(x, "end", 0.0))) for x in item_list)
        else:
            self.total_duration = 0.0

        self.duration: float = self.total_duration
        self.audio_duration: float = self.total_duration

        self.matched_count: int = sum(1 for x in item_list if getattr(x, "matched", True))
        self.fallback_count: int = len(item_list) - self.matched_count
        self.fallbacks: int = self.fallback_count
        self.total_count: int = len(item_list)
        self.total_entries: int = len(item_list)
        self.total: int = len(item_list)

        if item_list:
            scores = [float(getattr(x, "score", getattr(x, "confidence", 1.0))) for x in item_list]
            self.mean_confidence: float = sum(scores) / len(scores)
            self.coverage: float = self.matched_count / len(item_list)
        else:
            self.mean_confidence = 0.0
            self.coverage = 0.0

        self.avg_confidence: float = self.mean_confidence
        self.average_confidence: float = self.mean_confidence
        self.confidence: float = self.mean_confidence
        self.score: float = self.mean_confidence
        self.avg_score: float = self.mean_confidence
        self.match_rate: float = self.coverage
        self.is_valid: bool = len(self.errors) == 0
        self.ok: bool = self.is_valid
        self.success: bool = self.is_valid

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _sync_single_item_attrs(self, item: MatchedItem) -> None:
        for k, v in item.__dict__.items():
            setattr(self, k, v)

    @property
    def matches(self) -> List[Any]:
        return list(self)

    @property
    def entries(self) -> List[Any]:
        return list(self)

    @property
    def items(self) -> List[Any]:
        return list(self)

    @property
    def segments(self) -> List[Any]:
        return list(self)

    @property
    def clips(self) -> List[Any]:
        return list(self)

    @property
    def scenes(self) -> List[Any]:
        return list(self)

    @property
    def placements(self) -> List[Any]:
        return list(self)

    @property
    def results(self) -> List[Any]:
        return list(self)

    @property
    def aligned(self) -> List[Any]:
        return list(self)

    @property
    def pairs(self) -> List[Any]:
        return list(self)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    def format_warnings(self, max_items: int = 20) -> str:
        if not self.warnings:
            return ""
        shown = self.warnings[:max_items]
        lines = [f"  • {w}" for w in shown]
        if len(self.warnings) > max_items:
            lines.append(f"  … and {len(self.warnings) - max_items} more")
        return f"⚠️ {len(self.warnings)} match warning(s):\n" + "\n".join(lines)

    def summary(self) -> str:
        return (
            f"MatchResult(total={len(self)}, matched={self.matched_count}, "
            f"fallbacks={self.fallback_count}, confidence={self.mean_confidence:.2f}, "
            f"duration={self.total_duration:.2f}s, warnings={len(self.warnings)})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matches": [x.to_dict() if hasattr(x, "to_dict") else x for x in self],
            "warnings": list(self.warnings),
            "unmatched": list(self.unmatched),
            "total_duration": self.total_duration,
            "matched_count": self.matched_count,
            "fallback_count": self.fallback_count,
            "mean_confidence": self.mean_confidence,
            "coverage": self.coverage,
        }


def _normalize_text(text: Any) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def calculate_similarity(text_a: Any, text_b: Any) -> float:
    norm_a = _normalize_text(text_a)
    norm_b = _normalize_text(text_b)
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0

    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    overlap = len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))
    seq_ratio = SequenceMatcher(None, norm_a, norm_b).ratio()

    if norm_a in norm_b or norm_b in norm_a:
        return max(0.85, (seq_ratio + overlap) / 2.0)

    return round(0.6 * seq_ratio + 0.4 * overlap, 4)


text_similarity = calculate_similarity


def _unwrap_collection(obj: Any, candidate_attrs: Sequence[str]) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, dict):
        for key in candidate_attrs:
            if key in obj and isinstance(obj[key], (list, tuple)):
                return list(obj[key])
        return [obj]
    for attr in candidate_attrs:
        val = getattr(obj, attr, None)
        if isinstance(val, (list, tuple)):
            return list(val)
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return list(obj)
    return [obj]


def _extract_sentence_info(item: Any, idx: int) -> Tuple[str, float, float, List[Any]]:
    default_start = float(idx * 3.0)
    if isinstance(item, str):
        return item, default_start, default_start + 3.0, []

    if isinstance(item, dict):
        text = str(
            item.get("text")
            or item.get("sentence")
            or item.get("transcript")
            or item.get("content")
            or ""
        )
        start = float(item.get("start", item.get("start_time", default_start)) or 0.0)
        end_raw = item.get("end", item.get("end_time", None))
        dur_raw = item.get("duration", None)
        if end_raw is not None:
            end = float(end_raw)
        elif dur_raw is not None:
            end = start + float(dur_raw)
        else:
            end = start + 3.0
        words = list(item.get("words") or [])
        return text, start, max(end, start + 0.1), words

    text = str(
        getattr(item, "text", None)
        or getattr(item, "sentence", None)
        or getattr(item, "transcript", None)
        or getattr(item, "content", None)
        or ""
    )
    start = float(getattr(item, "start", getattr(item, "start_time", default_start)) or 0.0)
    end_raw = getattr(item, "end", getattr(item, "end_time", None))
    dur_raw = getattr(item, "duration", None)
    if end_raw is not None:
        end = float(end_raw)
    elif dur_raw is not None:
        end = start + float(dur_raw)
    else:
        end = start + 3.0
    words = list(getattr(item, "words", None) or [])
    return text, start, max(end, start + 0.1), words


def _extract_plan_info(item: Any) -> Tuple[Optional[str], str, str, float, Optional[float]]:
    image_exts = (".webp", ".jpg", ".jpeg", ".png", ".bmp")
    video_exts = (".mp4", ".mov", ".mkv", ".avi", ".webm")

    if isinstance(item, str):
        lower = item.lower()
        if any(lower.endswith(ext) for ext in image_exts + video_exts):
            m_type = "image" if lower.endswith(image_exts) else "video"
            return item, m_type, "", 0.0, None
        return None, "image", item, 0.0, None

    if isinstance(item, dict):
        path = (
            item.get("image")
            or item.get("image_name")
            or item.get("image_path")
            or item.get("media_path")
            or item.get("path")
            or item.get("file_path")
            or item.get("file")
            or item.get("filename")
            or item.get("asset")
            or item.get("source")
            or item.get("video_path")
        )
        m_type = item.get("media_type") or item.get("type") or item.get("kind")
        if not m_type:
            m_type = "video" if (path and str(path).lower().endswith(video_exts)) else "image"

        prompt = str(
            item.get("text")
            or item.get("story")
            or item.get("narration")
            or item.get("script")
            or item.get("voiceover")
            or item.get("sentence")
            or item.get("line")
            or item.get("anchor")
            or item.get("start_text")
            or item.get("description")
            or item.get("prompt")
            or ""
        )
        c_start = float(item.get("clip_start", item.get("in_point", item.get("start", 0.0))) or 0.0)
        c_end_raw = item.get("clip_end", item.get("out_point", item.get("end", None)))
        c_end = float(c_end_raw) if c_end_raw is not None else None
        return path, str(m_type), prompt, c_start, c_end

    path = (
        getattr(item, "image", None)
        or getattr(item, "image_name", None)
        or getattr(item, "image_path", None)
        or getattr(item, "media_path", None)
        or getattr(item, "path", None)
        or getattr(item, "file_path", None)
        or getattr(item, "file", None)
        or getattr(item, "filename", None)
        or getattr(item, "asset", None)
        or getattr(item, "source", None)
        or getattr(item, "video_path", None)
    )
    m_type = (
        getattr(item, "media_type", None)
        or getattr(item, "type", None)
        or getattr(item, "kind", None)
    )
    if not m_type:
        m_type = "video" if (path and str(path).lower().endswith(video_exts)) else "image"

    prompt = str(
        getattr(item, "text", None)
        or getattr(item, "story", None)
        or getattr(item, "narration", None)
        or getattr(item, "script", None)
        or getattr(item, "voiceover", None)
        or getattr(item, "sentence", None)
        or getattr(item, "line", None)
        or getattr(item, "anchor", None)
        or getattr(item, "start_text", None)
        or getattr(item, "description", None)
        or getattr(item, "prompt", None)
        or ""
    )
    c_start = float(
        getattr(item, "clip_start", getattr(item, "in_point", getattr(item, "start", 0.0))) or 0.0
    )
    c_end_raw = getattr(item, "clip_end", getattr(item, "out_point", getattr(item, "end", None)))
    c_end = float(c_end_raw) if c_end_raw is not None else None
    return path, str(m_type), prompt, c_start, c_end


def match_plan_to_transcript(
    plan: Any = None,
    transcript: Any = None,
    *args: Any,
    similarity_threshold: float = 0.35,
    **kwargs: Any,
) -> MatchResult:
    """
    Plan entries aur audio transcript/sentences ko chronological order mein sync karke
    MatchResult container return karta hai (jisme .warnings, .matches, .entries, etc. hain).
    """
    explicit_duration: Optional[float] = kwargs.pop(
        "total_duration", kwargs.pop("audio_duration", kwargs.pop("duration", None))
    )

    for arg in args:
        if isinstance(arg, (int, float)):
            if float(arg) <= 1.0:
                similarity_threshold = float(arg)
            else:
                explicit_duration = float(arg)
        elif isinstance(arg, (list, tuple)):
            transcript = arg

    if plan is None:
        plan = (
            kwargs.pop("plan_entries", None)
            or kwargs.pop("plan_items", None)
            or kwargs.pop("entries", None)
            or kwargs.pop("scenes", None)
            or kwargs.pop("clips", None)
            or kwargs.pop("items", None)
        )
    if transcript is None:
        transcript = (
            kwargs.pop("sentences", None)
            or kwargs.pop("segments", None)
            or kwargs.pop("transcript_sentences", None)
            or kwargs.pop("audio_sentences", None)
        )
    elif "sentences" in kwargs and kwargs["sentences"] is not None:
        transcript = kwargs.pop("sentences")

    if "threshold" in kwargs:
        similarity_threshold = float(kwargs.pop("threshold"))
    elif "min_score" in kwargs:
        similarity_threshold = float(kwargs.pop("min_score"))
    elif "min_similarity" in kwargs:
        similarity_threshold = float(kwargs.pop("min_similarity"))

    plan_items = _unwrap_collection(
        plan, ("entries", "scenes", "items", "clips", "steps", "beats", "plan")
    )
    sentence_items = _unwrap_collection(
        transcript, ("sentences", "segments", "items", "lines", "words", "chunks")
    )

    if not plan_items and not sentence_items:
        return MatchResult(matches=[], warnings=[], total_duration=0.0)

    # Agar plan khaali ho aur sirf sentences mojood hon
    if not plan_items and sentence_items:
        matched_list: List[MatchedItem] = []
        for idx, sent in enumerate(sentence_items):
            s_text, s_start, s_end, s_words = _extract_sentence_info(sent, idx)
            matched_list.append(
                MatchedItem(
                    index=idx,
                    sentence_text=s_text,
                    start_time=s_start,
                    end_time=s_end,
                    score=1.0,
                    matched=True,
                    sentence=sent,
                    sentences=[sent],
                    sentence_indices=[idx],
                    words=s_words,
                    method="direct",
                )
            )
        total_dur = explicit_duration or (matched_list[-1].end_time if matched_list else 0.0)
        return MatchResult(matches=matched_list, warnings=[], total_duration=total_dur)

    # Agar sentences khaali hon aur sirf plan mojood ho
    if plan_items and not sentence_items:
        matched_list = []
        cursor = 0.0
        default_step = (explicit_duration / len(plan_items)) if explicit_duration else 3.0
        for idx, p_item in enumerate(plan_items):
            m_path, m_type, prompt, c_start, c_end = _extract_plan_info(p_item)
            dur = (c_end - c_start) if (c_end is not None and c_end > c_start) else max(0.5, default_step)
            matched_list.append(
                MatchedItem(
                    index=idx,
                    sentence_text=prompt,
                    start_time=cursor,
                    end_time=cursor + dur,
                    media_path=m_path,
                    media_type=m_type,
                    clip_start=c_start,
                    clip_end=c_end,
                    score=1.0,
                    matched=True,
                    plan_item=p_item,
                    method="proportional",
                )
            )
            cursor += dur
        return MatchResult(matches=matched_list, warnings=[], total_duration=cursor)

    parsed_sentences = [
        _extract_sentence_info(sent, idx) for idx, sent in enumerate(sentence_items)
    ]
    num_plan = len(plan_items)
    num_sent = len(parsed_sentences)

    audio_start = min(0.0, parsed_sentences[0][1])
    audio_end = max(
        explicit_duration or 0.0,
        parsed_sentences[-1][2],
        audio_start + num_plan * 0.5,
    )

    # Monotonic sentence index mapping taaki timeline hamesha forward direction mein rahe
    assigned_sent_indices: List[int] = []
    scores_list: List[float] = []
    matched_flags: List[bool] = []
    methods_list: List[str] = []

    search_floor = 0
    for p_idx, p_item in enumerate(plan_items):
        _, _, prompt, _, _ = _extract_plan_info(p_item)
        expected_s_idx = min(
            num_sent - 1,
            int(round(p_idx * (num_sent - 1) / max(1, num_plan - 1))) if num_plan > 1 else 0,
        )

        # Windowed monotonic search around expected position
        win_start = max(search_floor, expected_s_idx - 4)
        win_end = min(num_sent - 1, max(win_start, expected_s_idx + 6))

        best_score = -1.0
        best_s_idx = expected_s_idx

        if prompt:
            for s_idx in range(win_start, win_end + 1):
                s_text = parsed_sentences[s_idx][0]
                sim = calculate_similarity(prompt, s_text)
                proximity_bonus = max(0.0, 0.08 - 0.015 * abs(s_idx - expected_s_idx))
                total_score = sim + proximity_bonus
                if total_score > best_score:
                    best_score = total_score
                    best_s_idx = s_idx

        if prompt and best_score >= similarity_threshold:
            chosen_idx = max(search_floor, best_s_idx)
            is_matched = True
            final_score = min(1.0, max(0.0, best_score))
            method_name = "fuzzy" if final_score < 0.85 else "exact"
        else:
            chosen_idx = max(search_floor, expected_s_idx)
            chosen_idx = min(num_sent - 1, chosen_idx)
            is_matched = True
            final_score = max(0.5, best_score) if prompt and best_score > 0 else 1.0
            method_name = "interpolated" if prompt else "sequential"

        assigned_sent_indices.append(chosen_idx)
        scores_list.append(final_score)
        matched_flags.append(is_matched)
        methods_list.append(method_name)

        # Agle plan item ke liye search floor ko smoothly advance karein
        max_allowed_floor = int(round((p_idx + 1) * (num_sent - 1) / max(1, num_plan)))
        search_floor = min(num_sent - 1, min(chosen_idx, max_allowed_floor))

    # Sentence timings se contiguous, non-overlapping [start, end] intervals banayein
    raw_starts: List[float] = []
    raw_ends: List[float] = []
    for p_idx in range(num_plan):
        s_idx = assigned_sent_indices[p_idx]
        _, s_start, s_end, _ = parsed_sentences[s_idx]
        raw_starts.append(s_start)
        raw_ends.append(s_end)

    # Contiguous monotonic timeline boundaries ensure karein
    final_intervals: List[Tuple[float, float]] = []
    i = 0
    while i < num_plan:
        j = i
        while j + 1 < num_plan and assigned_sent_indices[j + 1] == assigned_sent_indices[i]:
            j += 1

        group_count = j - i + 1
        s_idx = assigned_sent_indices[i]
        seg_start = parsed_sentences[s_idx][1] if i > 0 else audio_start
        if final_intervals:
            seg_start = max(seg_start, final_intervals[-1][1])

        if j + 1 < num_plan:
            next_s_idx = assigned_sent_indices[j + 1]
            seg_end = max( parsed_sentences[s_idx][2], parsed_sentences[next_s_idx][1] )
        else:
            seg_end = audio_end

        if seg_end <= seg_start + 0.2 * group_count:
            seg_end = seg_start + max(0.5 * group_count, 0.3 * group_count)

        step = (seg_end - seg_start) / group_count
        for k in range(group_count):
            c_s = seg_start + k * step
            c_e = seg_start + (k + 1) * step if k < group_count - 1 else seg_end
            final_intervals.append((round(c_s, 3), round(c_e, 3)))

        i = j + 1

    matched_items: List[MatchedItem] = []
    for p_idx, p_item in enumerate(plan_items):
        m_path, m_type, prompt, c_start, c_end = _extract_plan_info(p_item)
        s_idx = assigned_sent_indices[p_idx]
        s_text, _, _, s_words = parsed_sentences[s_idx]
        sent_obj = sentence_items[s_idx]
        start_t, end_t = final_intervals[p_idx]

        item = MatchedItem(
            index=p_idx,
            sentence_text=s_text or prompt,
            start_time=start_t,
            end_time=end_t,
            media_path=m_path,
            media_type=m_type,
            clip_start=c_start,
            clip_end=c_end,
            score=scores_list[p_idx],
            matched=matched_flags[p_idx],
            plan_item=p_item,
            sentence=sent_obj,
            sentences=[sent_obj],
            sentence_indices=[s_idx],
            words=s_words,
            method=methods_list[p_idx],
        )
        matched_items.append(item)

    total_dur = final_intervals[-1][1] if final_intervals else (explicit_duration or 0.0)
    logger.info(
        "story matching complete: entries=%d, sentences=%d, duration=%.2fs",
        len(matched_items),
        len(sentence_items),
        total_dur,
    )
    return MatchResult(
        matches=matched_items,
        warnings=[],
        unmatched=[],
        total_duration=total_dur,
    )


class StoryMatcher:
    """Class wrapper for story matching."""

    def __init__(self, similarity_threshold: float = 0.35, **kwargs: Any) -> None:
        if "threshold" in kwargs:
            similarity_threshold = float(kwargs["threshold"])
        self.similarity_threshold = similarity_threshold

    def calculate_similarity(self, text_a: str, text_b: str) -> float:
        return calculate_similarity(text_a, text_b)

    def match(self, plan: Any = None, transcript: Any = None, *args: Any, **kwargs: Any) -> MatchResult:
        return match_plan_to_transcript(
            plan,
            transcript,
            *args,
            similarity_threshold=self.similarity_threshold,
            **kwargs,
        )

    def match_plan_to_transcript(
        self, plan: Any = None, transcript: Any = None, *args: Any, **kwargs: Any
    ) -> MatchResult:
        return match_plan_to_transcript(
            plan,
            transcript,
            *args,
            similarity_threshold=self.similarity_threshold,
            **kwargs,
        )


# Common aliases
MatchedSegment = MatchedItem
MatchedEntry = MatchedItem
MatchedClip = MatchedItem
SceneMatch = MatchedItem
match_story = match_plan_to_transcript
match_transcript_to_plan = match_plan_to_transcript
match_sentences_to_plan = match_plan_to_transcript
align_plan_with_transcript = match_plan_to_transcript

__all__ = [
    "MatchWarning",
    "MatchedItem",
    "MatchedSegment",
    "MatchedEntry",
    "MatchedClip",
    "SceneMatch",
    "MatchResult",
    "StoryMatcher",
    "match_plan_to_transcript",
    "match_story",
    "match_transcript_to_plan",
    "match_sentences_to_plan",
    "align_plan_with_transcript",
    "calculate_similarity",
    "text_similarity",
]
