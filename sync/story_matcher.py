from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


class MatchResult:
    """Plan scene/item aur transcript sentence/timing ke match ka result store karta hai."""

    def __init__(
        self,
        index: int = 0,
        sentence_text: str = "",
        start_time: float = 0.0,
        end_time: float = 0.0,
        media_path: Optional[str] = None,
        media_type: str = "video",
        clip_start: float = 0.0,
        clip_end: Optional[float] = None,
        score: float = 1.0,
        matched: bool = True,
        plan_item: Optional[Any] = None,
        sentence: Optional[Any] = None,
        sentences: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Alias support agar caller alag keyword names pass kare
        if "start" in kwargs and start_time == 0.0:
            start_time = float(kwargs.pop("start") or 0.0)
        if "end" in kwargs and end_time == 0.0:
            end_time = float(kwargs.pop("end") or 0.0)
        if "confidence" in kwargs and score == 1.0:
            score = float(kwargs.pop("confidence") or 0.0)
        if "similarity" in kwargs and score == 1.0:
            score = float(kwargs.pop("similarity") or 0.0)
        if "scene" in kwargs and plan_item is None:
            plan_item = kwargs.pop("scene")
        if "clip" in kwargs and plan_item is None:
            plan_item = kwargs.pop("clip")
        if "item" in kwargs and plan_item is None:
            plan_item = kwargs.pop("item")
        if "text" in kwargs and not sentence_text:
            sentence_text = str(kwargs.pop("text") or "")
        if "matched_text" in kwargs and not sentence_text:
            sentence_text = str(kwargs.pop("matched_text") or "")
        if "path" in kwargs and media_path is None:
            media_path = kwargs.pop("path")
        if "file_path" in kwargs and media_path is None:
            media_path = kwargs.pop("file_path")

        self.index: int = int(kwargs.pop("scene_index", kwargs.pop("plan_index", index)))
        self.scene_index: int = self.index
        self.plan_index: int = self.index

        self.sentence_text: str = str(sentence_text or "")
        self.text: str = self.sentence_text
        self.matched_text: str = self.sentence_text

        self.start_time: float = float(start_time or 0.0)
        self.end_time: float = float(
            end_time if end_time and end_time > self.start_time else self.start_time + 3.0
        )
        self.start: float = self.start_time
        self.end: float = self.end_time

        self.media_path: Optional[str] = media_path
        self.path: Optional[str] = media_path
        self.file_path: Optional[str] = media_path
        self.media_type: str = str(media_type or "video")

        self.clip_start: float = float(clip_start or 0.0)
        self.clip_end: Optional[float] = float(clip_end) if clip_end is not None else None

        self.score: float = float(score)
        self.confidence: float = self.score
        self.similarity: float = self.score

        self.matched: bool = bool(matched)
        self.is_matched: bool = self.matched
        self.is_fallback: bool = not self.matched

        self.plan_item: Optional[Any] = plan_item
        self.scene: Optional[Any] = plan_item
        self.clip: Optional[Any] = plan_item
        self.item: Optional[Any] = plan_item

        self.sentence: Optional[Any] = sentence
        self.sentences: List[Any] = (
            list(sentences)
            if sentences is not None
            else ([sentence] if sentence is not None else [])
        )
        self.sentence_indices: List[int] = list(kwargs.pop("sentence_indices", []))
        self.sentence_index: Optional[int] = kwargs.pop(
            "sentence_index",
            self.sentence_indices[0] if self.sentence_indices else self.index,
        )

        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}

        # Extra fields ko direct attribute ke roop mein attach karein
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end_time) - float(self.start_time))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "scene_index": self.scene_index,
            "sentence_text": self.sentence_text,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "media_path": self.media_path,
            "media_type": self.media_type,
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "score": self.score,
            "confidence": self.confidence,
            "matched": self.matched,
            "sentence_index": self.sentence_index,
            "sentence_indices": self.sentence_indices,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"MatchResult(index={self.index}, start={self.start_time:.2f}, "
            f"end={self.end_time:.2f}, score={self.score:.2f}, matched={self.matched})"
        )


def _normalize_text(text: Any) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def calculate_similarity(text_a: Any, text_b: Any) -> float:
    """Do text strings ke beech similarity score (0.0 se 1.0) nikalta hai."""
    norm_a = _normalize_text(text_a)
    norm_b = _normalize_text(text_b)
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0

    # Token overlap + SequenceMatcher ratio ka weighted mix
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    overlap = len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))
    seq_ratio = SequenceMatcher(None, norm_a, norm_b).ratio()

    if norm_a in norm_b or norm_b in norm_a:
        return max(0.85, (seq_ratio + overlap) / 2.0)

    return round(0.6 * seq_ratio + 0.4 * overlap, 4)


text_similarity = calculate_similarity


def _unwrap_collection(obj: Any, candidate_attrs: Sequence[str]) -> List[Any]:
    """Container object (jaise Plan ya Transcript) se items ki list nikalta hai."""
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


def _extract_sentence_info(item: Any, idx: int) -> Tuple[str, float, float]:
    """Transcript sentence ya segment se (text, start_time, end_time) nikalta hai."""
    default_start = float(idx * 3.0)
    if isinstance(item, str):
        return item, default_start, default_start + 3.0

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
        return text, start, max(end, start + 0.1)

    text = str(
        getattr(item, "text", None)
        or getattr(item, "sentence", None)
        or getattr(item, "transcript", None)
        or getattr(item, "content", None)
        or ""
    )
    start = float(
        getattr(item, "start", getattr(item, "start_time", default_start)) or 0.0
    )
    end_raw = getattr(item, "end", getattr(item, "end_time", None))
    dur_raw = getattr(item, "duration", None)
    if end_raw is not None:
        end = float(end_raw)
    elif dur_raw is not None:
        end = start + float(dur_raw)
    else:
        end = start + 3.0
    return text, start, max(end, start + 0.1)


def _extract_plan_info(item: Any) -> Tuple[Optional[str], str, str, float, Optional[float]]:
    """Plan item/scene se (media_path, media_type, script_text, clip_start, clip_end) nikalta hai."""
    if isinstance(item, str):
        if any(item.lower().endswith(ext) for ext in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".jpg", ".jpeg", ".png", ".webp")):
            m_type = "image" if item.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) else "video"
            return item, m_type, "", 0.0, None
        return None, "video", item, 0.0, None

    if isinstance(item, dict):
        path = (
            item.get("media_path")
            or item.get("path")
            or item.get("file_path")
            or item.get("file")
            or item.get("source")
            or item.get("video_path")
            or item.get("image_path")
        )
        m_type = str(item.get("media_type") or item.get("type") or item.get("kind") or "video")
        prompt = str(
            item.get("text")
            or item.get("narration")
            or item.get("script")
            or item.get("voiceover")
            or item.get("sentence")
            or item.get("line")
            or item.get("description")
            or item.get("prompt")
            or ""
        )
        c_start = float(item.get("clip_start", item.get("in_point", item.get("start", 0.0))) or 0.0)
        c_end_raw = item.get("clip_end", item.get("out_point", item.get("end", None)))
        c_end = float(c_end_raw) if c_end_raw is not None else None
        return path, m_type, prompt, c_start, c_end

    path = (
        getattr(item, "media_path", None)
        or getattr(item, "path", None)
        or getattr(item, "file_path", None)
        or getattr(item, "file", None)
        or getattr(item, "source", None)
        or getattr(item, "video_path", None)
        or getattr(item, "image_path", None)
    )
    m_type = str(
        getattr(item, "media_type", None)
        or getattr(item, "type", None)
        or getattr(item, "kind", None)
        or "video"
    )
    prompt = str(
        getattr(item, "text", None)
        or getattr(item, "narration", None)
        or getattr(item, "script", None)
        or getattr(item, "voiceover", None)
        or getattr(item, "sentence", None)
        or getattr(item, "line", None)
        or getattr(item, "description", None)
        or getattr(item, "prompt", None)
        or ""
    )
    c_start = float(
        getattr(item, "clip_start", getattr(item, "in_point", getattr(item, "start", 0.0)))
        or 0.0
    )
    c_end_raw = getattr(item, "clip_end", getattr(item, "out_point", getattr(item, "end", None)))
    c_end = float(c_end_raw) if c_end_raw is not None else None
    return path, m_type, prompt, c_start, c_end


def match_plan_to_transcript(
    plan: Any = None,
    transcript: Any = None,
    similarity_threshold: float = 0.40,
    **kwargs: Any,
) -> List[MatchResult]:
    """
    Plan scenes/items ko audio transcript sentences ke saath sync aur match karta hai.
    Positional aur keyword arguments dono ko flexibly handle karta hai.
    """
    if plan is None:
        plan = (
            kwargs.pop("plan_items", None)
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
    if "threshold" in kwargs:
        similarity_threshold = float(kwargs.pop("threshold"))
    elif "min_score" in kwargs:
        similarity_threshold = float(kwargs.pop("min_score"))
    elif "min_similarity" in kwargs:
        similarity_threshold = float(kwargs.pop("min_similarity"))

    plan_items = _unwrap_collection(
        plan, ("scenes", "items", "clips", "steps", "beats", "plan", "entries")
    )
    sentence_items = _unwrap_collection(
        transcript, ("sentences", "segments", "items", "lines", "words", "chunks")
    )

    if not plan_items and not sentence_items:
        return []

    # Agar plan khaali ho par transcript mojood ho
    if not plan_items and sentence_items:
        results: List[MatchResult] = []
        for idx, sent in enumerate(sentence_items):
            s_text, s_start, s_end = _extract_sentence_info(sent, idx)
            results.append(
                MatchResult(
                    index=idx,
                    sentence_text=s_text,
                    start_time=s_start,
                    end_time=s_end,
                    score=1.0,
                    matched=True,
                    sentence=sent,
                    sentences=[sent],
                    sentence_indices=[idx],
                )
            )
        return results

    # Agar transcript khaali ho par plan mojood ho
    if plan_items and not sentence_items:
        results = []
        cursor = 0.0
        for idx, p_item in enumerate(plan_items):
            m_path, m_type, prompt, c_start, c_end = _extract_plan_info(p_item)
            dur = (c_end - c_start) if (c_end is not None and c_end > c_start) else 3.0
            results.append(
                MatchResult(
                    index=idx,
                    sentence_text=prompt,
                    start_time=cursor,
                    end_time=cursor + dur,
                    media_path=m_path,
                    media_type=m_type,
                    clip_start=c_start,
                    clip_end=c_end,
                    score=0.0,
                    matched=False,
                    plan_item=p_item,
                )
            )
            cursor += dur
        return results

    parsed_sentences = [
        _extract_sentence_info(sent, idx) for idx, sent in enumerate(sentence_items)
    ]
    used_sentence_indices = set()
    results = []

    for p_idx, p_item in enumerate(plan_items):
        m_path, m_type, prompt, c_start, c_end = _extract_plan_info(p_item)

        best_score = -1.0
        best_s_idx: Optional[int] = None

        for s_idx, (s_text, _, _) in enumerate(parsed_sentences):
            if prompt:
                score = calculate_similarity(prompt, s_text)
                if s_idx not in used_sentence_indices:
                    score += 0.05
            else:
                score = 1.0 if s_idx == p_idx else 0.0

            if score > best_score:
                best_score = score
                best_s_idx = s_idx

        if best_s_idx is not None and best_score >= similarity_threshold:
            chosen_s_idx = best_s_idx
            matched_flag = True
            final_score = max(0.0, min(1.0, best_score))
        else:
            # Sequential proportional fallback jab similarity threshold se kam ho
            chosen_s_idx = min(
                len(parsed_sentences) - 1,
                int(round(p_idx * (len(parsed_sentences) / max(1, len(plan_items))))),
            )
            matched_flag = bool(not prompt)
            final_score = max(0.0, best_score) if prompt else 1.0

        used_sentence_indices.add(chosen_s_idx)
        s_text, s_start, s_end = parsed_sentences[chosen_s_idx]
        chosen_sent_obj = sentence_items[chosen_s_idx]

        results.append(
            MatchResult(
                index=p_idx,
                sentence_text=s_text or prompt,
                start_time=s_start,
                end_time=s_end,
                media_path=m_path,
                media_type=m_type,
                clip_start=c_start,
                clip_end=c_end,
                score=final_score,
                matched=matched_flag,
                plan_item=p_item,
                sentence=chosen_sent_obj,
                sentences=[chosen_sent_obj],
                sentence_indices=[chosen_s_idx],
            )
        )

    return results


class StoryMatcher:
    """Class wrapper jo match_plan_to_transcript aur match_story methods provide karta hai."""

    def __init__(self, similarity_threshold: float = 0.40, **kwargs: Any) -> None:
        if "threshold" in kwargs:
            similarity_threshold = float(kwargs["threshold"])
        self.similarity_threshold = similarity_threshold

    def calculate_similarity(self, text_a: str, text_b: str) -> float:
        return calculate_similarity(text_a, text_b)

    def match(
        self,
        plan_or_sentences: Any = None,
        transcript_or_media: Any = None,
        **kwargs: Any,
    ) -> List[MatchResult]:
        return match_plan_to_transcript(
            plan=plan_or_sentences,
            transcript=transcript_or_media,
            similarity_threshold=self.similarity_threshold,
            **kwargs,
        )

    def match_plan_to_transcript(
        self,
        plan: Any = None,
        transcript: Any = None,
        **kwargs: Any,
    ) -> List[MatchResult]:
        return match_plan_to_transcript(
            plan=plan,
            transcript=transcript,
            similarity_threshold=self.similarity_threshold,
            **kwargs,
        )

    def match_story(
        self,
        plan_or_sentences: Any = None,
        transcript_or_media: Any = None,
        **kwargs: Any,
    ) -> List[MatchResult]:
        return match_plan_to_transcript(
            plan=plan_or_sentences,
            transcript=transcript_or_media,
            similarity_threshold=self.similarity_threshold,
            **kwargs,
        )


# Additional helper aliases taaki kisi bhi naam se import hone par fail na ho
match_story = match_plan_to_transcript
match_transcript_to_plan = match_plan_to_transcript
match_sentences_to_plan = match_plan_to_transcript
align_plan_with_transcript = match_plan_to_transcript


__all__ = [
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
