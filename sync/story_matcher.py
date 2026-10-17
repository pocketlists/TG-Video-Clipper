from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    # Type checking imports sirf IDE ke liye hain, runtime par circular import nahi karenge
    from audio.sentences import Sentence
    from planning.plan_parser import PlanItem

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Audio sentence/segment aur media clip ke beech match ka result store karta hai."""
    index: int = 0
    sentence_text: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    media_path: Optional[str] = None
    media_type: str = "video"
    clip_start: float = 0.0
    clip_end: Optional[float] = None
    score: float = 1.0
    matched: bool = True
    plan_item: Optional[Any] = None
    sentence: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end_time) - float(self.start_time))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "sentence_text": self.sentence_text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "media_path": self.media_path,
            "media_type": self.media_type,
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "score": self.score,
            "matched": self.matched,
            "metadata": self.metadata,
        }


class StoryMatcher:
    """Transcript sentences aur planned media/story beats ko sync aur match karne wala engine."""

    def __init__(self, similarity_threshold: float = 0.45) -> None:
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def calculate_similarity(self, text_a: str, text_b: str) -> float:
        norm_a = self._normalize_text(text_a)
        norm_b = self._normalize_text(text_b)
        if not norm_a or not norm_b:
            return 0.0
        if norm_a == norm_b:
            return 1.0
        if norm_a in norm_b or norm_b in norm_a:
            return 0.85
        return SequenceMatcher(None, norm_a, norm_b).ratio()

    @staticmethod
    def _extract_sentence_fields(item: Any, fallback_idx: int) -> Tuple[str, float, float]:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("sentence") or "")
            start = float(item.get("start", item.get("start_time", fallback_idx * 3.0)))
            end = float(item.get("end", item.get("end_time", start + 3.0)))
            return text, start, end

        text = str(getattr(item, "text", getattr(item, "sentence", str(item))))
        start = float(getattr(item, "start", getattr(item, "start_time", fallback_idx * 3.0)))
        end = float(getattr(item, "end", getattr(item, "end_time", start + 3.0)))
        return text, start, end

    @staticmethod
    def _extract_media_fields(item: Any) -> Tuple[Optional[str], str, str, float, Optional[float]]:
        if isinstance(item, str):
            return item, "video", "", 0.0, None

        if isinstance(item, dict):
            path = item.get("media_path") or item.get("path") or item.get("file")
            m_type = str(item.get("media_type") or item.get("type") or "video")
            prompt = str(
                item.get("text")
                or item.get("script")
                or item.get("sentence")
                or item.get("description")
                or ""
            )
            c_start = float(item.get("clip_start", item.get("start", 0.0)) or 0.0)
            c_end_raw = item.get("clip_end", item.get("end", None))
            c_end = float(c_end_raw) if c_end_raw is not None else None
            return path, m_type, prompt, c_start, c_end

        path = (
            getattr(item, "media_path", None)
            or getattr(item, "path", None)
            or getattr(item, "file", None)
        )
        m_type = str(getattr(item, "media_type", getattr(item, "type", "video")))
        prompt = str(
            getattr(item, "text", None)
            or getattr(item, "script", None)
            or getattr(item, "sentence", None)
            or getattr(item, "description", "")
        )
        c_start = float(getattr(item, "clip_start", getattr(item, "start", 0.0)) or 0.0)
        c_end_raw = getattr(item, "clip_end", getattr(item, "end", None))
        c_end = float(c_end_raw) if c_end_raw is not None else None
        return path, m_type, prompt, c_start, c_end

    def match(
        self,
        sentences: Sequence[Any],
        media_or_plan: Optional[Sequence[Any]] = None,
    ) -> List[MatchResult]:
        results: List[MatchResult] = []
        if not sentences:
            return results

        items = list(media_or_plan) if media_or_plan else []
        used_indices = set()

        for idx, sent in enumerate(sentences):
            s_text, s_start, s_end = self._extract_sentence_fields(sent, idx)

            best_score = -1.0
            best_item_idx: Optional[int] = None

            for m_idx, item in enumerate(items):
                _, _, prompt, _, _ = self._extract_media_fields(item)
                if prompt:
                    score = self.calculate_similarity(s_text, prompt)
                    if m_idx not in used_indices:
                        score += 0.05
                else:
                    score = 1.0 if m_idx == idx else 0.0

                if score > best_score:
                    best_score = score
                    best_item_idx = m_idx

            if best_item_idx is not None and items:
                chosen_Idx = (
                    best_item_idx
                    if best_score >= self.similarity_threshold
                    else (idx % len(items))
                )
                used_indices.add(chosen_Idx)
                chosen_item = items[chosen_Idx]
                m_path, m_type, _, c_start, c_end = self._extract_media_fields(chosen_item)
                matched_flag = best_score >= self.similarity_threshold or bool(m_path)
                final_score = max(0.0, min(1.0, best_score)) if best_score > 0 else 1.0
            else:
                chosen_item = None
                m_path, m_type, c_start, c_end = None, "video", 0.0, None
                matched_flag = False
                final_score = 0.0

            results.append(
                MatchResult(
                    index=idx,
                    sentence_text=s_text,
                    start_time=s_start,
                    end_time=s_end,
                    media_path=m_path,
                    media_type=m_type,
                    clip_start=c_start,
                    clip_end=c_end,
                    score=final_score,
                    matched=matched_flag,
                    plan_item=chosen_item,
                    sentence=sent,
                )
            )

        return results

    def match_story(
        self,
        sentences: Sequence[Any],
        media_or_plan: Optional[Sequence[Any]] = None,
    ) -> List[MatchResult]:
        return self.match(sentences, media_or_plan)


def match_story(
    sentences: Sequence[Any],
    media_or_plan: Optional[Sequence[Any]] = None,
    threshold: float = 0.45,
) -> List[MatchResult]:
    """Direct helper function agar dusre modules function import karte hain."""
    matcher = StoryMatcher(similarity_threshold=threshold)
    return matcher.match(sentences, media_or_plan)


__all__ = ["MatchResult", "StoryMatcher", "match_story"]
