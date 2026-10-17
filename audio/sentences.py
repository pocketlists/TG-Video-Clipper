
"""
audio/sentences.py

Turns a TranscriptionResult into sentence-level units with monotonic,
non-overlapping start/end times that stay inside the original audio
duration. Prefers word-level timings (Whisper); falls back to text-based
splitting of segment timings (Gemini without word info).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from audio.transcriber import TranscriptionResult, TranscriptSegment, WordTiming
from utils.error_handler import PipelineError, STAGE_TRANSCRIBE
from utils.logger import get_logger

logger = get_logger("audio.sentences")


_HARD_TERMINATORS = {".", "!", "?", "।", "॥"}
_SOFT_TERMINATORS = {"…", ";", ":"}

_MAX_WORDS_PER_SENTENCE = 22
_MIN_SENTENCE_SECONDS = 0.25


@dataclass
class Sentence:
    start: float
    end: float
    text: str
    words: List[WordTiming]

    def duration(self) -> float:
        return self.end - self.start

    def normalized(self) -> str:
        return normalize_for_match(self.text)


_ASCII_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_UNICODE_PUNCT = "\u2018\u2019\u201c\u201d\u2026\u2014\u2013\u0964\u0965"
_PUNCT_RE = re.compile("[" + re.escape(_ASCII_PUNCT + _UNICODE_PUNCT) + "]+")
_WS_RE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """Normalize text for matching without removing its actual words."""
    if not text:
        return ""
    lowered = text.lower()
    stripped = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", stripped).strip()


def _ends_sentence(word_text: str) -> bool:
    if not word_text:
        return False
    return word_text[-1] in _HARD_TERMINATORS


def _split_words_into_sentences(
    words: List[WordTiming],
) -> List[List[WordTiming]]:
    sentences: List[List[WordTiming]] = []
    current: List[WordTiming] = []

    for word in words:
        current.append(word)

        if _ends_sentence(word.text):
            sentences.append(current)
            current = []
            continue

        if len(current) >= _MAX_WORDS_PER_SENTENCE:
            sentences.append(current)
            current = []

    if current:
        sentences.append(current)

    return sentences


def _merge_short_sentences(
    groups: List[List[WordTiming]],
) -> List[List[WordTiming]]:
    if not groups:
        return []

    merged: List[List[WordTiming]] = [groups[0]]

    for group in groups[1:]:
        previous = merged[-1]
        span = (
            group[-1].end - previous[0].start
            if group and previous else 0.0
        )
        previous_span = previous[-1].end - previous[0].start

        if previous_span < _MIN_SENTENCE_SECONDS and span > 0:
            merged[-1] = previous + group
        else:
            merged.append(group)

    return merged


def _from_words(
    words: List[WordTiming],
    duration: float,
) -> List[Sentence]:
    words = [word for word in words if word.text.strip()]
    if not words:
        return []

    groups = _merge_short_sentences(
        _split_words_into_sentences(words)
    )

    sentences: List[Sentence] = []
    previous_end = 0.0

    for group in groups:
        start = max(float(group[0].start), previous_end)
        end = max(float(group[-1].end), start + 0.05)

        if duration > 0:
            start = min(start, duration)
            end = min(end, duration)

        if end <= start:
            continue

        text = " ".join(word.text for word in group).strip()
        sentences.append(
            Sentence(
                start=start,
                end=end,
                text=text,
                words=list(group),
            )
        )
        previous_end = end

    return sentences


def _split_segment_by_text(
    seg: TranscriptSegment,
) -> List[Sentence]:
    text = seg.text.strip()
    if not text:
        return []

    pieces = _split_text_by_punctuation(text)
    if not pieces:
        pieces = [text]

    total_chars = sum(len(piece) for piece in pieces) or 1
    segment_duration = max(float(seg.end) - float(seg.start), 0.05)

    sentences: List[Sentence] = []
    cursor = float(seg.start)

    for index, piece in enumerate(pieces):
        share = len(piece) / total_chars
        piece_end = cursor + segment_duration * share

        if index == len(pieces) - 1:
            piece_end = float(seg.end)

        if piece_end <= cursor:
            continue

        sentences.append(
            Sentence(
                start=cursor,
                end=piece_end,
                text=piece.strip(),
                words=[],
            )
        )
        cursor = piece_end

    return sentences


def _split_text_by_punctuation(text: str) -> List[str]:
    """Split by sentence punctuation, then split excessively long pieces."""
    tokens = re.findall(r"[^.!?।॥…]+[.!?।॥…]*", text)
    cleaned = [token.strip() for token in tokens if token.strip()]

    output: List[str] = []

    for piece in cleaned:
        words = piece.split()

        if len(words) <= _MAX_WORDS_PER_SENTENCE:
            output.append(piece)
            continue

        for offset in range(0, len(words), _MAX_WORDS_PER_SENTENCE):
            output.append(
                " ".join(words[offset:offset + _MAX_WORDS_PER_SENTENCE])
            )

    return output


def _from_segments(
    segments: List[TranscriptSegment],
    duration: float,
) -> List[Sentence]:
    sentences: List[Sentence] = []
    previous_end = 0.0

    for segment in segments:
        for sentence in _split_segment_by_text(segment):
            start = max(float(sentence.start), previous_end)
            end = max(float(sentence.end), start + 0.05)

            if duration > 0:
                start = min(start, duration)
                end = min(end, duration)

            if end <= start:
                continue

            sentences.append(
                Sentence(
                    start=start,
                    end=end,
                    text=sentence.text,
                    words=sentence.words,
                )
            )
            previous_end = end

    return sentences


# ---------------------------------------------------------------------------
# Post-processing: remove empty sentences, clamp timestamps to audio
# duration, and enforce chronological non-overlapping spans.
# ---------------------------------------------------------------------------

def _sanitize(
    sentences: Iterable[Sentence],
    duration: float,
) -> List[Sentence]:
    result: List[Sentence] = []
    previous_end = 0.0

    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0

    if duration < 0:
        duration = 0.0

    for sentence in sentences:
        text = sentence.text.strip()
        if not text:
            continue

        if duration > 0 and previous_end >= duration:
            break

        try:
            raw_start = float(sentence.start)
            raw_end = float(sentence.end)
        except (TypeError, ValueError):
            logger.warning("Skipping sentence with invalid timestamps")
            continue

        # Prevent negative timestamps and overlap with the previous sentence.
        start = max(0.0, raw_start, previous_end)

        if duration > 0:
            start = min(start, duration)

        # Preserve the original minimum-span behavior, then clamp to audio.
        end = max(raw_end, start + 0.05)

        if duration > 0:
            end = min(end, duration)

        # This also skips zero-duration spans after clamping.
        if end <= start:
            continue

        result.append(
            Sentence(
                start=start,
                end=end,
                text=text,
                words=sentence.words,
            )
        )
        previous_end = end

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_sentences(
    transcription: TranscriptionResult,
) -> List[Sentence]:
    """
    Produce sentence-level units from a TranscriptionResult.

    Word-level timings are preferred. If unavailable, segment timings are
    divided among punctuation-based sentence fragments by character count.

    Returned sentence spans are chronological, non-overlapping, and clamped
    to the transcription duration when that duration is positive.
    """
    if transcription is None:
        raise PipelineError(
            STAGE_TRANSCRIBE,
            "transcription is None",
        )

    if transcription.is_empty():
        raise PipelineError(
            STAGE_TRANSCRIBE,
            "transcription is empty — cannot build sentences",
        )

    duration = max(float(transcription.duration), 0.0)

    sentences: List[Sentence] = []

    if transcription.words:
        sentences = _from_words(transcription.words, duration)

    if not sentences:
        sentences = _from_segments(transcription.segments, duration)

    if not sentences and transcription.words:
        sentences = _from_words(transcription.words, duration)

    sentences = _sanitize(sentences, duration)

    if not sentences:
        raise PipelineError(
            STAGE_TRANSCRIBE,
            "could not derive any non-empty sentences from the transcript",
        )

    if duration > 0 and sentences[-1].end > duration + 0.01:
        logger.warning(
            f"last sentence ends at {sentences[-1].end:.2f}s but audio is "
            f"{duration:.2f}s — clamped"
        )

    logger.info(
        f"built {len(sentences)} sentences "
        f"(engine={transcription.engine}, "
        f"words_available={len(transcription.words)})"
    )

    return sentences
