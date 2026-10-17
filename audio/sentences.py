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
from typing import Iterable, List, Optional

from audio.transcriber import TranscriptionResult, TranscriptSegment, WordTiming
from utils.error_handler import PipelineError, STAGE_TRANSCRIBE
from utils.logger import get_logger


logger = get_logger("audio.sentences")


# Sentence-ending punctuation that is safe across Hindi/Hinglish/English.
# Devanagari danda "।" and double danda "॥" are included; ASCII ". ! ?"
# covers the rest. Ellipsis "…" is treated as a soft break, not a hard one.
_HARD_TERMINATORS = {".", "!", "?", "।", "॥"}
_SOFT_TERMINATORS = {"…", ";", ":"}

# Max words per sentence when the engine gives us no reliable punctuation
# (e.g. Whisper sometimes drops full stops on Hindi audio). Chosen so that
# a typical narration gets one image per 12-16 spoken words — comfortably
# within what one still frame can hold before it feels static.
_MAX_WORDS_PER_SENTENCE = 22

# Minimum duration for a sentence. Anything below gets merged forward —
# avoids 0.05 s flashes that break sync.
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


# ---------------------------------------------------------------------------
# Text normalization used by both this module and sync/story_matcher.py
# ---------------------------------------------------------------------------

# All ASCII punctuation + curly Unicode quotes + ellipsis / em-dash /
# en-dash + Devanagari danda. re.escape() handles every char-class
# escaping so there is no raw-string fragility and no SyntaxWarning.
_ASCII_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
_UNICODE_PUNCT = "\u2018\u2019\u201c\u201d\u2026\u2014\u2013\u0964\u0965"
_PUNCT_RE = re.compile("[" + re.escape(_ASCII_PUNCT + _UNICODE_PUNCT) + "]+")
_WS_RE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Meaning preserved;
    only surface differences removed. Devanagari characters are preserved."""
    if not text:
        return ""
    lowered = text.lower()
    stripped = _PUNCT_RE.sub(" ", lowered)
    collapsed = _WS_RE.sub(" ", stripped).strip()
    return collapsed


# ---------------------------------------------------------------------------
# Sentence splitting on a word list
# ---------------------------------------------------------------------------

def _ends_sentence(word_text: str) -> bool:
    if not word_text:
        return False
    return word_text[-1] in _HARD_TERMINATORS


def _split_words_into_sentences(words: List[WordTiming]) -> List[List[WordTiming]]:
    sentences: List[List[WordTiming]] = []
    current: List[WordTiming] = []
    for w in words:
        current.append(w)
        if _ends_sentence(w.text):
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
        prev = merged[-1]
        span = (group[-1].end - prev[0].start) if group and prev else 0.0
        prev_span = prev[-1].end - prev[0].start
        if prev_span < _MIN_SENTENCE_SECONDS and span > 0:
            merged[-1] = prev + group
        else:
            merged.append(group)
    return merged


def _from_words(words: List[WordTiming], duration: float) -> List[Sentence]:
    words = [w for w in words if w.text.strip()]
    if not words:
        return []
    groups = _merge_short_sentences(_split_words_into_sentences(words))

    sentences: List[Sentence] = []
    prev_end = 0.0
    for group in groups:
        start = max(group[0].start, prev_end)
        end = max(group[-1].end, start + 0.05)
        if duration > 0:
            end = min(end, duration)
        if end <= start:
            continue
        text = " ".join(w.text for w in group).strip()
        sentences.append(Sentence(start=start, end=end, text=text, words=list(group)))
        prev_end = end
    return sentences


# ---------------------------------------------------------------------------
# Fallback: split segment text by punctuation and share segment duration
# proportionally to character count
# ---------------------------------------------------------------------------

def _split_segment_by_text(seg: TranscriptSegment) -> List[Sentence]:
    text = seg.text.strip()
    if not text:
        return []

    pieces = _split_text_by_punctuation(text)
    if not pieces:
        pieces = [text]

    total_chars = sum(len(p) for p in pieces) or 1
    seg_duration = max(seg.end - seg.start, 0.05)
    sentences: List[Sentence] = []
    cursor = seg.start
    for i, piece in enumerate(pieces):
        share = len(piece) / total_chars
        piece_end = cursor + seg_duration * share
        if i == len(pieces) - 1:
            piece_end = seg.end
        if piece_end <= cursor:
            continue
        sentences.append(Sentence(
            start=cursor,
            end=piece_end,
            text=piece.strip(),
            words=[],
        ))
        cursor = piece_end
    return sentences


def _split_text_by_punctuation(text: str) -> List[str]:
    # Keep the terminator inside the piece (so "Hi. Bye." -> ["Hi.", "Bye."])
    tokens = re.findall(r"[^.!?।॥…]+[.!?।॥…]*", text)
    cleaned = [t.strip() for t in tokens if t.strip()]
    if len(cleaned) <= 1:
        return cleaned
    out: List[str] = []
    buffer = ""
    buffer_words = 0
    for piece in cleaned:
        buffer = (buffer + " " + piece).strip() if buffer else piece
        buffer_words += len(piece.split())
        if buffer_words >= _MAX_WORDS_PER_SENTENCE:
            out.append(buffer)
            buffer = ""
            buffer_words = 0
    if buffer:
        out.append(buffer)
    return out


def _from_segments(segments: List[TranscriptSegment], duration: float) -> List[Sentence]:
    sentences: List[Sentence] = []
    prev_end = 0.0
    for seg in segments:
        for sentence in _split_segment_by_text(seg):
            start = max(sentence.start, prev_end)
            end = max(sentence.end, start + 0.05)
            if duration > 0:
                end = min(end, duration)
            if end <= start:
                continue
            sentences.append(Sentence(
                start=start, end=end, text=sentence.text, words=sentence.words,
            ))
            prev_end = end
    return sentences


# ---------------------------------------------------------------------------
# Post-processing: remove empties, clamp to [0, duration], enforce monotonic
# ---------------------------------------------------------------------------

def _sanitize(sentences: Iterable[Sentence], duration: float) -> List[Sentence]:
    result: List[Sentence] = []
    prev_end = 0.0
    for s in sentences:
        if not s.text.strip():
            continue
        start = max(0.0, s.start, prev_end)
        end = max(s.end, start + 0.05)
        if duration > 0:
            end = min(end, duration)
        if end <= start:
            continue
        result.append(Sentence(
            start=start, end=end, text=s.text.strip(), words=s.words,
        ))
        prev_end = end
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_sentences(transcription: TranscriptionResult) -> List[Sentence]:
    """
    Produce sentence-level units from a TranscriptionResult.

    - If word timings are available, sentences are cut from words (most
      accurate). Fallback punctuation (।॥ etc) and a max word count guard
      ensure Hindi/Hinglish narration without punctuation still splits.
    - If only segment timings exist (Gemini without word info), each
      segment is split by punctuation, and segment duration is shared
      proportionally by character count.
    - Every returned sentence has 0 <= start < end <= duration and does
      not overlap the next sentence.
    """
    if transcription is None:
        raise PipelineError(STAGE_TRANSCRIBE, "transcription is None")

    if transcription.is_empty():
        raise PipelineError(
            STAGE_TRANSCRIBE, "transcription is empty — cannot build sentences"
        )

    duration = max(transcription.duration, 0.0)

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
        f"built {len(sentences)} sentences (engine={transcription.engine}, "
        f"words_available={len(transcription.words)})"
    )
    return sentences
