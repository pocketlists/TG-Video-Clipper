"""
Unit tests for audio/sentences.py and sync/story_matcher.py.

Gemini is never called — every test either exercises the fuzzy fallback
directly or monkeypatches the Gemini matcher to a deterministic result.
No network, no models, no ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio.sentences import (
    Sentence,
    build_sentences,
    normalize_for_match,
)
from audio.transcriber import (
    TranscriptionResult,
    TranscriptSegment,
    WordTiming,
)
from planning.plan_parser import EditPlan, PlanImage
from sync.story_matcher import (
    _is_monotonic,
    _split_equal_spans,
    _finalize_spans,
    _assign_raw_spans,
    _fuzzy_match,
    MatchedScene,
    match_plan_to_transcript,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word(start: float, end: float, text: str) -> WordTiming:
    return WordTiming(start=start, end=end, text=text)


def _sentence(start: float, end: float, text: str) -> Sentence:
    return Sentence(start=start, end=end, text=text, words=[])


def _plan(files_and_lines):
    images = [
        {
            "file": f,
            "story_line": line,
            "keep": True,
            "order": i + 1,
        }
        for i, (f, line) in enumerate(files_and_lines)
    ]
    return EditPlan(
        version=1,
        images=[PlanImage(**{
            "file": i["file"],
            "story_line": i["story_line"],
            "keep": True,
            "order": i["order"],
        }) for i in images],
        warnings=[],
        raw={"version": 1, "images": images},
    )


# ---------------------------------------------------------------------------
# normalize_for_match
# ---------------------------------------------------------------------------

def test_normalize_basic():
    assert normalize_for_match("Kael, the Brave!") == "kael the brave"


def test_normalize_preserves_devanagari():
    assert normalize_for_match("काएल जंगल में गया।") == "काएल जंगल में गया"


def test_normalize_whitespace_and_punct():
    assert normalize_for_match("  A   B. C?  ") == "a b c"


def test_normalize_empty():
    assert normalize_for_match("") == ""
    assert normalize_for_match(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_sentences — word-based path
# ---------------------------------------------------------------------------

def _transcription_from_words(words):
    seg = TranscriptSegment(
        start=words[0].start, end=words[-1].end,
        text=" ".join(w.text for w in words), words=words,
    )
    return TranscriptionResult(
        segments=[seg], words=words, duration=words[-1].end,
        engine="whisper-local", model="base",
    )


def test_build_sentences_splits_on_period():
    words = [
        _word(0.0, 0.4, "Kael"),
        _word(0.4, 0.8, "entered"),
        _word(0.8, 1.2, "the"),
        _word(1.2, 1.6, "forest."),
        _word(1.8, 2.2, "A"),
        _word(2.2, 2.6, "dragon"),
        _word(2.6, 3.0, "appeared!"),
    ]
    sentences = build_sentences(_transcription_from_words(words))
    assert len(sentences) == 2
    assert sentences[0].text == "Kael entered the forest."
    assert sentences[1].text == "A dragon appeared!"
    assert sentences[0].start == 0.0
    assert abs(sentences[0].end - 1.6) < 1e-6
    assert abs(sentences[1].start - 1.8) < 1e-6


def test_build_sentences_splits_on_devanagari_danda():
    words = [
        _word(0.0, 0.5, "काएल"),
        _word(0.5, 1.0, "जंगल"),
        _word(1.0, 1.5, "में।"),
        _word(1.7, 2.2, "फिर"),
        _word(2.2, 2.7, "वो"),
        _word(2.7, 3.2, "गया।"),
    ]
    sentences = build_sentences(_transcription_from_words(words))
    assert len(sentences) == 2


def test_build_sentences_caps_long_sentence_without_punctuation():
    # 30 words, no punctuation — should split by MAX_WORDS_PER_SENTENCE
    words = [_word(i * 0.1, (i + 1) * 0.1, f"w{i}") for i in range(30)]
    sentences = build_sentences(_transcription_from_words(words))
    assert len(sentences) >= 2
    # each sentence must be strictly inside the audio duration
    for s in sentences:
        assert s.end <= 3.0 + 1e-6
        assert s.start < s.end


def test_build_sentences_monotonic():
    words = [_word(i * 0.2, (i + 1) * 0.2, f"w{i}.") for i in range(10)]
    sentences = build_sentences(_transcription_from_words(words))
    for prev, nxt in zip(sentences, sentences[1:]):
        assert prev.end <= nxt.start + 1e-6


def test_build_sentences_clamped_to_duration():
    words = [_word(0.0, 0.5, "a."), _word(0.5, 0.9, "b.")]
    tr = TranscriptionResult(
        segments=[TranscriptSegment(0.0, 0.9, "a. b.", words)],
        words=words, duration=0.8,  # shorter than last end
        engine="whisper-local", model="base",
    )
    sentences = build_sentences(tr)
    for s in sentences:
        assert s.end <= 0.8 + 1e-6


# ---------------------------------------------------------------------------
# build_sentences — segment-only fallback (Gemini without words)
# ---------------------------------------------------------------------------

def test_build_sentences_segment_fallback():
    seg = TranscriptSegment(
        start=0.0, end=4.0,
        text="Kael entered the forest. A dragon appeared. He drew his sword!",
        words=[],
    )
    tr = TranscriptionResult(
        segments=[seg], words=[], duration=4.0,
        engine="gemini", model="x",
    )
    sentences = build_sentences(tr)
    assert len(sentences) == 3
    assert all(s.start < s.end for s in sentences)
    assert sentences[0].start == 0.0
    assert abs(sentences[-1].end - 4.0) < 1e-6


def test_build_sentences_rejects_empty_transcript():
    seg = TranscriptSegment(start=0.0, end=1.0, text="", words=[])
    tr = TranscriptionResult(
        segments=[seg], words=[], duration=1.0,
        engine="gemini", model="x",
    )
    with pytest.raises(Exception, match="empty"):
        build_sentences(tr)


# ---------------------------------------------------------------------------
# Monotonic validation
# ---------------------------------------------------------------------------

def test_is_monotonic_true():
    ok, _ = _is_monotonic({0: [0], 1: [1, 2], 2: [3]}, 3)
    assert ok


def test_is_monotonic_false_on_overlap():
    ok, reason = _is_monotonic({0: [0, 1], 1: [1]}, 2)
    assert not ok
    assert "plan[1]" in reason


def test_is_monotonic_allows_gaps():
    ok, _ = _is_monotonic({0: [0], 2: [4]}, 3)
    assert ok


# ---------------------------------------------------------------------------
# Fuzzy match (no Gemini)
# ---------------------------------------------------------------------------

def test_fuzzy_match_exact():
    sentences = [
        _sentence(0.0, 1.0, "Kael entered the forest."),
        _sentence(1.0, 2.0, "A dragon appeared."),
    ]
    images = [
        PlanImage(file="a.jpg", story_line="Kael entered the forest.", keep=True),
        PlanImage(file="b.jpg", story_line="A dragon appeared.", keep=True),
    ]
    result = _fuzzy_match(images, sentences, threshold=0.5)
    assert result[0] == [0]
    assert result[1] == [1]


def test_fuzzy_match_skips_when_below_threshold():
    sentences = [_sentence(0.0, 1.0, "Something completely different")]
    images = [PlanImage(file="a.jpg", story_line="xyz qrs", keep=True)]
    result = _fuzzy_match(images, sentences, threshold=0.7)
    assert result == {}


def test_fuzzy_match_advances_cursor():
    sentences = [
        _sentence(0.0, 1.0, "alpha"),
        _sentence(1.0, 2.0, "alpha"),
    ]
    images = [
        PlanImage(file="a.jpg", story_line="alpha", keep=True),
        PlanImage(file="b.jpg", story_line="alpha", keep=True),
    ]
    result = _fuzzy_match(images, sentences, threshold=0.5)
    assert result[0] == [0]
    assert result[1] == [1]


# ---------------------------------------------------------------------------
# Span assignment for unmatched plan entries
# ---------------------------------------------------------------------------

def test_assign_raw_spans_no_unmatched():
    sentences = [
        _sentence(0.0, 1.0, "a"),
        _sentence(1.0, 2.0, "b"),
    ]
    assignment = {0: [0], 1: [1]}
    spans, unmatched = _assign_raw_spans(2, assignment, sentences, 2.0)
    assert unmatched == []
    assert spans == [(0.0, 1.0), (1.0, 2.0)]


def test_assign_raw_spans_fills_gap():
    sentences = [
        _sentence(0.0, 1.0, "a"),
        _sentence(1.0, 2.0, "b"),
        _sentence(2.0, 3.0, "c"),
    ]
    # plan index 1 is unmatched between two matched neighbours
    assignment = {0: [0], 2: [2]}
    spans, unmatched = _assign_raw_spans(3, assignment, sentences, 3.0)
    assert unmatched == [1]
    # slot should sit between 1.0 and 2.0
    assert spans[1][0] >= 1.0 - 1e-6
    assert spans[1][1] <= 2.0 + 1e-6
    assert spans[1][0] < spans[1][1]


def test_assign_raw_spans_leading_unmatched():
    sentences = [_sentence(1.0, 2.0, "a")]
    assignment = {1: [0]}
    spans, unmatched = _assign_raw_spans(2, assignment, sentences, 2.0)
    assert unmatched == [0]
    # leading slot starts at 0.0 and ends at first matched start
    assert spans[0][0] == 0.0
    assert abs(spans[0][1] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Shared-span splitting
# ---------------------------------------------------------------------------

def test_split_equal_spans_two_images_share_sentence():
    scenes = [
        MatchedScene(
            image=PlanImage(file="a.jpg", keep=True),
            start=0.0, end=2.0, sentence_indices=[0],
            method="gemini", confidence=1.0,
        ),
        MatchedScene(
            image=PlanImage(file="b.jpg", keep=True),
            start=0.0, end=2.0, sentence_indices=[0],
            method="gemini", confidence=1.0,
        ),
    ]
    raw = [(0.0, 2.0), (0.0, 2.0)]
    _split_equal_spans(scenes, raw)
    assert raw[0] == (0.0, 1.0)
    assert raw[1] == (1.0, 2.0)


def test_split_equal_spans_noop_when_distinct():
    scenes = [
        MatchedScene(
            image=PlanImage(file="a.jpg", keep=True),
            start=0.0, end=1.0, sentence_indices=[0],
            method="gemini", confidence=1.0,
        ),
        MatchedScene(
            image=PlanImage(file="b.jpg", keep=True),
            start=1.0, end=2.0, sentence_indices=[1],
            method="gemini", confidence=1.0,
        ),
    ]
    raw = [(0.0, 1.0), (1.0, 2.0)]
    _split_equal_spans(scenes, raw)
    assert raw == [(0.0, 1.0), (1.0, 2.0)]


# ---------------------------------------------------------------------------
# Final span hold model
# ---------------------------------------------------------------------------

def test_finalize_spans_extends_to_duration():
    spans = [(0.0, 1.0), (1.5, 2.0)]
    out = _finalize_spans(spans, 2, duration=5.0)
    # scene 0 held until scene 1 begins
    assert out[0] == (0.0, 1.5)
    # last scene extended to full duration
    assert out[1] == (1.5, 5.0)


def test_finalize_spans_trims_overlap():
    spans = [(0.0, 2.0), (1.0, 3.0)]
    out = _finalize_spans(spans, 2, duration=3.0)
    assert out[0][1] <= out[1][0] + 1e-6
    assert out[1][0] >= 2.0 - 1e-6


def test_finalize_spans_first_starts_at_zero():
    spans = [(0.3, 1.0), (1.0, 2.0)]
    out = _finalize_spans(spans, 2, duration=2.0)
    assert out[0][0] == 0.0


# ---------------------------------------------------------------------------
# match_plan_to_transcript — end-to-end with mocked Gemini
# ---------------------------------------------------------------------------

def test_match_end_to_end_fuzzy(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("gemini unavailable in tests")
    monkeypatch.setattr(
        "sync.story_matcher._gemini_match", _boom,
    )

    sentences = [
        _sentence(0.0, 1.5, "Kael entered the dark forest."),
        _sentence(1.5, 3.0, "A dragon appeared from the sky."),
        _sentence(3.0, 4.5, "He drew his sword."),
    ]
    plan = _plan([
        ("a.jpg", "Kael entered the dark forest."),
        ("b.jpg", "A dragon appeared from the sky."),
        ("c.jpg", "He drew his sword."),
    ])
    result = match_plan_to_transcript(plan, sentences)
    assert result.method == "fuzzy"
    assert len(result.scenes) == 3
    assert all(s.end > s.start for s in result.scenes)
    assert result.scenes[0].start == 0.0
    assert abs(result.scenes[-1].end - 4.5) < 1e-6


def test_match_end_to_end_gemini(monkeypatch):
    def _fake_gemini(images, sentences, cfg):
        return {0: [0], 1: [1, 2]}

    monkeypatch.setattr(
        "sync.story_matcher._gemini_match", _fake_gemini,
    )

    sentences = [
        _sentence(0.0, 1.0, "One."),
        _sentence(1.0, 2.0, "Two."),
        _sentence(2.0, 3.0, "Three."),
    ]
    plan = _plan([
        ("a.jpg", "One."),
        ("b.jpg", "Two and three."),
    ])
    result = match_plan_to_transcript(plan, sentences)
    assert result.method == "gemini"
    assert result.scenes[0].sentence_indices == [0]
    assert result.scenes[1].sentence_indices == [1, 2]
    # scene 0 must end where scene 1 begins
    assert abs(result.scenes[0].end - result.scenes[1].start) < 1e-6


def test_match_rejects_empty_sentences():
    plan = _plan([("a.jpg", "x")])
    with pytest.raises(Exception, match="no sentences"):
        match_plan_to_transcript(plan, [])


def test_match_rejects_empty_plan(monkeypatch):
    monkeypatch.setattr(
        "sync.story_matcher._gemini_match",
        lambda *a, **k: {},
    )
    plan = EditPlan(version=1, images=[], warnings=[], raw={})
    with pytest.raises(Exception):
        match_plan_to_transcript(plan, [_sentence(0.0, 1.0, "x")])


def test_match_warns_when_nothing_aligns(monkeypatch):
    monkeypatch.setattr(
        "sync.story_matcher._gemini_match",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no gemini")),
    )
    sentences = [_sentence(0.0, 2.0, "completely unrelated")]
    plan = _plan([
        ("a.jpg", "zzz qqq xxx"),
        ("b.jpg", "yyy www vvv"),
    ])
    result = match_plan_to_transcript(plan, sentences)
    # even split fallback must fire
    assert result.method == "even"
    assert len(result.scenes) == 2
    assert result.scenes[0].start == 0.0
    assert abs(result.scenes[-1].end - 2.0) < 1e-6
    assert any("evenly" in w for w in result.warnings)


def test_match_handles_shared_sentence(monkeypatch):
    # Both images map to the same sentence; span must be split evenly.
    monkeypatch.setattr(
        "sync.story_matcher._gemini_match",
        lambda images, sentences, cfg: {0: [0], 1: [0]},
    )
    sentences = [_sentence(0.0, 2.0, "One long line.")]
    plan = _plan([
        ("a.jpg", "One long line."),
        ("b.jpg", "One long line."),
    ])
    result = match_plan_to_transcript(plan, sentences)
    assert result.method == "gemini"
    assert abs(result.scenes[0].start - 0.0) < 1e-6
    assert abs(result.scenes[1].start - 1.0) < 1e-6
    assert abs(result.scenes[1].end - 2.0) < 1e-6
