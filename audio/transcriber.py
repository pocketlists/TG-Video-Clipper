
"""
Offline audio transcription using local OpenAI Whisper.
No transcription API calls. CPU only.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from config.settings import transcriber_config
from utils.error_handler import PipelineError, STAGE_TRANSCRIBE
from utils.logger import get_logger

logger = get_logger("audio.transcriber")

ProgressFn = Optional[Callable[[str], None]]


@dataclass
class WordTiming:
    start: float
    end: float
    text: str


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: List[WordTiming] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    segments: List[TranscriptSegment]
    words: List[WordTiming]
    duration: float
    engine: str
    model: str
    language: Optional[str] = None

    def is_empty(self) -> bool:
        return not self.segments or not any(
            segment.text.strip() for segment in self.segments
        )


# Cache loaded models for reuse within the same runner process.
_MODEL_CACHE: dict = {}


def _probe_duration(path: Path) -> float:
    """Get audio duration through local ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 0.0

    if result.returncode != 0 or not result.stdout:
        return 0.0

    try:
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0) or 0)
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def _get_whisper_model(model_name: str):
    """Load Whisper explicitly on CPU."""
    if model_name not in _MODEL_CACHE:
        try:
            import whisper
        except ImportError as exc:
            raise PipelineError(
                STAGE_TRANSCRIBE,
                "Local Whisper is not installed.",
                hint="Install the openai-whisper package and FFmpeg.",
                cause=exc,
            ) from exc

        logger.info("Loading local Whisper model %s on CPU", model_name)

        try:
            _MODEL_CACHE[model_name] = whisper.load_model(
                model_name,
                device="cpu",
            )
        except Exception as exc:
            raise PipelineError(
                STAGE_TRANSCRIBE,
                f"Could not load local Whisper model: {exc}",
                hint=(
                    "The model may not be cached yet. On the first run, "
                    "download its weights and cache them for later runs."
                ),
                cause=exc,
            ) from exc

    return _MODEL_CACHE[model_name]


def _transcribe_local(
    audio_path: Path,
    model_name: str,
) -> TranscriptionResult:
    model = _get_whisper_model(model_name)

    try:
        # CPU-only inference; do not use half-precision FP16.
        raw = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            fp16=False,
            verbose=False,
        )
    except Exception as exc:
        raise PipelineError(
            STAGE_TRANSCRIBE,
            f"Local Whisper transcription failed: {exc}",
            hint="Check the audio file, FFmpeg installation, and available RAM.",
            cause=exc,
        ) from exc

    segments: List[TranscriptSegment] = []
    all_words: List[WordTiming] = []

    for item in raw.get("segments") or []:
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
        except (TypeError, ValueError):
            continue

        segment_words: List[WordTiming] = []

        for word in item.get("words") or []:
            try:
                timing = WordTiming(
                    start=float(word.get("start", 0.0)),
                    end=float(word.get("end", 0.0)),
                    text=str(word.get("word", "")).strip(),
                )
            except (TypeError, ValueError):
                continue

            if timing.text:
                segment_words.append(timing)
                all_words.append(timing)

        text = str(item.get("text", "")).strip()

        if not text and segment_words:
            text = " ".join(word.text for word in segment_words)

        if text and end >= start:
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                    words=segment_words,
                )
            )

    if not segments:
        raise PipelineError(
            STAGE_TRANSCRIBE,
            "Local Whisper returned no transcript.",
            hint="Check whether the audio contains clear speech.",
        )

    duration = _probe_duration(audio_path) or segments[-1].end

    return TranscriptionResult(
        segments=segments,
        words=all_words,
        duration=duration,
        engine="whisper-local-cpu",
        model=model_name,
        language=str(raw.get("language") or "") or None,
    )


def transcribe(
    audio_path: Path,
    progress: ProgressFn = None,
) -> TranscriptionResult:
    """
    Public entry point.

    Audio transcription runs locally on CPU.
    No OpenAI transcription API, Gemini transcription API,
    network upload, or cloud transcription fallback is used.
    """
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise PipelineError(
            STAGE_TRANSCRIBE,
            f"Audio file not found: {audio_path}",
        )

    if audio_path.stat().st_size == 0:
        raise PipelineError(
            STAGE_TRANSCRIBE,
            f"Audio file is empty: {audio_path}",
        )

    cfg = transcriber_config()
    model_name = cfg.local_whisper_model or "base"

    if progress:
        progress(f"Local Whisper CPU ({model_name})")

    result = _transcribe_local(audio_path, model_name)

    if result.is_empty():
        raise PipelineError(
            STAGE_TRANSCRIBE,
            "Local Whisper returned an empty transcript.",
        )

    logger.info(
        "Offline CPU transcription complete: model=%s, segments=%d, words=%d",
        result.model,
        len(result.segments),
        len(result.words),
    )

    return result
