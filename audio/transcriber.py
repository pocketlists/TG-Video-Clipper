"""
audio/transcriber.py
Local Whisper primary, Gemini Files-API fallback. No OpenAI, no Sarvam.
Returns a single TranscriptionResult — no module-level globals carry
state between calls (fixes the _LAST_WHISPER_WORDS bug).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from config.settings import TranscriberConfig, transcriber_config
from utils.error_handler import PipelineError, STAGE_TRANSCRIBE
from utils.logger import get_logger


logger = get_logger("audio.transcriber")

ProgressFn = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

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
        return not self.segments or not any(s.text.strip() for s in self.segments)


# ---------------------------------------------------------------------------
# Whisper model cache (resource, not state)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}


def _get_whisper_model(name: str):
    if name not in _MODEL_CACHE:
        try:
            import whisper as _whisper_pkg  # openai-whisper pip package
        except ImportError as exc:
            raise PipelineError(
                STAGE_TRANSCRIBE,
                f"openai-whisper is not installed ({exc})",
                hint="pip install openai-whisper",
            ) from exc
        logger.info(f"loading local Whisper model '{name}'")
        _MODEL_CACHE[name] = _whisper_pkg.load_model(name)
    return _MODEL_CACHE[name]


# ---------------------------------------------------------------------------
# ffprobe duration
# ---------------------------------------------------------------------------

def _probe_duration(path: Path) -> float:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 0.0
    if proc.returncode != 0 or not proc.stdout:
        return 0.0
    try:
        data = json.loads(proc.stdout)
        return float(data.get("format", {}).get("duration", 0) or 0)
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Local Whisper
# ---------------------------------------------------------------------------

def _transcribe_local(audio_path: Path, cfg: TranscriberConfig) -> TranscriptionResult:
    model = _get_whisper_model(cfg.local_whisper_model)
    raw = model.transcribe(str(audio_path), word_timestamps=True)

    segments: List[TranscriptSegment] = []
    words: List[WordTiming] = []
    for seg in raw.get("segments") or []:
        try:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        seg_words: List[WordTiming] = []
        for w in seg.get("words") or []:
            try:
                wt = WordTiming(
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    text=str(w.get("word", "")).strip(),
                )
            except (TypeError, ValueError):
                continue
            if wt.text:
                seg_words.append(wt)
                words.append(wt)
        text = str(seg.get("text", "")).strip()
        if not text and seg_words:
            text = " ".join(w.text for w in seg_words)
        if text:
            segments.append(TranscriptSegment(
                start=seg_start, end=seg_end, text=text, words=seg_words,
            ))

    if not segments:
        raise PipelineError(STAGE_TRANSCRIBE, "local Whisper returned no segments")

    duration = _probe_duration(audio_path) or (
        segments[-1].end if segments else 0.0
    )
    return TranscriptionResult(
        segments=segments,
        words=words,
        duration=duration,
        engine="whisper-local",
        model=cfg.local_whisper_model,
        language=str(raw.get("language") or "") or None,
    )


# ---------------------------------------------------------------------------
# Gemini fallback
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```\s*$")
_RATE_LIMIT_HINTS = ("429", "rate limit", "resource_exhausted", "quota", "exhausted")


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    return _FENCE_RE.sub("", text).strip()


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in _RATE_LIMIT_HINTS)


def _state_name(obj) -> str:
    state = getattr(obj, "state", None)
    if state is None:
        return ""
    return str(getattr(state, "name", state))


def _gemini_generate(client, model: str, uploaded, prompt: str, genai_types):
    return client.models.generate_content(
        model=model,
        contents=[uploaded, prompt],
        config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
    )


def _transcribe_gemini(audio_path: Path, cfg: TranscriberConfig) -> TranscriptionResult:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise PipelineError(STAGE_TRANSCRIBE, "GEMINI_API_KEY not set")

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise PipelineError(
            STAGE_TRANSCRIBE, f"google-genai not installed ({exc})"
        ) from exc

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = client.files.upload(file=str(audio_path))
        waited = 0.0
        while _state_name(uploaded) == "PROCESSING" and waited < cfg.timeout_seconds:
            time.sleep(2.0)
            waited += 2.0
            uploaded = client.files.get(name=uploaded.name)
        if _state_name(uploaded) == "FAILED":
            raise PipelineError(STAGE_TRANSCRIBE, "Gemini file processing failed")

        prompt = (
            "Transcribe this audio. Return ONLY JSON of the form "
            '{"language":"<iso>",'
            '"segments":[{"start":0.0,"end":2.5,"text":"...",'
            '"words":[{"start":0.0,"end":0.4,"text":"..."}]}]}. '
            "Include per-word timings when you can detect them; omit the "
            "'words' array on a segment when uncertain. Do not invent text."
        )

        last_error: Optional[BaseException] = None
        response = None
        for model in cfg.gemini_fallback_models:
            try:
                response = _gemini_generate(client, model, uploaded, prompt, genai_types)
                cfg_model_used = model
                break
            except Exception as exc:
                last_error = exc
                if _is_rate_limit(exc):
                    logger.warning(f"gemini '{model}' rate-limited, trying next")
                else:
                    logger.warning(f"gemini '{model}' failed: {exc}")
                continue
        if response is None:
            raise PipelineError(
                STAGE_TRANSCRIBE,
                f"all Gemini transcribe models failed: {last_error}",
                cause=last_error,
            )

        raw_text = _strip_fences(response.text or "")
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise PipelineError(
                STAGE_TRANSCRIBE,
                f"Gemini returned non-JSON: {raw_text[:200]!r}",
                cause=exc,
            ) from exc

        segments: List[TranscriptSegment] = []
        words: List[WordTiming] = []
        for entry in (data.get("segments") or []):
            try:
                seg_start = float(entry["start"])
                seg_end = float(entry["end"])
            except (KeyError, TypeError, ValueError):
                continue
            seg_words: List[WordTiming] = []
            for w in entry.get("words") or []:
                try:
                    wt = WordTiming(
                        start=float(w["start"]),
                        end=float(w["end"]),
                        text=str(w.get("text", "")).strip(),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if wt.text:
                    seg_words.append(wt)
                    words.append(wt)
            text = str(entry.get("text", "")).strip()
            if not text and seg_words:
                text = " ".join(w.text for w in seg_words)
            if text:
                segments.append(TranscriptSegment(
                    start=seg_start, end=seg_end, text=text, words=seg_words,
                ))

        if not segments:
            raise PipelineError(
                STAGE_TRANSCRIBE, "Gemini returned no usable segments"
            )

        duration = _probe_duration(audio_path) or segments[-1].end
        return TranscriptionResult(
            segments=segments,
            words=words,
            duration=duration,
            engine="gemini",
            model=cfg_model_used,
            language=str(data.get("language") or "") or None,
        )

    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def transcribe(audio_path: Path, progress: ProgressFn = None) -> TranscriptionResult:
    """
    Try local Whisper first; if it raises, try Gemini. Never modifies the
    original audio file. Never leaves module-level state behind.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise PipelineError(
            STAGE_TRANSCRIBE, f"audio file not found: {audio_path}"
        )
    if audio_path.stat().st_size == 0:
        raise PipelineError(STAGE_TRANSCRIBE, f"audio file is empty: {audio_path}")

    cfg = transcriber_config()
    local_error: Optional[BaseException] = None

    try:
        if progress:
            progress("local Whisper")
        result = _transcribe_local(audio_path, cfg)
        if not result.is_empty():
            logger.info(
                f"transcribed via local Whisper ({result.model}): "
                f"{len(result.segments)} segments, {len(result.words)} words"
            )
            return result
        logger.warning("local Whisper returned empty text; trying Gemini")
    except Exception as exc:
        local_error = exc
        logger.warning(f"local Whisper failed: {exc}")

    try:
        if progress:
            progress("Gemini (fallback)")
        result = _transcribe_gemini(audio_path, cfg)
        if result.is_empty():
            raise PipelineError(STAGE_TRANSCRIBE, "Gemini also returned empty text")
        logger.info(
            f"transcribed via Gemini ({result.model}): "
            f"{len(result.segments)} segments, {len(result.words)} words"
        )
        return result
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            STAGE_TRANSCRIBE,
            "both local Whisper and Gemini transcription failed",
            hint="check that the audio is a valid media file",
            cause=exc if local_error is None else Exception(
                f"local={local_error}; gemini={exc}"
            ),
        ) from exc
