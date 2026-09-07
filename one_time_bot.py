#!/usr/bin/env python3
"""
One-Time Manga Recap Video Renderer (Telegram)

Bot start hota hai, GitHub Actions workflow start hote hi owner ko
Telegram par ek ping bhej deta hai ("main chal raha hoon"), phir files
ka wait karta hai. Files kisi bhi order mein, kisi bhi naam se aa sakti
hain — bot khud content dekh kar samajh leta hai ki kaunsi file kya hai:

    🖼️  ZIP (bina folder ke bhi chalega, andar jitni bhi images hongi
        wo sab uthaa li jayengi) — ya seedha loose image files
    🎧  Audio (aapka khud ka generate kiya hua voiceover)
    📝 .txt — bot khud (heuristic + Gemini) decide karta hai ki ye
        "image prompts / description" hai ya "narration script"
    🎵  Doosra audio file (agar pehla voiceover already mil chuka hai)
        → background music maana jaata hai

Jab zaroori files (images + audio/script) mil jaati hain, ~15 second
baad render khud-ba-khud shuru ho jaata hai (ya /render se turant).

Is version mein:
    - timeline.json (agar diya ho) ko HAMESHA use kiya jaata hai,
      sanity check sirf warning dega, reject nahi karega
    - Auto-timeline fallback bhi improve kiya hai
"""

import os
import re
import sys
import json
import uuid
import random
import logging
import asyncio
import subprocess
import shutil
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from functools import wraps

import requests
from dotenv import load_dotenv
from pydub import AudioSegment
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from sfx_engine import build_sfx_events
from video_editor import (
    prepare_scenes,
    render_video,
    natural_sort_key,
    QUALITY_PRESETS,
    QUALITY_ORDER,
    DEFAULT_QUALITY,
    quality_label,
    VideoEditorError,
)

# Force unbuffered stdout
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------------------------
load_dotenv()

API_ID = int(os.getenv("API_ID", "0") or "0")
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID")

GEMINI_MODELS_POOL = [
    m.strip() for m in os.getenv(
        "GEMINI_MODELS_POOL",
        "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash-lite,"
        "gemini-3-flash,gemini-3.6-flash,gemini-2.5-flash,gemini-2-flash"
    ).split(",") if m.strip()
]

WHISPER_LOCAL_MODEL = os.getenv("WHISPER_LOCAL_MODEL", "base")

DEBOUNCE_SECONDS = 15

if not all([API_ID, API_HASH, BOT_TOKEN]):
    raise ValueError("API_ID, API_HASH, BOT_TOKEN .env mein zaroor daalein!")
if not OPENAI_API_KEY or not GEMINI_API_KEY:
    raise ValueError("OPENAI_API_KEY aur GEMINI_API_KEY bhi chahiye!")

# ---------------------------------------------------------------------------
# Initialize APIs
# ---------------------------------------------------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Work Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "work"
INPUT_PANELS_DIR = WORK_DIR / "input_panels"
TEMP_DIR = WORK_DIR / "temp_workspace"
OUTPUT_DIR = WORK_DIR / "output"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pyrogram Client
# ---------------------------------------------------------------------------
app = Client(
    "one_time_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
)

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def ensure_work_dirs():
    for d in [INPUT_PANELS_DIR, TEMP_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for f in TEMP_DIR.glob("*"):
        if f.is_file():
            f.unlink()


def retry_with_backoff(max_retries=3, initial_delay=2.0, backoff_factor=2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
            if last_exc:
                raise last_exc
        return wrapper
    return decorator


class PipelineError(Exception):
    """User-facing pipeline error."""
    pass


# ---------------------------------------------------------------------------
# Startup notification
# ---------------------------------------------------------------------------

def _bot_api_call(method: str, payload: dict, timeout: int = 15) -> dict:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {}
    if resp.status_code != 200 or not data.get("ok"):
        raise RuntimeError(f"Telegram API {method} fail: status={resp.status_code} body={resp.text}")
    return data


def notify_owner_startup():
    if not OWNER_CHAT_ID:
        logger.warning("OWNER_CHAT_ID set nahi hai — startup ping skip.")
        return
    try:
        _bot_api_call("sendMessage", {
            "chat_id": OWNER_CHAT_ID,
            "text": "✅ Bot ka workflow start ho gaya hai, ab ye chal raha hai!\n\nFiles bhejo — kisi bhi order mein: ZIP (images), audio, (optional) prompts.txt. Jab sab mil jaayega, render khud shuru ho jaayega."
        })
        logger.info(f"✅ Startup ping Telegram par bhej diya (chat_id={OWNER_CHAT_ID})")
    except Exception as e:
        logger.error(f"❌ Startup ping FAIL ho gaya: {e}")


# ---------------------------------------------------------------------------
# Progress Reporter
# ---------------------------------------------------------------------------

class ProgressReporter:
    STAGES = [
        ("images", "🖼️ Images standardize ho rahi hain"),
        ("audio", "🎧 Audio taiyar ho raha hai"),
        ("transcribe", "⏱️ Transcription (timings nikal rahe hain)"),
        ("timeline", "📋 timeline.json load ho rahi hai"),
        ("sfx", "🔊 Sound effects dhoonde ja rahe hain"),
        ("assemble", "🎬 Clips + final video assemble ho raha hai"),
        ("done", "✅ Ho gaya!"),
    ]

    def __init__(self, chat_id: int, message_id: int, loop):
        self.chat_id = chat_id
        self.message_id = message_id
        self.loop = loop
        self._last_edit = 0.0
        self._min_gap = 3.0
        self._lock = asyncio.Lock()
        self._last_text = None

    @classmethod
    async def create(cls, chat_id: int, initial_text: str) -> "ProgressReporter":
        msg = await app.send_message(chat_id, initial_text)
        return cls(chat_id, msg.id, asyncio.get_event_loop())

    def _render(self, stage_key: str, detail: str = "") -> str:
        lines = ["🎬 <b>Render Progress</b>\n"]
        reached = True
        for key, label in self.STAGES:
            if key == stage_key:
                lines.append(f"▶️ {label}" + (f" — {detail}" if detail else ""))
                reached = False
            elif reached:
                lines.append(f"✅ {label}")
            else:
                lines.append(f"⏳ {label}")
        return "\n".join(lines)

    def update_sync(self, stage_key: str, detail: str = ""):
        text = self._render(stage_key, detail)
        if text == self._last_text:
            return
        now = time.time()
        if now - self._last_edit < self._min_gap and stage_key != "done":
            return
        self._last_edit = now
        self._last_text = text
        try:
            fut = asyncio.run_coroutine_threadsafe(self._edit(text), self.loop)
            fut.result(timeout=10)
        except Exception as e:
            logger.warning(f"Progress edit fail (ignored): {e}")

    async def _edit(self, text: str):
        try:
            await app.edit_message_text(self.chat_id, self.message_id, text)
        except Exception as e:
            logger.debug(f"edit_message_text skip: {e}")

    async def finish(self, final_text: str):
        try:
            await app.edit_message_text(self.chat_id, self.message_id, final_text)
        except Exception as e:
            logger.warning(f"Final progress edit fail: {e}")


# ---------------------------------------------------------------------------
# Pipeline Functions
# ---------------------------------------------------------------------------

class ScriptGenerator:
    def __init__(self, model: Optional[str] = None):
        self.model = model or GEMINI_MODELS_POOL[0]

    @retry_with_backoff()
    def generate_full_script(self, raw_text: str) -> str:
        prompt = f"""
        You are a professional manga/manhwa recap scriptwriter.
        Given the raw story text below, create a dramatic, engaging narration script.
        Write in natural spoken language, 2-4 minutes worth of narration.
        Do not include any formatting, just plain text paragraphs.

        Raw story:
        {raw_text}
        """
        response = gemini_client.models.generate_content(model=self.model, contents=prompt)
        return (response.text or "").strip()


@retry_with_backoff()
def generate_tts(text: str, output_path: Path):
    logger.info("🎙️ AI Voiceover generate ho raha hai...")
    if SARVAM_API_KEY:
        try:
            url = "https://api.sarvam.ai/text-to-speech"
            headers = {"Authorization": f"Bearer {SARVAM_API_KEY}", "Content-Type": "application/json"}
            payload = {"text": text, "language_code": "hi-IN", "voice": "default", "format": "wav"}
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                output_path.write_bytes(res.content)
                logger.info("✅ Sarvam AI TTS Success!")
                return
        except Exception as e:
            logger.warning(f"Sarvam fail: {e}, OpenAI fallback...")

    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text,
        response_format="mp3",
    )
    mp3_path = output_path.with_suffix(".mp3")
    response.stream_to_file(mp3_path)
    AudioSegment.from_mp3(mp3_path).export(output_path, format="wav")
    mp3_path.unlink()
    logger.info("✅ OpenAI TTS Success!")


def seg_get(seg, key, default=None):
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
    return raw.strip()


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(tok in msg for tok in ("429", "rate limit", "resource_exhausted", "quota"))


def _gemini_generate_with_rotation(contents, config=None, max_retries_per_model: int = 2):
    last_exc: Optional[Exception] = None
    for model in GEMINI_MODELS_POOL:
        delay = 3.0
        for attempt in range(1, max_retries_per_model + 1):
            try:
                if config is not None:
                    return gemini_client.models.generate_content(model=model, contents=contents, config=config)
                return gemini_client.models.generate_content(model=model, contents=contents)
            except Exception as e:
                last_exc = e
                if _is_rate_limit_error(e):
                    logger.warning(f"Gemini model '{model}' rate-limited — agle model par switch ho raha hai.")
                    break
                if attempt == max_retries_per_model:
                    logger.warning(f"Gemini model '{model}' fail ({e}) — agla model try ho raha hai.")
                    break
                time.sleep(delay)
                delay *= 2.0
    raise last_exc


_local_whisper_model = None
_LAST_WHISPER_WORDS: List[Dict] = []


def get_local_whisper_model():
    global _local_whisper_model
    if _local_whisper_model is None:
        import whisper as local_whisper_pkg
        logger.info(f"📦 Local (open-source) Whisper model '{WHISPER_LOCAL_MODEL}' load ho raha hai...")
        _local_whisper_model = local_whisper_pkg.load_model(WHISPER_LOCAL_MODEL)
    return _local_whisper_model


@retry_with_backoff()
def local_whisper_segments(audio_path: Path) -> List[Dict]:
    logger.info("⏱️ Local Whisper se timings nikal rahe hain (word-level)...")
    model = get_local_whisper_model()
    result = model.transcribe(str(audio_path), word_timestamps=True)
    raw_segments = result.get("segments", []) or []
    segments = [
        {
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": (seg.get("text", "") or "").strip(),
        }
        for seg in raw_segments
    ]
    if not segments:
        raise ValueError("Local Whisper ne khaali segments diye")
    words = []
    for seg in raw_segments:
        for w in (seg.get("words") or []):
            try:
                words.append({
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                    "text": (w.get("word", "") or "").strip(),
                })
            except (TypeError, ValueError):
                continue
    global _LAST_WHISPER_WORDS
    _LAST_WHISPER_WORDS = words
    return segments


@retry_with_backoff()
def gemini_transcribe_segments(audio_path: Path) -> List[Dict]:
    logger.info("⏱️ Gemini se audio timings nikal rahe hain...")
    uploaded = gemini_client.files.upload(file=str(audio_path))
    try:
        waited = 0.0
        while getattr(uploaded.state, "name", uploaded.state) == "PROCESSING" and waited < 60:
            time.sleep(2)
            waited += 2
            uploaded = gemini_client.files.get(name=uploaded.name)

        prompt = (
            "Transcribe this audio with timestamps. Output ONLY a JSON array "
            "of objects like: [{\"start\": 0.0, \"end\": 2.5, \"text\": \"...\"}]. "
            "No other text, no markdown fences."
        )
        response = _gemini_generate_with_rotation(
            contents=[uploaded, prompt],
            config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
        )
        raw = strip_json_fences(response.text or "")
        data = json.loads(raw)
        segments = [
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": str(item.get("text", "")).strip(),
            }
            for item in data
        ]
        if not segments:
            raise ValueError("Gemini ne khaali segments list wapas ki")
        return segments
    finally:
        try:
            gemini_client.files.delete(name=uploaded.name)
        except Exception:
            pass


@retry_with_backoff()
def openai_whisper_segments(audio_path: Path) -> List[Dict]:
    logger.info("⏱️ (Fallback) OpenAI Whisper se timings nikal rahe hain...")
    with open(audio_path, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    raw_segments = seg_get(transcript, "segments", []) or []
    return [
        {
            "start": float(seg_get(s, "start", 0.0)),
            "end": float(seg_get(s, "end", 0.0)),
            "text": (seg_get(s, "text", "") or "").strip(),
        }
        for s in raw_segments
    ]


def get_transcript_segments(audio_path: Path, progress: Optional[ProgressReporter] = None) -> List[Dict]:
    global _LAST_WHISPER_WORDS
    _LAST_WHISPER_WORDS = []
    errors = []
    try:
        if progress:
            progress.update_sync("transcribe", "local Whisper")
        segments = local_whisper_segments(audio_path)
        return segments
    except Exception as e:
        errors.append(f"local Whisper: {e}")
        logger.warning(f"⚠️ Local Whisper fail ho gaya ({e}), OpenAI Whisper API try kar rahe hain...")

    try:
        if progress:
            progress.update_sync("transcribe", "OpenAI Whisper (fallback)")
        segments = openai_whisper_segments(audio_path)
        return segments
    except Exception as e:
        errors.append(f"OpenAI Whisper: {e}")
        logger.warning(f"⚠️ OpenAI Whisper bhi fail ho gaya ({e}), Gemini try kar rahe hain...")

    try:
        if progress:
            progress.update_sync("transcribe", "Gemini (last resort)")
        segments = gemini_transcribe_segments(audio_path)
        return segments
    except Exception as e:
        errors.append(f"Gemini: {e}")

    raise PipelineError(
        "Audio transcribe nahi ho paaya, teeno methods fail ho gaye:\n" + "\n".join(errors)
    )


def fill_segment_gaps(raw_segments: List[Dict], total_duration: float) -> List[Dict]:
    segs = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text", "")}
        for s in raw_segments
    ]
    n = len(segs)
    if n == 0:
        return segs
    for i in range(n):
        segs[i]["end"] = segs[i + 1]["start"] if i + 1 < n else total_duration
    segs[0]["start"] = 0.0
    return segs


def _extract_leading_number(name: str) -> Optional[int]:
    match = re.match(r'(\d+)', name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def load_timeline(work_dir: Path, images: List[str], total_duration: float) -> List[Dict]:
    """timeline.json load karta hai, ab images ko numeric-prefix se match karta hai."""
    timeline_path = work_dir / "timeline.json"
    if not timeline_path.exists():
        raise PipelineError("timeline.json nahi mili!")
    try:
        raw = json.loads(timeline_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise PipelineError(f"timeline.json parse nahi ho payi: {e}")

    if not isinstance(raw, list) or not raw:
        raise PipelineError("timeline.json ek non-empty JSON array honi chahiye.")

    num_to_files: Dict[int, str] = {}
    for img in images:
        num = _extract_leading_number(img)
        if num is not None and num not in num_to_files:
            num_to_files[num] = img

    missing = []
    scenes = []
    for item in raw:
        img_ref = str(item.get("image"))
        actual = img_ref if img_ref in images else None
        if actual is None:
            num = _extract_leading_number(img_ref)
            if num is not None and num in num_to_files:
                actual = num_to_files[num]
        if actual is None:
            missing.append(img_ref)
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError) as e:
            raise PipelineError(f"timeline.json ka ek entry galat hai (start/end missing ya invalid): {item} ({e})")
        scenes.append({
            "start": start,
            "end": max(end, start + 0.05),
            "image_filename": actual,
            "text": str(item.get("reason", "")),
        })

    if missing:
        raise PipelineError(
            "timeline.json mein ye images numeric-prefix se bhi nahi mili: " + ", ".join(missing[:10]) +
            (" ... aur bhi" if len(missing) > 10 else "")
        )

    scenes.sort(key=lambda s: s["start"])
    scenes[0]["start"] = 0.0
    scenes[-1]["end"] = max(total_duration, scenes[-1]["start"] + 0.05)
    for i in range(1, len(scenes)):
        if scenes[i]["start"] < scenes[i - 1]["end"]:
            scenes[i]["start"] = scenes[i - 1]["end"]

    logger.info(f"📋 timeline.json se {len(scenes)} scenes load hui (numeric-prefix matching).")
    return scenes


def timeline_is_sane(scenes: List[Dict], total_duration: float) -> Tuple[bool, str]:
    """Ab sirf warning ke liye — rejection nahi."""
    n = len(scenes)
    if n == 0 or total_duration <= 0:
        return True, ""
    expected_avg = total_duration / n
    max_dur = max(s["end"] - s["start"] for s in scenes)
    cap = max(12.0, 2.5 * expected_avg)
    if max_dur > cap:
        return False, (
            f"sabse lambi scene {max_dur:.1f}s ki hai jabki {n} images ke "
            f"hisaab se expected average sirf ~{expected_avg:.1f}s hai "
            f"(sanity cap {cap:.1f}s) — ye warning hai, reject nahi."
        )
    return True, ""


def auto_generate_timeline(images: List[str], words: List[Dict], total_duration: float) -> List[Dict]:
    """Improved auto-timeline: pauses kam hain to boundaries evenly distribute hoti hain."""
    n = len(images)
    if n == 0:
        raise PipelineError("Auto-timeline ke liye koi image nahi mili.")

    gap_centers: List[float] = []
    sorted_words = sorted(words, key=lambda w: w["start"]) if words else []
    for i in range(len(sorted_words) - 1):
        gap_start = sorted_words[i]["end"]
        gap_end = sorted_words[i + 1]["start"]
        if gap_end > gap_start:
            gap_centers.append((gap_start + gap_end) / 2.0)
    gap_centers.sort()

    def nearest_gap_center(target: float) -> Optional[float]:
        if not gap_centers:
            return None
        import bisect
        idx = bisect.bisect_left(gap_centers, target)
        candidates = []
        if idx < len(gap_centers):
            candidates.append(gap_centers[idx])
        if idx > 0:
            candidates.append(gap_centers[idx - 1])
        return min(candidates, key=lambda c: abs(c - target))

    rough_boundaries = [total_duration * i / n for i in range(1, n)]

    snapped: List[float] = []
    for rb in rough_boundaries:
        snapped_point = nearest_gap_center(rb)
        snapped.append(snapped_point if snapped_point is not None else rb)

    # Ensure strictly increasing
    for i in range(1, len(snapped)):
        if snapped[i] <= snapped[i - 1]:
            snapped[i] = snapped[i - 1] + (total_duration / n)  # give a reasonable increment
            if snapped[i] >= total_duration:
                snapped[i] = snapped[i - 1] + 0.05

    boundaries = [0.0] + snapped + [total_duration]
    if boundaries[-2] >= boundaries[-1]:
        boundaries[-1] = boundaries[-2] + 0.05

    scenes = []
    for i, img in enumerate(images):
        scenes.append({
            "start": boundaries[i],
            "end": boundaries[i + 1],
            "image_filename": img,
            "text": "",
        })

    logger.info(f"📋 Auto-timeline generate hui: {n} images, {len(gap_centers)} natural pauses mile.")
    return scenes


# ---------------------------------------------------------------------------
# Run Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(work_dir: Path, quality: str = DEFAULT_QUALITY,
                  progress: Optional[ProgressReporter] = None) -> Path:
    global WORK_DIR, INPUT_PANELS_DIR, TEMP_DIR, OUTPUT_DIR
    WORK_DIR = work_dir
    INPUT_PANELS_DIR = WORK_DIR / "input_panels"
    TEMP_DIR = WORK_DIR / "temp_workspace"
    OUTPUT_DIR = WORK_DIR / "output"

    ensure_work_dirs()

    prompts_file = WORK_DIR / "prompts.txt"
    audio_file = TEMP_DIR / "voiceover.wav"
    output_mp4 = OUTPUT_DIR / "final_manga_recap.mp4"

    prompts_text = prompts_file.read_text(encoding="utf-8") if prompts_file.exists() else "No extra context."

    if progress:
        progress.update_sync("images", "scan ho raha hai")
    images = sorted(
        (f.name for f in INPUT_PANELS_DIR.glob("*") if f.suffix.lower() in IMAGE_EXTS),
        key=natural_sort_key,
    )
    if not images:
        raise PipelineError("Koi bhi image nahi mili! ZIP ya image files bhejo.")
    logger.info(f"🖼️ Total images: {len(images)}")

    if progress:
        progress.update_sync("audio", "voiceover check ho raha hai")
    custom_audio_found = False
    for ext in sorted(AUDIO_EXTS):
        possible = WORK_DIR / f"voiceover{ext}"
        if possible.exists():
            logger.info(f"🎧 Custom audio mil gaya: {possible.name}")
            AudioSegment.from_file(possible).export(audio_file, format="wav")
            custom_audio_found = True
            break

    if not custom_audio_found:
        script_file = WORK_DIR / "script.txt"
        if script_file.exists():
            logger.info("🤖 script.txt se AI voice ban rahi hai...")
            if progress:
                progress.update_sync("audio", "AI voiceover generate ho raha hai")
            narration_text = script_file.read_text(encoding="utf-8")
            generate_tts(narration_text, audio_file)
        else:
            raise PipelineError("Na custom audio mila, na script/story text!")

    segments = get_transcript_segments(audio_file, progress)
    total_duration = len(AudioSegment.from_file(audio_file)) / 1000.0
    gapped_segments = fill_segment_gaps(segments, total_duration)

    if progress:
        progress.update_sync("timeline", "image sequence taiyar ho rahi hai")

    scenes = None
    if (WORK_DIR / "timeline.json").exists():
        try:
            scenes = load_timeline(WORK_DIR, images, total_duration)
            ok, reason = timeline_is_sane(scenes, total_duration)
            if not ok:
                logger.warning(f"⚠️ timeline.json sanity warning ({reason}) — par use kar rahe hain.")
        except Exception as e:
            logger.warning(f"⚠️ timeline.json load fail ({e}), auto-generate karenge.")
            scenes = None
    if scenes is None:
        scenes = auto_generate_timeline(images, _LAST_WHISPER_WORDS, total_duration)

    scenes = prepare_scenes(scenes)

    if progress:
        progress.update_sync("sfx", "sound effects dhoonde ja rahe hain")
    sfx_events = build_sfx_events(
        gapped_segments, prompts_text, TEMP_DIR,
        progress_callback=(lambda msg: progress.update_sync("sfx", msg)) if progress else None,
    )

    bgm_path = next(iter(sorted(WORK_DIR.glob("bgm.*"))), None)
    try:
        if progress:
            progress.update_sync("assemble", f"{quality} mein render ho raha hai")
        render_video(
            scenes, audio_file, bgm_path, output_mp4,
            INPUT_PANELS_DIR, TEMP_DIR,
            quality=quality,
            sfx_events=sfx_events,
            progress_callback=(lambda msg: progress.update_sync("assemble", msg)) if progress else None,
        )
    except VideoEditorError as e:
        raise PipelineError(str(e))
    return output_mp4


# ---------------------------------------------------------------------------
# Smart file-type detection
# ---------------------------------------------------------------------------

def sniff_kind(path: Path) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except Exception:
        return None
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return "zip"
    if head.startswith(b"\xff\xd8\xff") or head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if head[0:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio"
    if head.startswith(b"ID3") or head[0:2] == b"\xff\xfb" or head.startswith(b"OggS") or head.startswith(b"fLaC"):
        return "audio"
    try:
        head.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return None


def classify_incoming_file(path: Path, filename_hint: str) -> str:
    ext = Path(filename_hint).suffix.lower()
    if ext == ".zip":
        return "zip"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext == ".json":
        return "timeline"
    if ext == ".txt":
        return "text"
    return sniff_kind(path) or "unknown"


def try_parse_timeline(text: str) -> Optional[list]:
    try:
        data = json.loads(text)
    except Exception:
        return None
    if isinstance(data, list) and data and all(
        isinstance(item, dict) and "image" in item and "start" in item and "end" in item
        for item in data
    ):
        return data
    return None


def extract_images_from_zip(zip_path: Path, dest_dir: Path) -> int:
    extract_tmp = zip_path.parent / f"extract_{uuid.uuid4().hex[:8]}"
    extract_tmp.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_tmp)
    except zipfile.BadZipFile:
        shutil.rmtree(extract_tmp, ignore_errors=True)
        raise ValueError("Ye valid ZIP file nahi hai (corrupt ho sakti hai).")

    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(extract_tmp.rglob("*"), key=lambda p: natural_sort_key(p.name)):
        if not f.is_file() or f.name.startswith(".") or "__MACOSX" in f.parts:
            continue
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        target = dest_dir / f.name
        if target.exists():
            target = dest_dir / f"{f.stem}_{uuid.uuid4().hex[:6]}{f.suffix}"
        shutil.move(str(f), str(target))
        count += 1
    shutil.rmtree(extract_tmp, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    return count


def classify_text_role(text: str, already_have_script: bool) -> str:
    sample = text.strip()
    if not sample:
        return "prompts"

    lines = [l.strip() for l in sample.splitlines() if l.strip()]
    keywords = ["panel", "image", "scene", "shot", "background", "character design", "art style"]
    if lines:
        short_ratio = sum(1 for l in lines if len(l) < 60) / len(lines)
        bullet_ratio = sum(1 for l in lines if l.startswith(("-", "*", "•")) or re.match(r"^\d+[.)]", l)) / len(lines)
        keyword_hits = sum(1 for l in lines for kw in keywords if kw in l.lower())
        if len(lines) >= 3 and (bullet_ratio > 0.4 or (short_ratio > 0.7 and keyword_hits >= 2)):
            return "prompts"
        if len(lines) >= 3 and short_ratio < 0.3 and not already_have_script:
            return "script"

    try:
        prompt = (
            "Classify this text as either 'prompts' (image descriptions/panel "
            "notes for an artist) or 'script' (spoken narration for a video). "
            "Reply with ONLY one word: prompts or script.\n\nText:\n" + sample[:2000]
        )
        response = _gemini_generate_with_rotation(contents=prompt)
        verdict = (response.text or "").strip().lower()
        if "script" in verdict:
            return "script"
        return "prompts"
    except Exception as e:
        logger.warning(f"classify_text_role Gemini fallback fail ({e}), defaulting to 'prompts'.")
        return "prompts"


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------

sessions: Dict[int, dict] = {}


def get_session(chat_id: int) -> dict:
    if chat_id not in sessions:
        sessions[chat_id] = {
            "work_dir": BASE_DIR / "work" / str(chat_id),
            "images_count": 0,
            "has_audio": False,
            "has_bgm": False,
            "has_script": False,
            "has_timeline": False,
            "prompts_chars": 0,
            "debounce_task": None,
            "processing": False,
            "quality": None,
            "quality_prompt_sent": False,
        }
    return sessions[chat_id]


def reset_session(chat_id: int):
    sess = sessions.pop(chat_id, None)
    if sess:
        task = sess.get("debounce_task")
        if task and not task.done():
            task.cancel()
    shutil.rmtree(BASE_DIR / "work" / str(chat_id), ignore_errors=True)


def is_session_ready(sess: dict) -> bool:
    return (
        sess["images_count"] > 0
        and (sess["has_audio"] or sess["has_script"])
    )


def build_status_text(sess: dict) -> str:
    ready = is_session_ready(sess)
    lines = [
        f"{'✅' if sess['images_count'] > 0 else '⏳'} Images: {sess['images_count']}",
        f"{'✅' if sess['has_audio'] else '⏳'} Voiceover audio"
        + ("" if sess["has_audio"] else " (ya niche wala script bhejo)"),
        f"{'✅' if sess['has_script'] else '➖'} Script/story text"
        + (" (optional, audio ke bina zaroori)" if not sess["has_audio"] else " (optional)"),
        f"{'✅' if sess['has_timeline'] else '➖'} timeline.json (optional — na ho to bot khud "
        f"filename-order + audio-pause se sequence banayega)",
        f"{'✅' if sess['prompts_chars'] > 0 else '➖'} Image prompts (optional, ab sirf SFX-context ke liye)",
        f"{'✅' if sess['has_bgm'] else '➖'} Background music (optional)",
        f"{'✅ ' + sess['quality'] if sess.get('quality') else '⏳'} Video quality"
        + ("" if sess.get("quality") else " (sab files mil jaane par button se chunni hogi)"),
    ]
    if ready and not sess.get("quality"):
        footer = "\n\n🎚️ Saari zaroori files mil gayi — quality-select buttons ka wait karo (ya /render bhejo)."
    elif ready and sess.get("quality"):
        footer = "\n\n🚀 Sab ready — render chal raha hai / shuru hone wala hai."
    else:
        footer = "\n\n⏳ Abhi aur files chahiye (kam se kam: images + audio (ya script) + timeline.json)."
    return "\n".join(lines) + footer


def quality_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(f"🎚️ {quality_label(q)}", callback_data=f"quality:{q}")]
        for q in QUALITY_ORDER
    ]
    return InlineKeyboardMarkup(buttons)


def schedule_auto_render(chat_id: int):
    sess = get_session(chat_id)
    old_task = sess.get("debounce_task")
    if old_task and not old_task.done():
        old_task.cancel()
    sess["debounce_task"] = asyncio.create_task(debounced_render(chat_id))


async def debounced_render(chat_id: int):
    try:
        await asyncio.sleep(DEBOUNCE_SECONDS)
    except asyncio.CancelledError:
        return
    await try_start_pipeline(chat_id, force=False)


async def try_start_pipeline(chat_id: int, force: bool):
    sess = get_session(chat_id)
    if sess["processing"]:
        if force:
            await app.send_message(chat_id, "⏳ Render pehle se chal raha hai...")
        return
    ready = is_session_ready(sess)
    if not ready:
        if force:
            await app.send_message(chat_id, "❌ Abhi render nahi ho sakta.\n\n" + build_status_text(sess))
        return

    if not sess.get("quality"):
        if force or not sess.get("quality_prompt_sent"):
            sess["quality_prompt_sent"] = True
            await app.send_message(
                chat_id,
                "🎬 Saari zaroori files mil gayi!\n\nRender shuru karne se pehle video quality chuno (neeche button dabao):",
                reply_markup=quality_keyboard(),
            )
        return

    await start_render(chat_id, sess["quality"])


async def start_render(chat_id: int, quality: str):
    sess = get_session(chat_id)
    sess["processing"] = True
    progress = await ProgressReporter.create(
        chat_id,
        f"🎬 Video generation shuru — quality: {quality} (kuch minute lag sakte hain)..."
    )
    try:
        output_video = await asyncio.to_thread(run_pipeline, sess["work_dir"], quality, progress)
        await progress.finish("✅ Render complete! Video upload ho raha hai...")
        await app.send_video(
            chat_id=chat_id,
            video=str(output_video),
            caption=f"✅ Aapka recap video ready! ({quality})",
            supports_streaming=True,
        )
        await progress.finish("✅ Ho gaya — video upar bheji ja chuki hai!")
    except PipelineError as e:
        logger.exception("Pipeline error")
        await progress.finish(f"❌ Render fail ho gaya:\n\n{e}")
    except Exception as e:
        logger.exception("Unexpected pipeline error")
        await progress.finish(f"❌ Kuch anjaan error aa gaya:\n\n{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(sess["work_dir"], ignore_errors=True)
        sessions.pop(chat_id, None)
        os._exit(0)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@app.on_message(filters.command("start"))
async def start_handler(client, message):
    await message.reply_text(
        "✅ Bot chal raha hai!\n\n"
        f"Aapka chat ID: {message.chat.id}\n"
        "(Ise OWNER_CHAT_ID GitHub secret mein daal do taaki agli baar "
        "workflow start hote hi aapko yahin ping mil jaaye.)\n\n"
        "Ab bas files bhejo — kisi bhi order mein, jitni marzi ek saath:\n"
        "🖼️ ZIP (sirf images, folder ho ya na ho, farq nahi padta) — ya loose images "
        "(filename mein number ho, jaise 1_xyz.jpg, 2_xyz.jpg — sequence isi se banegi)\n"
        "🎧 Audio (aapka khud ka voiceover)\n"
        "📋 (Optional) timeline.json — [{image, start, end}, ...] format mein "
        "custom image-sequence. Na do to bot khud filename-number se sequence "
        "banayega aur audio ke natural pauses par timing snap karega — AI se "
        "sequence kabhi guess nahi karwaya jaata.\n"
        "📝 (Optional) extra story-context wali .txt file\n"
        "🎵 (Optional) doosra audio file = background music\n\n"
        "Sab files mil jaane ke baad bot khud tumhe QUALITY CHUNNE ke liye "
        "buttons dega (360p/480p/720p/1080p — koi message type nahi karna "
        "padega, bas button dabao). Button dabate hi render turant shuru "
        "ho jaata hai.\n"
        "Agar FREESOUND_API_KEY set hai to important moments par CC0 "
        "sound-effects bhi automatically add hote hain.\n\n"
        "Commands: /status /render /reset"
    )


@app.on_message(filters.command("status") & filters.private)
async def status_cmd(client, message):
    await message.reply_text(build_status_text(get_session(message.chat.id)))


@app.on_message(filters.command("reset") & filters.private)
async def reset_cmd(client, message):
    reset_session(message.chat.id)
    await message.reply_text("🔄 Session clear kar di. Naye sirey se files bhejo.")


@app.on_message(filters.command("render") & filters.private)
async def render_cmd(client, message):
    await try_start_pipeline(message.chat.id, force=True)


@app.on_callback_query(filters.regex(r"^quality:"))
async def quality_callback(client, callback_query):
    chat_id = callback_query.message.chat.id
    quality = callback_query.data.split(":", 1)[1]
    if quality not in QUALITY_PRESETS:
        await callback_query.answer("❌ Invalid quality option.", show_alert=True)
        return

    sess = get_session(chat_id)
    if sess["processing"]:
        await callback_query.answer("⏳ Render pehle se chal raha hai...", show_alert=True)
        return

    sess["quality"] = quality
    await callback_query.answer(f"✅ {quality} select ho gaya!")
    try:
        await callback_query.message.edit_text(
            f"✅ Quality select ho gayi: {quality}\n🎬 Render shuru ho raha hai..."
        )
    except Exception:
        pass
    await try_start_pipeline(chat_id, force=True)


@app.on_message(filters.private & (filters.document | filters.audio | filters.voice | filters.photo))
async def handle_media(client, message):
    chat_id = message.chat.id
    sess = get_session(chat_id)
    work_dir = sess["work_dir"]
    (work_dir / "input_panels").mkdir(parents=True, exist_ok=True)

    if message.document:
        file_name = message.document.file_name or f"file_{uuid.uuid4().hex[:6]}"
    elif message.audio:
        file_name = message.audio.file_name or f"audio_{uuid.uuid4().hex[:6]}.mp3"
    elif message.voice:
        file_name = f"voice_{uuid.uuid4().hex[:6]}.ogg"
    elif message.photo:
        file_name = f"photo_{uuid.uuid4().hex[:6]}.jpg"
    else:
        return

    ext = Path(file_name).suffix.lower()
    dl_path = work_dir / f"incoming_{uuid.uuid4().hex[:8]}{ext}"
    note = None
    try:
        await client.download_media(message, file_name=str(dl_path))
        kind = classify_incoming_file(dl_path, file_name)

        if kind == "zip":
            try:
                n = extract_images_from_zip(dl_path, work_dir / "input_panels")
            except ValueError as ve:
                note = f"❌ {ve}"
                n = 0
            if n:
                sess["images_count"] += n
                note = f"✅ {n} images mili is ZIP se (total {sess['images_count']})."
            elif not note:
                note = "❌ ZIP mein koi image nahi mili (.jpg/.jpeg/.png/.webp/.bmp)."

        elif kind == "image":
            target = work_dir / "input_panels" / (file_name if ext else f"{uuid.uuid4().hex[:8]}.jpg")
            if target.exists():
                target = work_dir / "input_panels" / f"{target.stem}_{uuid.uuid4().hex[:6]}{target.suffix}"
            shutil.move(str(dl_path), str(target))
            sess["images_count"] += 1
            note = f"🖼️ Image add ho gayi (total {sess['images_count']})."

        elif kind == "audio":
            if not sess["has_audio"]:
                voiceover_path = work_dir / f"voiceover{ext if ext in AUDIO_EXTS else '.ogg'}"
                shutil.move(str(dl_path), str(voiceover_path))
                sess["has_audio"] = True
                note = "🎧 Voiceover audio mil gaya!"
            else:
                bgm_path = work_dir / f"bgm{ext if ext in AUDIO_EXTS else '.mp3'}"
                shutil.move(str(dl_path), str(bgm_path))
                sess["has_bgm"] = True
                note = "🎵 Background music mil gaya!"

        elif kind == "timeline":
            text = dl_path.read_text(encoding="utf-8", errors="ignore")
            dl_path.unlink(missing_ok=True)
            parsed = try_parse_timeline(text)
            if parsed is None:
                note = "❌ Ye .json valid timeline format mein nahi hai (har item mein image/start/end chahiye)."
            else:
                (work_dir / "timeline.json").write_text(text, encoding="utf-8")
                sess["has_timeline"] = True
                note = f"📋 timeline.json mil gayi ({len(parsed)} entries) — image sequence isi se aayegi."

        elif kind == "text":
            text = dl_path.read_text(encoding="utf-8", errors="ignore")
            dl_path.unlink(missing_ok=True)
            parsed = try_parse_timeline(text)
            if parsed is not None:
                (work_dir / "timeline.json").write_text(text, encoding="utf-8")
                sess["has_timeline"] = True
                note = f"📋 timeline.json mil gayi ({len(parsed)} entries) — image sequence isi se aayegi."
            else:
                role = await asyncio.to_thread(classify_text_role, text, sess["has_script"])
                if role == "prompts":
                    with open(work_dir / "prompts.txt", "a", encoding="utf-8") as f:
                        f.write(text.strip() + "\n")
                    sess["prompts_chars"] += len(text)
                    note = "📝 Context note kar liya (SFX-detection ko story samajhne mein help karega)."
                else:
                    (work_dir / "script.txt").write_text(text, encoding="utf-8")
                    sess["has_script"] = True
                    note = "📜 Script/story mil gaya (agar audio na bheja to isi se voice banegi)."

        else:
            dl_path.unlink(missing_ok=True)
            note = f"🤔 {file_name} samajh nahi aayi — ZIP, image, audio, timeline .json ya .txt bhejo."

    except Exception as e:
        logger.exception("File handling error")
        note = f"❌ File process karte waqt error: {e}"

    await message.reply_text(f"{note}\n\n{build_status_text(sess)}")
    if note and not note.startswith("❌"):
        schedule_auto_render(chat_id)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("🤖 Bot starting... workflow start hote hi owner ko Telegram par ping jaayega.")
    notify_owner_startup()
    app.run()
