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
    - Video ab 60 FPS mein render hota hai (smooth motion)
    - Har image force-standardize hoti hai poori 1920x1080 canvas par
      (chhoti/badi/alag-aspect-ratio images bhi ab sahi 1080p frame
      banaengi, stretch ya crop distortion nahi)
    - Live progress: Telegram par EK hi status message baar baar
      edit hota hai (spam nahi), jisme stage-by-stage % dikhta hai —
      images ready, transcription, AI sync, clip N/Total, final render
    - Better error handling: har stage try/except mein wrapped hai,
      asli error Telegram par bhi jaata hai (generic crash nahi)
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
from typing import List, Dict, Optional
from functools import wraps

import requests
from dotenv import load_dotenv
from pydub import AudioSegment
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# SFX aur video-rendering ab do independent, plug-and-play modules mein
# hain — future upgrade ke liye sirf wahi files replace/edit karni
# hongi, ye main bot file touch nahi karni padegi.
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

# Force unbuffered stdout so logs show up immediately in GitHub Actions
# (bina isske "print"/log lines buffer mein atak jaate hain aur
# Actions log mein der se ya kabhi kabhi bilkul nahi dikhte)
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

# Owner ka Telegram chat id — startup-ping isi id par jaayega.
# /start bhejo bot ko ek baar, wo reply mein chat id de dega, use yaha /
# GitHub secret OWNER_CHAT_ID mein daal do.
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID")

# --- Gemini MODEL ROTATION -------------------------------------------------
# Gemini ka use jaan-boojh kar KAM se KAM rakha gaya hai: transcription
# pehle hamesha LOCAL Whisper try karta hai (koi API call hi nahi),
# Gemini sirf teesre/last-resort fallback ke roop mein aur ambiguous
# text-classification ke liye use hota hai. SFX detection ka Gemini use
# ab sfx_engine.py ke andar hai (poori tarah independent).
#
# Jab bhi Gemini call karna padta hai, ye SIRF EK model try nahi karta —
# GEMINI_MODELS_POOL mein se ek-ek karke try karta hai. Jaise hi koi
# model rate-limit (429/quota) de, turant AGLE model par switch ho jaata
# hai (wait kiye bina) — kyunki Google har model ko ALAG RPM/TPM/RPD
# quota deta hai, ek model ka quota khatam hone ka matlab ye nahi ki
# baaki models bhi khatam hain. Isi tarah effectively rate-limit
# "bypass" hoti hai — bina kisi ek model par zyada dependent hue.
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
# NOTE: purana "google-generativeai" package Google ne deprecate kar diya
# hai (unstable ho sakta hai), isliye naya unified "google-genai" SDK use
# kar rahe hain.
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


# NOTE: natural_sort_key ab video_editor.py se import hota hai (upar) —
# duplicate implementation yahan jaan-boojh kar nahi rakhi, taaki sort
# logic ek hi jagah maintain ho.

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
    """User-facing pipeline error — jo message isme diya jayega, wahi
    seedha Telegram status message par dikhega, isliye readable rakho."""
    pass


# ---------------------------------------------------------------------------
# Startup notification (FIXED)
# ---------------------------------------------------------------------------
# Purana issue: agar ye Pyrogram client se (app.send_message) bheja jaaye,
# to har naye GitHub Actions run mein session/peer-cache khaali hota hai,
# aur bot chat_id ko resolve nahi kar paata (PEER_ID_INVALID) — chat id sahi
# hone ke bawajood error aata hai. Fix: seedha Telegram Bot HTTP API se
# sendMessage call karo — usko peer-cache ki zaroorat nahi hoti, sirf itna
# chahiye ki user ne bot ko kabhi bhi (kisi bhi purane run mein) /start
# kiya ho.

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
        logger.warning(
            "⚠️ OWNER_CHAT_ID set nahi hai — startup ping skip. Bot ko ek baar "
            "/start bhejo, reply mein chat id milega, use OWNER_CHAT_ID secret mein daal do."
        )
        return
    try:
        _bot_api_call("sendMessage", {
            "chat_id": OWNER_CHAT_ID,
            "text": (
                "✅ Bot ka workflow start ho gaya hai, ab ye chal raha hai!\n\n"
                "Files bhejo — kisi bhi order mein: ZIP (images), audio, "
                "(optional) prompts.txt. Jab sab mil jaayega, render khud shuru ho jaayega.\n"
                "Manually shuru karne ke liye /render bhejo."
            ),
        })
        logger.info(f"✅ Startup ping Telegram par bhej diya (chat_id={OWNER_CHAT_ID})")
    except Exception as e:
        logger.error(f"❌ Startup ping FAIL ho gaya: {e}")


# ---------------------------------------------------------------------------
# Progress Reporter — ek hi message edit hota hai, spam nahi hota
# ---------------------------------------------------------------------------

class ProgressReporter:
    """Ek single Telegram message ko baar baar edit karta hai taaki user ko
    live progress dikhe bina inbox spam kiye. Edits ko throttle bhi karta
    hai (min gap) taaki Telegram rate-limit na lage."""

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
        self._min_gap = 3.0  # seconds — isse zyada baar edit nahi karenge
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
        """Kisi bhi thread se safely call karne ke liye (thread-pool mein
        chalne wale ffmpeg/whisper code se progress bhejne ka tarika)."""
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
            # Telegram "message not modified" jaisi cheezein ignore karo
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
    """Available hai agar kabhi raw story text ko polished narration script
    mein expand karna ho. Filhaal auto-flow ise trigger nahi karta (aap
    hamesha khud audio ya ready script bhejte ho), lekin function ready hai."""

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
    """OpenAI SDK ke transcription segment object dict ya pydantic model
    dono ho sakte hain — dono se safely value nikalta hai."""
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
    """Har Gemini call jo bot khud karta hai (transcription fallback,
    text-role classification) isi se guzarta hai. GEMINI_MODELS_POOL
    mein se ek-ek model try karta hai; jaise hi koi model rate-limit de,
    turant agle model par switch (wait kiye bina) — kyunki har model ki
    quota (RPM/TPM/RPD) alag hoti hai. Exception sirf tab raise hoti hai
    jab SAARE models fail ho jaayein."""
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
                    logger.warning(
                        f"Gemini model '{model}' rate-limited — agle model par switch ho raha hai."
                    )
                    break
                if attempt == max_retries_per_model:
                    logger.warning(f"Gemini model '{model}' fail ({e}) — agla model try ho raha hai.")
                    break
                time.sleep(delay)
                delay *= 2.0
    raise last_exc


_local_whisper_model = None  # lazy-loaded singleton — model load slow hai, ek hi baar karo
_LAST_WHISPER_WORDS: List[Dict] = []  # local_whisper_segments() ke word-level output ka cache — auto_generate_timeline() isse padhta hai


def get_local_whisper_model():
    """openai-whisper (open-source pip package, import whisper) — ye
    OpenAI ki paid transcription API se bilkul alag hai, koi API key ya
    rate-limit nahi lagta, poora model local chalta hai."""
    global _local_whisper_model
    if _local_whisper_model is None:
        import whisper as local_whisper_pkg
        logger.info(f"📦 Local (open-source) Whisper model '{WHISPER_LOCAL_MODEL}' load ho raha hai...")
        _local_whisper_model = local_whisper_pkg.load_model(WHISPER_LOCAL_MODEL)
    return _local_whisper_model


@retry_with_backoff()
def local_whisper_segments(audio_path: Path) -> List[Dict]:
    """Primary transcription path — open-source Whisper, poori tarah
    local (GitHub Actions runner par hi chalta hai, ffmpeg pehle se
    installed hai workflow mein). Na koi API limit, na koi cost.
    word_timestamps=True rakha hai taaki auto_generate_timeline() ko
    fine-grained word-level pauses milein (image-boundary snapping ke
    liye) — sentence-level se bhi zyada precise natural cut-points."""
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
    # Word-level timestamps ek global cache mein rakhte hain (segments
    # list se return nahi karte taaki har existing caller — jo sirf
    # start/end/text expect karta hai — bina change kiye chalta rahe).
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
    """Extra fallback: Gemini audio understanding (agar local Whisper
    kisi wajah se fail ho jaaye)."""
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
    """Fallback path — sirf tab use hota hai jab local aur Gemini dono
    fail ho jaayein."""
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
    """Audio -> timed segments (image-sync ke liye use hote hain, video mein
    subtitles burn nahi hote — sirf timing/text ka internal use hai). Priority order:
      1) Local open-source Whisper — na API key, na rate-limit, na cost.
      2) OpenAI Whisper API — fallback.
      3) Gemini — last-resort fallback.
    Teeno fail ho jaayein to PipelineError raise hoti hai (upar tak
    readable message pahunchta hai, silent crash nahi hota)."""
    # Naye run se pehle purana word-cache saaf karo — warna local Whisper
    # is baar fail ho jaaye (OpenAI/Gemini fallback chale) to auto-timeline
    # galti se pichle audio file ke stale words use kar legi.
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
    """Whisper sirf jahan speech hai wahi ke segments deta hai — beech mein
    jo pauses/silences hote hain wo kisi segment mein cover nahi hote.
    Fix: har gap ko pichli line ki image tak extend kar dete hain, aur
    shuru/end ke silence ko bhi cover kar dete hain — taaki poora audio
    duration hamesha kisi na kisi image se covered rahe."""
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


def load_timeline(work_dir: Path, images: List[str], total_duration: float) -> List[Dict]:
    """Image-sequence ab Gemini se guess nahi karwate — user jo timeline.json
    deta hai (kisi aur AI/tool se banaya hua, [{image,start,end,...}] format
    mein) usi ko ground-truth maan kar seedha use karte hain. Ye REQUIRED
    hai, koi fallback/auto-sync nahi — is file ke bina render start hi
    nahi hota (upstream mein bhi check hai, yahan bhi defensive re-check)."""
    timeline_path = work_dir / "timeline.json"
    if not timeline_path.exists():
        raise PipelineError(
            "timeline.json nahi mili! Ye ab required hai — image sequence "
            "isi file se aati hai, Gemini se guess nahi karwaya jaata."
        )
    try:
        raw = json.loads(timeline_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise PipelineError(f"timeline.json parse nahi ho payi: {e}")

    if not isinstance(raw, list) or not raw:
        raise PipelineError("timeline.json ek non-empty JSON array honi chahiye.")

    valid_images = set(images)
    missing = sorted({str(item.get("image")) for item in raw if str(item.get("image")) not in valid_images})
    if missing:
        raise PipelineError(
            "timeline.json mein ye images ZIP mein maujood nahi hain: " + ", ".join(missing[:15]) +
            (" ... aur bhi" if len(missing) > 15 else "")
        )

    scenes = []
    for item in raw:
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError) as e:
            raise PipelineError(f"timeline.json ka ek entry galat hai (start/end missing ya invalid): {item} ({e})")
        scenes.append({
            "start": start,
            "end": max(end, start + 0.05),
            "image_filename": str(item["image"]),
            "text": str(item.get("reason", "")),
        })

    scenes.sort(key=lambda s: s["start"])
    # Safety clamp — real audio ke 0 se end tak poori coverage guarantee:
    # dusre AI/tool ki file mein chhoti rounding-mismatch ho sakti hai.
    scenes[0]["start"] = 0.0
    scenes[-1]["end"] = max(total_duration, scenes[-1]["start"] + 0.05)
    for i in range(1, len(scenes)):
        if scenes[i]["start"] < scenes[i - 1]["end"]:
            scenes[i]["start"] = scenes[i - 1]["end"]

    logger.info(f"📋 timeline.json se {len(scenes)} scenes load hui (Gemini image-sync bypass).")
    return scenes


def auto_generate_timeline(images: List[str], words: List[Dict], total_duration: float) -> List[Dict]:
    """timeline.json na ho to bot khud image-sequence + timing banata hai —
    Gemini ka is decision mein koi role nahi (na sequence mein, na timing
    mein), isliye 25-ke-baad-28 jaisi galtiyan yahan possible hi nahi hain.

    Logic:
      1) Images ka sequence sirf filename ke number se aata hai (caller
         se already natural_sort_key se sorted aata hai) — koi AI guess
         nahi.
      2) Total audio duration ko N images mein rough equal-split karte
         hain — N-1 internal boundaries.
      3) Har boundary ko sabse nikatam Whisper word-gap (pause) ke center
         par "snap" karte hain, taaki cut kisi word ke beech mein na aaye.
         Agar us boundary ke aas-paas koi pause hi nahi hai (continuous
         bolna chal raha hai), rough-split boundary hi as-is rehta hai.
      4) Ek boundary snap hone ka asar agli image ke start par bhi padta
         hai (chain), isliye final list left-to-right mein consistent
         (non-decreasing, gap-free) bana kar return karte hain.
    """
    n = len(images)
    if n == 0:
        raise PipelineError("Auto-timeline ke liye koi image nahi mili.")

    # Consecutive words ke beech ke gaps — yahi hamare "pause" candidates
    # hain. Har gap ka center hi snap-target hota hai.
    gap_centers: List[float] = []
    sorted_words = sorted(words, key=lambda w: w["start"]) if words else []
    for i in range(len(sorted_words) - 1):
        gap_start = sorted_words[i]["end"]
        gap_end = sorted_words[i + 1]["start"]
        if gap_end > gap_start:
            gap_centers.append((gap_start + gap_end) / 2.0)
    gap_centers.sort()

    def nearest_gap_center(target: float) -> Optional[float]:
        """Binary-search se target ke sabse nikatam gap-center dhoondo.
        Koi gap hi na ho to None (caller rough-split boundary rakhega)."""
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

    # Rough equal-split boundaries (N-1 internal cut points).
    rough_boundaries = [total_duration * i / n for i in range(1, n)]

    # Har boundary ko nearest pause-center par snap karo.
    snapped: List[float] = []
    for rb in rough_boundaries:
        snapped_point = nearest_gap_center(rb)
        snapped.append(snapped_point if snapped_point is not None else rb)

    # Consistency guarantee — snapping ke baad bhi boundaries strictly
    # badhte kram mein rahein (do boundaries ek hi gap par snap ho sakti
    # hain agar images bahut chhoti hon ya pauses sparse hon), warna
    # scenes overlap/negative-duration ho jaayenge.
    for i in range(1, len(snapped)):
        if snapped[i] <= snapped[i - 1]:
            snapped[i] = snapped[i - 1] + 0.05

    boundaries = [0.0] + snapped + [total_duration]
    # Aakhri clamp — agar upar wale +0.05 nudge se total_duration cross ho
    # gaya ho (bahut zyada images, bahut kam audio — edge case), to end
    # ko hi wapas total_duration par le aao taaki video duration na badhe.
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

    logger.info(
        f"📋 Auto-timeline generate hui: {n} images, "
        f"{len(gap_centers)} natural pauses mile, "
        f"{sum(1 for rb, s in zip(rough_boundaries, snapped) if s != rb)} boundaries snap hui."
    )
    return scenes


# ---------------------------------------------------------------------------
# NOTE: SFX detection/search/download ab sfx_engine.py mein hai
# (build_sfx_events — import upar). Image standardize, zoom/pan effects
# aur final FFmpeg assembly ab video_editor.py mein hain (prepare_scenes,
# render_video — import upar). Ye main bot file sirf inhe orchestrate
# karti hai, implementation details ab yahan nahi hain.
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

    # Hybrid audio: pehle khud ka bheja hua voiceover dhoondo
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

    # Image-sequence: agar user ne timeline.json di hai to wahi ground-
    # truth maani jaati hai (backward compatible). Warna bot khud
    # auto_generate_timeline() se sequence + timing banata hai — filename
    # number se sort + Whisper word-level pauses se natural snapping.
    # Dono cases mein Gemini ka is decision mein koi role nahi.
    if progress:
        progress.update_sync("timeline", "image sequence taiyar ho rahi hai")
    if (WORK_DIR / "timeline.json").exists():
        scenes = load_timeline(WORK_DIR, images, total_duration)
    else:
        scenes = auto_generate_timeline(images, _LAST_WHISPER_WORDS, total_duration)
    # Safety net: agar timeline mein kahin consecutive entries same image
    # ki hon (source AI/tool ne pre-merge nahi kiya), tab bhi animation
    # restart wala bug na aaye — idempotent hai, already-merged timeline
    # par koi asar nahi padta. (merge + effect-assignment ab video_editor
    # ke andar hai.)
    scenes = prepare_scenes(scenes)

    if progress:
        progress.update_sync("sfx", "sound effects dhoonde ja rahe hain")
    # SFX poori tarah independent module se — is call ke fail hone se
    # bhi render kabhi nahi rukta, sfx_engine khud [] guarantee karta hai.
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
        # video_editor apni khud ki exception type use karta hai (taaki
        # wo module bhi independent rahe) — yahan PipelineError mein
        # translate karte hain taaki Telegram error-handling ek jaisi rahe.
        raise PipelineError(str(e))
    return output_mp4


# ---------------------------------------------------------------------------
# Smart file-type detection (order-independent, name-independent)
# ---------------------------------------------------------------------------

def sniff_kind(path: Path) -> Optional[str]:
    """Extension pe bharosa na ho (random/missing filename) to file ke
    andar jhaank kar type pehchano."""
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
    """Kisi bhi naam se aayi file ke andar bhi timeline JSON ho sakti hai
    (Telegram kabhi extension preserve nahi karta) — isliye .json
    extension ke bhajar bhi content dekh kar pehchan lete hain: ek list
    jiske har item mein image/start/end keys hon."""
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
    """ZIP ke andar images ho ya folder ke andar ho, dono chalega — sab
    flatten karke seedha dest_dir mein daal deta hai."""
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
    """Random filename ke saath aayi .txt file "image prompts/description"
    hai ya "narration script" — pehle ek sasta heuristic try karo, fir
    zaroorat pade to Gemini se poochho."""
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

    # Ambiguous — Gemini se final decision lelo
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
            "quality": None,             # user ka button-choice — 360p/480p/720p/1080p
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
    """Render se pehle quality chunne ke liye tappable buttons — koi
    message type nahi karna padta. Lowest (360p) se highest (1080p)
    tak, poore trade-off (speed vs quality) ke saath label kiya gaya."""
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
    """Files complete hote hi seedha render shuru NAHI hota — pehle
    quality-select buttons bheje jaate hain (Telegram inline keyboard),
    render sirf tab shuru hota hai jab user koi ek button dabaye (dekho
    quality_callback). Isse user ko kabhi message type nahi karna
    padta."""
    sess = get_session(chat_id)
    if sess["processing"]:
        if force:
            await app.send_message(chat_id, "⏳ Render pehle se chal raha hai...")
        return
    ready = is_session_ready(sess)
    if not ready:
        if force:
            await app.send_message(
                chat_id,
                "❌ Abhi render nahi ho sakta.\n\n" + build_status_text(sess),
            )
        return

    if not sess.get("quality"):
        if force or not sess.get("quality_prompt_sent"):
            sess["quality_prompt_sent"] = True
            await app.send_message(
                chat_id,
                "🎬 Saari zaroori files mil gayi!\n\n"
                "Render shuru karne se pehle video quality chuno (neeche button dabao):",
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
        os._exit(0)  # one-time bot: ek video ban gaya, ab GitHub Actions job khatam


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
    """Quality-select button ka tap yahan handle hota hai — koi message
    type nahi hota, seedha button-tap se render shuru ho jaata hai."""
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
    app.run()  # blocking: connect, idle, aur updates process karta rehta hai
