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
from PIL import Image
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from pyrogram import Client, filters

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

GEMINI_SYNC_MODEL = os.getenv("GEMINI_SYNC_MODEL", "gemini-3.5-flash-lite")
GEMINI_SCRIPT_MODEL = os.getenv("GEMINI_SCRIPT_MODEL", "gemini-3.5-flash-lite")
GEMINI_TRANSCRIBE_MODEL = os.getenv("GEMINI_TRANSCRIBE_MODEL", GEMINI_SCRIPT_MODEL)
WHISPER_LOCAL_MODEL = os.getenv("WHISPER_LOCAL_MODEL", "base")

# ---------------------------------------------------------------------------
# Video Output Settings (aap yaha se video quality tune kar sakte ho)
# ---------------------------------------------------------------------------
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 60           # 30 se 60 kar diya gaya — smoother motion
VIDEO_CRF = 20           # kam = better quality, bada file (18-23 sweet spot)
VIDEO_PRESET = "medium"  # fast se medium — quality thoda better, GitHub Actions runner par bhi affordable

EFFECTS = [
    "zoom_in", "zoom_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
    "zoomin_pan_left", "zoomin_pan_right",
    "zoomout_pan_left", "zoomout_pan_right",
    "diagonal_zoom_in", "diagonal_zoom_out",
]

# Animation smoothness tuning: motion ab clip ki poori duration par
# ease-in/ease-out (smoothstep) curve se normalize hoti hai, fixed
# per-frame rate se nahi — isse lambe segments par zoom beech mein
# "freeze" (max zoom tak pahunch kar ruk jaana) nahi hota, aur chhote
# segments mein motion negligible nahi lagti. Har clip apni duration mein
# hi shuru se ant tak smoothly complete hoti hai.
ZOOM_AMOUNT = 0.16       # 1.0 -> 1.16 tak zoom (subtle-cinematic, jerky nahi)
PAN_ZOOM_LEVEL = 1.14    # pan effects ke dauraan held zoom (room-to-pan)

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


def natural_sort_key(name: str):
    """Filename ke andar jitne bhi number hain unhe REAL integer maan kar
    sort karta hai — '2_x.jpg' hamesha '10_x.jpg' se PEHLE aayega (plain
    alphabetical sorted() mein '10_' '2_' se pehle aa jaata, jo galat
    order deta tha). 1, 2, 3 ... 9, 10, 11 ... jitni bhi images ho (7 ho
    ya 700), number ke hisaab se hi strict sequence banegi."""
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

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
        ("sync", "🧠 AI image-sync ho raha hai"),
        ("clips", "🎬 Clips render ho rahe hain"),
        ("assemble", "🔗 Final video assemble ho raha hai"),
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
        self.model = model or GEMINI_SCRIPT_MODEL

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


_local_whisper_model = None  # lazy-loaded singleton — model load slow hai, ek hi baar karo


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
    installed hai workflow mein). Na koi API limit, na koi cost."""
    logger.info("⏱️ Local Whisper se timings nikal rahe hain...")
    model = get_local_whisper_model()
    result = model.transcribe(str(audio_path), word_timestamps=False)
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
        response = gemini_client.models.generate_content(
            model=GEMINI_TRANSCRIBE_MODEL,
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


def smart_sync_with_gemini(segments: List[Dict], image_files: List[str], prompts_text: str) -> List[Dict]:
    logger.info("🧠 Gemini AI sync kar raha hai...")
    segment_data = [
        {"id": i, "start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"]}
        for i, s in enumerate(segments)
    ]
    prompt = f"""
    You are a master manhwa/manga video editor.
    You are given narration segments (with text) and a list of available
    image filenames, plus optional story/image context.
    The "Available Images" list below is already given to you in the
    CORRECT chronological story order (sorted by the number in each
    filename, e.g. 1_..., 2_..., 3_... up to however many images there
    are). Treat this order as the ground-truth story sequence.
    Assign EXACTLY one image filename to each segment id, based on what the
    text is describing at that moment.
    Rules:
    1. Use images logically based on the narration text and the image context/prompts.
    2. Follow the given image order as the default forward flow — as segment
       id increases, move forward through the image list in the same order
       given. Only reuse an earlier image (hold on it a bit longer) or skip
       ahead if the narration text clearly demands it; never jump backward
       or reorder the images arbitrarily.
    3. Every segment id from the input must appear exactly once in the output.
    4. Only use filenames from the "Available Images" list — do not invent names.
    5. Output ONLY a JSON array like: [{{"id": 0, "image_filename": "001.jpg"}}, ...]. No other text.

    Segments: {json.dumps(segment_data, ensure_ascii=False)}
    Available Images (already in correct 1,2,3... story order): {image_files}
    Story / Image Context (may be generic if not provided): {prompts_text}
    """
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_SYNC_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )
        content = strip_json_fences(response.text or "")
        mapping_list = json.loads(content)
        mapping = {int(item["id"]): item["image_filename"] for item in mapping_list}
    except Exception as e:
        logger.warning(f"Gemini sync JSON parse fail ({e}), sequential fallback use ho raha hai.")
        mapping = {}

    valid_images = set(image_files)
    sync_data = []
    last_effect = None
    for i, seg in enumerate(segments):
        img = mapping.get(i)
        if not img or img not in valid_images:
            img = image_files[i % len(image_files)]
        available = [e for e in EFFECTS if e != last_effect]
        chosen = random.choice(available)
        last_effect = chosen
        sync_data.append({
            "start": seg["start"],
            "end": seg["end"],
            "image_filename": img,
            "text": seg.get("text", ""),
            "effect": chosen,
        })
    return sync_data


class ImageProcessor:
    def __init__(self, fps=VIDEO_FPS):
        self.fps = fps

    def get_image_dimensions(self, image_path):
        with Image.open(image_path) as img:
            return img.size

    def choose_effect(self, image_path, previous_effect=None):
        width, height = self.get_image_dimensions(image_path)
        is_vertical = height > width * 1.2
        is_horizontal = width > height * 1.2
        if is_vertical:
            candidates = ["pan_up", "pan_down", "zoom_in", "zoom_out", "diagonal_zoom_in", "diagonal_zoom_out"]
        elif is_horizontal:
            candidates = ["pan_left", "pan_right", "zoom_in", "zoom_out",
                          "zoomin_pan_left", "zoomin_pan_right", "zoomout_pan_left", "zoomout_pan_right"]
        else:
            candidates = list(EFFECTS)
        if previous_effect in candidates and len(candidates) > 1:
            candidates.remove(previous_effect)
        return random.choice(candidates)

    def generate_zoompan_filter(self, effect, duration, fps):
        total_frames = max(int(duration * fps), 1)
        d = f"d={total_frames}"
        s = f"s={VIDEO_WIDTH}x{VIDEO_HEIGHT}"
        fps_str = f"fps={fps}"

        # Eased progress (smoothstep: slow-start / slow-end), normalized
        # to THIS clip's own duration — motion always starts at frame 0
        # and finishes exactly at the last frame, chahe clip 0.5s ka ho
        # ya 15s ka.
        #
        # Purana code fixed per-frame rate (zoom+0.0015 har frame) use
        # karta tha jo clip ki duration se independent tha — lambe
        # segments par zoom apni max/min limit tak pahunch kar beech mein
        # hi "freeze" ho jaata tha (static image, phir achanak cut —
        # yahi sabse badi jerkiness ki wajah thi). Aur "zoom_out" /
        # "zoomout_pan_*" effects mein to zoom start hi apni minimum
        # limit (1.0) se hota tha, isliye wo poori clip mein bilkul bhi
        # move hi nahi karte the (silently static) — kabhi static, kabhi
        # moving clips ke beech achanak cut hona hi "jerky" feel de raha
        # tha. Ab har effect apni poori duration mein guaranteed, smooth,
        # ease-in/ease-out motion dega.
        t = f"(on/{total_frames})"
        et = f"({t}*{t}*(3-2*{t}))"  # smoothstep(t) -> 0..1, eased

        z_max = 1.0 + ZOOM_AMOUNT
        pz = PAN_ZOOM_LEVEL

        if effect == "zoom_in":
            z, x, y = f"(1.0+{ZOOM_AMOUNT}*{et})", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
        elif effect == "zoom_out":
            z, x, y = f"({z_max}-{ZOOM_AMOUNT}*{et})", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
        elif effect == "pan_left":
            z, x, y = f"{pz}", f"((iw-iw/zoom)*{et})", "(ih-ih/zoom)/2"
        elif effect == "pan_right":
            z, x, y = f"{pz}", f"((iw-iw/zoom)*(1-{et}))", "(ih-ih/zoom)/2"
        elif effect == "pan_up":
            z, x, y = f"{pz}", "(iw-iw/zoom)/2", f"((ih-ih/zoom)*(1-{et}))"
        elif effect == "pan_down":
            z, x, y = f"{pz}", "(iw-iw/zoom)/2", f"((ih-ih/zoom)*{et})"
        elif effect == "zoomin_pan_left":
            z, x, y = f"(1.0+{ZOOM_AMOUNT}*{et})", f"((iw-iw/zoom)*{et})", "(ih-ih/zoom)/2"
        elif effect == "zoomin_pan_right":
            z, x, y = f"(1.0+{ZOOM_AMOUNT}*{et})", f"((iw-iw/zoom)*(1-{et}))", "(ih-ih/zoom)/2"
        elif effect == "zoomout_pan_left":
            z, x, y = f"({z_max}-{ZOOM_AMOUNT}*{et})", f"((iw-iw/zoom)*{et})", "(ih-ih/zoom)/2"
        elif effect == "zoomout_pan_right":
            z, x, y = f"({z_max}-{ZOOM_AMOUNT}*{et})", f"((iw-iw/zoom)*(1-{et}))", "(ih-ih/zoom)/2"
        elif effect == "diagonal_zoom_in":
            z, x, y = (f"(1.0+{ZOOM_AMOUNT}*{et})",
                        f"((iw-iw/zoom)*{et})",
                        f"((ih-ih/zoom)*{et})")
        elif effect == "diagonal_zoom_out":
            z, x, y = (f"({z_max}-{ZOOM_AMOUNT}*{et})",
                        f"((iw-iw/zoom)*(1-{et}))",
                        f"((ih-ih/zoom)*(1-{et}))")
        else:
            z, x, y = f"(1.0+{ZOOM_AMOUNT}*{et})", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
        return f"zoompan=z='{z}':x='{x}':y='{y}':{d}:{s}:{fps_str}"

    def standardize_image(self, image_path, output_path):
        """Har image ko — chaahe kitni bhi choti/badi/kisi bhi aspect
        ratio ki ho — exact 1920x1080 canvas par force karta hai:
        background blurred-cover + foreground fit-contain overlay,
        taaki na to image stretch ho na crop se content kate."""
        cmd = [
            "ffmpeg", "-y", "-i", str(image_path),
            "-filter_complex",
            "[0:v]split=2[bg][fg];"
            f"[bg]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},boxblur=10:5[bgblur];"
            f"[fg]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2[fgpad];"
            "[bgblur][fgpad]overlay=(W-w)/2:(H-h)/2:format=auto[out]",
            "-map", "[out]",
            "-frames:v", "1",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise PipelineError(f"Image standardize fail ({image_path.name}): {result.stderr[-500:]}")

    def create_clip(self, image_path, output_path, duration, effect):
        logger.info(f"  → Effect: {effect} on {image_path.name}")
        standardized = TEMP_DIR / f"std_{image_path.stem}.png"
        self.standardize_image(image_path, standardized)
        filter_str = self.generate_zoompan_filter(effect, duration, self.fps)
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(standardized),
            "-sws_flags", "lanczos+accurate_rnd",
            "-filter_complex", f"[0:v]{filter_str}[v]",
            "-map", "[v]",
            "-c:v", "libx264", "-preset", VIDEO_PRESET, "-crf", str(VIDEO_CRF),
            "-t", str(duration),
            "-r", str(self.fps),
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        standardized.unlink(missing_ok=True)
        if result.returncode != 0:
            raise PipelineError(f"Clip render fail ({image_path.name}): {result.stderr[-500:]}")


def assemble_final_video(sync_data, audio_path, bgm_path: Optional[Path], output_path,
                          progress: Optional[ProgressReporter] = None):
    logger.info("🎬 Final assembly shuru...")
    img_proc = ImageProcessor()
    clip_paths = []
    fallback_images = sorted(
        (p for p in INPUT_PANELS_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: natural_sort_key(p.name),
    )
    total = len(sync_data)

    for idx, item in enumerate(sync_data):
        if progress:
            progress.update_sync("clips", f"{idx + 1}/{total}")
        img_file = item["image_filename"]
        img_path = INPUT_PANELS_DIR / img_file
        if not img_path.exists() and fallback_images:
            img_path = fallback_images[idx % len(fallback_images)]
            logger.warning(f"Image {img_file} missing, using {img_path.name}")
        dur = float(item["end"]) - float(item["start"])
        dur = max(dur, 0.5)
        clip_out = TEMP_DIR / f"clip_{idx:03d}.mp4"
        img_proc.create_clip(img_path, clip_out, dur, item["effect"])
        clip_paths.append(clip_out)

    if progress:
        progress.update_sync("assemble", "clips concat ho rahe hain")

    concat_txt = TEMP_DIR / "inputs.txt"
    with open(concat_txt, "w") as f:
        for c in clip_paths:
            f.write(f"file '{c.resolve()}'\n")
    temp_video = TEMP_DIR / "no_subs.mp4"
    res = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(temp_video)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise PipelineError(f"Clips concat fail: {res.stderr[-500:]}")

    if progress:
        progress.update_sync("assemble", "audio mix ho raha hai (no subtitles)")

    # NOTE: subtitles intentionally NOT burned into the video anymore —
    # video par koi subtitle/caption text overlay nahi aayega, sirf clean visuals.
    final_cmd = ["ffmpeg", "-y", "-i", str(temp_video), "-i", str(audio_path)]
    filter_complex = f"[0:v]fps={VIDEO_FPS}[v_out]"
    if bgm_path and bgm_path.exists():
        final_cmd.extend(["-stream_loop", "-1", "-i", str(bgm_path)])
        filter_complex += ";[2:a]volume=0.08[bgm];[1:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[a_mix]"
        audio_map = "[a_mix]"
    else:
        audio_map = "1:a"
    final_cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v_out]",
        "-map", audio_map,
        "-c:v", "libx264", "-preset", VIDEO_PRESET, "-crf", str(VIDEO_CRF),
        "-r", str(VIDEO_FPS),
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    ])
    logger.info("🔥 Final render chal raha hai...")
    res = subprocess.run(final_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Render error:\n{res.stderr}")
        raise PipelineError(f"Final render fail ho gaya: {res.stderr[-800:]}")
    logger.info(f"🎉 Video ready: {output_path.name}")


def run_pipeline(work_dir: Path, progress: Optional[ProgressReporter] = None) -> Path:
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

    if progress:
        progress.update_sync("sync")
    sync_data = smart_sync_with_gemini(gapped_segments, images, prompts_text)

    bgm_path = next(iter(sorted(WORK_DIR.glob("bgm.*"))), None)
    assemble_final_video(sync_data, audio_file, bgm_path, output_mp4, progress)
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
    if ext == ".txt":
        return "text"
    return sniff_kind(path) or "unknown"


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
        response = gemini_client.models.generate_content(model=GEMINI_SYNC_MODEL, contents=prompt)
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
            "prompts_chars": 0,
            "debounce_task": None,
            "processing": False,
        }
    return sessions[chat_id]


def reset_session(chat_id: int):
    sess = sessions.pop(chat_id, None)
    if sess:
        task = sess.get("debounce_task")
        if task and not task.done():
            task.cancel()
    shutil.rmtree(BASE_DIR / "work" / str(chat_id), ignore_errors=True)


def build_status_text(sess: dict) -> str:
    ready = sess["images_count"] > 0 and (sess["has_audio"] or sess["has_script"])
    lines = [
        f"{'✅' if sess['images_count'] > 0 else '⏳'} Images: {sess['images_count']}",
        f"{'✅' if sess['has_audio'] else '⏳'} Voiceover audio"
        + ("" if sess["has_audio"] else " (ya niche wala script bhejo)"),
        f"{'✅' if sess['has_script'] else '➖'} Script/story text"
        + (" (optional, audio ke bina zaroori)" if not sess["has_audio"] else " (optional)"),
        f"{'✅' if sess['prompts_chars'] > 0 else '➖'} Image prompts (optional)",
        f"{'✅' if sess['has_bgm'] else '➖'} Background music (optional)",
    ]
    footer = (
        "\n\n🚀 Zaroori sab kuch mil gaya — thodi der mein render khud shuru hoga (/render se abhi karo)."
        if ready else
        "\n\n⏳ Abhi aur files chahiye (kam se kam: images + audio, ya images + script)."
    )
    return "\n".join(lines) + footer


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
    ready = sess["images_count"] > 0 and (sess["has_audio"] or sess["has_script"])
    if not ready:
        if force:
            await app.send_message(
                chat_id,
                "❌ Abhi render nahi ho sakta.\n\n" + build_status_text(sess),
            )
        return

    sess["processing"] = True
    progress = await ProgressReporter.create(
        chat_id,
        "🎬 Saari zaroori files mil gayi! Video generation shuru (5-10 min, 1080p @ 60fps)..."
    )
    try:
        output_video = await asyncio.to_thread(run_pipeline, sess["work_dir"], progress)
        await progress.finish("✅ Render complete! Video upload ho raha hai...")
        await app.send_video(
            chat_id=chat_id,
            video=str(output_video),
            caption="✅ Aapka recap video ready! (1080p, 60fps)",
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
        "🖼️ ZIP (sirf images, folder ho ya na ho, farq nahi padta) — ya loose images\n"
        "🎧 Audio (aapka khud ka voiceover)\n"
        "📝 (Optional) image-prompts wali .txt file\n"
        "🎵 (Optional) doosra audio file = background music\n\n"
        "Sab mil jaane ke ~15 second baad video khud-ba-khud banna shuru ho jayega.\n"
        "Video 1080p @ 60fps mein render hota hai.\n\n"
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

        elif kind == "text":
            text = dl_path.read_text(encoding="utf-8", errors="ignore")
            dl_path.unlink(missing_ok=True)
            role = await asyncio.to_thread(classify_text_role, text, sess["has_script"])
            if role == "prompts":
                with open(work_dir / "prompts.txt", "a", encoding="utf-8") as f:
                    f.write(text.strip() + "\n")
                sess["prompts_chars"] += len(text)
                note = "📝 Image-prompts note kar liye (image matching mein help karega)."
            else:
                (work_dir / "script.txt").write_text(text, encoding="utf-8")
                sess["has_script"] = True
                note = "📜 Script/story mil gaya (agar audio na bheja to isi se voice banegi)."

        else:
            dl_path.unlink(missing_ok=True)
            note = f"🤔 {file_name} samajh nahi aayi — ZIP, image, audio ya .txt bhejo."

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
