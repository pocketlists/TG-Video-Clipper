from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import requests
from pyrogram import Client, filters, idle
from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from audio.sentences import build_sentences
from audio.transcriber import transcribe
from config.settings import intake_config, load_secrets
from media.detector import DetectionResult, classify
from media.extractor import ExtractionError, extract_archive
from media.image_cleaner import clean_images
from planning.plan_parser import parse_plan_file
from planning.plan_validator import remap_plan_archives, validate_plan
from render.video_renderer import available_qualities, quality_label, render
from sync.story_matcher import match_plan_to_transcript
from sync.timeline import build_scenes_from_legacy_timeline, build_scenes_from_match
from utils.error_handler import STAGE_PLAN, STAGE_SYNC, PipelineError
from utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("telegram.bot")

if os.getenv("TG_DEBUG", "").strip() in ("1", "true", "yes"):
    logging.getLogger("pyrogram").setLevel(logging.INFO)

SECRETS = load_secrets()
INTAKE = intake_config()

BASE_DIR = Path(__file__).parent.parent
WORK_ROOT = BASE_DIR / "work"

IDLE_TIMEOUT_MIN = int(os.getenv("IDLE_TIMEOUT_MIN", "45") or 45)
DOWNLOAD_TIMEOUT_S = int(os.getenv("DOWNLOAD_TIMEOUT_S", "900") or 900)
DOWNLOAD_RETRIES = 3

# in_memory=True: runner par stale .session file / AUTH_KEY problems nahi aate.
app = Client(
    "story_video_bot",
    api_id=SECRETS.api_id,
    api_hash=SECRETS.api_hash,
    bot_token=SECRETS.bot_token,
    in_memory=True,
    max_concurrent_transmissions=4,
)

SESSIONS: Dict[int, dict] = {}
STARTED_AT = time.monotonic()
LAST_ACTIVITY = [time.monotonic()]
CPU_SLOTS = asyncio.Semaphore(2)


def touch() -> None:
    LAST_ACTIVITY[0] = time.monotonic()


# --------------------------------------------------------------------------
# sessions / status
# --------------------------------------------------------------------------

def get_session(chat_id: int) -> dict:
    if chat_id not in SESSIONS:
        work_dir = WORK_ROOT / str(chat_id)
        sess = {
            "work_dir": work_dir,
            "input_dir": work_dir / "input",
            "incoming_dir": work_dir / "incoming",
            "temp_dir": work_dir / "temp",
            "output_dir": work_dir / "output",
            "images_count": 0,
            "has_audio": False,
            "has_bgm": False,
            "has_plan": False,
            "narration_path": None,
            "bgm_path": None,
            "plan_path": None,
            "plan_kind": None,
            "quality": None,
            "debounce_task": None,
            "processing": False,
            "quality_prompt_sent": False,
            "pending_downloads": 0,
            "seen_message_ids": set(),
            "ingest_tasks": set(),
        }
        for d in (sess["input_dir"], sess["incoming_dir"], sess["temp_dir"], sess["output_dir"]):
            d.mkdir(parents=True, exist_ok=True)
        SESSIONS[chat_id] = sess
    return SESSIONS[chat_id]


def reset_session(chat_id: int) -> None:
    sess = SESSIONS.pop(chat_id, None)
    if sess is None:
        return
    task = sess.get("debounce_task")
    if task and not task.done():
        task.cancel()
    for t in list(sess.get("ingest_tasks") or ()):
        if not t.done():
            t.cancel()
    shutil.rmtree(sess["work_dir"], ignore_errors=True)


def is_ready(sess: dict) -> bool:
    return sess["images_count"] > 0 and sess["has_audio"] and sess["has_plan"]


def build_status_text(sess: dict) -> str:
    def mark(ok: bool) -> str:
        return "✅" if ok else "⏳"

    lines = [
        f"{mark(sess['images_count'] > 0)} Images: {sess['images_count']}",
        f"{mark(sess['has_audio'])} Narration audio",
        f"{mark(sess['has_plan'])} edit_plan.json / timeline.json",
        f"{'✅' if sess['has_bgm'] else '➖'} Background music (optional)",
        f"{'✅ ' + sess['quality'] if sess.get('quality') else '⏳'} Video quality",
    ]
    if sess.get("pending_downloads"):
        lines.append(f"📥 Downloading: {sess['pending_downloads']} file(s) in progress")
    seen = sess.get("seen_message_ids")
    if seen:
        lines.append(f"📩 Files received this session: {len(seen)}")
    if is_ready(sess) and not sess.get("quality"):
        lines.append("\n🎚️ All files received — pick a quality to start render.")
    elif is_ready(sess) and sess.get("quality"):
        lines.append("\n🚀 Ready — render will begin shortly.")
    else:
        lines.append("\n⏳ Waiting for more files.")
    return "\n".join(lines)


def quality_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(f"🎚️ {quality_label(q)}", callback_data=f"quality:{q}")]
        for q in available_qualities()
    ]
    return InlineKeyboardMarkup(rows)


def _safe_name(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\-() \[\]]+", "_", name, flags=re.UNICODE).strip(" .")
    return name[:120] or f"file_{uuid.uuid4().hex[:6]}"


def target_suffix(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix and len(suffix) <= 5 and suffix.replace(".", "").isalnum():
        return suffix
    return ".bin"


async def _safe_edit(msg, text: str) -> None:
    if msg is None:
        return
    try:
        await msg.edit_text(text)
    except Exception as exc:
        logger.debug(f"edit skipped: {exc}")


# --------------------------------------------------------------------------
# progress helpers
# --------------------------------------------------------------------------

class Progress:
    """Thread-safe progress updater used by the render pipeline thread."""

    def __init__(self, client: Client, chat_id: int, message_id: int, loop):
        self.client = client
        self.chat_id = chat_id
        self.message_id = message_id
        self.loop = loop
        self._last_edit = 0.0
        self._last_text: Optional[str] = None

    def update(self, stage: str, detail: str = "") -> None:
        touch()
        text = f"🎬 {stage}" + (f" — {detail}" if detail else "")
        logger.info(text)
        if text == self._last_text:
            return
        now = time.time()
        if now - self._last_edit < 3.0:
            return
        self._last_edit = now
        self._last_text = text
        try:
            fut = asyncio.run_coroutine_threadsafe(self._edit(text), self.loop)
            fut.result(timeout=5.0)
        except Exception as exc:
            logger.debug(f"progress update skipped: {exc}")

    async def _edit(self, text: str) -> None:
        try:
            await self.client.edit_message_text(self.chat_id, self.message_id, text)
        except Exception:
            pass

    async def finish(self, text: str) -> None:
        try:
            await self.client.edit_message_text(self.chat_id, self.message_id, text)
        except Exception as exc:
            logger.debug(f"final progress edit failed: {exc}")


class TransferProgress:
    """Async callback for download_media / send_video: edits one status message."""

    def __init__(self, status_msg, icon: str, label: str):
        self.msg = status_msg
        self.icon = icon
        self.label = label
        self.last = 0.0

    async def __call__(self, current: int, total: int) -> None:
        touch()
        now = time.monotonic()
        if current < total and now - self.last < 4.0:
            return
        self.last = now
        total = total or 1
        pct = int(current * 100 / total)
        filled = pct // 10
        bar = "█" * filled + "░" * (10 - filled)
        await _safe_edit(
            self.msg,
            f"{self.icon} {self.label}\n{bar} {pct}%  ({current / 1048576:.1f}/{total / 1048576:.1f} MB)",
        )


# --------------------------------------------------------------------------
# telegram HTTP helper (works without peer cache)
# --------------------------------------------------------------------------

def tg_http_send(chat_id: int, text: str) -> None:
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{SECRETS.bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning(f"telegram http send non-200: {resp.status_code} {resp.text[:200]}")
    except Exception as exc:
        logger.warning(f"telegram http send failed: {exc}")


def notify_owner_startup(username: str) -> None:
    if not SECRETS.owner_chat_id:
        logger.info("OWNER_CHAT_ID not set — startup ping skipped")
        return
    tg_http_send(
        SECRETS.owner_chat_id,
        f"✅ Bot is up and running (@{username}) — Telegram connection OK.\n\n"
        "Send files in any order:\n"
        "• ZIP of images (or loose images)\n"
        "• Narration audio\n"
        "• edit_plan.json\n\n"
        "When everything is present, a quality picker will appear.\n"
        "Commands: /ping /status /render /reset",
    )


async def _heartbeat() -> None:
    while True:
        await asyncio.sleep(60)
        now = time.monotonic()
        idle_min = (now - LAST_ACTIVITY[0]) / 60
        busy = any(s.get("processing") or s.get("pending_downloads") for s in SESSIONS.values())
        logger.info(
            f"💓 alive {int((now - STARTED_AT) / 60)}m | sessions={len(SESSIONS)} | "
            f"busy={busy} | idle={idle_min:.1f}m"
        )
        if not busy and idle_min >= IDLE_TIMEOUT_MIN:
            logger.warning(f"idle for {idle_min:.0f} min — shutting down to save Actions minutes")
            if SECRETS.owner_chat_id:
                await asyncio.to_thread(
                    tg_http_send,
                    SECRETS.owner_chat_id,
                    f"💤 {IDLE_TIMEOUT_MIN} min se koi activity nahi — bot band ho gaya. "
                    "Dobara workflow run karo.",
                )
            sys.stdout.flush()
            os._exit(0)


# --------------------------------------------------------------------------
# auto-render orchestration
# --------------------------------------------------------------------------

def schedule_auto_render(chat_id: int) -> None:
    sess = get_session(chat_id)
    old = sess.get("debounce_task")
    if old and not old.done():
        old.cancel()
    sess["debounce_task"] = asyncio.create_task(debounced_render(chat_id))


async def debounced_render(chat_id: int) -> None:
    try:
        await asyncio.sleep(INTAKE.debounce_seconds)
    except asyncio.CancelledError:
        return
    if chat_id not in SESSIONS:
        return
    await try_start_pipeline(chat_id, force=False)


async def try_start_pipeline(chat_id: int, force: bool) -> None:
    sess = get_session(chat_id)
    if sess["processing"]:
        if force:
            await app.send_message(chat_id, "⏳ A render is already running.")
        return
    if sess.get("pending_downloads"):
        if force:
            await app.send_message(
                chat_id, f"⏳ {sess['pending_downloads']} file(s) abhi download ho rahi hain — thoda ruko."
            )
        return
    if not is_ready(sess):
        if force:
            await app.send_message(chat_id, "❌ Cannot render yet:\n\n" + build_status_text(sess))
        return

    if not sess.get("quality"):
        if force or not sess.get("quality_prompt_sent"):
            sess["quality_prompt_sent"] = True
            await app.send_message(
                chat_id,
                "🎬 All required files received.\n\nChoose the render quality:",
                reply_markup=quality_keyboard(),
            )
        return

    await start_render(chat_id, sess["quality"])


def _parse_plan(plan_path: Path, kind: str, media_dir: Path, progress: Progress):
    progress.update("validating plan")
    if kind == "edit_plan":
        plan = parse_plan_file(plan_path)
        # edit_plan.json file ko change NAHI kiya jata — sirf memory me
        # plan ki zip->folder mapping theek ki jati hai:
        #  • plan me zip ka FULL naam hai aur wahi folder disk par hai
        #    -> bilkul waisa hi rehne do (exact match).
        #  • plan me full naam NAHI hai (zip missing / chhota naam /
        #    alag format) -> chapter number se match karo: uploaded zip
        #    "Monster Streaming Star_Ch1.zip", "Chapter 1.zip",
        #    "Ch2.zip" ... jo bhi format ho, uske chapter number wala
        #    folder us chapter ki entries ko mil jata hai.
        for note in remap_plan_archives(plan, media_dir):
            logger.info(f"plan remap: {note}")
        report = validate_plan(plan, media_dir)
        if not report.ok:
            raise PipelineError(
                STAGE_PLAN,
                report.user_summary(),
                hint="fix the reported items in edit_plan.json",
            )
        if report.warnings():
            logger.warning("plan warnings:\n" + report.user_summary())
        return plan

    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(STAGE_PLAN, f"legacy timeline.json cannot be parsed: {exc}", cause=exc) from exc

    if not isinstance(data, list) or not data:
        raise PipelineError(STAGE_PLAN, "legacy timeline.json must be a non-empty array")
    return data


def run_pipeline(sess: dict, quality: str, progress: Progress) -> Path:
    input_dir: Path = sess["input_dir"]
    temp_dir: Path = sess["temp_dir"]
    output_dir: Path = sess["output_dir"]
    narration: Path = sess["narration_path"]
    plan_path: Path = sess["plan_path"]
    plan_kind: str = sess["plan_kind"]
    bgm_path: Optional[Path] = sess.get("bgm_path")

    progress.update("transcribing audio")
    transcription = transcribe(narration, progress=lambda s: progress.update("transcribing audio", s))
    if transcription.is_empty():
        raise PipelineError(STAGE_SYNC, "narration transcript is empty")

    parsed = _parse_plan(plan_path, plan_kind, input_dir, progress)

    if plan_kind == "edit_plan":
        progress.update("aligning plan to narration")
        sentences = build_sentences(transcription)
        match = match_plan_to_transcript(
            parsed,
            sentences,
            progress=lambda s: progress.update("aligning plan to narration", s),
        )
        for w in getattr(match, "warnings", []):
            logger.warning(f"matcher: {w}")
        scenes = build_scenes_from_match(parsed, match, sentences)
    else:
        progress.update("adapting legacy timeline")
        scenes = build_scenes_from_legacy_timeline(parsed, transcription.duration)

    progress.update("rendering", f"quality={quality}")
    output_path = output_dir / "final.mp4"
    return render(
        scenes=scenes,
        narration_path=narration,
        output_path=output_path,
        media_dir=input_dir,
        temp_dir=temp_dir,
        quality=quality,
        bgm_path=bgm_path,
        progress=lambda s: progress.update("rendering", s),
    )


async def _deliver_video(chat_id: int, output_path: Path, quality: str, msg) -> None:
    size_mb = output_path.stat().st_size / 1048576
    caption = f"✅ Render complete ({quality}) — {size_mb:.1f} MB"
    try:
        await app.send_video(
            chat_id=chat_id,
            video=str(output_path),
            caption=caption,
            supports_streaming=True,
            progress=TransferProgress(msg, "📤", f"Uploading video ({size_mb:.0f} MB)"),
        )
    except Exception as exc:
        logger.exception("send_video failed; falling back to send_document")
        await _safe_edit(msg, f"⚠️ Video upload failed ({type(exc).__name__}) — document ke taur par bhej raha hoon…")
        await app.send_document(
            chat_id=chat_id,
            document=str(output_path),
            caption=caption,
            progress=TransferProgress(msg, "📤", f"Uploading file ({size_mb:.0f} MB)"),
        )
    await _safe_edit(msg, "✅ Video delivered.")


async def start_render(chat_id: int, quality: str) -> None:
    sess = get_session(chat_id)
    sess["processing"] = True
    touch()
    success = False

    msg = await app.send_message(chat_id, f"🎬 Starting render ({quality})…")
    progress = Progress(app, chat_id, msg.id, asyncio.get_running_loop())

    try:
        output_path = await asyncio.to_thread(run_pipeline, sess, quality, progress)
        await _safe_edit(msg, "📤 Render done — uploading…")
        await _deliver_video(chat_id, output_path, quality, msg)
        success = True
    except PipelineError as exc:
        logger.error(f"pipeline error: {exc}")
        await progress.finish(exc.user_text() + "\n\n🔁 Fix karke /render bhejo, ya /reset.")
    except Exception as exc:
        logger.exception("unexpected pipeline failure")
        await progress.finish(
            f"❌ Unexpected error: {type(exc).__name__}: {exc}\n\n🔁 /render se retry ya /reset."
        )
    finally:
        sess["processing"] = False
        touch()

    if success:
        shutil.rmtree(sess["work_dir"], ignore_errors=True)
        SESSIONS.pop(chat_id, None)
        logger.info("render delivered; exiting workflow")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------

@app.on_message(filters.all, group=-1)
async def log_every_update(client: Client, message):
    touch()
    media = getattr(message.media, "value", message.media) if message.media else None
    kind = media or ("command" if (message.text or "").startswith("/") else "text")
    chat = getattr(message.chat, "id", None)
    doc = getattr(message, "document", None) or getattr(message, "audio", None)
    fname = getattr(doc, "file_name", None)
    logger.info(
        f"📩 incoming update: chat={chat} kind={kind} "
        f"mid={getattr(message, 'id', None)} file={fname} "
        f"group={getattr(message, 'media_group_id', None)}"
    )


@app.on_message(filters.command("start"))
async def start_handler(client: Client, message):
    await message.reply_text(
        "✅ Bot is running.\n\n"
        f"Your chat ID: {message.chat.id}\n"
        "Set it as OWNER_CHAT_ID in GitHub Secrets for startup pings.\n\n"
        "Send in any order:\n"
        "🖼️ ZIP of images (or loose images)\n"
        "🎧 Narration audio\n"
        "📋 edit_plan.json (v1)\n"
        "🎵 (Optional) a second audio = background music\n\n"
        "When everything is present, a quality picker will appear. "
        "Click a quality to start the render.\n\n"
        "Commands: /ping  /status  /render  /reset"
    )


@app.on_message(filters.command("ping"))
async def ping_handler(client: Client, message):
    up = int((time.monotonic() - STARTED_AT) / 60)
    await message.reply_text(f"🏓 pong — bot alive, uptime {up} min")


@app.on_message(filters.command("status") & filters.private)
async def status_handler(client: Client, message):
    await message.reply_text(build_status_text(get_session(message.chat.id)))


@app.on_message(filters.command("reset") & filters.private)
async def reset_handler(client: Client, message):
    reset_session(message.chat.id)
    await message.reply_text("🔄 Session cleared. Upload a fresh set of files.")


@app.on_message(filters.command("render") & filters.private)
async def render_handler(client: Client, message):
    await try_start_pipeline(message.chat.id, force=True)


@app.on_callback_query(filters.regex(r"^quality:"))
async def quality_callback(client: Client, callback):
    touch()
    chat_id = callback.message.chat.id
    quality = callback.data.split(":", 1)[1]
    if quality not in available_qualities():
        await callback.answer("Invalid quality.", show_alert=True)
        return
    sess = get_session(chat_id)
    if sess["processing"]:
        await callback.answer("A render is already running.", show_alert=True)
        return
    sess["quality"] = quality
    try:
        await callback.answer(f"{quality} selected")
        await callback.message.edit_text(f"✅ Quality: {quality}\n🎬 Starting render…")
    except Exception:
        pass
    await try_start_pipeline(chat_id, force=True)


async def _download(client: Client, message, dest: Path, label: str, status) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Optional[BaseException] = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            path = await asyncio.wait_for(
                client.download_media(
                    message,
                    file_name=str(dest),
                    progress=TransferProgress(status, "📥", f"Downloading {label}"),
                ),
                timeout=DOWNLOAD_TIMEOUT_S,
            )
            if path and Path(path).is_file():
                return Path(path)
            last_exc = RuntimeError("download returned no file")
        except asyncio.TimeoutError as exc:
            last_exc = exc
            logger.warning(f"download timeout ({label}) attempt {attempt}/{DOWNLOAD_RETRIES}")
        except Exception as exc:
            last_exc = exc
            logger.warning(f"download error ({label}) attempt {attempt}/{DOWNLOAD_RETRIES}: {exc}")
        await _safe_edit(status, f"🔁 Retry {attempt}/{DOWNLOAD_RETRIES}: {label}")
        await asyncio.sleep(2 * attempt)
    raise RuntimeError(f"{type(last_exc).__name__}: {last_exc}")


def _max_upload_bytes() -> int:
    return INTAKE.max_upload_mb * 1024 * 1024


# ---------------------------------------------------------------------------
# Media intake — v3 hardening (bot up tha, /ping-/status chal rahe the,
# lekin bheji gayi files register nahi ho rahi thi):
#   1. Media handler ab group=1 me hai aur filter sirf `filters.private`
#      hai — document/audio/voice/photo ki pehchan handler ke ANDAR
#      hoti hai. Koi composed filter ya group-0 ka koi dusra handler
#      (ek group me sirf EK handler chalta hai) file ko rok nahi sakta.
#   2. Edited messages ke liye bhi wahi handler hai: album ki caption
#      edit hone par ya file bot ke connect hone ke aas-paas bheji gayi
#      ho to edit update se bhi file le li jati hai. Ek hi message do
#      baar process nahi hota (seen_message_ids guard).
#   3. Ingest ek alag task me chalta hai — 20 MB download ke dauran
#      dispatcher dusre updates (/status, /ping) ke liye free rehta
#      hai, aur status me "Downloading: N" sahi dikhta hai.
#   4. Pehla "📥 Received" reply ab send_message se jata hai (reply/
#      quote par nirbhar nahi); wo fail ho to HTTP fallback se bhi
#      bheja jata hai, aur ingest phir bhi rukta nahi.
# ---------------------------------------------------------------------------

def _has_media(message) -> bool:
    return bool(
        getattr(message, "document", None)
        or getattr(message, "audio", None)
        or getattr(message, "voice", None)
        or getattr(message, "photo", None)
    )


def _spawn_ingest(client: Client, message, *, via_edit: bool) -> None:
    sess = get_session(message.chat.id)
    seen = sess.setdefault("seen_message_ids", set())
    mid = getattr(message, "id", None)
    if mid is not None:
        if mid in seen:
            logger.info(f"media duplicate skipped: mid={mid} via_edit={via_edit}")
            return
        seen.add(mid)
    task = asyncio.create_task(_handle_media_impl(client, message, via_edit=via_edit))
    tasks = sess.setdefault("ingest_tasks", set())
    tasks.add(task)
    task.add_done_callback(tasks.discard)


@app.on_message(filters.private, group=1)
async def handle_media(client: Client, message):
    if _has_media(message):
        _spawn_ingest(client, message, via_edit=False)


@app.on_edited_message(filters.private, group=1)
async def handle_edited_media(client: Client, message):
    if _has_media(message):
        _spawn_ingest(client, message, via_edit=True)


async def _handle_media_impl(client: Client, message, *, via_edit: bool = False):
    chat_id = message.chat.id
    sess = get_session(chat_id)

    if message.document:
        filename = message.document.file_name or f"doc_{uuid.uuid4().hex[:8]}"
        size = message.document.file_size or 0
    elif message.audio:
        filename = message.audio.file_name or f"audio_{uuid.uuid4().hex[:8]}.mp3"
        size = message.audio.file_size or 0
    elif message.voice:
        filename = f"voice_{uuid.uuid4().hex[:8]}.ogg"
        size = message.voice.file_size or 0
    elif message.photo:
        filename = f"photo_{uuid.uuid4().hex[:8]}.jpg"
        size = message.photo.file_size or 0
    else:
        return

    safe = _safe_name(filename)
    size_mb = size / 1048576
    logger.info(
        f"📥 handle_media fired: {safe} ({size_mb:.1f} MB) chat={chat_id} "
        f"mid={getattr(message, 'id', None)} via_edit={via_edit}"
    )

    # Instant acknowledgement — user ko turant pata chale ki bot ne file
    # dekh li. send_message (quote/reply par nirbhar nahi); fail ho to
    # HTTP fallback; dono fail hon to bhi ingest aage badhta hai.
    status = None
    try:
        status = await app.send_message(
            chat_id, f"📥 Received: {safe} ({size_mb:.1f} MB)\nDownload shuru…"
        )
    except Exception as exc:
        logger.warning(f"ack send_message failed for {safe}: {exc}")
        await asyncio.to_thread(
            tg_http_send, chat_id, f"📥 Received: {safe} ({size_mb:.1f} MB) — download shuru…"
        )

    if _max_upload_bytes() and size > _max_upload_bytes():
        await _safe_edit(
            status, f"❌ File too large ({size_mb:.0f} MB) — max {INTAKE.max_upload_mb} MB per upload."
        )
        return

    sess["pending_downloads"] += 1
    saved: Optional[Path] = None
    try:
        dest = sess["incoming_dir"] / uuid.uuid4().hex[:8] / safe
        try:
            saved = await _download(client, message, dest, safe, status)
        except Exception as exc:
            logger.error(f"download failed for {safe}: {exc}")
            await _safe_edit(status, f"❌ Download failed: {exc}")
            return

        await _safe_edit(status, f"🔎 Checking {safe}…")
        try:
            detection = await asyncio.to_thread(classify, saved)
        except Exception as exc:
            logger.exception("classify failed")
            await _safe_edit(status, f"❌ Could not inspect file: {exc}")
            shutil.rmtree(saved.parent, ignore_errors=True)
            return

        try:
            note = await _ingest_detection(chat_id, sess, saved, safe, detection, status)
        except Exception as exc:
            logger.exception("ingest failed")
            note = f"❌ {safe}: {type(exc).__name__}: {exc}"
    finally:
        sess["pending_downloads"] = max(0, sess["pending_downloads"] - 1)

    await _safe_edit(status, f"{note}\n\n{build_status_text(sess)}")
    if is_ready(sess):
        schedule_auto_render(chat_id)


async def _ingest_detection(
    chat_id: int,
    sess: dict,
    saved: Path,
    original_name: str,
    detection: DetectionResult,
    status,
) -> str:
    category = detection.category
    role = detection.role
    input_dir: Path = sess["input_dir"]
    incoming: Path = sess["incoming_dir"]

    if category == "archive":
        await _safe_edit(status, f"📦 Extracting {original_name}…")
        async with CPU_SLOTS:
            try:
                results = await asyncio.to_thread(extract_archive, saved, input_dir)
            except ExtractionError as exc:
                return f"❌ Could not extract {original_name}: {exc}"
            image_paths: List[Path] = [r.path for r in results if r.category == "image"]
            if not image_paths:
                return f"❌ Archive {original_name} contained no recognisable images."
            await _safe_edit(status, f"🖼️ Validating {len(image_paths)} image(s) from {original_name}…")
            report = await asyncio.to_thread(clean_images, image_paths)
        shutil.rmtree(saved.parent, ignore_errors=True)  # zip delete — disk bachao
        sess["images_count"] += len(report.clean)
        note = (
            f"📦 {original_name}: {len(image_paths)} file(s) extracted; "
            f"{len(report.clean)} validated, {len(report.rejected)} rejected."
        )
        if report.rejected:
            note += "\n" + "\n".join(f"  • {r.path.name}: {r.reason}" for r in report.rejected[:5])
        return note

    if category == "image":
        target = input_dir / original_name
        if target.exists():
            target = input_dir / f"{target.stem}_{uuid.uuid4().hex[:6]}{target.suffix}"
        shutil.move(str(saved), str(target))
        report = await asyncio.to_thread(clean_images, [target])
        if report.rejected:
            target.unlink(missing_ok=True)
            return f"❌ {original_name} rejected: {report.rejected[0].reason}"
        sess["images_count"] += 1
        return f"🖼️ Image added ({sess['images_count']} total)."

    if category == "audio":
        if not sess["has_audio"]:
            dest = incoming / f"narration{target_suffix(original_name)}"
            if dest.exists():
                dest = incoming / f"narration_{uuid.uuid4().hex[:6]}{dest.suffix}"
            shutil.move(str(saved), str(dest))
            sess["has_audio"] = True
            sess["narration_path"] = dest
            return "🎧 Narration audio received."
        dest = incoming / f"bgm{target_suffix(original_name)}"
        if dest.exists():
            dest = incoming / f"bgm_{uuid.uuid4().hex[:6]}{dest.suffix}"
        shutil.move(str(saved), str(dest))
        sess["has_bgm"] = True
        sess["bgm_path"] = dest
        return "🎵 Background music received."

    if role == "edit_plan":
        if sess["has_plan"]:
            saved.unlink(missing_ok=True)
            return "⚠️ A plan is already loaded. Send /reset to replace it."
        dest = sess["work_dir"] / "edit_plan.json"
        shutil.move(str(saved), str(dest))
        sess["has_plan"] = True
        sess["plan_path"] = dest
        sess["plan_kind"] = "edit_plan"
        return "📋 edit_plan.json received."

    if role == "timeline_legacy":
        if sess["has_plan"]:
            saved.unlink(missing_ok=True)
            return "⚠️ A plan is already loaded. Send /reset to replace it."
        dest = sess["work_dir"] / "timeline.json"
        shutil.move(str(saved), str(dest))
        sess["has_plan"] = True
        sess["plan_path"] = dest
        sess["plan_kind"] = "timeline_legacy"
        return "📋 timeline.json (legacy format) received."

    if role == "json_other":
        saved.unlink(missing_ok=True)
        return f"❌ {original_name} is JSON but does not match edit_plan v1 or legacy timeline."

    if category in ("text", "structured", "subtitle"):
        saved.unlink(missing_ok=True)
        return f"📝 {original_name} received ({category}) — ignored by pipeline."

    saved.unlink(missing_ok=True)
    return f"❌ {original_name} is not a supported input (detected: {category})."


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------

async def _serve() -> None:
    logger.info("connecting to Telegram…")
    await app.start()
    me = await app.get_me()
    logger.info(f"✅ connected as @{me.username} (id={me.id}) — waiting for updates")
    await asyncio.to_thread(notify_owner_startup, me.username or "bot")
    hb = asyncio.create_task(_heartbeat())
    try:
        await idle()
    finally:
        hb.cancel()
        await app.stop()


def main() -> None:
    # FIX v2 (do crash, ek ke baad ek):
    #   1) app.run(_serve()) -> TypeError: Run.run() takes 1 positional
    #      argument but 2 were given. Kurigram/Pyrogram ke naye version
    #      me Client.run() koi coroutine argument NAHI leta.
    #   2) asyncio.run(_serve()) -> RuntimeError: "... got Future ...
    #      attached to a different loop". Wajah: Client import time par
    #      apna event loop (app.loop) cache kar leta hai, aur Telegram
    #      session ke saare future usi loop par bante hain. asyncio.run()
    #      ek NAYA loop banata hai — dono loop alag, isliye crash.
    # Hal: bot ko USI loop par chalao jo Client ne pehle se rakha hai
    # (app.loop). _serve() already app.start() / idle() / app.stop()
    # sambhalta hai, aur ab har task/future ek hi loop par rahega.
    loop = app.loop
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
