from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import requests
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from audio.sentences import build_sentences
from audio.transcriber import transcribe
from config.settings import intake_config, load_secrets
from media.detector import DetectionResult, classify
from media.extractor import ExtractionError, extract_archive
from media.image_cleaner import clean_images
from planning.plan_parser import PlanParseError, parse_plan_file
from planning.plan_validator import validate_plan
from render.video_renderer import available_qualities, quality_label, render
from sync.story_matcher import match_plan_to_transcript
from sync.timeline import build_scenes_from_legacy_timeline, build_scenes_from_match
from utils.error_handler import (
    STAGE_INTAKE,
    STAGE_PLAN,
    STAGE_RENDER,
    STAGE_SYNC,
    STAGE_UPLOAD,
    PipelineError,
    wrap,
)
from utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("telegram.bot")

SECRETS = load_secrets()
INTAKE = intake_config()

BASE_DIR = Path(__file__).parent.parent
WORK_ROOT = BASE_DIR / "work"

app = Client(
    "story_video_bot",
    api_id=SECRETS.api_id,
    api_hash=SECRETS.api_hash,
    bot_token=SECRETS.bot_token,
)

SESSIONS: Dict[int, dict] = {}


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


class Progress:
    def __init__(self, client: Client, chat_id: int, message_id: int, loop):
        self.client = client
        self.chat_id = chat_id
        self.message_id = message_id
        self.loop = loop
        self._last_edit = 0.0
        self._last_text: Optional[str] = None

    def update(self, stage: str, detail: str = "") -> None:
        text = f"🎬 {stage}" + (f" — {detail}" if detail else "")
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


def notify_owner_startup() -> None:
    if not SECRETS.owner_chat_id:
        logger.info("OWNER_CHAT_ID not set — startup ping skipped")
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{SECRETS.bot_token}/sendMessage",
            json={
                "chat_id": SECRETS.owner_chat_id,
                "text": (
                    "✅ Bot is up and running.\n\n"
                    "Send files in any order:\n"
                    "• ZIP of images (or loose images)\n"
                    "• Narration audio\n"
                    "• edit_plan.json\n\n"
                    "When everything is present, a quality picker will appear."
                ),
            },
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning(f"startup ping non-200: {resp.status_code}")
    except Exception as exc:
        logger.warning(f"startup ping failed: {exc}")


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
    result = render(
        scenes=scenes,
        narration_path=narration,
        output_path=output_path,
        media_dir=input_dir,
        temp_dir=temp_dir,
        quality=quality,
        bgm_path=bgm_path,
        progress=lambda s: progress.update("rendering", s),
    )
    return result


async def start_render(chat_id: int, quality: str) -> None:
    sess = get_session(chat_id)
    sess["processing"] = True

    msg = await app.send_message(chat_id, f"🎬 Starting render ({quality})…")
    progress = Progress(app, chat_id, msg.id, asyncio.get_event_loop())

    try:
        output_path = await asyncio.to_thread(run_pipeline, sess, quality, progress)
        progress.update("uploading")
        try:
            await app.send_video(
                chat_id=chat_id,
                video=str(output_path),
                caption=f"✅ Render complete ({quality})",
                supports_streaming=True,
            )
            await progress.finish("✅ Video delivered.")
        except Exception as exc:
            logger.exception("video upload failed")
            await progress.finish(
                f"❌ Rendered, but Telegram upload failed ({type(exc).__name__}). "
                "The file may be larger than the Bot API upload limit (50 MB). Try a lower quality."
            )
    except PipelineError as exc:
        logger.error(f"pipeline error: {exc}")
        await progress.finish(exc.user_text())
    except Exception as exc:
        logger.exception("unexpected pipeline failure")
        await progress.finish(f"❌ Unexpected error: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(sess["work_dir"], ignore_errors=True)
        SESSIONS.pop(chat_id, None)
        logger.info("one-time render finished; exiting workflow")
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(0)


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
        "Commands: /status  /render  /reset"
    )


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


async def _save_document(client: Client, message, dest: Path) -> Optional[Path]:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        await client.download_media(message, file_name=str(dest))
        return dest
    except Exception as exc:
        logger.exception("download failed")
        await message.reply_text(f"❌ Download failed: {exc}")
        return None


def _max_upload_bytes() -> int:
    return INTAKE.max_upload_mb * 1024 * 1024


@app.on_message(filters.private & (filters.document | filters.audio | filters.voice | filters.photo))
async def handle_media(client: Client, message):
    chat_id = message.chat.id
    sess = get_session(chat_id)
    incoming = sess["incoming_dir"]

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

    if _max_upload_bytes() and size > _max_upload_bytes():
        await message.reply_text(
            f"❌ File too large ({size // (1024*1024)} MB) — max {INTAKE.max_upload_mb} MB per upload."
        )
        return

    suffix = Path(filename).suffix.lower()
    dl_path = incoming / f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    saved = await _save_document(client, message, dl_path)
    if saved is None:
        return

    try:
        detection = classify(saved)
    except Exception as exc:
        logger.exception("classify failed")
        await message.reply_text(f"❌ Could not inspect file: {exc}")
        saved.unlink(missing_ok=True)
        return

    note = await _ingest_detection(chat_id, sess, saved, filename, detection)
    await message.reply_text(f"{note}\n\n{build_status_text(sess)}")
    if is_ready(sess):
        schedule_auto_render(chat_id)


async def _ingest_detection(
    chat_id: int,
    sess: dict,
    saved: Path,
    original_name: str,
    detection: DetectionResult,
) -> str:
    category = detection.category
    role = detection.role
    input_dir = sess["input_dir"]
    incoming = sess["incoming_dir"]

    if category == "archive":
        try:
            results = extract_archive(saved, input_dir)
        except ExtractionError as exc:
            return f"❌ Could not extract {original_name}: {exc}"
        image_paths: List[Path] = [r.path for r in results if r.category == "image"]
        if not image_paths:
            return f"❌ Archive {original_name} contained no recognisable images."
        report = clean_images(image_paths)
        sess["images_count"] += len(report.clean)
        note = (
            f"📦 Extracted {len(image_paths)} file(s) from archive; "
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
        report = clean_images([target])
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


def target_suffix(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix and len(suffix) <= 5 and suffix.replace(".", "").isalnum():
        return suffix
    return ".bin"


def main() -> None:
    notify_owner_startup()
    app.run()


if __name__ == "__main__":
    main()
