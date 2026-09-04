#!/usr/bin/env python3
"""
One-Time Manga Recap Video Renderer (Telegram)
-----------------------------------------------
Run karo, Telegram se ZIP wait karo, video banao, bhejo, exit ho jao.
24/7 bot nahi hai – sirf jab aap chahe tab chalega.
"""

import os
import json
import random
import logging
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
from PIL import Image
import google.generativeai as genai
from openai import OpenAI
from pyrogram import Client, filters

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not all([API_ID, API_HASH, BOT_TOKEN]):
    raise ValueError("API_ID, API_HASH, BOT_TOKEN .env mein zaroor daalein!")
if not OPENAI_API_KEY or not GEMINI_API_KEY:
    raise ValueError("OPENAI_API_KEY aur GEMINI_API_KEY bhi chahiye!")

# ----------------------------
# Initialize APIs
# ----------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-3.5-flash-lite")
# ----------------------------
# Work Directories
# ----------------------------
BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "work"
INPUT_PANELS_DIR = WORK_DIR / "input_panels"
TEMP_DIR = WORK_DIR / "temp_workspace"
OUTPUT_DIR = WORK_DIR / "output"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Pyrogram Client (One-Time)
# ----------------------------
app = Client(
    "one_time_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ----------------------------
# Utility Functions
# ----------------------------
def ensure_work_dirs():
    for d in [INPUT_PANELS_DIR, TEMP_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for f in TEMP_DIR.glob("*"):
        if f.is_file(): f.unlink()

def cleanup_work_dir():
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

def retry_with_backoff(max_retries=3, initial_delay=2.0, backoff_factor=2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator

# ----------------------------
# Pipeline Functions (Same as before)
# ----------------------------
class ScriptGenerator:
    def __init__(self, provider="gemini"):
        self.provider = provider
        if provider == "gemini":
            self.model = genai.GenerativeModel("gemini-1.5-flash")

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
        response = self.model.generate_content(prompt)
        return response.text.strip()

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
        response_format="mp3"
    )
    mp3_path = output_path.with_suffix(".mp3")
    response.stream_to_file(mp3_path)
    AudioSegment.from_mp3(mp3_path).export(output_path, format="wav")
    mp3_path.unlink()
    logger.info("✅ OpenAI TTS Success!")

@retry_with_backoff()
def get_whisper_segments(audio_path: Path) -> List[Dict]:
    logger.info("⏱️ Whisper API se timings nikal rahe hain...")
    with open(audio_path, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    segments = transcript.segments

    srt_path = TEMP_DIR / "subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as srt:
        for i, seg in enumerate(segments):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            srt.write(f"{i+1}\n{start} --> {end}\n{seg['text'].strip()}\n\n")
    logger.info(f"✅ Subtitles saved: {srt_path.name}")
    return segments

def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

@retry_with_backoff()
def smart_sync_with_gemini(segments, image_files, prompts_text):
    logger.info("🧠 Gemini AI sync kar raha hai...")
    segment_data = [{"id": i, "start": s["start"], "end": s["end"], "text": s["text"]} for i, s in enumerate(segments)]
    prompt = f"""
    You are a master manhwa video editor.
    I have audio segments with exact timestamps, a list of image filenames, and story context.
    Assign EXACTLY one image filename to each audio segment.
    Output ONLY a JSON array: [{{"start": 0.0, "end": 2.5, "image_filename": "001.jpg", "text": "..."}}]
    Rules:
    1. Use images logically based on text.
    2. Prefer sequential flow, but reuse if needed.
    3. No extra text, no markdown.
    Audio Segments: {json.dumps(segment_data)}
    Available Images: {image_files}
    Story Context (Prompts): {prompts_text}
    """
    response = gemini_model.generate_content(prompt)
    content = response.text.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    sync_data = json.loads(content)

    last_effect = None
    for item in sync_data:
        available = [e for e in EFFECTS if e != last_effect]
        chosen = random.choice(available)
        item["effect"] = chosen
        last_effect = chosen
    return sync_data

# Ken Burns Effects
EFFECTS = [
    "zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down",
    "zoom_in_pan_left", "zoom_in_pan_right", "zoom_out_pan_left", "zoom_out_pan_right",
    "diagonal_zoom_in", "diagonal_zoom_out"
]

class ImageProcessor:
    def __init__(self, fps=30):
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
            candidates = ["pan_left", "pan_right", "zoom_in", "zoom_out", "zoom_in_pan_left", "zoom_in_pan_right", "zoom_out_pan_left", "zoom_out_pan_right"]
        else:
            candidates = EFFECTS
        if previous_effect in candidates and len(candidates) > 1:
            candidates.remove(previous_effect)
        return random.choice(candidates)

    def generate_zoompan_filter(self, effect, duration, fps):
        total_frames = int(duration * fps)
        d = f"d={total_frames}"
        s = "s=1920x1080"
        fps_str = f"fps={fps}"
        if effect == "zoom_in":
            z, x, y = "min(zoom+0.0015,1.5)", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
        elif effect == "zoom_out":
            z, x, y = "max(zoom-0.0015,1.0)", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
        elif effect == "pan_left":
            z, x, y = "1.2", f"min( (iw-iw/zoom)*(on/{total_frames}) , iw-iw/zoom )", "(ih-ih/zoom)/2"
        elif effect == "pan_right":
            z, x, y = "1.2", f"max( (iw-iw/zoom)*(1 - on/{total_frames}) , 0 )", "(ih-ih/zoom)/2"
        elif effect == "pan_up":
            z, x, y = "1.2", "(iw-iw/zoom)/2", f"max( (ih-ih/zoom)*(1 - on/{total_frames}) , 0 )"
        elif effect == "pan_down":
            z, x, y = "1.2", "(iw-iw/zoom)/2", f"min( (ih-ih/zoom)*(on/{total_frames}) , ih-ih/zoom )"
        elif effect == "zoom_in_pan_left":
            z, x, y = "min(zoom+0.0015,1.5)", f"(iw-iw/zoom)*(on/{total_frames})", "(ih-ih/zoom)/2"
        elif effect == "zoom_in_pan_right":
            z, x, y = "min(zoom+0.0015,1.5)", f"(iw-iw/zoom)*(1 - on/{total_frames})", "(ih-ih/zoom)/2"
        elif effect == "zoom_out_pan_left":
            z, x, y = "max(zoom-0.0015,1.0)", f"(iw-iw/zoom)*(on/{total_frames})", "(ih-ih/zoom)/2"
        elif effect == "zoom_out_pan_right":
            z, x, y = "max(zoom-0.0015,1.0)", f"(iw-iw/zoom)*(1 - on/{total_frames})", "(ih-ih/zoom)/2"
        elif effect == "diagonal_zoom_in":
            z, x, y = "min(zoom+0.0015,1.5)", f"(iw-iw/zoom)*(on/{total_frames})", f"(ih-ih/zoom)*(on/{total_frames})"
        elif effect == "diagonal_zoom_out":
            z, x, y = "max(zoom-0.0015,1.0)", f"(iw-iw/zoom)*(1 - on/{total_frames})", f"(ih-ih/zoom)*(1 - on/{total_frames})"
        else:
            z, x, y = "min(zoom+0.0015,1.5)", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
        return f"zoompan=z='{z}':x='{x}':y='{y}':{d}:{s}:{fps_str}"

    def standardize_image(self, image_path, output_path):
        cmd = [
            "ffmpeg", "-y", "-i", str(image_path),
            "-filter_complex",
            "[0:v]split=2[bg][fg];"
            "[bg]scale=1920:1080:force_original_aspect_ratio=increase,"
            "crop=1920:1080,boxblur=20:5[bgblur];"
            "[fg]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2[fgpad];"
            "[bgblur][fgpad]overlay=(W-w)/2:(H-h)/2[out]",
            "-frames:v", "1",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def create_clip(self, image_path, output_path, duration, effect):
        logger.info(f"  → Effect: {effect} on {image_path.name}")
        standardized = TEMP_DIR / f"std_{image_path.stem}.png"
        self.standardize_image(image_path, standardized)
        filter_str = self.generate_zoompan_filter(effect, duration, self.fps)
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(standardized),
            "-filter_complex", f"[0:v]{filter_str}[v]",
            "-map", "[v]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "21",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        standardized.unlink(missing_ok=True)

def assemble_final_video(sync_data, audio_path, bgm_path, output_path):
    logger.info("🎬 Final assembly shuru...")
    img_proc = ImageProcessor()
    clip_paths = []
    for idx, item in enumerate(sync_data):
        img_file = item["image_filename"]
        img_path = INPUT_PANELS_DIR / img_file
        if not img_path.exists():
            img_path = next(INPUT_PANELS_DIR.glob("*.*"))
            logger.warning(f"Image {img_file} missing, using {img_path.name}")
        dur = float(item["end"]) - float(item["start"])
        dur = max(dur, 0.5)
        clip_out = TEMP_DIR / f"clip_{idx:03d}.mp4"
        img_proc.create_clip(img_path, clip_out, dur, item["effect"])
        clip_paths.append(clip_out)

    concat_txt = TEMP_DIR / "inputs.txt"
    with open(concat_txt, "w") as f:
        for c in clip_paths:
            f.write(f"file '{c.resolve()}'\n")
    temp_video = TEMP_DIR / "no_subs.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(temp_video)], check=True, capture_output=True)

    srt_path = TEMP_DIR / "subtitles.srt"
    escaped_srt = str(srt_path).replace('\\', '/').replace(':', '\\:')
    final_cmd = ["ffmpeg", "-y", "-i", str(temp_video), "-i", str(audio_path)]
    filter_complex = f"[0:v]subtitles='{escaped_srt}':force_style='FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1'[v_sub]"
    if bgm_path and bgm_path.exists():
        final_cmd.extend(["-stream_loop", "-1", "-i", str(bgm_path)])
        filter_complex += ";[2:a]volume=0.08[bgm];[1:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[a_mix]"
        audio_map = "[a_mix]"
    else:
        audio_map = "1:a"
    final_cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v_sub]",
        "-map", audio_map,
        "-c:v", "libx264", "-preset", "fast", "-crf", "21",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path)
    ])
    logger.info("🔥 Final render chal raha hai...")
    res = subprocess.run(final_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Render error:\n{res.stderr}")
        raise RuntimeError("Final render failed!")
    logger.info(f"🎉 Video ready: {output_path.name}")

def run_pipeline(workdir: Path) -> Path:
    global WORK_DIR, INPUT_PANELS_DIR, TEMP_DIR, OUTPUT_DIR
    WORK_DIR = workdir
    INPUT_PANELS_DIR = WORK_DIR / "input_panels"
    TEMP_DIR = WORK_DIR / "temp_workspace"
    OUTPUT_DIR = WORK_DIR / "output"

    ensure_work_dirs()

    prompts_file = WORK_DIR / "prompts.txt"
    bgm_file = WORK_DIR / "bgm.mp3"
    audio_file = TEMP_DIR / "voiceover.wav"
    output_mp4 = OUTPUT_DIR / "final_manga_recap.mp4"

    prompts_text = prompts_file.read_text(encoding="utf-8") if prompts_file.exists() else "No extra context."

    images = [f.name for f in INPUT_PANELS_DIR.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not images:
        raise FileNotFoundError("input_panels folder khali hai! Images daalo.")
    logger.info(f"🖼️ Total images: {len(images)}")

    # Hybrid audio
    custom_audio_found = False
    for ext in [".mp3", ".wav", ".m4a"]:
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
            narration_text = script_file.read_text(encoding="utf-8")
            generate_tts(narration_text, audio_file)
        else:
            story_file = WORK_DIR / "story.txt"
            if story_file.exists():
                logger.info("📖 story.txt se script generate karke TTS kar rahe hain...")
                gen = ScriptGenerator(provider="gemini")
                narration_text = gen.generate_full_script(story_file.read_text(encoding="utf-8"))
                generate_tts(narration_text, audio_file)
            else:
                raise FileNotFoundError("Na custom audio, na script.txt, na story.txt!")

    segments = get_whisper_segments(audio_file)
    sync_data = smart_sync_with_gemini(segments, images, prompts_text)
    assemble_final_video(sync_data, audio_file, bgm_file, output_mp4)
    return output_mp4

# ----------------------------
# Telegram Handler (One-Time)
# ----------------------------
@app.on_message(filters.document & filters.private)
async def handle_zip(client, message):
    doc = message.document
    if not doc.file_name.lower().endswith(".zip"):
        await message.reply_text("❌ Sirf ZIP file bhejo!")
        return

    # Clean and prepare work dir
    cleanup_work_dir()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    status_msg = await message.reply_text("📥 ZIP download ho rahi hai...")
    zip_path = WORK_DIR / "upload.zip"
    await client.download_media(message, file_name=str(zip_path))

    try:
        await status_msg.edit_text("📂 Extract ho raha hai...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(WORK_DIR)
        zip_path.unlink()

        if not (WORK_DIR / "input_panels").exists():
            await status_msg.edit_text("❌ ZIP mein 'input_panels' folder nahi mila!")
            await app.stop()
            return

        await status_msg.edit_text("🎬 Video generation shuru... (5-10 min lag sakte hain)")
        output_video = run_pipeline(WORK_DIR)

        await status_msg.edit_text("📤 Video upload ho rahi hai...")
        await client.send_video(
            chat_id=message.chat.id,
            video=str(output_video),
            caption="✅ Aapka recap video ready!",
            supports_streaming=True
        )
        await status_msg.delete()
    except Exception as e:
        logger.exception("Pipeline error")
        await status_msg.edit_text(f"❌ Error: {e}")
    finally:
        # Cleanup work dir
        cleanup_work_dir()
        # Stop the bot after handling
        await app.stop()

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    print("🤖 One-Time Bot started. ZIP file bhejo...")
    app.run()
