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

Force unbuffered stdout so logs show up immediately in GitHub Actions
(bina isske "print"/log lines buffer mein atak jaate hain aur
Actions log mein der se ya kabhi kabhi bilkul nahi dikhte)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

Load Environment Variables

load_dotenv()

APIID = int(os.getenv("APIID"))
APIHASH = os.getenv("APIHASH")
BOTTOKEN = os.getenv("BOTTOKEN")

OPENAIAPIKEY = os.getenv("OPENAIAPIKEY")
GEMINIAPIKEY = os.getenv("GEMINIAPIKEY")
SARVAMAPIKEY = os.getenv("SARVAMAPIKEY")

Owner ka Telegram chat id — startup-ping isi id par jaayega.
/start bhejo bot ko ek baar, wo reply mein chat id de dega, use yaha /
GitHub secret OWNERCHATID mein daal do.
OWNERCHATID = os.getenv("OWNERCHATID")

GEMINISYNCMODEL = os.getenv("GEMINISYNCMODEL", "gemini-3.5-flash-lite")
GEMINISCRIPTMODEL = os.getenv("GEMINISCRIPTMODEL", "gemini-3.5-flash")

if not all([APIID, APIHASH, BOT_TOKEN]):
    raise ValueError("APIID, APIHASH, BOT_TOKEN .env mein zaroor daalein!")
if not OPENAIAPIKEY or not GEMINIAPIKEY:
    raise ValueError("OPENAIAPIKEY aur GEMINIAPIKEY bhi chahiye!")

Initialize APIs

openaiclient = OpenAI(apikey=OPENAIAPIKEY)
NOTE: purana "google-generativeai" package Google ne deprecate kar diya
hai (unstable ho sakta hai), isliye naya unified "google-genai" SDK use
kar rahe hain.
geminiclient = genai.Client(apikey=GEMINIAPIKEY)

Work Directories

BASE_DIR = Path(file).parent
WORKDIR = BASEDIR / "work"
INPUTPANELSDIR = WORKDIR / "inputpanels"
TEMPDIR = WORKDIR / "temp_workspace"
OUTPUTDIR = WORKDIR / "output"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(name)

Pyrogram Client

app = Client(
    "onetimebot",
    apiid=APIID,
    apihash=APIHASH,
    bottoken=BOTTOKEN
)

Utility Functions

def ensureworkdirs():
    for d in [INPUTPANELSDIR, TEMPDIR, OUTPUTDIR]:
        d.mkdir(parents=True, exist_ok=True)
    for f in TEMP_DIR.glob("*"):
        if f.is_file():
            f.unlink()

def retrywithbackoff(maxretries=3, initialdelay=2.0, backoff_factor=2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(args, *kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(args, *kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator

Startup notification (FIXED)

Purana issue: agar ye Pyrogram client se (app.send_message) bheja jaaye,
to har naye GitHub Actions run mein session/peer-cache khaali hota hai,
aur bot chatid ko resolve nahi kar paata (PEERID_INVALID) — chat id sahi
hone ke bawajood error aata hai. Fix: seedha Telegram Bot HTTP API se
sendMessage call karo — usko peer-cache ki zaroorat nahi hoti, sirf itna
chahiye ki user ne bot ko kabhi bhi (kisi bhi purane run mein) /start
kiya ho.
def notifyownerstartup():
    if not OWNERCHATID:
        logger.warning(
            "⚠️ OWNERCHATID set nahi hai — startup ping skip. Bot ko ek baar "
            "/start bhejo, reply mein chat id milega, use OWNERCHATID secret mein daal do."
        )
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={
                "chatid": OWNERCHAT_ID,
                "text": (
                    "✅ Bot ka workflow start ho gaya hai, ab ye chal raha hai!\n\n"
                    "Files bhejo — kisi bhi order mein: ZIP (images), audio, "
                    "(optional) prompts.txt. Jab sab mil jaayega, render khud shuru ho jaayega.\n"
                    "Manually shuru karne ke liye /render bhejo."
                ),
            },
            timeout=15,
        )
        data = {}
        try:
            data = resp.json()
        except Exception:
            pass
        if resp.status_code == 200 and data.get("ok"):
            logger.info(f"✅ Startup ping Telegram par bhej diya (chatid={OWNERCHAT_ID})")
        else:
            logger.error(f"❌ Startup ping FAIL ho gaya! status={resp.status_code} body={resp.text}")
    except Exception as e:
        logger.error(f"❌ Startup ping exception: {e}")

Pipeline Functions

class ScriptGenerator:
    """Available hai agar kabhi raw story text ko polished narration script
    mein expand karna ho. Filhaal auto-flow ise trigger nahi karta (aap
    hamesha khud audio ya ready script bhejte ho), lekin function ready hai."""
    def init(self, model: Optional[str] = None):
        self.model = model or GEMINISCRIPTMODEL

    @retrywithbackoff()
    def generatefullscript(self, raw_text: str) -> str:
        prompt = f"""
        You are a professional manga/manhwa recap scriptwriter.
        Given the raw story text below, create a dramatic, engaging narration script.
        Write in natural spoken language, 2-4 minutes worth of narration.
        Do not include any formatting, just plain text paragraphs.

        Raw story:
        {raw_text}
        """
        response = geminiclient.models.generatecontent(model=self.model, contents=prompt)
        return (response.text or "").strip()

@retrywithbackoff()
def generatetts(text: str, outputpath: Path):
    logger.info("🎙️ AI Voiceover generate ho raha hai...")
    if SARVAMAPIKEY:
        try:
            url = "https://api.sarvam.ai/text-to-speech"
            headers = {"Authorization": f"Bearer {SARVAMAPIKEY}", "Content-Type": "application/json"}
            payload = {"text": text, "language_code": "hi-IN", "voice": "default", "format": "wav"}
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                outputpath.writebytes(res.content)
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
    mp3path = outputpath.with_suffix(".mp3")
    response.streamtofile(mp3_path)
    AudioSegment.frommp3(mp3path).export(output_path, format="wav")
    mp3_path.unlink()
    logger.info("✅ OpenAI TTS Success!")

def segget(seg, key, default=None):
    """OpenAI SDK ke transcription segment object dict ya pydantic model
    dono ho sakte hain — dono se safely value nikalta hai."""
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)

GEMINITRANSCRIBEMODEL = os.getenv("GEMINITRANSCRIBEMODEL", GEMINISCRIPTMODEL)
WHISPERLOCALMODEL = os.getenv("WHISPERLOCALMODEL", "base")

def formatsrttime(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def writesrt(segments: List[Dict]) -> Path:
    srtpath = TEMPDIR / "subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as srt:
        for i, seg in enumerate(segments):
            start = formatsrttime(seg["start"])
            end = formatsrttime(seg["end"])
            srt.write(f"{i+1}\n{start} --> {end}\n{seg['text']}\n\n")
    logger.info(f"✅ Subtitles saved: {srt_path.name}")
    return srt_path

def stripjson_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith(""):
        raw = re.sub(r"^[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\s*$", "", raw)
    return raw.strip()

localwhisper_model = None  # lazy-loaded singleton — model load slow hai, ek hi baar karo

def getlocalwhispermodel():
    """openai-whisper (open-source pip package, import whisper) — ye
    OpenAI ki paid transcription API se bilkul alag hai, koi API key ya
    rate-limit nahi lagta, poora model local chalta hai. Isi tarah ka
    approach jo tumhari caption-engine reference file use karti hai."""
    global localwhisper_model
    if localwhisper_model is None:
        import whisper as localwhisperpkg
        logger.info(f"📦 Local (open-source) Whisper model '{WHISPERLOCALMODEL}' load ho raha hai...")
        localwhispermodel = localwhisperpkg.loadmodel(WHISPERLOCALMODEL)
    return localwhisper_model

@retrywithbackoff()
def localwhispersegments(audiopath: Path) -> List[Dict]:
    """Primary transcription path — open-source Whisper, poori tarah
    local (GitHub Actions runner par hi chalta hai, ffmpeg pehle se
    installed hai workflow mein). Na koi API limit, na koi cost."""
    logger.info("⏱️ Local Whisper se timings nikal rahe hain...")
    model = getlocalwhispermodel()
    result = model.transcribe(str(audiopath), wordtimestamps=False)
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

@retrywithbackoff()
def geminitranscribesegments(audiopath: Path) -> List[Dict]:
    """Extra fallback: Gemini audio understanding (agar local Whisper
    kisi wajah se fail ho jaaye). Isme bhi Gemini API rate-limit lag
    sakti hai, isliye ye ab primary nahi, sirf last-resort hai."""
    logger.info("⏱️ Gemini se audio timings nikal rahe hain...")
    uploaded = geminiclient.files.upload(file=str(audiopath))
    try:
        # Chhote audio ke liye usually turant ACTIVE ho jaata hai, lekin
        # safety ke liye thoda poll kar lete hain.
        waited = 0.0
        while getattr(uploaded.state, "name", uploaded.state) == "PROCESSING" and waited , \"end\": , \"text\": }."
        )
        response = geminiclient.models.generatecontent(
            model=GEMINITRANSCRIBEMODEL,
            contents=[uploaded, prompt],
            config=genaitypes.GenerateContentConfig(responsemime_type="application/json"),
        )
        raw = stripjson_fences(response.text or "")
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

@retrywithbackoff()
def openaiwhispersegments(audiopath: Path) -> List[Dict]:
    """Fallback path — sirf tab use hota hai jab Gemini fail ho jaaye
    (e.g. OpenAI credits/limit khatam ho jaane par bhi bot chalta rahe)."""
    logger.info("⏱️ (Fallback) OpenAI Whisper se timings nikal rahe hain...")
    with open(audio_path, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            responseformat="verbosejson",
            timestamp_granularities=["segment"]
        )
    rawsegments = seg_get(transcript, "segments", []) or []
    return [
        {
            "start": float(segget(s, "start", 0.0)),
            "end": float(segget(s, "end", 0.0)),
            "text": (segget(s, "text", "") or "").strip(),
        }
        for s in raw_segments
    ]

def gettranscriptsegments(audio_path: Path) -> List[Dict]:
    """Audio -> timed segments (subtitles ke liye, aur image-sync ke liye
    bhi use hote hain). Priority order:
      1) Local open-source Whisper — na API key, na rate-limit, na cost
         (jaise caption-engine reference file karti hai).
      2) OpenAI Whisper API — fallback, agar local model kisi wajah se
         (missing dependency, corrupt audio, etc.) fail ho jaaye.
      3) Gemini — last-resort fallback.
    """
    try:
        segments = localwhispersegments(audiopath)
    except Exception as e:
        logger.warning(f"⚠️ Local Whisper fail ho gaya ({e}), OpenAI Whisper API try kar rahe hain...")
        try:
            segments = openaiwhispersegments(audiopath)
        except Exception as e2:
            logger.warning(f"⚠️ OpenAI Whisper bhi fail ho gaya ({e2}), Gemini try kar rahe hain...")
            segments = geminitranscribesegments(audiopath)
    writesrt(segments)
    return segments

def fillsegmentgaps(rawsegments: List[Dict], totalduration: float) -> List[Dict]:
    """Whisper sirf jahan speech hai wahi ke segments deta hai — beech mein
    jo pauses/silences (blank spots) hote hain wo kisi segment mein cover
    nahi hote. Isse agar hum seedha segment start/end pe image clip banate
    hain to un silences ke doraan koi bhi image screen pe nahi hoti aur
    audio-video sync toot jaata hai / video audio se chhota reh jaata hai.

    Fix: har gap ko pichli line ki image tak extend kar dete hain (jab tak
    agla dialogue start nahi hota, wahi image screen par rehti hai), aur
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
        segs[i]["end"] = segs[i + 1]["start"] if i + 1  List[Dict]:
    logger.info("🧠 Gemini AI sync kar raha hai...")
    segment_data = [
        {"id": i, "start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"]}
        for i, s in enumerate(segments)
    ]
    prompt = f"""
    You are a master manhwa/manga video editor.
    You are given narration segments (with text) and a list of available
    image filenames, plus optional story/image context.
    Assign EXACTLY one image filename to each segment id, based on what the
    text is describing at that moment.
    Rules:
    1. Use images logically based on the narration text and the image context/prompts.
    2. Prefer sequential forward flow through the images, but you may reuse one if needed.
    3. Every segment id from the input must appear exactly once in the output.
    4. Only use filenames from the "Available Images" list — do not invent names.
    5. Output ONLY a JSON array like: [{{"id": 0, "image_filename": "001.jpg"}}, ...]. No other text.

    Segments: {json.dumps(segmentdata, ensureascii=False)}
    Available Images: {image_files}
    Story / Image Context (may be generic if not provided): {prompts_text}
    """
    response = geminiclient.models.generatecontent(
        model=GEMINISYNCMODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            responsemimetype="application/json",
            temperature=0.4,
        ),
    )
    content = (response.text or "").strip()
    if content.startswith(""):
        content = content.split("\n", 1)[1].rsplit("", 1)[0]
    try:
        mapping_list = json.loads(content)
        mapping = {int(item["id"]): item["imagefilename"] for item in mappinglist}
    except Exception as e:
        logger.warning(f"Gemini sync JSON parse fail ({e}), sequential fallback use ho raha hai.")
        mapping = {}

    validimages = set(imagefiles)
    sync_data = []
    last_effect = None
    for i, seg in enumerate(segments):
        img = mapping.get(i)
        if not img or img not in valid_images:
            img = imagefiles[i % len(imagefiles)]
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
    def init(self, fps=30):
        self.fps = fps

    def getimagedimensions(self, image_path):
        with Image.open(image_path) as img:
            return img.size

    def chooseeffect(self, imagepath, previous_effect=None):
        width, height = self.getimagedimensions(image_path)
        is_vertical = height > width * 1.2
        is_horizontal = width > height * 1.2
        if is_vertical:
            candidates = ["panup", "pandown", "zoomin", "zoomout", "diagonalzoomin", "diagonalzoomout"]
        elif is_horizontal:
            candidates = ["panleft", "panright", "zoomin", "zoomout", "zoominpanleft", "zoominpanright", "zoomoutpanleft", "zoomoutpanright"]
        else:
            candidates = EFFECTS
        if previous_effect in candidates and len(candidates) > 1:
            candidates.remove(previous_effect)
        return random.choice(candidates)

    def generatezoompanfilter(self, effect, duration, fps):
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
        elif effect == "zoominpan_left":
            z, x, y = "min(zoom+0.0015,1.5)", f"(iw-iw/zoom)*(on/{total_frames})", "(ih-ih/zoom)/2"
        elif effect == "zoominpan_right":
            z, x, y = "min(zoom+0.0015,1.5)", f"(iw-iw/zoom)*(1 - on/{total_frames})", "(ih-ih/zoom)/2"
        elif effect == "zoomoutpan_left":
            z, x, y = "max(zoom-0.0015,1.0)", f"(iw-iw/zoom)*(on/{total_frames})", "(ih-ih/zoom)/2"
        elif effect == "zoomoutpan_right":
            z, x, y = "max(zoom-0.0015,1.0)", f"(iw-iw/zoom)*(1 - on/{total_frames})", "(ih-ih/zoom)/2"
        elif effect == "diagonalzoomin":
            z, x, y = "min(zoom+0.0015,1.5)", f"(iw-iw/zoom)(on/{total_frames})", f"(ih-ih/zoom)(on/{total_frames})"
        elif effect == "diagonalzoomout":
            z, x, y = "max(zoom-0.0015,1.0)", f"(iw-iw/zoom)(1 - on/{total_frames})", f"(ih-ih/zoom)(1 - on/{total_frames})"
        else:
            z, x, y = "min(zoom+0.0015,1.5)", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
        return f"zoompan=z='{z}':x='{x}':y='{y}':{d}:{s}:{fps_str}"

    # ✅ FIXED standardize_image (IndentationError fix — body ab method ke
    # andar properly indented hai; Error 234 wala filter fix bhi barkarar)
    def standardizeimage(self, imagepath, output_path):
        cmd = [
            "ffmpeg", "-y", "-i", str(image_path),
            "-filter_complex",
            # Background: Scale to cover, crop to exact size, blur
            "[0:v]split=2[bg][fg];"
            "[bg]scale=1920:1080:forceoriginalaspect_ratio=increase,"
            "crop=1920:1080,boxblur=10:5[bgblur];"
            # Foreground: Scale to fit, pad to exact size
            "[fg]scale=1920:1080:forceoriginalaspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2[fgpad];"
            # Overlay
            "[bgblur][fgpad]overlay=(W-w)/2:(H-h)/2[out]",
            "-frames:v", "1",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    # ✅ Ye method class ke andar hai (duplicate definition hata di gayi)
    def createclip(self, imagepath, output_path, duration, effect):
        logger.info(f"  → Effect: {effect} on {image_path.name}")
        standardized = TEMPDIR / f"std{image_path.stem}.png"
        self.standardizeimage(imagepath, standardized)
        filterstr = self.generatezoompan_filter(effect, duration, self.fps)
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(standardized),
            "-filtercomplex", f"[0:v]{filterstr}[v]",
            "-map", "[v]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "21",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        standardized.unlink(missing_ok=True)

def assemblefinalvideo(syncdata, audiopath, bgmpath: Optional[Path], outputpath):
    logger.info("🎬 Final assembly shuru...")
    img_proc = ImageProcessor()
    clip_paths = []
    fallbackimages = sorted(INPUTPANELS_DIR.glob("."))
    for idx, item in enumerate(sync_data):
        imgfile = item["imagefilename"]
        imgpath = INPUTPANELSDIR / imgfile
        if not imgpath.exists() and fallbackimages:
            imgpath = fallbackimages[idx % len(fallback_images)]
            logger.warning(f"Image {imgfile} missing, using {imgpath.name}")
        dur = float(item["end"]) - float(item["start"])
        dur = max(dur, 0.5)
        clipout = TEMPDIR / f"clip_{idx:03d}.mp4"
        imgproc.createclip(imgpath, clipout, dur, item["effect"])
        clippaths.append(clipout)

    concattxt = TEMPDIR / "inputs.txt"
    with open(concat_txt, "w") as f:
        for c in clip_paths:
            f.write(f"file '{c.resolve()}'\n")
    tempvideo = TEMPDIR / "no_subs.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concattxt), "-c", "copy", str(tempvideo)], check=True, capture_output=True)

    srtpath = TEMPDIR / "subtitles.srt"
    escapedsrt = str(srtpath).replace('\\', '/').replace(':', '\\:')
    finalcmd = ["ffmpeg", "-y", "-i", str(tempvideo), "-i", str(audio_path)]
    filtercomplex = f"[0:v]subtitles='{escapedsrt}':forcestyle='FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1'[vsub]"
    if bgmpath and bgmpath.exists():
        finalcmd.extend(["-streamloop", "-1", "-i", str(bgm_path)])
        filtercomplex += ";[2:a]volume=0.08[bgm];[1:a][bgm]amix=inputs=2:duration=first:dropouttransition=2[a_mix]"
        audiomap = "[amix]"
    else:
        audio_map = "1:a"
    final_cmd.extend([
        "-filtercomplex", filtercomplex,
        "-map", "[v_sub]",
        "-map", audio_map,
        "-c:v", "libx264", "-preset", "fast", "-crf", "21",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path)
    ])
    logger.info("🔥 Final render chal raha hai...")
    res = subprocess.run(finalcmd, captureoutput=True, text=True)
    if res.returncode != 0:
        logger.error(f"Render error:\n{res.stderr}")
        raise RuntimeError("Final render failed!")
    logger.info(f"🎉 Video ready: {output_path.name}")

def run_pipeline(workdir: Path) -> Path:
    global WORKDIR, INPUTPANELSDIR, TEMPDIR, OUTPUT_DIR
    WORK_DIR = workdir
    INPUTPANELSDIR = WORKDIR / "inputpanels"
    TEMPDIR = WORKDIR / "temp_workspace"
    OUTPUTDIR = WORKDIR / "output"

    ensureworkdirs()

    promptsfile = WORKDIR / "prompts.txt"
    audiofile = TEMPDIR / "voiceover.wav"
    outputmp4 = OUTPUTDIR / "finalmangarecap.mp4"

    promptstext = promptsfile.readtext(encoding="utf-8") if promptsfile.exists() else "No extra context."

    images = sorted(f.name for f in INPUTPANELSDIR.glob(".") if f.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError("Koi bhi image nahi mili! ZIP ya image files bhejo.")
    logger.info(f"🖼️ Total images: {len(images)}")

    # Hybrid audio: pehle khud ka bheja hua voiceover dhoondo
    customaudiofound = False
    for ext in sorted(AUDIO_EXTS):
        possible = WORK_DIR / f"voiceover{ext}"
        if possible.exists():
            logger.info(f"🎧 Custom audio mil gaya: {possible.name}")
            AudioSegment.fromfile(possible).export(audiofile, format="wav")
            customaudiofound = True
            break

    if not customaudiofound:
        scriptfile = WORKDIR / "script.txt"
        if script_file.exists():
            logger.info("🤖 script.txt se AI voice ban rahi hai...")
            narrationtext = scriptfile.read_text(encoding="utf-8")
            generatetts(narrationtext, audio_file)
        else:
            raise FileNotFoundError("Na custom audio mila, na script/story text!")

    segments = gettranscriptsegments(audio_file)
    totalduration = len(AudioSegment.fromfile(audio_file)) / 1000.0
    gappedsegments = fillsegmentgaps(segments, totalduration)

    syncdata = smartsyncwithgemini(gappedsegments, images, promptstext)

    bgmpath = next((p for p in sorted(WORKDIR.glob("bgm.*"))), None)
    assemblefinalvideo(syncdata, audiofile, bgmpath, outputmp4)
    return output_mp4

Smart file-type detection (order-independent, name-independent)

def sniff_kind(path: Path) -> Optional[str]:
    """Extension pe bharosa na ho (random/missing filename) to file ke
    andar jhaank kar type pehchano — bilkul waise jaise koi insaan
    file khol kar dekh leta hai ki ye kya hai."""
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

def classifyincomingfile(path: Path, filename_hint: str) -> str:
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

def extractimagesfromzip(zippath: Path, dest_dir: Path) -> int:
    """ZIP ke andar images ho ya folder ke andar ho, dono chalega — sab
    flatten karke seedha dest_dir mein daal deta hai."""
    extracttmp = zippath.parent / f"extract{uuid.uuid4().hex[:8]}"
    extracttmp.mkdir(parents=True, existok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_tmp)
    except zipfile.BadZipFile:
        shutil.rmtree(extracttmp, ignoreerrors=True)
        raise ValueError("Ye valid ZIP file nahi hai (corrupt ho sakti hai).")

    destdir.mkdir(parents=True, existok=True)
    count = 0
    for f in sorted(extract_tmp.rglob("*")):
        if not f.isfile() or f.name.startswith(".") or "_MACOSX" in f.parts:
            continue
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        target = dest_dir / f.name
        if target.exists():
            target = destdir / f"{f.stem}{uuid.uuid4().hex[:6]}{f.suffix}"
        shutil.move(str(f), str(target))
        count += 1
    shutil.rmtree(extracttmp, ignoreerrors=True)
    zippath.unlink(missingok=True)
    return count

def classifytextrole(text: str, alreadyhavescript: bool) -> str:
    """Random filename ke saath aayi .txt file "image prompts/description"
    hai ya "narration script" — pehle ek sasta heuristic try karo, fir
    zaroorat pade to Gemini se poochho (jaisa AI-assist maanga gaya tha)."""
    sample = text.strip()
    if not sample:
        return "prompts"

    lines = [l.strip() for l in sample.splitlines() if l.strip()]
    if lines:
        short_ratio = sum(1 for l in lines if len(l) = 3 and (bulletratio > 0.4 or (shortratio > 0.7 and keyword_hits >= 2)):
            return "prompts"
        if len(lines) >= 3 and shortratio  dict:
    if chat_id not in sessions:
        sessions[chat_id] = {
            "workdir": BASEDIR / "work" / str(chat_id),
            "images_count": 0,
            "has_audio": False,
            "has_bgm": False,
            "has_script": False,
            "prompts_chars": 0,
            "debounce_task": None,
            "processing": False,
        }
    return sessions[chat_id]

def resetsession(chatid: int):
    sess = sessions.pop(chat_id, None)
    if sess:
        task = sess.get("debounce_task")
        if task and not task.done():
            task.cancel()
    shutil.rmtree(BASEDIR / "work" / str(chatid), ignore_errors=True)

def buildstatustext(sess: dict) -> str:
    ready = sess["imagescount"] > 0 and (sess["hasaudio"] or sess["has_script"])
    lines = [
        f"{'✅' if sess['imagescount'] > 0 else '⏳'} Images: {sess['imagescount']}",
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

def scheduleautorender(chat_id: int):
    sess = getsession(chatid)
    oldtask = sess.get("debouncetask")
    if oldtask and not oldtask.done():
        old_task.cancel()
    sess["debouncetask"] = asyncio.createtask(debouncedrender(chat_id))

async def debouncedrender(chat_id: int):
    try:
        await asyncio.sleep(DEBOUNCE_SECONDS)
    except asyncio.CancelledError:
        return
    await trystartpipeline(chat_id, force=False)

async def trystartpipeline(chat_id: int, force: bool):
    sess = getsession(chatid)
    if sess["processing"]:
        if force:
            await app.sendmessage(chatid, "⏳ Render pehle se chal raha hai...")
        return
    ready = sess["imagescount"] > 0 and (sess["hasaudio"] or sess["has_script"])
    if not ready:
        if force:
            await app.send_message(
                chat_id,
                "❌ Abhi render nahi ho sakta.\n\n" + buildstatustext(sess),
            )
        return

    sess["processing"] = True
    statusmsg = await app.sendmessage(chat_id, "🎬 Saari zaroori files mil gayi! Video generation shuru (5-10 min)...")
    try:
        outputvideo = await asyncio.tothread(runpipeline, sess["workdir"])
        await app.send_video(
            chatid=chatid,
            video=str(output_video),
            caption="✅ Aapka recap video ready!",
            supports_streaming=True,
        )
        await status_msg.delete()
    except Exception as e:
        logger.exception("Pipeline error")
        await statusmsg.edittext(f"❌ Error: {e}")
    finally:
        shutil.rmtree(sess["workdir"], ignoreerrors=True)
        sessions.pop(chat_id, None)
        os._exit(0)  # one-time bot: ek video ban gaya, ab GitHub Actions job khatam

Handlers

@app.on_message(filters.command("start"))
async def start_handler(client, message):
    await message.reply_text(
        "✅ Bot chal raha hai!\n\n"
        f"Aapka chat ID: {message.chat.id}\n"
        "(Ise OWNERCHATID GitHub secret mein daal do taaki agli baar "
        "workflow start hote hi aapko yahin ping mil jaaye.)\n\n"
        "Ab bas files bhejo — kisi bhi order mein, jitni marzi ek saath:\n"
        "🖼️ ZIP (sirf images, folder ho ya na ho, farq nahi padta) — ya loose images\n"
        "🎧 Audio (aapka khud ka voiceover)\n"
        "📝 (Optional) image-prompts wali .txt file\n"
        "🎵 (Optional) doosra audio file = background music\n\n"
        "Sab mil jaane ke ~15 second baad video khud-ba-khud banna shuru ho jayega.\n\n"
        "Commands: /status /render /reset"
    )

@app.on_message(filters.command("status") & filters.private)
async def status_cmd(client, message):
    await message.replytext(buildstatustext(getsession(message.chat.id)))

@app.on_message(filters.command("reset") & filters.private)
async def reset_cmd(client, message):
    reset_session(message.chat.id)
    await message.reply_text("🔄 Session clear kar di. Naye sirey se files bhejo.")

@app.on_message(filters.command("render") & filters.private)
async def render_cmd(client, message):
    await trystartpipeline(message.chat.id, force=True)

@app.on_message(filters.private & (filters.document | filters.audio | filters.voice | filters.photo))
async def handle_media(client, message):
    chat_id = message.chat.id
    sess = getsession(chatid)
    workdir = sess["workdir"]
    (workdir / "inputpanels").mkdir(parents=True, exist_ok=True)

    if message.document:
        filename = message.document.filename or f"file_{uuid.uuid4().hex[:6]}"
    elif message.audio:
        filename = message.audio.filename or f"audio_{uuid.uuid4().hex[:6]}.mp3"
    elif message.voice:
        filename = f"voice{uuid.uuid4().hex[:6]}.ogg"
    elif message.photo:
        filename = f"photo{uuid.uuid4().hex[:6]}.jpg"
    else:
        return

    ext = Path(file_name).suffix.lower()
    dlpath = workdir / f"incoming_{uuid.uuid4().hex[:8]}{ext}"
    note = None
    try:
        await client.downloadmedia(message, filename=str(dl_path))
        kind = classifyincomingfile(dlpath, filename)

        if kind == "zip":
            try:
                n = extractimagesfromzip(dlpath, workdir / "inputpanels")
            except ValueError as ve:
                note = f"❌ {ve}"
                n = 0
            if n:
                sess["images_count"] += n
                note = f"✅ {n} images mili is ZIP se (total {sess['images_count']})."
            elif not note:
                note = "❌ ZIP mein koi image nahi mili (.jpg/.jpeg/.png/.webp/.bmp)."

        elif kind == "image":
            target = workdir / "inputpanels" / (file_name if ext else f"{uuid.uuid4().hex[:8]}.jpg")
            if target.exists():
                target = workdir / "inputpanels" / f"{target.stem}_{uuid.uuid4().hex[:6]}{target.suffix}"
            shutil.move(str(dl_path), str(target))
            sess["images_count"] += 1
            note = f"🖼️ Image add ho gayi (total {sess['images_count']})."

        elif kind == "audio":
            if not sess["has_audio"]:
                voiceoverpath = workdir / f"voiceover{ext if ext in AUDIO_EXTS else '.ogg'}"
                shutil.move(str(dlpath), str(voiceoverpath))
                sess["has_audio"] = True
                note = "🎧 Voiceover audio mil gaya!"
            else:
                bgmpath = workdir / f"bgm{ext if ext in AUDIO_EXTS else '.mp3'}"
                shutil.move(str(dlpath), str(bgmpath))
                sess["has_bgm"] = True
                note = "🎵 Background music mil gaya!"

        elif kind == "text":
            text = dlpath.readtext(encoding="utf-8", errors="ignore")
            dlpath.unlink(missingok=True)
            role = await asyncio.tothread(classifytextrole, text, sess["hasscript"])
            if role == "prompts":
                with open(work_dir / "prompts.txt", "a", encoding="utf-8") as f:
                    f.write(text.strip() + "\n")
                sess["prompts_chars"] += len(text)
                note = "📝 Image-prompts note kar liye (image matching mein help karega)."
            else:
                (workdir / "script.txt").writetext(text, encoding="utf-8")
                sess["has_script"] = True
                note = "📜 Script/story mil gaya (agar audio na bheja to isi se voice banegi)."

        else:
            dlpath.unlink(missingok=True)
            note = f"🤔 {file_name} samajh nahi aayi — ZIP, image, audio ya .txt bhejo."

    except Exception as e:
        logger.exception("File handling error")
        note = f"❌ File process karte waqt error: {e}"

    await message.replytext(f"{note}\n\n{buildstatus_text(sess)}")
    if note and not note.startswith(("❌", "")):
        scheduleautorender(chat_id)

Main Entry Point

if name == "main":
    logger.info("🤖 Bot starting... workflow start hote hi owner ko Telegram par ping jaayega.")
    notifyownerstartup()
    app.run()  # blocking: connect, idle, aur updates process karta rehta hai
```
