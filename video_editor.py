#!/usr/bin/env python3
"""
VIDEO EDITOR
------------
Standalone plug-and-play video rendering/assembly engine — image
standardization, zoom/pan effects, quality presets aur final FFmpeg
assembly (narration + optional BGM + optional SFX).

one_time_bot.py isse sirf ye import karta hai:

    from video_editor import (
        prepare_scenes, render_video, natural_sort_key,
        QUALITY_PRESETS, DEFAULT_QUALITY, VideoEditorError,
    )

Future mein rendering upgrade karna ho (naye effects, alag encoder,
better quality ladder, etc.) to sirf yahi file replace/edit karo —
one_time_bot.py ko touch karne ki zaroorat nahi.
"""

import random
import re
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class VideoEditorError(Exception):
    """User-facing rendering error — jo message isme diya jayega, wahi
    seedha Telegram status message par dikhega, isliye readable rakho."""
    pass


# ---------------------------------------------------------------------------
# QUALITY PRESETS — Telegram bot render se pehle inme se ek chunwaega
# (buttons ke through, koi typing nahi). Lowest = 360p.
# ---------------------------------------------------------------------------

QUALITY_PRESETS: Dict[str, Dict] = {
    "360p": {
        "width": 640, "height": 360, "fps": 30, "crf": 24, "preset": "fast",
        "label": "360p — sabse fast, sabse chhoti file",
    },
    "480p": {
        "width": 854, "height": 480, "fps": 30, "crf": 23, "preset": "fast",
        "label": "480p — fast, kam storage",
    },
    "720p": {
        "width": 1280, "height": 720, "fps": 60, "crf": 21, "preset": "medium",
        "label": "720p — balanced quality/speed",
    },
    "1080p": {
        "width": 1920, "height": 1080, "fps": 60, "crf": 20, "preset": "medium",
        "label": "1080p — best quality (sabse slow)",
    },
}
# Ladder lowest -> highest, taaki quality-select buttons isi order mein
# consistently render hon.
QUALITY_ORDER = ["360p", "480p", "720p", "1080p"]
DEFAULT_QUALITY = "1080p"


def available_qualities() -> List[str]:
    """Bot ye call karke pata karta hai ki kaunse quality options
    Telegram button-picker mein dikhane hain (lowest 360p se highest
    tak)."""
    return list(QUALITY_ORDER)


def quality_label(quality: str) -> str:
    return QUALITY_PRESETS.get(quality, QUALITY_PRESETS[DEFAULT_QUALITY])["label"]


# ---------------------------------------------------------------------------
# Animation tuning (subtle-cinematic zoom/pan, jerky nahi)
# ---------------------------------------------------------------------------

EFFECTS = [
    "zoom_in", "zoom_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
    "zoomin_pan_left", "zoomin_pan_right",
    "zoomout_pan_left", "zoomout_pan_right",
    "diagonal_zoom_in", "diagonal_zoom_out",
]
ZOOM_AMOUNT = 0.16       # 1.0 -> 1.16 tak zoom (subtle-cinematic, jerky nahi)
PAN_ZOOM_LEVEL = 1.14    # pan effects ke dauraan held zoom (room-to-pan)

# FFmpeg ke zoompan filter ka ek well-known issue hai: crop/zoom position
# seedhe TARGET resolution (jo aksar 640-1920px chhoti hoti hai) ke pixel
# grid par compute hoti hai. Ek subtle, slow zoom/pan mein per-frame shift
# aksar 1 pixel se bhi kam hota hai — zoompan use nearest integer pixel
# par round karta hai, isliye kai consecutive frames ek hi position par
# "hold" ho jaate hain, phir achanak 1-2px jump hota hai. Yehi asli
# "jittery / keyframe thik se na chalna / halka wobble-shake" wala symptom
# hai — chahe encoded fps (60) bilkul sahi ho, kyunki ye timing ka bug
# nahi, pixel-rounding ka bug hai.
# Fix: zoompan se PEHLE image ko ek kaafi bade (supersampled) canvas par
# scale karo — isse crop-position ki granularity NxN gunaa fine ho jaati
# hai, rounding-error negligible ban jaata hai, motion buttery-smooth
# lagti hai. zoompan khud s= se wapas asli target resolution par le aata
# hai, isliye final output size/encode-cost same rehta hai — sirf ek extra
# (cheap, ek-baar-wala) upscale step add hota hai.
ZOOMPAN_SUPERSAMPLE = 4


def natural_sort_key(name: str):
    """Filename ke andar jitne bhi number hain unhe REAL integer maan
    kar sort karta hai — '2_x.jpg' hamesha '10_x.jpg' se PEHLE aayega."""
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# ---------------------------------------------------------------------------
# Scene preparation — merge consecutive same-image segments + assign
# a non-repeating effect per merged scene.
# ---------------------------------------------------------------------------

def merge_consecutive_scenes(sync_data: List[Dict]) -> List[Dict]:
    """Jab lagataar segments ko same image assign hoti hai, unhe EK hi
    continuous 'scene' mein merge karta hai, taaki us poore span ke
    liye sirf EK clip bane jisme animation shuru se ant tak bina reset
    hue chalti hai."""
    if not sync_data:
        return []
    merged = [dict(sync_data[0])]
    for seg in sync_data[1:]:
        if seg["image_filename"] == merged[-1]["image_filename"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + seg.get("text", "")).strip()
        else:
            merged.append(dict(seg))
    return merged


def assign_scene_effects(scenes: List[Dict]) -> List[Dict]:
    """Har MERGED scene ko ek baar effect milta hai (na ki har raw
    segment ko) — isliye jab tak image nahi badalti, animation bhi wahi
    ek continuous motion rehta hai. Lagataar do scenes ko kabhi same
    effect nahi milta."""
    last_effect = None
    for scene in scenes:
        available = [e for e in EFFECTS if e != last_effect]
        chosen = random.choice(available)
        last_effect = chosen
        scene["effect"] = chosen
    return scenes


def prepare_scenes(scenes: List[Dict]) -> List[Dict]:
    """Public helper: timeline se load hui scenes ko render-ready banata
    hai (merge + effect assignment) ek hi call mein."""
    return assign_scene_effects(merge_consecutive_scenes(scenes))


# ---------------------------------------------------------------------------
# Image standardization + zoom/pan clip rendering
# ---------------------------------------------------------------------------

class ImageProcessor:
    def __init__(self, quality: str, temp_dir):
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[DEFAULT_QUALITY])
        self.width = preset["width"]
        self.height = preset["height"]
        self.fps = preset["fps"]
        self.crf = preset["crf"]
        self.ffmpeg_preset = preset["preset"]
        self.temp_dir = Path(temp_dir)

    def get_image_dimensions(self, image_path):
        from PIL import Image
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
        s = f"s={self.width}x{self.height}"
        fps_str = f"fps={fps}"

        # Eased progress (smoothstep: slow-start / slow-end), normalized
        # to THIS clip's own duration — motion hamesha frame 0 par shuru
        # hoti hai aur exact last frame par khatam, chahe clip 0.5s ka
        # ho ya 15s ka (koi mid-clip "freeze" nahi).
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

        # Supersample-then-zoompan: zoompan ke andar iw/ih is bade canvas
        # ke honge, isliye upar ke saare x/y/zoom expressions (jo iw/ih pe
        # depend karte hain) automatically zyada fine-grained pixel-math
        # par evaluate honge — koi expression change nahi karni padi.
        pre_w = self.width * ZOOMPAN_SUPERSAMPLE
        pre_h = self.height * ZOOMPAN_SUPERSAMPLE
        prescale = f"scale={pre_w}:{pre_h}:flags=lanczos"
        return f"{prescale},zoompan=z='{z}':x='{x}':y='{y}':{d}:{s}:{fps_str}"

    def standardize_image(self, image_path, output_path):
        """Har image ko — chaahe kitni bhi choti/badi/kisi bhi aspect
        ratio ki ho — exact target canvas par force karta hai:
        background blurred-cover + foreground fit-contain overlay,
        taaki na to image stretch ho na crop se content kate."""
        cmd = [
            "ffmpeg", "-y", "-i", str(image_path),
            "-filter_complex",
            "[0:v]split=2[bg][fg];"
            f"[bg]scale={self.width}:{self.height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={self.width}:{self.height},boxblur=10:5[bgblur];"
            f"[fg]scale={self.width}:{self.height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2[fgpad];"
            "[bgblur][fgpad]overlay=(W-w)/2:(H-h)/2:format=auto[out]",
            "-map", "[out]",
            "-frames:v", "1",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise VideoEditorError(f"Image standardize fail ({image_path.name}): {result.stderr[-500:]}")

    def create_clip(self, image_path, output_path, duration, effect):
        logger.info(f"  → Effect: {effect} on {image_path.name}")
        standardized = self.temp_dir / f"std_{image_path.stem}.png"
        self.standardize_image(image_path, standardized)
        filter_str = self.generate_zoompan_filter(effect, duration, self.fps)
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(standardized),
            "-sws_flags", "lanczos+accurate_rnd",
            "-filter_complex", f"[0:v]{filter_str}[v]",
            "-map", "[v]",
            "-c:v", "libx264", "-preset", self.ffmpeg_preset, "-crf", str(self.crf),
            "-t", str(duration),
            "-r", str(self.fps),
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        standardized.unlink(missing_ok=True)
        if result.returncode != 0:
            raise VideoEditorError(f"Clip render fail ({image_path.name}): {result.stderr[-500:]}")


# ---------------------------------------------------------------------------
# Final assembly — clips concat + narration + optional BGM + optional SFX
# ---------------------------------------------------------------------------

def render_video(
    scenes: List[Dict],
    audio_path,
    bgm_path,
    output_path,
    input_panels_dir,
    temp_dir,
    quality: str = DEFAULT_QUALITY,
    sfx_events: Optional[List[Dict]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    PUBLIC API — the only rendering entry point one_time_bot.py needs.

    scenes: prepare_scenes() se aayi hui list [{"image_filename","start",
            "end","effect",...}, ...]
    audio_path: narration wav
    bgm_path: optional background-music path (ya None)
    output_path: final .mp4 kahan likhna hai
    input_panels_dir: uploaded images kahan hain
    temp_dir: scratch workspace (clips, standardized frames yahin banti hain)
    quality: "360p" | "480p" | "720p" | "1080p"
    sfx_events: sfx_engine.build_sfx_events() ka output (ya None/[])
    progress_callback: optional fn(str) -> None

    Raises VideoEditorError on unrecoverable ffmpeg failure.
    """
    input_panels_dir = Path(input_panels_dir)
    temp_dir = Path(temp_dir)
    output_path = Path(output_path)
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[DEFAULT_QUALITY])

    logger.info(f"🎬 Final assembly shuru... (quality={quality})")
    img_proc = ImageProcessor(quality, temp_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    clip_paths = []
    fallback_images = sorted(
        (p for p in input_panels_dir.glob("*") if p.suffix.lower() in image_exts),
        key=lambda p: natural_sort_key(p.name),
    )
    total = len(scenes)

    for idx, item in enumerate(scenes):
        if progress_callback:
            progress_callback(f"clip {idx + 1}/{total}")
        img_file = item["image_filename"]
        img_path = input_panels_dir / img_file
        if not img_path.exists() and fallback_images:
            img_path = fallback_images[idx % len(fallback_images)]
            logger.warning(f"Image {img_file} missing, using {img_path.name}")
        dur = float(item["end"]) - float(item["start"])
        dur = max(dur, 0.5)
        clip_out = temp_dir / f"clip_{idx:03d}.mp4"
        img_proc.create_clip(img_path, clip_out, dur, item["effect"])
        clip_paths.append(clip_out)

    if progress_callback:
        progress_callback("clips concat ho rahe hain")

    concat_txt = temp_dir / "inputs.txt"
    with open(concat_txt, "w") as f:
        for c in clip_paths:
            f.write(f"file '{c.resolve()}'\n")
    temp_video = temp_dir / "no_subs.mp4"
    res = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(temp_video)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise VideoEditorError(f"Clips concat fail: {res.stderr[-500:]}")

    if progress_callback:
        progress_callback("audio mix ho raha hai")

    # NOTE: subtitles intentionally NOT burned into the video — sirf
    # clean visuals. Audio mix 3 tarah ke tracks tak combine karta hai:
    # narration (hamesha) + optional BGM + optional sparse SFX clips
    # (har ek apne exact timestamp par adelay se place hota hai).
    final_cmd = ["ffmpeg", "-y", "-i", str(temp_video), "-i", str(audio_path)]
    filter_parts = [f"[0:v]fps={preset['fps']}[v_out]"]
    audio_labels = ["[1:a]"]
    next_input_idx = 2

    if bgm_path and Path(bgm_path).exists():
        final_cmd.extend(["-stream_loop", "-1", "-i", str(bgm_path)])
        filter_parts.append(f"[{next_input_idx}:a]volume=0.08[bgm]")
        audio_labels.append("[bgm]")
        next_input_idx += 1

    SFX_VOLUME = 0.5
    for i, sfx in enumerate(sfx_events or []):
        final_cmd.extend(["-i", str(sfx["path"])])
        delay_ms = max(0, round(sfx["timestamp"] * 1000))
        label = f"sfx{i}"
        filter_parts.append(
            f"[{next_input_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"adelay=delays={delay_ms}:all=1,volume={SFX_VOLUME}[{label}]"
        )
        audio_labels.append(f"[{label}]")
        next_input_idx += 1

    if len(audio_labels) > 1:
        # normalize=0 zaroori hai: amix ka default (normalize=1) poore mix
        # ko hamesha 1/N se scale karta hai (N = total input tracks),
        # chahe wo tracks us waqt bol/bajj rahe hon ya nahi — isliye jaise
        # hi SFX/BGM track count video ke ek hisse se doosre mein badalta
        # hai, narration ka loudness bhi achanak badal jaata hai (empirically
        # verified: 1 track = -21dB, 2 tracks = -27dB, 3 tracks = -30dB).
        # normalize=0 ke saath amix sirf plain additive sum karta hai, gain
        # already har track ke apne volume= filter se control hoti hai.
        filter_parts.append(
            "".join(audio_labels) + f"amix=inputs={len(audio_labels)}:duration=first:dropout_transition=2:normalize=0[a_mix]"
        )
        audio_map = "[a_mix]"
    else:
        audio_map = "1:a"

    final_cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", "[v_out]",
        "-map", audio_map,
        "-c:v", "libx264", "-preset", preset["preset"], "-crf", str(preset["crf"]),
        "-r", str(preset["fps"]),
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    ])
    logger.info(f"🔥 Final render chal raha hai... ({quality})")
    res = subprocess.run(final_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Render error:\n{res.stderr}")
        raise VideoEditorError(f"Final render fail ho gaya: {res.stderr[-800:]}")
    logger.info(f"🎉 Video ready: {output_path.name} ({quality})")
    return output_path


if __name__ == "__main__":
    print("video_editor.py ek importable module hai — isse directly run nahi karte.")
    print("Use: from video_editor import prepare_scenes, render_video")
