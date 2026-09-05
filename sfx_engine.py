#!/usr/bin/env python3
"""
SFX ENGINE
----------
Standalone plug-and-play SFX detection and Freesound CC0 engine.

Ye file poori tarah INDEPENDENT hai — one_time_bot.py isse sirf ek
function import karta hai:

    from sfx_engine import build_sfx_events

Future mein SFX system upgrade karna ho (better detector, naya
provider, AI-generated SFX, etc.) to sirf yahi file replace/edit karo —
one_time_bot.py ko touch karne ki zaroorat nahi.

============================================================
PUBLIC API (sirf yahi bahar se use hota hai)
============================================================

    build_sfx_events(
        segments,               # transcript segments: [{"start","end","text"}, ...]
        prompts_text,           # extra story-context (SFX detection ko madad karta hai)
        temp_dir,                # yahin par downloaded .mp3 clips save hote hain
        api_key=None,            # Freesound key (default: env FREESOUND_API_KEY)
        progress_callback=None,  # optional: fn(str) -> None, human-readable status
    ) -> List[Dict]

    Har event: {"timestamp": float, "path": str, "keyword": str, "duration": float}

    NEVER raises. NEVER crashes. Har failure sirf UTNA hi SFX skip karta
    hai, poore video render ko kabhi nahi rokta:
        - FREESOUND_API_KEY missing        -> []
        - GEMINI_API_KEY missing            -> []
        - Gemini detection poori fail       -> []
        - Freesound search ek keyword ke liye fail -> sirf wo SFX skip
        - download fail                     -> sirf wo SFX skip
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1) CONFIGURATION — sab SFX constants yahin hain, bot mein scattered nahi
# ---------------------------------------------------------------------------

FREESOUND_SEARCH_URL = "https://freesound.org/apiv2/search/text/"
# CC0 = "no rights reserved" — YouTube monetization ke liye sabse safe.
# Attribution / Attribution-NonCommercial / ShareAlike / unknown license
# waale sounds YAHAN HAMESHA reject honge, chahe bot khud kuch bhi na
# check kare — licensing enforcement is module ki responsibility hai.
FREESOUND_LICENSE_FILTER = 'license:"Creative Commons 0"'

SFX_MAX_DURATION_SEC = float(os.getenv("SFX_MAX_DURATION_SEC", "12.0"))
SFX_VOLUME = float(os.getenv("SFX_VOLUME", "0.5"))
# Do SFX ke beech itna minimum gap zaroor rahega (chahe Gemini zyada bhi
# flag kar de) — hard safety check, Gemini par bharosa nahi karte.
SFX_MIN_GAP_SECONDS = float(os.getenv("SFX_MIN_GAP_SECONDS", "3.0"))

GEMINI_SFX_BATCH_SIZE = int(os.getenv("SFX_GEMINI_BATCH_SIZE", "25"))
PROMPTS_TEXT_MAX_CHARS = int(os.getenv("SFX_PROMPTS_TEXT_MAX_CHARS", "4000"))

# Gemini MODEL ROTATION — jaise hi ek model rate-limit (429/quota) de
# de, turant agla model try hota hai (wait kiye bina). Google har model
# ko ALAG RPM/RPD quota deta hai, isliye ek model ka quota khatam hone
# ka matlab ye nahi ki baaki sab bhi khatam hain. Comma-separated env
# var se override ho sakta hai; default list latest-se-purane models
# tak covers karti hai.
GEMINI_SFX_MODELS = [
    m.strip() for m in os.getenv(
        "GEMINI_SFX_MODELS",
        "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash-lite,"
        "gemini-3-flash,gemini-3.6-flash,gemini-2.5-flash,gemini-2-flash"
    ).split(",") if m.strip()
]


# ---------------------------------------------------------------------------
# 2) INTERNAL HELPERS (main bot ko in details ki zaroorat kabhi nahi)
# ---------------------------------------------------------------------------

def _strip_json_fences(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
    return raw.strip()


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(tok in msg for tok in ("429", "rate limit", "resource_exhausted", "quota"))


def _call_gemini_json_with_rotation(prompt: str, gemini_api_key: str,
                                     max_retries_per_model: int = 2) -> Optional[list]:
    """Gemini ko JSON-mode mein call karta hai, aur agar koi model
    rate-limited nikle to GEMINI_SFX_MODELS list mein se agle model par
    turant switch ho jaata hai. Sirf tab None return hota hai jab SAARE
    models fail/rate-limited ho jaayein — caller isse "detection fail"
    maan kar [] return karega, crash nahi hoga."""
    try:
        from google import genai
        from google.genai import types as genai_types
    except Exception as e:
        logger.warning(f"SFX: google-genai import fail ({e}) — SFX detection skip ho rahi hai.")
        return None

    client = genai.Client(api_key=gemini_api_key)
    last_exc: Optional[Exception] = None

    for model in GEMINI_SFX_MODELS:
        delay = 3.0
        for attempt in range(1, max_retries_per_model + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
                )
                content = _strip_json_fences(response.text or "")
                return json.loads(content)
            except Exception as e:
                last_exc = e
                if _is_rate_limit_error(e):
                    logger.warning(
                        f"SFX: Gemini model '{model}' rate-limited — agle model par switch "
                        f"ho raha hai (is model ka quota khatam, baaki models abhi fresh hain)."
                    )
                    break  # is model par wait karna bekaar hai, seedha agle model try karo
                if attempt == max_retries_per_model:
                    logger.warning(f"SFX: Gemini model '{model}' fail ({e}) — agla model try ho raha hai.")
                    break
                time.sleep(delay)
                delay *= 2.0

    logger.warning(f"SFX: SAARE Gemini models fail/rate-limited ho gaye (last error: {last_exc}).")
    return None


# ---------------------------------------------------------------------------
# 3) SFX DETECTION (Gemini) — genuinely important moments only, spam nahi
# ---------------------------------------------------------------------------

def detect_sfx_moments(segments: List[Dict], prompts_text: str, gemini_api_key: str) -> List[Dict]:
    """Narration transcript padh kar sirf GENUINELY strong, distinct
    sound-effect moments flag karta hai (explosion, gunshot, door slam,
    glass break, thunder, footsteps, etc.) — zyada tar segments mein
    KUCH bhi flag nahi hoga, jaan-boojh kar (taaki video mein SFX spam
    na lage). Har batch call independently rate-limit-safe hai (model
    rotation ke through)."""
    if not segments:
        return []
    capped_prompts_text = prompts_text or "No extra context."
    if len(capped_prompts_text) > PROMPTS_TEXT_MAX_CHARS:
        capped_prompts_text = capped_prompts_text[:PROMPTS_TEXT_MAX_CHARS] + "\n...(truncated for length)"

    events = []
    for batch_start in range(0, len(segments), GEMINI_SFX_BATCH_SIZE):
        batch = segments[batch_start: batch_start + GEMINI_SFX_BATCH_SIZE]
        segment_data = [
            {"id": batch_start + j, "start": round(s["start"], 2), "text": s.get("text", "")}
            for j, s in enumerate(batch)
        ]
        prompt = f"""
        You are a sound designer reviewing a video's narration transcript.
        Flag ONLY segments where a strong, distinct, one-shot sound effect
        would clearly belong (explosion, gunshot, punch/slap/hit, door
        slam/creak/knock, running footsteps, thunder, glass breaking,
        scream, wind gust, heavy rain, car crash/engine revving, siren,
        heartbeat, phone ringing, and similar).
        Be VERY selective — most segments should get NOTHING flagged. This
        is meant to be sparse, occasional emphasis, never continuous or
        frequent. If in doubt, leave it out.
        For each flagged segment, give a short 2-4 word English search
        keyword suitable for searching a sound-effects library (e.g.
        "glass shattering", "door slam", "thunder crack").
        Output ONLY a JSON array (can be empty: []) like:
        [{{"id": 3, "keyword": "door slam"}}]

        Segments (this batch only): {json.dumps(segment_data, ensure_ascii=False)}
        Story context (may be generic if not provided): {capped_prompts_text}
        """
        flagged = _call_gemini_json_with_rotation(prompt, gemini_api_key)
        if flagged is None:
            logger.warning(
                f"SFX detection batch [{segment_data[0]['id']}-{segment_data[-1]['id']}] "
                f"fail ho gaya, skip kiya."
            )
            continue
        by_id = {s["id"]: s for s in segment_data}
        for item in flagged or []:
            sid = item.get("id")
            kw = item.get("keyword")
            if sid in by_id and kw:
                events.append({"timestamp": by_id[sid]["start"], "keyword": kw})

    events.sort(key=lambda e: e["timestamp"])
    sparse = []
    for ev in events:
        if not sparse or ev["timestamp"] - sparse[-1]["timestamp"] >= SFX_MIN_GAP_SECONDS:
            sparse.append(ev)
    if len(events) != len(sparse):
        logger.info(
            f"SFX: {len(events)} flagged -> {len(sparse)} kept after "
            f"{SFX_MIN_GAP_SECONDS}s min-gap spam guard."
        )
    return sparse


# ---------------------------------------------------------------------------
# 4) SFX PROVIDER — abhi Freesound (CC0-only). Future providers
#    (Pixabay, Internet Archive, local library, ...) yahan add ho
#    saktein hain bina build_sfx_events() ko chhue.
# ---------------------------------------------------------------------------

def _freesound_find_cc0(keyword: str, freesound_api_key: str, max_retries: int = 3) -> Optional[Dict]:
    """Freesound par search karta hai, SIRF CC0-licensed results leta
    hai. Attribution / Attribution-NonCommercial / ShareAlike / unknown
    license kabhi accept nahi hoti — filter khud query mein hardcoded
    hai, is licensing check ko main bot par kabhi depend nahi kiya
    jaata."""
    params = {
        "query": keyword,
        "token": freesound_api_key,
        "filter": f'{FREESOUND_LICENSE_FILTER} duration:[0.1 TO {SFX_MAX_DURATION_SEC}]',
        "fields": "id,name,previews,duration,license",
        "sort": "rating_desc",
        "page_size": 5,
    }
    delay = 3.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(FREESOUND_SEARCH_URL, params=params, timeout=20)
            if resp.status_code == 429:
                raise RuntimeError("429 rate limited")
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return None
            top = results[0]
            # Defensive re-check: filter chahe query mein ho, yahan bhi
            # CC0 confirm karo — kabhi bhi "trust the query alone" nahi.
            license_str = str(top.get("license", "")).lower()
            if "0" not in license_str and "publicdomain" not in license_str.replace(" ", ""):
                logger.warning(f"SFX: '{keyword}' result CC0 nahi nikla ({license_str}), skip.")
                return None
            previews = top.get("previews") or {}
            preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
            if not preview_url:
                return None
            return {
                "id": top.get("id"),
                "name": top.get("name"),
                "url": preview_url,
                "duration": float(top.get("duration", 0.0) or 0.0),
            }
        except Exception as e:
            if attempt == max_retries:
                logger.warning(f"SFX: Freesound search fail for '{keyword}' ({e}), skip.")
                return None
            wait = min(60.0, 20.0 * attempt) if _is_rate_limit_error(e) else delay
            delay *= 2.0
            time.sleep(wait)
    return None


def _download_sfx_clip(url: str, dest_path: Path) -> bool:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest_path.write_bytes(resp.content)
        return True
    except Exception as e:
        logger.warning(f"SFX: download fail ({e}), is clip ko skip kiya ja raha hai.")
        return False


# ---------------------------------------------------------------------------
# 5) PUBLIC API — one_time_bot.py sirf ISE call karta hai
# ---------------------------------------------------------------------------

def build_sfx_events(
    segments: List[Dict],
    prompts_text: str,
    temp_dir,
    api_key: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """detect -> Freesound search (CC0-only) -> download, poora pipeline
    ek hi call mein. Har stage par har failure sirf US EK sfx (ya poore
    SFX step) ko skip karti hai — poore video ka render KABHI is wajah
    se nahi rukta.

    Args:
        segments: transcript segments [{"start","end","text"}, ...]
        prompts_text: extra story context (SFX detection ko madad)
        temp_dir: downloaded .mp3 clips yahan save honge
        api_key: Freesound API key (default: env FREESOUND_API_KEY)
        progress_callback: optional fn(str) -> None, human-readable status

    Returns:
        List[Dict], har ek: {"timestamp": float, "path": str,
                              "keyword": str, "duration": float}
    """
    temp_dir = Path(temp_dir)
    freesound_key = api_key or os.getenv("FREESOUND_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not freesound_key:
        logger.info("SFX: FREESOUND_API_KEY set nahi hai — SFX step disabled, [] return, render jaari.")
        return []
    if not gemini_key:
        logger.info("SFX: GEMINI_API_KEY set nahi hai — SFX detection disabled, [] return, render jaari.")
        return []

    try:
        if progress_callback:
            progress_callback("sound-effect moments dhoonde ja rahe hain")
        moments = detect_sfx_moments(segments, prompts_text, gemini_key)
    except Exception as e:
        logger.warning(f"SFX: detection poori tarah fail ho gayi ({e}) — [] return, render jaari rahega.")
        return []

    events = []
    for i, moment in enumerate(moments):
        try:
            if progress_callback:
                progress_callback(f"'{moment['keyword']}' ke liye sound dhoonda ja raha hai ({i + 1}/{len(moments)})")
            found = _freesound_find_cc0(moment["keyword"], freesound_key)
            if not found:
                continue
            dest = temp_dir / f"sfx_{i:03d}.mp3"
            if _download_sfx_clip(found["url"], dest):
                logger.info(f"🔊 SFX @ {moment['timestamp']:.1f}s: '{moment['keyword']}' -> {found['name']} (CC0)")
                events.append({
                    "timestamp": moment["timestamp"],
                    "path": str(dest),
                    "keyword": moment["keyword"],
                    "duration": found.get("duration", 0.0),
                })
        except Exception as e:
            logger.warning(f"SFX: '{moment.get('keyword')}' event skip ho gaya ({e}).")
            continue

    return events


# Bina kisi application execute kiye is module ko import karna safe hai
# (koi module-level side-effect, koi client init, koi bot-start nahi).
if __name__ == "__main__":
    print("sfx_engine.py ek importable module hai — isse directly run nahi karte.")
    print("Use: from sfx_engine import build_sfx_events")
