"""
media/detector.py
Combines extension, magic-byte header, parsed JSON content and ffprobe
metadata to classify a single file. Returns a DetectionResult; never
raises on a bad file — problems are reported in result.warnings.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from media.formats import (
    CATEGORY_IMAGE, CATEGORY_AUDIO, CATEGORY_VIDEO, CATEGORY_ARCHIVE,
    CATEGORY_JSON, CATEGORY_STRUCTURED, CATEGORY_SUBTITLE, CATEGORY_TEXT,
    CATEGORY_FONT, CATEGORY_UNKNOWN,
    specs_for_filename, specs_for_header, category_from_extension_only,
    SUBTITLE_EXTENSIONS, TEXT_EXTENSIONS, STRUCTURED_EXTENSIONS,
    JSON_EXTENSIONS,
)


ROLE_IMAGE           = "image"
ROLE_ARCHIVE         = "archive"
ROLE_AUDIO           = "audio"
ROLE_VIDEO           = "video"
ROLE_EDIT_PLAN       = "edit_plan"
ROLE_TIMELINE_LEGACY = "timeline_legacy"
ROLE_JSON_OTHER      = "json_other"
ROLE_TEXT            = "text"
ROLE_SUBTITLE        = "subtitle"
ROLE_FONT            = "font"
ROLE_STRUCTURED      = "structured"
ROLE_UNKNOWN         = "unknown"

ROLE_BY_CATEGORY = {
    CATEGORY_IMAGE:      ROLE_IMAGE,
    CATEGORY_AUDIO:      ROLE_AUDIO,
    CATEGORY_VIDEO:      ROLE_VIDEO,
    CATEGORY_ARCHIVE:    ROLE_ARCHIVE,
    CATEGORY_TEXT:       ROLE_TEXT,
    CATEGORY_SUBTITLE:   ROLE_SUBTITLE,
    CATEGORY_FONT:       ROLE_FONT,
    CATEGORY_STRUCTURED: ROLE_STRUCTURED,
    CATEGORY_JSON:       ROLE_JSON_OTHER,
    CATEGORY_UNKNOWN:    ROLE_UNKNOWN,
}

AUDIO_NARRATION_HINTS = (
    "narration", "voice", "voiceover", "tts", "speech", "dialogue",
)
AUDIO_MUSIC_HINTS = (
    "music", "bgm", "background", "soundtrack", "ost", "score", "instrumental",
)


@dataclass
class DetectionResult:
    path: Path
    filename: str
    size: int
    extension: str
    extension_category: str
    header_category: str
    category: str
    role: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extension helpers
# ---------------------------------------------------------------------------

_KNOWN_EXTS_CACHE: Optional[Tuple[str, ...]] = None


def _known_extensions() -> Tuple[str, ...]:
    global _KNOWN_EXTS_CACHE
    if _KNOWN_EXTS_CACHE is None:
        from media.formats import FORMATS
        exts = set()
        for spec in FORMATS:
            exts.update(spec.extensions)
        exts.update(SUBTITLE_EXTENSIONS)
        exts.update(TEXT_EXTENSIONS)
        exts.update(STRUCTURED_EXTENSIONS)
        exts.update(JSON_EXTENSIONS)
        _KNOWN_EXTS_CACHE = tuple(sorted(exts, key=len, reverse=True))
    return _KNOWN_EXTS_CACHE


def _display_extension(filename: str) -> str:
    lower = filename.lower()
    for ext in _known_extensions():
        if lower.endswith(ext):
            return ext
    return Path(filename).suffix.lower() or ""


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def _read_header(path: Path, n: int = 1024) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def _looks_like_text(header: bytes) -> bool:
    if not header:
        return False
    if b"\x00" in header[:512]:
        return False
    try:
        header.decode("utf-8")
        return True
    except UnicodeDecodeError:
        try:
            header[: max(1, len(header) - 4)].decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False


# ---------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------

def _probe(path: Path) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {}
    if proc.returncode != 0 or not proc.stdout:
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}


def _parse_fps(rate: Optional[str]) -> Optional[float]:
    if not rate or "/" not in rate:
        return None
    try:
        num, den = rate.split("/", 1)
        n, d = float(num), float(den)
        if d == 0:
            return None
        return round(n / d, 3)
    except (TypeError, ValueError):
        return None


def _audio_metadata(probe: Dict[str, Any]) -> Dict[str, Any]:
    fmt = probe.get("format", {})
    streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    first = streams[0] if streams else {}
    try:
        duration = float(fmt.get("duration", 0) or 0)
    except (TypeError, ValueError):
        duration = 0.0
    sample_rate: Optional[int]
    try:
        sample_rate = int(first.get("sample_rate", 0) or 0) or None
    except (TypeError, ValueError):
        sample_rate = None
    return {
        "duration":     duration,
        "channels":     first.get("channels"),
        "sample_rate":  sample_rate,
        "codec":        first.get("codec_name"),
        "audio_streams": len(streams),
    }


def _video_metadata(probe: Dict[str, Any]) -> Dict[str, Any]:
    fmt = probe.get("format", {})
    streams = probe.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    try:
        duration = float(fmt.get("duration", 0) or 0)
    except (TypeError, ValueError):
        duration = 0.0
    return {
        "duration":  duration,
        "width":     v.get("width"),
        "height":    v.get("height"),
        "fps":       _parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate")),
        "codec":     v.get("codec_name"),
        "has_audio": bool(audio_streams),
    }


# ---------------------------------------------------------------------------
# JSON inspection
# ---------------------------------------------------------------------------

def _inspect_json(path: Path) -> Tuple[Optional[str], Dict[str, Any], List[str]]:
    warnings: List[str] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, {}, [f"json: {exc}"]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, {}, [f"json parse error at {exc.pos}: {exc.msg}"]

    if isinstance(data, dict) and isinstance(data.get("images"), list) and data["images"]:
        entries = data["images"]
        if all(isinstance(e, dict) for e in entries):
            files_ok = all(isinstance(e.get("file"), str) and e["file"] for e in entries)
            keep_present = any("keep" in e for e in entries)
            if files_ok and keep_present:
                meta = {
                    "entries": len(entries),
                    "version": data.get("version"),
                    "kept":    sum(1 for e in entries if e.get("keep", True)),
                }
                return ROLE_EDIT_PLAN, meta, warnings
            if files_ok and not keep_present:
                warnings.append("plan: 'images' present but every entry lacks 'keep'")
                return ROLE_JSON_OTHER, {"entries": len(entries)}, warnings
            warnings.append("plan: 'images' entries missing required 'file' string")
            return ROLE_JSON_OTHER, {"entries": len(entries)}, warnings

    if isinstance(data, list) and data:
        if all(isinstance(e, dict)
               and "image" in e and "start" in e and "end" in e
               for e in data):
            return ROLE_TIMELINE_LEGACY, {"entries": len(data)}, warnings

    return ROLE_JSON_OTHER, {"top_type": type(data).__name__}, warnings


# ---------------------------------------------------------------------------
# Category decision
# ---------------------------------------------------------------------------

def _decide_category(
    ext_cat: str,
    hdr_cat: str,
    warnings: List[str],
) -> Tuple[str, float]:
    if ext_cat != CATEGORY_UNKNOWN and hdr_cat != CATEGORY_UNKNOWN:
        if ext_cat == hdr_cat:
            return ext_cat, 0.95
        warnings.append(f"ext/header mismatch: ext={ext_cat} header={hdr_cat}")
        return hdr_cat, 0.55
    if hdr_cat != CATEGORY_UNKNOWN:
        return hdr_cat, 0.75
    if ext_cat != CATEGORY_UNKNOWN:
        return ext_cat, 0.65
    return CATEGORY_UNKNOWN, 0.10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(path: Path) -> DetectionResult:
    path = Path(path)
    filename = path.name
    extension = _display_extension(filename)
    evidence: List[str] = []
    warnings: List[str] = []

    try:
        size = path.stat().st_size
    except OSError as exc:
        return DetectionResult(
            path=path, filename=filename, size=0, extension=extension,
            extension_category=CATEGORY_UNKNOWN, header_category=CATEGORY_UNKNOWN,
            category=CATEGORY_UNKNOWN, role=ROLE_UNKNOWN, confidence=0.0,
            evidence=evidence, warnings=[f"stat failed: {exc}"], metadata={},
        )

    ext_specs = specs_for_filename(filename)
    ext_cat = ext_specs[0].category if ext_specs else category_from_extension_only(filename)

    if size == 0:
        return DetectionResult(
            path=path, filename=filename, size=0, extension=extension,
            extension_category=ext_cat, header_category=CATEGORY_UNKNOWN,
            category=CATEGORY_UNKNOWN, role=ROLE_UNKNOWN, confidence=0.0,
            evidence=evidence, warnings=["empty file"], metadata={},
        )

    header = _read_header(path)
    hdr_specs = specs_for_header(header)
    hdr_cat = hdr_specs[0].category if hdr_specs else CATEGORY_UNKNOWN

    if ext_specs:
        evidence.append(f"ext:{extension} -> {ext_cat}")
    else:
        evidence.append(f"ext:{extension or 'none'} unrecognised")

    if hdr_specs:
        evidence.append(f"header -> {hdr_specs[0].name} ({hdr_cat})")
    else:
        evidence.append("header: no magic match")

    category, confidence = _decide_category(ext_cat, hdr_cat, warnings)
    metadata: Dict[str, Any] = {}
    role = ROLE_BY_CATEGORY.get(category, ROLE_UNKNOWN)

    if category == CATEGORY_JSON:
        json_role, json_meta, json_warn = _inspect_json(path)
        warnings.extend(json_warn)
        if json_role:
            role = json_role
            metadata.update(json_meta)
            confidence = min(1.0, confidence + 0.05)

    elif category == CATEGORY_AUDIO:
        probe = _probe(path)
        if probe:
            metadata.update(_audio_metadata(probe))
            confidence = min(1.0, confidence + 0.05)
            lowered = filename.lower()
            if any(h in lowered for h in AUDIO_MUSIC_HINTS):
                metadata["audio_hint"] = "music"
                evidence.append("name hint: music")
            elif any(h in lowered for h in AUDIO_NARRATION_HINTS):
                metadata["audio_hint"] = "narration"
                evidence.append("name hint: narration")
        else:
            warnings.append("ffprobe returned no data — audio may be corrupt or unsupported")

    elif category == CATEGORY_VIDEO:
        probe = _probe(path)
        if probe:
            metadata.update(_video_metadata(probe))
            confidence = min(1.0, confidence + 0.05)
        else:
            warnings.append("ffprobe returned no data — video may be corrupt or unsupported")

    elif category == CATEGORY_UNKNOWN:
        if _looks_like_text(header):
            category = CATEGORY_TEXT
            role = ROLE_TEXT
            confidence = max(confidence, 0.40)
            evidence.append("content: utf-8 text")
        else:
            role = ROLE_UNKNOWN
            warnings.append(f"unknown format (ext={extension or 'none'}, no magic match)")

    return DetectionResult(
        path=path, filename=filename, size=size, extension=extension,
        extension_category=ext_cat, header_category=hdr_cat,
        category=category, role=role, confidence=confidence,
        evidence=evidence, warnings=warnings, metadata=metadata,
    )


def detect_many(paths: List[Path]) -> List[DetectionResult]:
    return [classify(p) for p in paths]


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

@dataclass
class FileInventory:
    files: List[DetectionResult] = field(default_factory=list)

    def add(self, path: Path) -> DetectionResult:
        result = classify(path)
        self.files.append(result)
        return result

    def extend(self, paths: List[Path]) -> List[DetectionResult]:
        return [self.add(p) for p in paths]

    def by_category(self, category: str) -> List[DetectionResult]:
        return [f for f in self.files if f.category == category]

    def by_role(self, role: str) -> List[DetectionResult]:
        return [f for f in self.files if f.role == role]

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.files:
            counts[f.role] = counts.get(f.role, 0) + 1
        return counts

    def all_warnings(self) -> List[str]:
        return [f"{f.filename}: {w}" for f in self.files for w in f.warnings]

    def ambiguous(self) -> List[DetectionResult]:
        return [f for f in self.files if f.warnings or f.confidence < 0.65]
