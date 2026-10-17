#!/usr/bin/env python3
"""
media/formats.py
----------------
Format registry and content-signature detection.

Pure data + pure functions. No file I/O in this module — the caller
provides the leading bytes of a file, and this module tells you what
formats those bytes are consistent with.

This is the single source of truth for:
  - which extensions belong to which category
  - the magic-byte signatures used for content verification
  - which categories are processable given the current dependencies

Kept separate from detector.py (which scores candidates and assigns
roles) so that adding a new format is a one-file change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Category constants (used everywhere else in the project)
# ---------------------------------------------------------------------------

CATEGORY_IMAGE      = "image"
CATEGORY_AUDIO      = "audio"
CATEGORY_VIDEO      = "video"
CATEGORY_ARCHIVE    = "archive"
CATEGORY_JSON       = "json"
CATEGORY_STRUCTURED = "structured"   # yaml / xml / toml / ini
CATEGORY_SUBTITLE   = "subtitle"
CATEGORY_TEXT       = "text"
CATEGORY_FONT       = "font"
CATEGORY_UNKNOWN    = "unknown"


# ---------------------------------------------------------------------------
# FormatSpec — one row per logical format
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormatSpec:
    """One logical format: e.g. 'jpeg' with extensions (.jpg, .jpeg, ...)
    and one or more (offset, prefix) magic signatures."""
    name: str                             # human name ("jpeg", "png", ...)
    category: str                         # one of the CATEGORY_* constants
    extensions: Tuple[str, ...]           # lowercase, leading dot
    magic: Tuple[Tuple[int, bytes], ...]  # ((offset, prefix), ...)

    def matches_header(self, header: bytes) -> bool:
        for offset, prefix in self.magic:
            end = offset + len(prefix)
            if len(header) >= end and header[offset:end] == prefix:
                return True
        return False


# ---------------------------------------------------------------------------
# Registry — one row per format we can recognise.
# Order within this list is irrelevant; matches are computed per-file.
# ---------------------------------------------------------------------------

FORMATS: List[FormatSpec] = [
    # ---------------- Images ------------------------------------------------
    FormatSpec(
        "jpeg", CATEGORY_IMAGE,
        (".jpg", ".jpeg", ".jpe", ".jfif"),
        ((0, b"\xff\xd8\xff"),),
    ),
    FormatSpec(
        "png", CATEGORY_IMAGE, (".png",),
        ((0, b"\x89PNG\r\n\x1a\n"),),
    ),
    FormatSpec(
        "gif", CATEGORY_IMAGE, (".gif",),
        ((0, b"GIF87a"), (0, b"GIF89a")),
    ),
    FormatSpec(
        "webp", CATEGORY_IMAGE, (".webp",),
        ((0, b"RIFF"), (8, b"WEBP")),
    ),
    FormatSpec(
        "bmp", CATEGORY_IMAGE, (".bmp",),
        ((0, b"BM"),),
    ),
    FormatSpec(
        "tiff", CATEGORY_IMAGE, (".tif", ".tiff"),
        ((0, b"II*\x00"), (0, b"MM\x00*")),
    ),
    FormatSpec(
        "avif", CATEGORY_IMAGE, (".avif",),
        ((4, b"ftypavif"), (4, b"ftypavis")),
    ),
    FormatSpec(
        "heic", CATEGORY_IMAGE, (".heic", ".heif"),
        ((4, b"ftypheic"), (4, b"ftypheix"), (4, b"ftypheim"),
         (4, b"ftypheis"), (4, b"ftypmif1")),
    ),
    FormatSpec(
        "jxl", CATEGORY_IMAGE, (".jxl",),
        ((0, b"\xff\x0a"),
         (0, b"\x00\x00\x00\x0cJXL \r\n\x87\n")),
    ),
    FormatSpec(
        "ico", CATEGORY_IMAGE, (".ico",),
        ((0, b"\x00\x00\x01\x00"),),
    ),

    # ---------------- Audio -------------------------------------------------
    FormatSpec(
        "mp3", CATEGORY_AUDIO, (".mp3",),
        ((0, b"ID3"), (0, b"\xff\xfb"),
         (0, b"\xff\xf3"), (0, b"\xff\xf2")),
    ),
    FormatSpec(
        "wav", CATEGORY_AUDIO, (".wav",),
        ((0, b"RIFF"), (8, b"WAVE")),
    ),
    FormatSpec(
        "flac", CATEGORY_AUDIO, (".flac",),
        ((0, b"fLaC"),),
    ),
    FormatSpec(
        "ogg", CATEGORY_AUDIO, (".ogg", ".oga", ".opus"),
        ((0, b"OggS"),),
    ),
    FormatSpec(
        "m4a", CATEGORY_AUDIO, (".m4a", ".aac"),
        ((4, b"ftypM4A "), (4, b"ftypM4B "), (4, b"ftypmp42")),
    ),
    FormatSpec(
        "wma", CATEGORY_AUDIO, (".wma",),
        ((0, b"\x30\x26\xb2\x75\x8e\x66\xcf\x11"),),
    ),
    FormatSpec(
        "aiff", CATEGORY_AUDIO, (".aiff", ".aif"),
        ((0, b"FORM"), (8, b"AIFF")),
    ),
    FormatSpec(
        "amr", CATEGORY_AUDIO, (".amr",),
        ((0, b"#!AMR"),),
    ),
    FormatSpec(
        "midi", CATEGORY_AUDIO, (".mid", ".midi"),
        ((0, b"MThd"),),
    ),

    # ---------------- Video -------------------------------------------------
    # Note: mp4's magic (ftyp at offset 4) is intentionally broad — it also
    # matches m4a, mov, 3gp, avif, heic. That is correct: those are ISO-BMFF
    # containers. The detector picks the most *specific* match; extension
    # usually wins the tie-break.
    FormatSpec(
        "mp4", CATEGORY_VIDEO, (".mp4", ".m4v"),
        ((4, b"ftyp"),),
    ),
    FormatSpec(
        "mov", CATEGORY_VIDEO, (".mov",),
        ((4, b"ftypqt"),),
    ),
    FormatSpec(
        "mkv", CATEGORY_VIDEO, (".mkv", ".webm"),
        ((0, b"\x1a\x45\xdf\xa3"),),
    ),
    FormatSpec(
        "avi", CATEGORY_VIDEO, (".avi",),
        ((0, b"RIFF"), (8, b"AVI ")),
    ),
    FormatSpec(
        "mpeg", CATEGORY_VIDEO, (".mpeg", ".mpg"),
        ((0, b"\x00\x00\x01\xba"), (0, b"\x00\x00\x01\xb3")),
    ),
    # TS has no reliable magic at offset 0 — 'G' appears repeatedly. We
    # detect TS by extension only; content validation happens later via
    # ffprobe in detector.py.
    FormatSpec("ts", CATEGORY_VIDEO, (".ts",), ()),
    FormatSpec(
        "3gp", CATEGORY_VIDEO, (".3gp",),
        ((4, b"ftyp3gp"),),
    ),
    FormatSpec(
        "wmv", CATEGORY_VIDEO, (".wmv",),
        ((0, b"\x30\x26\xb2\x75\x8e\x66\xcf\x11"),),
    ),
    FormatSpec(
        "flv", CATEGORY_VIDEO, (".flv",),
        ((0, b"FLV\x01"),),
    ),

    # ---------------- Archive ----------------------------------------------
    FormatSpec(
        "zip", CATEGORY_ARCHIVE, (".zip",),
        ((0, b"PK\x03\x04"), (0, b"PK\x05\x06"), (0, b"PK\x07\x08")),
    ),
    FormatSpec(
        "gzip", CATEGORY_ARCHIVE, (".gz", ".tgz"),
        ((0, b"\x1f\x8b"),),
    ),
    # TAR's "ustar" magic lives at offset 257 — requires ≥262-byte header.
    FormatSpec(
        "tar", CATEGORY_ARCHIVE, (".tar",),
        ((257, b"ustar"),),
    ),
    FormatSpec(
        "7z", CATEGORY_ARCHIVE, (".7z",),
        ((0, b"7z\xbc\xaf\x27\x1c"),),
    ),
    FormatSpec(
        "rar", CATEGORY_ARCHIVE, (".rar",),
        ((0, b"Rar!\x1a\x07"),),   # covers v1.5 and v5 prefixes
    ),

    # ---------------- Fonts ------------------------------------------------
    FormatSpec(
        "ttf", CATEGORY_FONT, (".ttf",),
        ((0, b"\x00\x01\x00\x00"), (0, b"true")),
    ),
    FormatSpec(
        "otf", CATEGORY_FONT, (".otf",),
        ((0, b"OTTO"),),
    ),
    FormatSpec(
        "woff", CATEGORY_FONT, (".woff",),
        ((0, b"wOFF"),),
    ),
    FormatSpec(
        "woff2", CATEGORY_FONT, (".woff2",),
        ((0, b"wOF2"),),
    ),
]


# ---------------------------------------------------------------------------
# Extensions with no reliable magic — detected by content heuristics
# (see detector.py) or accepted when the extension is unambiguous.
# ---------------------------------------------------------------------------

SUBTITLE_EXTENSIONS   = (".srt", ".vtt", ".ass", ".ssa")
TEXT_EXTENSIONS       = (".txt", ".md", ".csv", ".tsv")
STRUCTURED_EXTENSIONS = (".yaml", ".yml", ".xml", ".toml", ".ini")
JSON_EXTENSIONS       = (".json",)


# ---------------------------------------------------------------------------
# Which formats can we actually *process* in the current environment?
# Detection recognises all of them; extractor/renderer raise a clear
# error if a format is detected but not processable. Never crash silently.
# ---------------------------------------------------------------------------

# Python stdlib gives zip / tar / gzip for free.
PROCESSABLE_ARCHIVES = frozenset({"zip", "tar", "gzip"})
# 7z and rar are *recognised* but require extra libs (py7zr, rarfile)
# that are not installed. Detector reports this clearly.

# Pillow (already a dependency) can open these without plugins.
PROCESSABLE_IMAGES_BASE = frozenset({
    "jpeg", "png", "gif", "webp", "bmp", "tiff",
})
# avif / heic / jxl / ico require optional Pillow plugins. Recognised;
# image_cleaner skips with a warning if Pillow cannot open them.

# FFmpeg (installed in workflow) handles all of these.
PROCESSABLE_AUDIO = frozenset({
    "mp3", "wav", "flac", "ogg", "m4a", "wma", "aiff", "amr", "midi",
})
PROCESSABLE_VIDEO = frozenset({
    "mp4", "mov", "mkv", "avi", "mpeg", "ts", "3gp", "wmv", "flv",
})


# ---------------------------------------------------------------------------
# Lookup helpers (pure functions, no I/O)
# ---------------------------------------------------------------------------

def _build_extension_index() -> Dict[str, List[FormatSpec]]:
    index: Dict[str, List[FormatSpec]] = {}
    for spec in FORMATS:
        for ext in spec.extensions:
            index.setdefault(ext, []).append(spec)
    return index


_EXT_INDEX = _build_extension_index()
_SORTED_EXTENSIONS = sorted(_EXT_INDEX.keys(), key=len, reverse=True)


def specs_for_filename(filename: str) -> List[FormatSpec]:
    """
    Return all format specs whose declared extensions match the tail of
    `filename`. Longest match wins (so ".tar.gz" is treated as gzip,
    which is correct — the tar layer is inside the gzip stream).

    Case-insensitive. Returns [] when no extension is recognised.
    """
    lower = filename.lower()
    for ext in _SORTED_EXTENSIONS:
        if lower.endswith(ext):
            return list(_EXT_INDEX[ext])
    return []


def specs_for_header(header: bytes) -> List[FormatSpec]:
    """Return every FormatSpec whose magic matches the given header."""
    matches: List[FormatSpec] = []
    for spec in FORMATS:
        if not spec.magic:
            continue
        if spec.matches_header(header):
            matches.append(spec)
    return matches


def category_from_extension_only(filename: str) -> str:
    """
    Cheap category guess from extension only. Used as a *hint*; never
    trusted alone. Returns CATEGORY_UNKNOWN if no extension is recognised.
    """
    lower = filename.lower()
    for ext in _SORTED_EXTENSIONS:
        if lower.endswith(ext):
            return _EXT_INDEX[ext][0].category
    for ext in SUBTITLE_EXTENSIONS:
        if lower.endswith(ext):
            return CATEGORY_SUBTITLE
    for ext in TEXT_EXTENSIONS:
        if lower.endswith(ext):
            return CATEGORY_TEXT
    for ext in STRUCTURED_EXTENSIONS:
        if lower.endswith(ext):
            return CATEGORY_STRUCTURED
    for ext in JSON_EXTENSIONS:
        if lower.endswith(ext):
            return CATEGORY_JSON
    return CATEGORY_UNKNOWN


def is_recognised_extension(filename: str) -> bool:
    return category_from_extension_only(filename) != CATEGORY_UNKNOWN


# ---------------------------------------------------------------------------
# Public extension tuples — re-exported so no other module hard-codes
# "which extensions count as images" etc.
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = tuple(
    ext
    for spec in FORMATS
    if spec.category == CATEGORY_IMAGE
    for ext in spec.extensions
)
AUDIO_EXTENSIONS = tuple(
    ext
    for spec in FORMATS
    if spec.category == CATEGORY_AUDIO
    for ext in spec.extensions
)
VIDEO_EXTENSIONS = tuple(
    ext
    for spec in FORMATS
    if spec.category == CATEGORY_VIDEO
    for ext in spec.extensions
)
ARCHIVE_EXTENSIONS = tuple(
    ext
    for spec in FORMATS
    if spec.category == CATEGORY_ARCHIVE
    for ext in spec.extensions
)
FONT_EXTENSIONS = tuple(
    ext
    for spec in FORMATS
    if spec.category == CATEGORY_FONT
    for ext in spec.extensions
)
