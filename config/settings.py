"""
config/settings.py
Central env + defaults reader. No .env loading — GitHub Secrets only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.error_handler import PipelineError, STAGE_INTAKE


_DEFAULTS_PATH = Path(__file__).parent / "defaults.json"


def _load_defaults() -> Dict[str, Any]:
    try:
        return json.loads(_DEFAULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(STAGE_INTAKE, f"cannot load config/defaults.json: {exc}") from exc


_DEFAULTS = _load_defaults()


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise PipelineError(STAGE_INTAKE, f"env {name} must be an integer, got {raw!r}")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise PipelineError(STAGE_INTAKE, f"env {name} must be a float, got {raw!r}")


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Secrets:
    api_id: int
    api_hash: str
    bot_token: str
    gemini_api_key: str
    owner_chat_id: Optional[int]


def load_secrets() -> Secrets:
    api_id_raw = _env_str("API_ID")
    api_hash = _env_str("API_HASH")
    bot_token = _env_str("BOT_TOKEN")
    gemini_api_key = _env_str("GEMINI_API_KEY")
    owner_chat_id_raw = _env_str("OWNER_CHAT_ID")

    missing = [
        name
        for name, value in (
            ("API_ID", api_id_raw),
            ("API_HASH", api_hash),
            ("BOT_TOKEN", bot_token),
            ("GEMINI_API_KEY", gemini_api_key),
        )
        if not value
    ]
    if missing:
        raise PipelineError(
            STAGE_INTAKE,
            f"missing required secrets: {', '.join(missing)}",
            hint="add them under GitHub repo Settings → Secrets and variables → Actions",
        )

    try:
        api_id = int(api_id_raw)  # type: ignore[arg-type]
    except ValueError as exc:
        raise PipelineError(STAGE_INTAKE, f"API_ID must be numeric, got {api_id_raw!r}") from exc

    owner_chat_id: Optional[int] = None
    if owner_chat_id_raw:
        try:
            owner_chat_id = int(owner_chat_id_raw)
        except ValueError as exc:
            raise PipelineError(
                STAGE_INTAKE, f"OWNER_CHAT_ID must be numeric, got {owner_chat_id_raw!r}"
            ) from exc

    return Secrets(
        api_id=api_id,
        api_hash=api_hash,      # type: ignore[arg-type]
        bot_token=bot_token,    # type: ignore[arg-type]
        gemini_api_key=gemini_api_key,  # type: ignore[arg-type]
        owner_chat_id=owner_chat_id,
    )


@dataclass(frozen=True)
class QualityPreset:
    key: str
    width: int
    height: int
    fps: int
    crf: int
    encoder_preset: str
    label: str


def quality_presets() -> Dict[str, QualityPreset]:
    out: Dict[str, QualityPreset] = {}
    for key, raw in _DEFAULTS["quality_presets"].items():
        out[key] = QualityPreset(
            key=key,
            width=int(raw["width"]),
            height=int(raw["height"]),
            fps=int(raw["fps"]),
            crf=int(raw["crf"]),
            encoder_preset=str(raw["preset"]),
            label=str(raw["label"]),
        )
    return out


def quality_order() -> List[str]:
    return list(_DEFAULTS["quality_order"])


def default_quality() -> str:
    return str(_DEFAULTS["default_quality"])


@dataclass(frozen=True)
class TranscriberConfig:
    local_whisper_model: str
    gemini_fallback_models: List[str]
    timeout_seconds: int


def transcriber_config() -> TranscriberConfig:
    raw = _DEFAULTS["transcriber"]
    return TranscriberConfig(
        local_whisper_model=_env_str("WHISPER_LOCAL_MODEL", raw["local_whisper_model"]) or raw["local_whisper_model"],
        gemini_fallback_models=_env_list("GEMINI_TRANSCRIBE_MODELS", raw["gemini_fallback_models"]),
        timeout_seconds=_env_int("TRANSCRIBE_TIMEOUT_SECONDS", int(raw["timeout_seconds"])),
    )


@dataclass(frozen=True)
class MatcherConfig:
    model: str
    fallback_models: List[str]
    batch_size: int
    min_confidence: float
    allow_fuzzy_fallback: bool


def matcher_config() -> MatcherConfig:
    raw = _DEFAULTS["matcher"]
    return MatcherConfig(
        model=_env_str("MATCHER_MODEL", raw["model"]) or raw["model"],
        fallback_models=_env_list("MATCHER_FALLBACK_MODELS", raw["fallback_models"]),
        batch_size=_env_int("MATCHER_BATCH_SIZE", int(raw["batch_size"])),
        min_confidence=_env_float("MATCHER_MIN_CONFIDENCE", float(raw["min_confidence"])),
        allow_fuzzy_fallback=bool(raw["allow_fuzzy_fallback"]),
    )


@dataclass(frozen=True)
class RenderConfig:
    supersample: int
    zoom_amount: float
    pan_zoom_level: float


def render_config() -> RenderConfig:
    raw = _DEFAULTS["render"]
    return RenderConfig(
        supersample=_env_int("RENDER_SUPERSAMPLE", int(raw["supersample"])),
        zoom_amount=_env_float("RENDER_ZOOM_AMOUNT", float(raw["zoom_amount"])),
        pan_zoom_level=_env_float("RENDER_PAN_ZOOM_LEVEL", float(raw["pan_zoom_level"])),
    )


@dataclass(frozen=True)
class VisionConfig:
    engine: str
    lang: str
    max_image_side: int
    timeout_per_image_seconds: int


def vision_config() -> VisionConfig:
    raw = _DEFAULTS["vision"]
    return VisionConfig(
        engine=_env_str("VISION_ENGINE", raw["engine"]) or raw["engine"],
        lang=_env_str("VISION_LANG", raw["lang"]) or raw["lang"],
        max_image_side=_env_int("VISION_MAX_IMAGE_SIDE", int(raw["max_image_side"])),
        timeout_per_image_seconds=_env_int("VISION_TIMEOUT_PER_IMAGE", int(raw["timeout_per_image_seconds"])),
    )


@dataclass(frozen=True)
class IntakeConfig:
    debounce_seconds: int
    max_upload_mb: int
    extract_max_files: int
    extract_max_total_mb: int
    extract_max_single_mb: int
    extract_max_depth: int


def intake_config() -> IntakeConfig:
    raw = _DEFAULTS["intake"]
    return IntakeConfig(
        debounce_seconds=_env_int("DEBOUNCE_SECONDS", int(raw["debounce_seconds"])),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", int(raw["max_upload_mb"])),
        extract_max_files=_env_int("EXTRACT_MAX_FILES", int(raw["extract_max_files"])),
        extract_max_total_mb=_env_int("EXTRACT_MAX_TOTAL_MB", int(raw["extract_max_total_mb"])),
        extract_max_single_mb=_env_int("EXTRACT_MAX_SINGLE_MB", int(raw["extract_max_single_mb"])),
        extract_max_depth=_env_int("EXTRACT_MAX_DEPTH", int(raw["extract_max_depth"])),
    )
