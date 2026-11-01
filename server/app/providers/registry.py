"""Model catalog and adapter lookup (MDL-1..5).

Selectors list only models whose provider has an API key in the environment
(MDL-3). Until a real adapter is registered for a model, it runs on the mock.
Prices are estimates from public pricing write-ups (October 2026); confirm
them on each provider's pricing page before relying on them.
"""
from __future__ import annotations

import os

from . import mock
from .base import ImageGenerator, ImageToVideo, TextToSpeech, VideoAnalyst, VideoSource, Writer

PROVIDER_ENV = {
    "google": "GEMINI_API_KEY",
    "youtube": "YOUTUBE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "higgsfield": "HIGGSFIELD_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
}

CATALOG = {
    "analyst": [
        {"id": "gemini-pro", "name": "Gemini Pro", "provider": "google", "per_video": 0.06, "limits": {"input": "YouTube URL", "public_videos_only": True}},
        {"id": "gemini-flash", "name": "Gemini Flash", "provider": "google", "per_video": 0.02, "limits": {"input": "YouTube URL", "public_videos_only": True}},
    ],
    # The writer is swappable: ChatGPT (OpenAI) by default, Claude (Anthropic) once its key is added.
    # per_video covers the angles and a short script; per_min adds the longer story of a longer video.
    "writer": [
        {"id": "gpt-5-5", "name": "GPT-5.5", "provider": "openai", "per_video": 0.30, "per_min": 0.03, "price": {"input_per_mtok": 5.0, "output_per_mtok": 30.0}},
        {"id": "gpt-5", "name": "GPT-5", "provider": "openai", "per_video": 0.10, "per_min": 0.01, "price": {"input_per_mtok": 1.25, "output_per_mtok": 10.0}},
        {"id": "claude-opus-5-5", "name": "Claude Opus 5.5", "provider": "anthropic", "per_video": 0.40, "per_min": 0.04, "price": {"input_per_mtok": 4.0, "output_per_mtok": 20.0}},
        {"id": "claude-sonnet-5-5", "name": "Claude Sonnet 5.5", "provider": "anthropic", "per_video": 0.20, "per_min": 0.02, "price": {"input_per_mtok": 2.0, "output_per_mtok": 10.0}},
        {"id": "gemini-pro-writer", "name": "Gemini Pro", "provider": "google", "per_video": 0.10, "per_min": 0.01},
    ],
    "voice": [
        {"id": "gemini-tts-flash", "name": "Gemini TTS Flash", "provider": "google", "cost": 0.01, "unit": "per 1k chars", "limits": {"cloning": False}},
        {"id": "gemini-tts-pro", "name": "Gemini TTS Pro", "provider": "google", "cost": 0.02, "unit": "per 1k chars", "limits": {"cloning": False, "multi_speaker": True}},
        {"id": "elevenlabs-v2", "name": "ElevenLabs Multilingual v2", "provider": "elevenlabs", "cost": 0.18, "unit": "per 1k chars", "limits": {"cloning": True}},
    ],
    "image": [
        {"id": "nano-banana-2", "name": "Nano Banana 2", "provider": "google", "cost": 0.05, "unit": "per image", "limits": {"reference_images": True}},
        {"id": "imagen-4", "name": "Imagen 4", "provider": "google", "cost": 0.04, "unit": "per image", "limits": {"reference_images": False}},
        {"id": "imagen-4-fast", "name": "Imagen 4 Fast", "provider": "google", "cost": 0.02, "unit": "per image", "limits": {"reference_images": False}},
    ],
    "video": [
        {"id": "veo-31-fast", "name": "Veo 3.1 Fast", "provider": "google", "cost": 0.15, "unit": "per second", "limits": {"max_clip": 8, "native_audio": True}},
        {"id": "veo-31", "name": "Veo 3.1", "provider": "google", "cost": 0.40, "unit": "per second", "limits": {"max_clip": 8, "native_audio": True}},
        {"id": "veo-31-lite", "name": "Veo 3.1 Lite", "provider": "google", "cost": 0.06, "unit": "per second", "limits": {"max_clip": 8}},
        {"id": "seedance-2", "name": "Seedance 2.0 (Higgsfield)", "provider": "higgsfield", "cost": 0.12, "unit": "per second", "limits": {"max_clip": 15}},
        {"id": "kling-3", "name": "Kling 3.0 (Higgsfield)", "provider": "higgsfield", "cost": 0.12, "unit": "per second", "limits": {"max_clip": 15, "price_confirmed": False}},
    ],
}

# Real adapters register here as they are written, e.g. REAL["writer"]["gpt-5-5"] = OpenAIWriter and
# REAL["writer"]["claude-opus-5-5"] = ClaudeWriter. Both implement base.Writer, so the rest of the app is unchanged.
REAL: dict[str, dict[str, type]] = {kind: {} for kind in CATALOG}


def mock_mode() -> bool:
    return os.getenv("ANGLE_STUDIO_MOCK", "1") == "1"


def has_key(provider: str) -> bool:
    return mock_mode() or bool(os.getenv(PROVIDER_ENV[provider]))


def available() -> dict[str, list[dict]]:
    return {kind: [m | {"configured": has_key(m["provider"])} for m in models] for kind, models in CATALOG.items()}


def model(kind: str, model_id: str) -> dict:
    for m in CATALOG[kind]:
        if m["id"] == model_id:
            return m
    raise KeyError(f"Unknown {kind} model {model_id!r}")


def _real(kind: str, model_id: str):
    return None if mock_mode() else REAL[kind].get(model_id)


def source() -> VideoSource:
    return mock.MockVideoSource()


def analyst(model_id: str) -> VideoAnalyst:
    m = model("analyst", model_id)
    return (cls := _real("analyst", model_id)) and cls() or mock.MockAnalyst(m["per_video"])


def writer(model_id: str) -> Writer:
    m = model("writer", model_id)
    # Two writer calls per video (angles, then script); split the estimate between them.
    return (cls := _real("writer", model_id)) and cls() or mock.MockWriter(m["per_video"] / 2)


def tts(model_id: str) -> TextToSpeech:
    m = model("voice", model_id)
    return (cls := _real("voice", model_id)) and cls() or mock.MockTTS(m["cost"])


def images(model_id: str) -> ImageGenerator:
    m = model("image", model_id)
    return (cls := _real("image", model_id)) and cls() or mock.MockImages(m["cost"])


def video(model_id: str) -> ImageToVideo:
    m = model("video", model_id)
    return (cls := _real("video", model_id)) and cls() or mock.MockVideo(m["cost"], m["limits"]["max_clip"])
