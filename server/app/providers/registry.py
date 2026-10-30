"""Model catalog and adapter lookup (MDL-1..5).

Selectors list only models whose provider has an API key in the environment
(MDL-3). Until a real adapter is registered for a model, it runs on the mock.
"""
from __future__ import annotations

import os

from . import mock
from .base import LLM, ImageGenerator, ImageToVideo, SpeechToText, TextToSpeech, VideoSource

CATALOG = {
    "voice": [
        {"id": "elevenlabs-v2", "name": "ElevenLabs Multilingual v2", "provider": "elevenlabs", "cost": 0.18, "unit": "per 1k chars", "limits": {"cloning": True, "languages": 29}},
        {"id": "openai-tts-hd", "name": "OpenAI TTS HD", "provider": "openai", "cost": 0.03, "unit": "per 1k chars", "limits": {"cloning": False}},
        {"id": "cartesia-sonic", "name": "Cartesia Sonic", "provider": "cartesia", "cost": 0.05, "unit": "per 1k chars", "limits": {"cloning": True}},
    ],
    "image": [
        {"id": "flux-11-pro", "name": "FLUX 1.1 Pro", "provider": "bfl", "cost": 0.04, "unit": "per image", "limits": {"aspect_ratios": ["9:16", "16:9"], "seeds": True, "reference_images": True}},
        {"id": "imagen-4", "name": "Imagen 4", "provider": "google", "cost": 0.04, "unit": "per image", "limits": {"aspect_ratios": ["9:16", "16:9"], "seeds": False, "reference_images": False}},
        {"id": "sdxl", "name": "Stable Diffusion XL", "provider": "stability", "cost": 0.01, "unit": "per image", "limits": {"aspect_ratios": ["9:16", "16:9"], "seeds": True, "reference_images": False}},
    ],
    "video": [
        {"id": "kling-21", "name": "Kling 2.1", "provider": "kling", "cost": 0.07, "unit": "per second", "limits": {"max_clip": 10, "character_reference": True, "lip_sync": False}},
        {"id": "runway-gen4", "name": "Runway Gen-4", "provider": "runway", "cost": 0.10, "unit": "per second", "limits": {"max_clip": 10, "character_reference": True, "lip_sync": False}},
        {"id": "veo-3", "name": "Veo 3", "provider": "google", "cost": 0.40, "unit": "per second", "limits": {"max_clip": 8, "character_reference": False, "lip_sync": True}},
    ],
}

# Real adapters register here as they are written, e.g. REAL["voice"]["elevenlabs-v2"] = ElevenLabsTTS
REAL: dict[str, dict[str, type]] = {"voice": {}, "image": {}, "video": {}}


def mock_mode() -> bool:
    return os.getenv("ANGLE_STUDIO_MOCK", "1") == "1"


def has_key(provider: str) -> bool:
    return mock_mode() or bool(os.getenv(f"{provider.upper()}_API_KEY"))


def available() -> dict[str, list[dict]]:
    return {kind: [m | {"configured": has_key(m["provider"])} for m in models] for kind, models in CATALOG.items()}


def model(kind: str, model_id: str) -> dict:
    for m in CATALOG[kind]:
        if m["id"] == model_id:
            return m
    raise KeyError(f"Unknown {kind} model {model_id!r}")


def tts(model_id: str) -> TextToSpeech:
    m = model("voice", model_id)
    if not mock_mode() and model_id in REAL["voice"]:
        return REAL["voice"][model_id]()
    return mock.MockTTS(m["cost"])


def images(model_id: str) -> ImageGenerator:
    m = model("image", model_id)
    if not mock_mode() and model_id in REAL["image"]:
        return REAL["image"][model_id]()
    return mock.MockImages(m["cost"])


def video(model_id: str) -> ImageToVideo:
    m = model("video", model_id)
    if not mock_mode() and model_id in REAL["video"]:
        return REAL["video"][model_id]()
    return mock.MockVideo(m["cost"], m["limits"]["max_clip"])


def llm() -> LLM:
    return mock.MockLLM()


def source() -> VideoSource:
    return mock.MockVideoSource()


def stt() -> SpeechToText:
    return mock.MockSpeechToText()
