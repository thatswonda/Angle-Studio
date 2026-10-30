"""One interface per capability (PRD 7.1, MDL-4).

Each provider is a small adapter that implements one of these. Adding a model
means writing one adapter and registering it in registry.py; nothing else in
the tool changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VideoInfo:
    title: str
    channel: str
    duration: float
    published: str
    captions: Optional[str]
    comments: list[dict]
    key_frames: list[dict]


@dataclass
class AudioResult:
    path: Path
    duration: float
    cost: float


@dataclass
class MediaResult:
    path: Path
    seed: Optional[int]
    cost: float


class VideoSource(ABC):
    """Reads metadata, captions, comments and key frames for a link (ING-1..4)."""

    @abstractmethod
    async def fetch(self, url: str) -> VideoInfo: ...


class SpeechToText(ABC):
    """Transcribes audio when no captions exist (ING-2)."""

    @abstractmethod
    async def transcribe(self, audio_path: Path) -> str: ...


class LLM(ABC):
    """Analysis and writing. Must return JSON matching the requested schema (PRD 7.3)."""

    @abstractmethod
    async def complete_json(self, system: str, prompt: str, schema: dict) -> dict: ...


class TextToSpeech(ABC):
    @abstractmethod
    async def synthesize(self, text: str, voice_id: str, out: Path) -> AudioResult: ...

    @abstractmethod
    def voices(self) -> list[dict]: ...


class ImageGenerator(ABC):
    @abstractmethod
    async def generate(self, prompt: str, aspect_ratio: str, out: Path,
                       seed: Optional[int] = None, reference_images: Optional[list[Path]] = None) -> MediaResult: ...


class ImageToVideo(ABC):
    max_clip_seconds: float = 10

    @abstractmethod
    async def animate(self, image: Path, motion_prompt: str, seconds: float, aspect_ratio: str, out: Path) -> MediaResult: ...
