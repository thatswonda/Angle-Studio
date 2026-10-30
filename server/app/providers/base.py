"""One interface per capability (PRD 7.1, MDL-4).

Each provider is a small adapter that implements one of these. Adding a model
means writing one adapter and registering it in registry.py; nothing else in
the tool changes.

The analysis is split in two: a VideoAnalyst (Gemini) watches the reference
video and writes a report; a Writer (Claude by default) never sees the video,
only that report, and writes the angles, scripts and metadata.
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
    views: int
    comments: list[dict]
    comments_scanned: int


@dataclass
class WatchReport:
    """What the analyst hands to the writer."""
    transcript: str
    scenes: list[dict]  # [{"t": "0:41", "description": "what is on screen"}]
    analysis: dict      # summary, hook, beats, pacing, why_it_worked, gaps
    cost: float


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
    """Metadata, stats and top comments for a link (ING-1, ING-4). YouTube Data API."""

    @abstractmethod
    async def fetch(self, url: str) -> VideoInfo: ...


class VideoAnalyst(ABC):
    """Watches the video from its link and returns transcript, scene descriptions,
    breakdown and gaps (ING-2, ING-3, ANG-1, ANG-2). Gemini takes a YouTube URL directly,
    so the app never downloads the video."""

    @abstractmethod
    async def watch(self, url: str, info: VideoInfo, manual_transcript: Optional[str] = None) -> WatchReport: ...


class Writer(ABC):
    """Angles, scripts, prompts and metadata. Must return JSON matching the schema (PRD 7.3)."""

    @abstractmethod
    async def complete_json(self, system: str, prompt: str, schema: dict) -> tuple[dict, float]: ...


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
    max_clip_seconds: float = 8

    @abstractmethod
    async def animate(self, image: Path, motion_prompt: str, seconds: float, aspect_ratio: str, out: Path) -> MediaResult: ...
