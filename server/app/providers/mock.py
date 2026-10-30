"""Mock adapters. They return realistic shapes and timings with no network,
so the whole pipeline runs before any real API is connected."""
from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Optional

from . import sample
from .base import (AudioResult, ImageGenerator, ImageToVideo, MediaResult, TextToSpeech,
                   VideoAnalyst, VideoInfo, VideoSource, WatchReport, Writer)

DELAY = 0.05  # seconds; keep tests fast. Raise it to feel real latency.


def _write_placeholder(out: Path, kind: str, meta: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mock": kind, **meta}))


class MockVideoSource(VideoSource):
    async def fetch(self, url: str) -> VideoInfo:
        await asyncio.sleep(DELAY)
        return VideoInfo(**sample.VIDEO)


class MockAnalyst(VideoAnalyst):
    def __init__(self, cost_per_video: float):
        self.cost = cost_per_video

    async def watch(self, url: str, info: VideoInfo, manual_transcript: Optional[str] = None) -> WatchReport:
        await asyncio.sleep(DELAY)
        return WatchReport(transcript=manual_transcript or sample.TRANSCRIPT, scenes=sample.SCENES,
                           analysis=sample.ANALYSIS, cost=self.cost)


class MockWriter(Writer):
    def __init__(self, cost_per_call: float):
        self.cost = cost_per_call

    async def complete_json(self, system: str, prompt: str, schema: dict) -> tuple[dict, float]:
        await asyncio.sleep(DELAY)
        name = schema.get("title")
        if name == "AngleList":
            return {"angles": sample.ANGLES}, self.cost
        if name == "Script":
            return sample.script_for(prompt), self.cost
        raise ValueError(f"MockWriter has no canned answer for schema {name!r}")


class MockTTS(TextToSpeech):
    def __init__(self, cost_per_1k: float):
        self.cost_per_1k = cost_per_1k

    def voices(self) -> list[dict]:
        return sample.VOICES

    async def synthesize(self, text: str, voice_id: str, out: Path) -> AudioResult:
        await asyncio.sleep(DELAY)
        speed = next((v["speed"] for v in sample.VOICES if v["id"] == voice_id), 1.0)
        duration = round(len(text.split()) / (2.6 * speed) + 0.5, 1)
        _write_placeholder(out, "audio", {"text": text, "voice": voice_id, "duration": duration})
        return AudioResult(path=out, duration=duration, cost=len(text) / 1000 * self.cost_per_1k)


class MockImages(ImageGenerator):
    def __init__(self, cost: float):
        self.cost = cost

    async def generate(self, prompt: str, aspect_ratio: str, out: Path,
                       seed: Optional[int] = None, reference_images: Optional[list[Path]] = None) -> MediaResult:
        await asyncio.sleep(DELAY)
        seed = seed if seed is not None else random.randint(0, 10**6)
        _write_placeholder(out, "image", {"prompt": prompt, "aspect_ratio": aspect_ratio, "seed": seed})
        return MediaResult(path=out, seed=seed, cost=self.cost)


class MockVideo(ImageToVideo):
    def __init__(self, cost_per_sec: float, max_clip: float):
        self.cost_per_sec = cost_per_sec
        self.max_clip_seconds = max_clip

    async def animate(self, image: Path, motion_prompt: str, seconds: float, aspect_ratio: str, out: Path) -> MediaResult:
        await asyncio.sleep(DELAY)
        gen = min(self.max_clip_seconds, max(5, -(-seconds // 5) * 5))
        _write_placeholder(out, "clip", {"image": str(image), "motion": motion_prompt, "generated": gen, "fitted": seconds})
        return MediaResult(path=out, seed=None, cost=gen * self.cost_per_sec)
