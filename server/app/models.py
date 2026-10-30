"""Project data model (PRD section 6).

The whole system revolves around one project JSON file that every stage reads
and writes. That is what makes each stage independently re-runnable.
"""
from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field

Status = Literal["pending", "generating", "ready", "failed"]
VideoType = Literal["voiceover", "character"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class Models(BaseModel):
    analyst: str = "gemini-pro"        # watches the reference video
    writer: str = "claude-opus-5-5"    # writes angles, scripts, prompts, metadata
    voice: str = "gemini-tts-flash"
    image: str = "nano-banana-2"
    video: str = "veo-31-fast"


class KeyFrame(BaseModel):
    t: str
    description: str
    image_path: Optional[str] = None


class Comment(BaseModel):
    id: str
    text: str
    likes: int = 0
    kind: Literal["question", "request", "confusion"] = "question"


class Source(BaseModel):
    url: str
    title: str = ""
    channel: str = ""
    duration: float = 0
    published: str = ""
    transcript: str = ""
    key_frames: list[KeyFrame] = []
    comments: list[Comment] = []


class Gap(BaseModel):
    kind: str
    text: str
    from_comment: Optional[str] = None


class Analysis(BaseModel):
    analyzed_by: str = ""
    summary: str
    hook: str
    beats: list[str]
    pacing: str = ""
    why_it_worked: list[str]
    gaps: list[Gap]


class Angle(BaseModel):
    id: str
    title: str
    pitch: str
    hook: str
    type: str
    suggested_style: str
    relatedness_score: int = Field(ge=0, le=100)
    relatedness_reason: str
    suggested_title: str
    suggested_tags: list[str]
    effort: Literal["Low", "Medium", "High"]
    potential: Literal["Low", "Medium", "High"]
    origin_comment: Optional[str] = None
    saved: bool = False


class ScriptLine(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ln"))
    speaker: Optional[str] = None  # character id for character videos
    text: str
    visual: str = ""
    audio_path: Optional[str] = None
    audio_status: Status = "pending"
    duration: Optional[float] = None


class Script(BaseModel):
    written_by: str = ""
    hook_options: list[str]
    chosen_hook: int = 0
    lines: list[ScriptLine]

    @property
    def full_text(self) -> str:
        return " ".join(line.text for line in self.lines)


class Style(BaseModel):
    template_id: str
    image_prefix: str
    mood: str = ""
    example_image: Optional[str] = None


class Character(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ch"))
    name: str
    description: str = ""
    notes: str = ""
    reference_images: list[str] = []
    voice_id: str = "v-aoede"


class Pacing(BaseModel):
    """How often the picture changes: simple ~8 s, moderate ~5 s, compact every 2 or 3 s per image."""
    mode: Literal["simple", "moderate", "compact"] = "moderate"
    compact_every: Literal[2, 3] = 3

    @property
    def every(self) -> float:
        return {"simple": 8, "moderate": 5}.get(self.mode, self.compact_every)


class Format(BaseModel):
    aspect_ratio: Literal["9:16", "16:9"] = "9:16"
    target_seconds: int = 60
    animated: bool = False
    pacing: Pacing = Field(default_factory=Pacing)


class Version(BaseModel):
    path: str
    model: str
    prompt: str = ""
    seed: Optional[int] = None
    created_at: float = Field(default_factory=time.time)


class Scene(BaseModel):
    id: str = Field(default_factory=lambda: new_id("sc"))
    order: int
    description: str
    line_ids: list[str]
    split_index: Optional[int] = None  # this scene shows part split_index of split_of of a long line
    split_of: Optional[int] = None
    character_ids: list[str] = []
    image_prompt: str = ""
    image_path: Optional[str] = None
    image_versions: list[Version] = []
    image_status: Status = "pending"
    duration: Optional[float] = None
    animate: bool = False
    motion_prompt: str = ""
    clip_path: Optional[str] = None
    clip_versions: list[Version] = []
    clip_status: Status = "pending"
    caption_style: str = "bold"
    error: Optional[str] = None


class Audio(BaseModel):
    voice_id: str = "v-aoede"
    music_path: Optional[str] = None
    music_volume: float = 0.25
    duck: bool = True
    clone_consent: bool = False


class Render(BaseModel):
    draft_path: Optional[str] = None
    final_path: Optional[str] = None
    cost_estimate: float = 0
    cost_actual: float = 0


class Checkpoints(BaseModel):
    angles: bool = False
    script: bool = False
    style: bool = False
    voice: bool = False
    format: bool = False
    scenes: bool = False
    clips: bool = False


class Project(BaseModel):
    id: str = Field(default_factory=lambda: new_id("prj"))
    title: str = "Untitled project"
    created_at: float = Field(default_factory=time.time)
    status: Literal["draft", "rendering", "rendered"] = "draft"
    template_id: Optional[str] = None
    video_type: VideoType = "voiceover"
    models: Models = Field(default_factory=Models)
    source: Optional[Source] = None
    analysis: Optional[Analysis] = None
    angles: list[Angle] = []
    angles_written_by: str = ""
    selected_angle_ids: list[str] = []
    user_direction: str = ""
    script: Optional[Script] = None
    style: Optional[Style] = None
    characters: list[Character] = []
    format: Format = Field(default_factory=Format)
    scenes: list[Scene] = []
    audio: Audio = Field(default_factory=Audio)
    render: Render = Field(default_factory=Render)
    checkpoints: Checkpoints = Field(default_factory=Checkpoints)

    def line(self, line_id: str) -> ScriptLine:
        assert self.script is not None
        return next(l for l in self.script.lines if l.id == line_id)

    def scene(self, scene_id: str) -> Scene:
        return next(s for s in self.scenes if s.id == scene_id)
