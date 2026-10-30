"""Pipeline stages. Each stage reads the project, does its work through the
provider adapters, and writes the project back, so any stage or single scene
can be re-run on its own."""
from __future__ import annotations

import asyncio
import json
import os
import re
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Callable

from . import assembly, storage
from .models import (Analysis, Angle, Character, Comment, KeyFrame, Project, Scene,
                     Script, ScriptLine, Source, Version)
from .providers import registry, sample

_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


@asynccontextmanager
async def editing(pid: str):
    """Load, yield, save under a per-project lock so parallel scene jobs don't clobber each other."""
    async with _locks[pid]:
        p = storage.load(pid)
        yield p
        storage.save(p)


class CheckpointError(Exception):
    """Raised when a downstream stage is asked to run before its upstream checkpoint is approved (PRJ-2)."""


class CostLimitError(Exception):
    def __init__(self, estimate: float, spent: float, limit: float):
        super().__init__(f"This step costs about ${estimate:.2f}; with ${spent:.2f} spent the project passes its ${limit:.2f} limit.")
        self.estimate, self.spent, self.limit = estimate, spent, limit


def cost_limit() -> float:
    return float(os.getenv("ANGLE_STUDIO_COST_LIMIT", "1.50"))


def require(p: Project, *checkpoints: str) -> None:
    missing = [c for c in checkpoints if not getattr(p.checkpoints, c)]
    if missing:
        raise CheckpointError(f"Approve {', '.join(missing)} first.")


def guard_cost(p: Project, estimate: float, confirmed: bool) -> None:
    if p.render.cost_actual + estimate > cost_limit() and not confirmed:
        raise CostLimitError(estimate, p.render.cost_actual, cost_limit())


# ---------- estimates (PRJ-3, ANM-6)

def line_voice(p: Project, line: ScriptLine) -> str:
    if p.video_type == "character" and line.speaker:
        return next((c.voice_id for c in p.characters if c.id == line.speaker), p.audio.voice_id)
    return p.audio.voice_id


def scene_seconds(p: Project, s: Scene) -> float:
    return round(sum(p.line(i).duration or len(p.line(i).text.split()) / 2.6 + 0.5 for i in s.line_ids), 1)


def estimate(p: Project) -> dict:
    lines = p.script.lines if p.script else []
    chars = sum(len(l.text) for l in lines if l.audio_status != "ready")
    n_img = sum(1 for s in p.scenes if s.image_status != "ready") if p.scenes else len(lines)
    anim = [s for s in p.scenes if s.animate and s.clip_status != "ready"]
    vm = registry.model("video", p.models.video)
    anim_secs = sum(min(vm["limits"]["max_clip"], max(5, -(-scene_seconds(p, s) // 5) * 5)) for s in anim)
    out = {
        "voice": chars / 1000 * registry.model("voice", p.models.voice)["cost"],
        "images": n_img * registry.model("image", p.models.image)["cost"],
        "video": anim_secs * vm["cost"],
        "render": 0.0,
    }
    out["total"] = sum(out.values())
    out["spent"] = p.render.cost_actual
    out["limit"] = cost_limit()
    return out


# ---------- SCR-6

def overlap(script_text: str, transcript: str) -> float:
    def tri(s: str) -> set[str]:
        w = re.sub(r"[^a-z0-9\s']", " ", s.lower()).split()
        return {" ".join(w[i:i + 3]) for i in range(len(w) - 2)}
    a, b = tri(script_text), tri(transcript)
    return len(a & b) / len(a) if a else 0.0


# ---------- stages

async def ingest(pid: str, url: str, manual_transcript: str | None, progress: Callable[[str], None]) -> None:
    cached = storage.kv_get("ingest_cache", url) if not manual_transcript else None  # ING-6
    if cached:
        progress("Loaded from cache")
        async with editing(pid) as p:
            p.source, p.analysis = Source(**cached["source"]), Analysis(**cached["analysis"])
            p.angles = [Angle(**a) for a in cached["angles"]]
        return
    progress("Fetching metadata and comments")
    info = await registry.source().fetch(url)
    progress("Reading captions")
    transcript = manual_transcript or info.captions or await registry.stt().transcribe(storage.project_dir(pid) / "source_audio")
    comments = [Comment(**c) for c in info.comments if c["kind"] in ("question", "request", "confusion")]  # ING-4
    source = Source(url=url, title=info.title, channel=info.channel, duration=info.duration, published=info.published,
                    transcript=transcript, key_frames=[KeyFrame(**k) for k in info.key_frames], comments=comments)
    llm = registry.llm()
    progress("Breaking down the video")
    payload = json.dumps({"metadata": {"title": info.title, "channel": info.channel}, "transcript": transcript,
                          "frames": info.key_frames, "comments": [c.model_dump() for c in comments]})
    analysis = Analysis(**await llm.complete_json(PROMPTS["analysis"], payload, Analysis.model_json_schema() | {"title": "Analysis"}))
    progress("Writing angle cards")
    raw = await llm.complete_json(PROMPTS["angles"], payload + "\n" + analysis.model_dump_json(),
                                  {"title": "AngleList", "type": "object", "properties": {"angles": {"type": "array", "items": Angle.model_json_schema()}}})
    angles = [Angle(**a) for a in raw["angles"]]  # pydantic validates; a real LLM call retries on ValidationError
    async with editing(pid) as p:
        p.source, p.analysis, p.angles = source, analysis, angles
    storage.kv_put("ingest_cache", url, {"source": source.model_dump(), "analysis": analysis.model_dump(), "angles": [a.model_dump() for a in angles]})


async def write_script(pid: str, progress: Callable[[str], None]) -> None:
    p = storage.load(pid)
    require(p, "angles")
    progress("Drafting hooks and lines")
    prompt = f"angles={','.join(p.selected_angle_ids)} seconds={p.format.target_seconds}\n{PROMPTS['script']}\nDirection: {p.user_direction}"
    raw = await registry.llm().complete_json(PROMPTS["system"], prompt, {"title": "Script"})
    async with editing(pid) as p:
        if p.video_type == "character" and not p.characters:
            p.characters = [Character(**c) for c in sample.DEFAULT_CHARACTERS]
        speakers = [c.id for c in p.characters] if p.video_type == "character" else [None]
        p.script = Script(hook_options=raw["hook_options"],
                          lines=[ScriptLine(text=l["text"], visual=l.get("visual", ""), speaker=speakers[i % len(speakers)]) for i, l in enumerate(raw["lines"])])
        p.scenes = []


async def generate_audio(pid: str, line_ids: list[str] | None, progress: Callable[[str], None]) -> None:
    p = storage.load(pid)
    require(p, "script", "style")
    tts = registry.tts(p.models.voice)
    ids = line_ids or [l.id for l in p.script.lines]
    for n, lid in enumerate(ids, 1):
        progress(f"Line {n} of {len(ids)}")
        async with editing(pid) as p:
            line = p.line(lid)
            line.audio_status = "generating"
            voice = line_voice(p, line)
            if voice == "v-own" and not p.audio.clone_consent:  # AUD-4
                line.audio_status = "failed"
                raise PermissionError("Confirm consent before using your cloned voice.")
        res = await tts.synthesize(line.text, voice, storage.project_dir(pid) / "audio" / f"{lid}.mp3")
        async with editing(pid) as p:
            line = p.line(lid)
            line.audio_path, line.duration, line.audio_status = str(res.path), res.duration, "ready"
            p.render.cost_actual += res.cost
            for s in p.scenes:  # AUD-3: audio sets each scene's length
                if lid in s.line_ids:
                    s.duration = scene_seconds(p, s)


async def build_scenes(pid: str, progress: Callable[[str], None]) -> None:
    """VIS-1: split the approved script into scenes, then generate every image."""
    async with editing(pid) as p:
        require(p, "format")
        p.scenes = [Scene(order=i, description=l.visual, line_ids=[l.id], image_prompt=l.visual,
                          character_ids=[l.speaker] if l.speaker else [], animate=p.format.animated,
                          motion_prompt="slow push in, gentle parallax") for i, l in enumerate(p.script.lines)]
        for s in p.scenes:
            s.duration = scene_seconds(p, s)
        ids = [s.id for s in p.scenes]
    progress(f"Generating {len(ids)} images")
    await asyncio.gather(*(generate_image(pid, sid, lambda _: None) for sid in ids))


async def generate_image(pid: str, scene_id: str, progress: Callable[[str], None]) -> None:
    async with editing(pid) as p:
        s = p.scene(scene_id)
        s.image_status, s.error = "generating", None
        prompt = f"{p.style.image_prefix}, {s.image_prompt}" if p.style else s.image_prompt
        refs = [r for c in p.characters if c.id in s.character_ids for r in c.reference_images]  # CHR-5
        gen = registry.images(p.models.image)
        model_id, aspect, n = p.models.image, p.format.aspect_ratio, len(s.image_versions) + 1
    try:
        res = await gen.generate(prompt, aspect, storage.project_dir(pid) / "images" / f"{scene_id}_v{n}.png", reference_images=refs)
    except Exception as e:
        async with editing(pid) as p:
            p.scene(scene_id).image_status, p.scene(scene_id).error = "failed", str(e)
        raise
    async with editing(pid) as p:
        s = p.scene(scene_id)
        s.image_versions.append(Version(path=str(res.path), model=model_id, prompt=prompt, seed=res.seed))  # VIS-3 keeps versions
        s.image_path, s.image_status = str(res.path), "ready"
        p.render.cost_actual += res.cost


async def animate(pid: str, scene_ids: list[str], progress: Callable[[str], None]) -> None:
    p = storage.load(pid)
    require(p, "scenes")
    vid = registry.video(p.models.video)
    for n, sid in enumerate(scene_ids, 1):
        progress(f"Clip {n} of {len(scene_ids)}")
        async with editing(pid) as p:
            s = p.scene(sid)
            s.clip_status = "generating"
            secs, image, motion, model_id = scene_seconds(p, s), s.image_path, s.motion_prompt, p.models.video
        res = await vid.animate(image, motion, secs, p.format.aspect_ratio, storage.project_dir(pid) / "clips" / f"{sid}_v{len(s.clip_versions) + 1}.mp4")
        async with editing(pid) as p:
            s = p.scene(sid)
            s.clip_versions.append(Version(path=str(res.path), model=model_id, prompt=motion))  # ANM-3 keeps takes
            s.clip_path, s.clip_status = str(res.path), "ready"
            p.render.cost_actual += res.cost


async def compile_video(pid: str, final: bool, progress: Callable[[str], None]) -> None:
    p = storage.load(pid)
    require(p, "scenes")
    if any(s.animate for s in p.scenes):
        require(p, "clips")
    out = storage.project_dir(pid) / ("final_1080p.mp4" if final else "draft_480p.mp4")
    progress("Assembling with FFmpeg")
    await assembly.render(p, out, height=1080 if final else 480, polish=final)
    async with editing(pid) as p:
        if final:
            p.render.final_path, p.status = str(out), "rendered"
        else:
            p.render.draft_path = str(out)


PROMPTS = {
    "system": "You write original short-form video scripts. Never reuse the reference video's wording. Return JSON only.",
    "analysis": "Break down the reference video: summary, hook, story beats with timecodes, pacing, why it likely worked, and gaps "
                "(unanswered questions, skipped perspectives, unexplored visuals, unfinished threads, audience requests from comments).",
    "angles": "Propose 8 to 12 original angles a viewer of the reference video would click next. For each give a relatedness score "
              "from topic, audience, format and keyword overlap, and a one-line reason. Titles and tags must honestly describe the new video.",
    "script": "Write a script for the chosen angle(s) at the target length. Give 3 hook options. One line per future scene, each with a visual.",
}
