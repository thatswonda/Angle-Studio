"""Live routes used by the web app when it is served by this server.

The browser keeps the project (as in the demo) and calls these for the real work: YouTube stats and
comments, Gemini watching the video, the writer (ChatGPT by default) writing angles and the story,
Gemini TTS, Nano Banana images and the FFmpeg render. Keys stay in the server environment.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import jobs, render, storage
from .providers import google, youtube
from .providers.google import GeminiWriter, ProviderError
from .providers.openai_writer import OpenAIWriter

router = APIRouter(prefix="/api/live")
MEDIA = storage.ROOT / "media"
STYLE_IDS = ["botanical", "infographic", "comic", "watercolor", "cinematic"]
ANGLE_TYPES = ["Deep dive", "Continuation", "Different storyline", "Perspective flip", "Different niche", "Different topic", "Different visuals"]
WRITER_NAMES = {"gpt-5-5": "GPT-5.5", "gpt-5": "GPT-5", "gemini-pro-writer": "Gemini Pro", "gemini-flash-writer": "Gemini Flash",
                "claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5"}
ANALYST_NAMES = {"gemini-pro": "Gemini Pro", "gemini-flash": "Gemini Flash"}
SYSTEM = ("You plan and write original short-form and mid-length videos for a solo creator. Never copy the reference "
          "video's wording, titles or thumbnails. Titles and tags must honestly describe the new video. Reply with JSON only.")


def has(env: str) -> bool:
    return bool(os.getenv(env))


def writer_for(model_id: str):
    if model_id.startswith("gpt"):
        return OpenAIWriter(model_id)
    if model_id.startswith("gemini"):
        return GeminiWriter(model_id)
    raise ProviderError("Claude is not connected yet. Add ANTHROPIC_API_KEY and a Claude writer adapter, or pick GPT-5.5 or Gemini Pro.")


def media_url(p: Path) -> str:
    return "/media/" + p.resolve().relative_to(MEDIA.resolve()).as_posix()


def _job(label: str, work) -> dict:
    j = jobs.submit("live", label, work)
    return {"id": j.id}


@router.get("/status")
def status():
    return {
        "live": True,
        "providers": {"google": has("GEMINI_API_KEY"), "youtube": has("YOUTUBE_API_KEY"), "openai": has("OPENAI_API_KEY"),
                      "anthropic": False, "higgsfield": False, "elevenlabs": False},
        "video_models": False,  # this Gemini key has no Veo models; animation stays off
        "writer": os.getenv("ANGLE_STUDIO_WRITER", "gpt-5-5"),
    }


@router.get("/jobs/{job_id}")
def job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "No such job. The server may have restarted; start the step again.")
    return {"status": j.status, "stage": j.stage, "error": j.error and j.error.split(": ", 1)[-1], "result": j.result}


# ---------- 1. Watch the reference video and write angle cards

class IngestBody(BaseModel):
    url: str
    transcript: Optional[str] = None
    analyst: str = "gemini-pro"
    writer: str = "gpt-5-5"


@router.post("/ingest")
async def ingest(body: IngestBody):
    youtube.video_id(body.url)  # fail fast on a bad link

    async def work(progress):
        progress("YouTube Data API: title, stats and comments")
        warning = None
        try:
            info = await youtube.fetch(body.url)
        except ProviderError as e:
            if "no public video" in str(e) or "doesn't look like" in str(e):
                raise
            # Keep going without comments: Gemini can still watch the video from its link.
            info = await youtube.basic_info(body.url)
            warning = f"Comments were skipped. {e}"
            progress("Comments skipped: the YouTube Data API refused the key")
        comments = info["comments"]
        an = ANALYST_NAMES.get(body.analyst, body.analyst)
        progress(f"{an} is watching the video")
        prompt = (
            "Watch this YouTube video carefully. The audience's top comments are below with ids.\n"
            f"Video: {info['title']} by {info['channel']}, {info['duration']} seconds.\n"
            f"Comments: {json.dumps([{k: c[k] for k in ('id', 'text', 'likes')} for c in comments])}\n"
            + (f"Use this transcript instead of transcribing: {body.transcript[:20000]}\n" if body.transcript else "")
            + "Return JSON with exactly these keys:\n"
            '{"transcript": "the full spoken transcript as plain text, with [m:ss] markers every few sentences",'
            ' "scenes": [{"t": "m:ss", "description": "what is on screen: shot type, subject, text overlays, edits"}] (8 to 12 key moments),'
            ' "summary": "one or two sentences", "hook": "how the first seconds hook the viewer",'
            ' "beats": [{"t": "m:ss", "text": "story beat"}] (5 to 8),'
            ' "pacing": "cut rhythm and structure", "why_it_worked": ["3 to 5 short reasons"],'
            ' "gaps": [{"kind": "Unanswered question | Unfinished thread | Audience request | Skipped perspective | Unexplored visual",'
            ' "text": "the gap", "from_comment": "comment id or null"}] (4 to 8)}'
        )
        report, cost = await google.generate_json(body.analyst, [google.youtube_part(body.url), {"text": prompt}], low_media=True)
        if isinstance(report, list):  # some models wrap the object in a list
            report = report[0] if report and isinstance(report[0], dict) else {}
        wn = WRITER_NAMES.get(body.writer, body.writer)
        progress(f"Handing the report to {wn}")
        progress(f"{wn} is writing angle cards")
        handoff = {"video": {k: info[k] for k in ("title", "channel", "duration", "views")}, "report": report, "comments": comments}
        angles_raw, wcost = await writer_for(body.writer).complete_json(SYSTEM, (
            "A model watched a reference video and wrote this report. Propose 10 original angles for a new video that a viewer "
            "of the reference video would want to watch next, so the new video can be recommended beside it. Mix the angle types "
            f"{ANGLE_TYPES}. Score relatedness 0-100 from topic, audience, format and keyword overlap.\n"
            f"Report: {json.dumps(handoff)[:60000]}\n"
            'Return {"angles": [{"type": one of the types, "title": short working title, "pitch": one line,'
            ' "reason": one line on why a viewer of the reference would watch it next, "score": 0-100,'
            f' "style": one of {STYLE_IDS}, "effort": "Low|Medium|High", "potential": "Low|Medium|High",'
            ' "origin": comment id it answers or null, "character": true if it suits a video with voiced characters,'
            ' "hooks": [three opening lines], "suggested_title": honest title using the reference video\'s key terms,'
            ' "tags": [5 tags]}]}'))
        angles = []
        ids = {c["id"] for c in comments}
        raw_list = angles_raw if isinstance(angles_raw, list) else (angles_raw.get("angles") or [])
        for i, a in enumerate([a for a in raw_list if isinstance(a, dict)][:12], 1):
            hooks = [h for h in (a.get("hooks") or []) if isinstance(h, str)][:3] or [a.get("pitch", "")]
            while len(hooks) < 3:
                hooks.append(hooks[-1])
            angles.append({
                "id": f"a{i}", "type": a.get("type") if a.get("type") in ANGLE_TYPES else "Deep dive",
                "style": a.get("style") if a.get("style") in STYLE_IDS else "botanical",
                "score": max(0, min(100, int(a.get("score") or 70))), "effort": a.get("effort") if a.get("effort") in ("Low", "Medium", "High") else "Medium",
                "potential": a.get("potential") if a.get("potential") in ("Low", "Medium", "High") else "Medium",
                "origin": a.get("origin") if a.get("origin") in ids else None, "character": bool(a.get("character")),
                "title": a.get("title", "Untitled angle"), "pitch": a.get("pitch", ""), "reason": a.get("reason", ""),
                "suggested_title": a.get("suggested_title", a.get("title", "")), "tags": [str(t) for t in (a.get("tags") or [])][:8],
                "hooks": hooks, "hook_visual": "", "beats": [],
            })
        if len(angles) < 3:
            raise ProviderError(f"{wn} returned too few angles. Try again.")
        return {
            "source": {"url": body.url, "title": info["title"], "channel": info["channel"], "duration": info["duration"],
                       "published": info["published"], "views": info["views"], "thumb": info["thumb"],
                       "transcript": report.get("transcript", "") if isinstance(report.get("transcript"), str) else json.dumps(report.get("transcript")),
                       "key_frames": [{"t": s.get("t", ""), "d": s.get("description", ""), "seed": 11 + i * 13} for i, s in enumerate(report.get("scenes") or [])],
                       "comments": [{k: c[k] for k in ("id", "text", "kind")} | {"likes": f"{c['likes'] / 1000:.1f}k" if c["likes"] >= 1000 else str(c["likes"])} for c in comments[:10]],
                       "comments_scanned": info["comments_scanned"]},
            "analysis": {"summary": report.get("summary", ""), "hook": report.get("hook", ""),
                         "beats": [[b.get("t", ""), b.get("text", "")] for b in (report.get("beats") or []) if isinstance(b, dict)],
                         "pacing": report.get("pacing", ""), "why_it_worked": [str(x) for x in (report.get("why_it_worked") or [])],
                         "gaps": [{"kind": g.get("kind", "Gap"), "text": g.get("text", ""), "from": g.get("from_comment") if g.get("from_comment") in ids else None}
                                  for g in (report.get("gaps") or []) if isinstance(g, dict)],
                         "analyzed_by": an},
            "angles": angles, "written_by": wn, "cost": round(cost + wcost, 4), "warning": warning,
        }

    return _job("Analyzing reference video", work)


# ---------- 2. Write the story

class StoryBody(BaseModel):
    writer: str = "gpt-5-5"
    angles: list[dict]
    analysis: dict = {}
    source_title: str = ""
    target_seconds: int = 60
    video_type: str = "voiceover"
    characters: list[dict] = []
    direction: str = ""
    tone: str = ""


@router.post("/story")
async def story(body: StoryBody):
    if not 15 <= body.target_seconds <= 600:
        raise HTTPException(422, "Length must be between 15 seconds and 10 minutes.")

    async def work(progress):
        wn = WRITER_NAMES.get(body.writer, body.writer)
        words = round(body.target_seconds * 2.5)
        char = body.video_type == "character"
        progress(f"{wn} is writing a {words}-word story")
        fmt = (f"Write it as dialogue, one line per speaker in the form 'Name: what they say', using only these characters: "
               f"{', '.join(c.get('name', '') + ' (' + c.get('description', '') + ')' for c in body.characters)}."
               if char else "Write it as narration in short paragraphs separated by a blank line.")
        data, cost = await writer_for(body.writer).complete_json(SYSTEM, (
            f"Write one continuous, original story for a {body.target_seconds}-second video (about {words} spoken words; stay within 10%).\n"
            f"Angle(s): {json.dumps([{k: a.get(k) for k in ('title', 'pitch', 'reason', 'hooks', 'type')} for a in body.angles])}\n"
            f"Reference video it should sit next to: {body.source_title}. What that video did: {json.dumps(body.analysis)[:6000]}\n"
            f"Tone: {body.tone or 'friendly, practical, no filler'}. Direction from the creator: {body.direction or 'none'}.\n"
            f"{fmt} Open with a strong hook, keep a new idea every few seconds, and end with a natural call to action.\n"
            'Return {"hook_options": [three different opening lines; the story starts with the first],'
            ' "story": "the full story",'
            ' "visuals": [{"sentence": "each sentence of the story, word for word, in order", "visual": "a short description of an original image that fits it"}]}'))
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else {}
        story_text = str(data.get("story", "")).strip()
        if len(story_text.split()) < words * 0.4:
            raise ProviderError(f"{wn} returned a story that is far too short. Try again.")
        hooks = [str(h) for h in (data.get("hook_options") or [])][:3]
        while len(hooks) < 3:
            hooks.append(story_text.split(".")[0] + ".")
        visuals = [v for v in (data.get("visuals") or []) if isinstance(v, dict) and v.get("sentence")]
        return {"hook_options": hooks, "story": story_text, "visuals": visuals, "written_by": wn, "cost": round(cost, 4)}

    return _job("Writing the story", work)


# ---------- 3. Voice, images

class SpeakBody(BaseModel):
    text: str
    voice: str = "v-aoede"
    model: str = "gemini-tts-flash"
    delivery: str = "warm, steady and conversational, like explaining something to a friend, at a medium pace"


@router.post("/tts")
async def tts(body: SpeakBody):
    if not body.text.strip():
        raise HTTPException(422, "Nothing to read.")
    out = MEDIA / "audio" / f"{uuid.uuid4().hex}.wav"
    try:
        seconds, cost = await google.speak(body.text, body.voice, body.model, out, body.delivery)
    except ProviderError as e:
        raise HTTPException(502, str(e))
    return {"url": media_url(out), "duration": round(seconds, 2), "cost": round(cost, 4)}


class ImageBody(BaseModel):
    prompt: str
    aspect: str = "9:16"
    model: str = "nano-banana-2"


@router.post("/image")
async def image(body: ImageBody):
    if body.aspect not in ("9:16", "16:9", "1:1", "4:5"):
        raise HTTPException(422, "Unsupported frame.")
    try:
        path, cost = await google.draw(body.prompt[:4000], body.aspect, body.model, MEDIA / "images" / uuid.uuid4().hex)
    except ProviderError as e:
        raise HTTPException(502, str(e))
    return {"url": media_url(path), "cost": cost}


# ---------- 4. Render

class RenderBody(BaseModel):
    final: bool = False
    aspect: str = "9:16"
    caption_style: str = "bold"
    scenes: list[dict[str, Any]]
    lines: list[dict[str, Any]]


@router.post("/render")
async def render_video(body: RenderBody):
    async def work(progress):
        url = await asyncio.to_thread(render.render, body.model_dump(), MEDIA, progress)
        return {"url": url}

    return _job("Rendering final MP4" if body.final else "Building draft", work)
