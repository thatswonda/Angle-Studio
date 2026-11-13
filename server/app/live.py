"""Live routes used by the web app when it is served by this server.

The browser keeps the project (as in the demo) and calls these for the real work: YouTube stats and
comments, Gemini watching the video, the writer (ChatGPT by default) writing angles and the story,
Gemini TTS, Nano Banana images and the FFmpeg render. Keys stay in the server environment.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from . import thumbs as thumbs_mod
from . import jobs, render, storage
from .providers import fal, google, youtube
from .providers.google import GeminiWriter, ProviderError
from .providers.anthropic_writer import ClaudeWriter
from .providers.openai_writer import OpenAIWriter

router = APIRouter(prefix="/api/live")
MEDIA = storage.ROOT / "media"
STYLE_IDS = ["photoreal", "cinematic", "documentary", "botanical", "clay", "animated3d", "anime", "comic", "watercolor", "infographic", "retro", "papercut", "lineart", "stickman", "toyworld", "wiresketch", "pixel", "silhouette", "cartoonexplain", "historypaint"]
# Ways to recreate what made the reference video go viral, so the new video is recommended beside it.
# Never a sequel or "part 2": each one is a stand-alone video built on the same proven idea.
ANGLE_TYPES = ["Fill a gap", "Same video, new niche", "Same video, new character", "Same format, new topic",
               "Same story, new look", "Similar storyline", "New perspective"]
ANGLE_GUIDE = {
    "Fill a gap": "makes the video viewers asked for or that the original left out (from the comments and gaps), in the same format and hook style",
    "Same video, new niche": "the same video idea that went viral, remade for a different niche or audience",
    "Same video, new character": "the same video with a different main character (another person, animal, historical figure, object or job)",
    "Same format, new topic": "the same structure, hook, pacing and payoff on a different but related topic people also search for",
    "Same story, new look": "the same story or concept remade in a clearly different visual style (animated, clay, stick figure, cinematic...)",
    "Similar storyline": "a fresh storyline with the same structure, emotional beats and payoff that made the original work",
    "New perspective": "the same events or idea from another character's or group's point of view",
}
# How the script talks to the viewer, chosen after the angle. The writer follows the one the creator picked.
SCRIPT_TYPES = {
    "story": "STORY: tell a story with a main character, a goal or problem, rising tension and a satisfying payoff.",
    "narration": "NARRATION in second person: speak straight to the viewer as an immersive what-if or experience "
                 "('What if you woke up one morning and everyone was gone? You walk to the store. Empty.'). Vivid moments the "
                 "viewer imagines living, building curiosity; no named main character.",
    "explainer": "EXPLAINER: a question up front, then clear simple steps, facts and examples, each point building on the last, "
                 "ending on a clear takeaway.",
    "documentary": "DOCUMENTARY: calm, factual narration with real facts, places, dates and numbers, a sense of discovery.",
    "list": "TOP LIST: a countdown of numbered items ('Number five...'), each a short punchy reveal, the strongest item last.",
    "pov": "POV in first person, present tense, as the person living it ('I wake up. The street is silent.').",
    "motivational": "MOTIVATIONAL SPEECH: direct, rhythmic, inspiring lines to the viewer, ending with a call to act.",
}


def script_rule(kind: str) -> str:
    return "Script type: " + SCRIPT_TYPES.get(kind, SCRIPT_TYPES["story"])


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
    if model_id.startswith("claude"):
        return ClaudeWriter(model_id)
    raise ProviderError(f"Unknown writing model {model_id}.")


FALLBACK_WRITER = "gemini-flash-writer"


async def write_json(writer_id: str, progress, prompt: str) -> tuple[dict, float, str, Optional[str]]:
    """Writes with the chosen model; if it fails (no key, no credit, unreachable, free tier), Gemini Flash writes instead."""
    name = WRITER_NAMES.get(writer_id, writer_id)
    try:
        data, cost = await writer_for(writer_id).complete_json(SYSTEM, prompt)
        return data, cost, name, None
    except ProviderError as e:
        if writer_id == FALLBACK_WRITER:
            raise
        progress(f"{name} could not write; Gemini Flash is writing instead")
        data, cost = await GeminiWriter(FALLBACK_WRITER).complete_json(SYSTEM, prompt)
        return data, cost, "Gemini Flash", f"{name} could not write ({e}) Gemini Flash wrote this instead."


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
                      "fal": has("FAL_KEY"), "anthropic": has("ANTHROPIC_API_KEY"), "higgsfield": False, "elevenlabs": False},
        "video_models": False,  # this Gemini key has no Veo models; animation stays off
        # Claude is the brain when its key is set; otherwise the configured writer
        "writer": os.getenv("ANGLE_STUDIO_WRITER") or ("claude-opus-5-5" if has("ANTHROPIC_API_KEY") else "gpt-5-5"),
    }


@router.get("/jobs/{job_id}")
def job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "No such job. The server may have restarted; start the step again.")
    return {"status": j.status, "stage": j.stage, "error": j.error and j.error.split(": ", 1)[-1], "result": j.result}


# ---------- 1. Watch the reference video and write angle cards
ANGLES_VERSION = 3  # bump when the angle prompt changes, so cached analyses get new angles without re-watching


async def write_angles(writer: str, progress, handoff: dict, ids: set) -> tuple[list, float, str, Optional[str]]:
    """The 4 angles most likely to go viral, from the analysis report."""
    angles_raw, wcost, wn, wwarn = await write_json(writer, progress, (
        "A model watched a video that went viral and wrote this report. The creator wants to recreate what made it go "
        "viral in a new, original video, so that it has the same chance of going viral and gets recommended next to the "
        "original. This is NOT a continuation, sequel or part 2: the new video stands on its own and does not pick up "
        "where the original stopped. Use these ways of recreating it:\n"
        + "\n".join(f"- {k}: {v}" for k, v in ANGLE_GUIDE.items()) + "\n"
        "Propose exactly 4 angles: the 4 ways of remaking this same video with the highest chance of going viral, best "
        "first, using at least 3 different types. Exploit what made the idea work and the gaps viewers left in the comments. "
        "Keep the original's proven ingredients (hook, structure, emotion, topic keywords) so the platform links the two "
        "videos, and change one clear thing. Never copy its wording, footage or characters.\n"
        "Describe each angle as a video concept (what the video is, what changes, why it can go viral), NOT as a story: "
        "the creator picks the script type afterwards (a story, a 'what if you...' narration, an explainer, a list, a POV...). "
        "Write the hooks so they work as opening lines for any script type.\n"
        f"Report: {json.dumps(handoff)[:60000]}\n"
        'Return {"angles": [{"type": one of the types, "title": short working title, "pitch": one line,'
        ' "reason": one line on why it can go viral like the original and be recommended next to it,'
        ' "keeps": what it keeps from the original (a few words), "changes": what is new (a few words),'
        ' "score": viral potential 0-100,'
        f' "style": one of {STYLE_IDS}, "effort": "Low|Medium|High", "potential": "Low|Medium|High",'
        ' "origin": comment id it answers or null, "character": true if it suits a video with voiced characters,'
        ' "hooks": [three opening lines], "suggested_title": honest title using the reference video\'s key terms,'
        ' "tags": [5 tags]}]}'))
    angles = []
    raw_list = angles_raw if isinstance(angles_raw, list) else (angles_raw.get("angles") or [])
    raw_list = sorted([a for a in raw_list if isinstance(a, dict)], key=lambda a: -(int(a.get("score") or 0) if str(a.get("score") or "0").isdigit() else 0))
    for i, a in enumerate(raw_list[:4], 1):
        hooks = [h for h in (a.get("hooks") or []) if isinstance(h, str)][:3] or [a.get("pitch", "")]
        while len(hooks) < 3:
            hooks.append(hooks[-1])
        angles.append({
            "id": f"a{i}", "type": a.get("type") if a.get("type") in ANGLE_TYPES else "Similar storyline",
            "keeps": str(a.get("keeps", ""))[:120], "changes": str(a.get("changes", ""))[:120],
            "style": a.get("style") if a.get("style") in STYLE_IDS else "botanical",
            "score": max(0, min(100, int(a.get("score") or 70))), "effort": a.get("effort") if a.get("effort") in ("Low", "Medium", "High") else "Medium",
            "potential": a.get("potential") if a.get("potential") in ("Low", "Medium", "High") else "Medium",
            "origin": a.get("origin") if a.get("origin") in ids else None, "character": bool(a.get("character")),
            "title": a.get("title", "Untitled angle"), "pitch": a.get("pitch", ""), "reason": a.get("reason", ""),
            "suggested_title": a.get("suggested_title", a.get("title", "")), "tags": [str(t) for t in (a.get("tags") or [])][:8],
            "hooks": hooks, "hook_visual": "", "beats": [],
        })
    return angles, wcost, wn, wwarn


class IngestBody(BaseModel):
    url: str
    transcript: Optional[str] = None
    analyst: str = "gemini-pro"
    writer: str = "gpt-5-5"


@router.post("/ingest")
async def ingest(body: IngestBody):
    youtube.video_id(body.url)  # fail fast on a bad link

    cache_key = f"{body.url}|{body.analyst}|{body.writer}"

    async def work(progress):
        cached = None if body.transcript else storage.kv_get("live_ingest", cache_key)
        if cached and cached.get("angles_v") == ANGLES_VERSION:  # analyzed before with the same models: no new cost
            progress("Loaded from cache")
            return {**cached, "cost": 0, "cached": True, "warning": None}
        if cached:  # analyzed before, but with the older angle style: keep the analysis, write new angles only
            wn = WRITER_NAMES.get(body.writer, body.writer)
            progress(f"Reusing the saved analysis; {wn} is picking the 4 angles most likely to go viral")
            src, an_ = cached["source"], cached["analysis"]
            handoff = cached.get("handoff") or {"video": {k: src.get(k) for k in ("title", "channel", "duration", "views")},
                                                "report": {"transcript": src.get("transcript", "")[:20000], "scenes": src.get("key_frames", []), **an_},
                                                "comments": src.get("comments", [])}
            angles, wcost, wn, wwarn = await write_angles(body.writer, progress, handoff, {c.get("id") for c in src.get("comments", [])})
            if len(angles) < 2:
                raise ProviderError(f"{wn} returned too few angles. Try again.")
            fresh = {**cached, "angles": angles, "written_by": wn, "angles_v": ANGLES_VERSION, "handoff": handoff}
            storage.kv_put("live_ingest", cache_key, fresh)
            return {**fresh, "cost": round(wcost, 4), "cached": False, "warning": wwarn}
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
            ' "text": "the gap", "from_comment": "comment id or null"}] (4 to 8),'
            ' "captions_look": ' + CAPTIONS_LOOK_SCHEMA + '}'
        )
        parts = [google.youtube_part(body.url), {"text": prompt}]
        try:
            report, cost = await google.generate_json(body.analyst, parts, low_media=True)
        except ProviderError as e:
            if body.analyst == "gemini-flash":
                raise
            progress(f"{an} is not available on this key; Gemini Flash is watching instead")
            report, cost = await google.generate_json("gemini-flash", parts, low_media=True)
            warning = " ".join(x for x in (warning, f"{an} could not watch the video ({e}) Gemini Flash watched it instead.") if x)
            an = "Gemini Flash"
        if isinstance(report, list):  # some models wrap the object in a list
            report = report[0] if report and isinstance(report[0], dict) else {}
        wn = WRITER_NAMES.get(body.writer, body.writer)
        progress(f"Handing the report to {wn}")
        progress(f"{wn} is picking the 4 angles most likely to go viral")
        handoff = {"video": {k: info.get(k) for k in ("title", "channel", "duration", "views", "tags")}, "report": report, "comments": comments}
        ids = {c["id"] for c in comments}
        angles, wcost, wn, wwarn = await write_angles(body.writer, progress, handoff, ids)
        warning = " ".join(x for x in (warning, wwarn) if x) or None
        if len(angles) < 2:
            raise ProviderError(f"{wn} returned too few angles. Try again.")
        result = {
            "source": {"url": body.url, "title": info["title"], "channel": info["channel"], "duration": info["duration"],
                       "published": info["published"], "views": info["views"], "thumb": info["thumb"],
                       "tags": info.get("tags", []), "description": info.get("description", ""),
                       "transcript": report.get("transcript", "") if isinstance(report.get("transcript"), str) else json.dumps(report.get("transcript")),
                       "key_frames": [{"t": s.get("t", ""), "d": s.get("description", ""), "seed": 11 + i * 13} for i, s in enumerate(report.get("scenes") or [])],
                       "comments": [{k: c[k] for k in ("id", "text", "kind")} | {"likes": f"{c['likes'] / 1000:.1f}k" if c["likes"] >= 1000 else str(c["likes"])} for c in comments[:10]],
                       "comments_scanned": info["comments_scanned"]},
            "analysis": {"summary": report.get("summary", ""), "hook": report.get("hook", ""),
                         "beats": [[b.get("t", ""), b.get("text", "")] for b in (report.get("beats") or []) if isinstance(b, dict)],
                         "pacing": report.get("pacing", ""), "why_it_worked": [str(x) for x in (report.get("why_it_worked") or [])],
                         "gaps": [{"kind": g.get("kind", "Gap"), "text": g.get("text", ""), "from": g.get("from_comment") if g.get("from_comment") in ids else None}
                                  for g in (report.get("gaps") or []) if isinstance(g, dict)],
                         "analyzed_by": an, "captions_look": clean_captions_look(report.get("captions_look"))},
            "angles": angles, "written_by": wn, "cost": round(cost + wcost, 4), "warning": warning,
            "angles_v": ANGLES_VERSION, "handoff": handoff,
        }
        if not body.transcript:
            storage.kv_put("live_ingest", cache_key, result)
        return result

    return _job("Analyzing reference video", work)


# ---------- The original video's look: its on-screen captions and its thumbnail

CAPTIONS_LOOK_SCHEMA = (
    '{"present": true or false (are there spoken-word captions burned into the video),'
    ' "words": how many words are on screen at once (1-8), "case": "upper | sentence | lower",'
    ' "text_color": "#RRGGBB", "highlight_color": "#RRGGBB colour of the word being spoken, or empty if none",'
    ' "box_color": "#RRGGBB of a solid box behind the words, or empty", "outline": true or false (dark outline round the letters),'
    ' "position": "top | middle | bottom", "size": "small | medium | large", "weight": "bold | regular",'
    ' "describe": "one short sentence describing the caption look, e.g. big white capitals, yellow active word, centre"}')
HEX = re.compile(r"#[0-9a-fA-F]{6}")


def clean_captions_look(c) -> dict | None:
    if not isinstance(c, dict) or c.get("present") is False:
        return None
    pick = lambda k, opts, d: c.get(k) if c.get(k) in opts else d  # noqa: E731
    col = lambda k: c.get(k) if isinstance(c.get(k), str) and HEX.fullmatch(c.get(k).strip()) else ""  # noqa: E731
    try:
        words = max(1, min(8, int(c.get("words") or 3)))
    except (TypeError, ValueError):
        words = 3
    return {"words": words, "case": pick("case", ("upper", "sentence", "lower"), "sentence"), "text_color": col("text_color") or "#FFFFFF",
            "highlight_color": col("highlight_color"), "box_color": col("box_color"), "outline": c.get("outline") is not False,
            "position": pick("position", ("top", "middle", "bottom"), "bottom"), "size": pick("size", ("small", "medium", "large"), "medium"),
            "weight": pick("weight", ("bold", "regular"), "bold"), "describe": str(c.get("describe") or "")[:160]}


class SourceCaptionsBody(BaseModel):
    url: str


@router.post("/source-captions")
async def source_captions(body: SourceCaptionsBody):
    """For videos analyzed before captions were read: Gemini Flash watches the original once (low resolution) for its caption look."""
    youtube.video_id(body.url)

    async def work(progress):
        hit = storage.kv_get("captions_look", body.url)
        if hit is not None:
            return {**hit, "cost": 0}
        progress("Gemini Flash is reading the original video's captions")
        data, cost = await google.generate_json("gemini-flash", [google.youtube_part(body.url), {"text": (
            "Look only at the words burned into this video that follow the speech (its captions or subtitles), not titles or "
            "signs. Return JSON: " + CAPTIONS_LOOK_SCHEMA)}], max_tokens=1024, low_media=True)
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else {}
        out = {"look": clean_captions_look(data)}
        storage.kv_put("captions_look", body.url, out)
        return {**out, "cost": round(cost, 4)}

    return _job("Reading the original's captions", work)


THUMB_LOOK_PROMPT = (
    "This is the thumbnail of a viral video. A creator will make a thumbnail of the SAME TYPE for their own new video "
    "(same layout and design, new picture and words). Describe how it is built. Return JSON: "
    '{"layout": "where the main subject sits and how big, framing (close-up face, wide, split screen, before/after...), the '
    'background, and how the eye moves", "subject": "who or what is shown and the expression or action", "mood": "the '
    'feeling and colour grade in a few words", "extras": "arrows, circles, emoji, borders, glow or props, or empty", '
    '"text": "the exact words printed on it, or empty", "words": number of printed words, "place": "where the words '
    'sit: top | bottom | left | right | center", "text_color": "#RRGGBB", "highlight_color": "#RRGGBB of a word printed in '
    'another colour, or empty", "box_color": "#RRGGBB of a solid box behind the words, or empty", "outline": true or false}')


async def source_thumb_look(thumb_url: str) -> tuple[dict | None, str, float]:
    """Reads the original video's thumbnail (once per link): (how it is built, the saved picture's URL, cost)."""
    if not thumb_url or not re.match(r"https://i\d?\.ytimg\.com/", thumb_url):
        return None, "", 0.0
    hit = storage.kv_get("thumb_look", thumb_url)
    if hit is not None:
        return hit.get("look"), hit.get("ref", ""), 0.0
    data = b""
    async with httpx.AsyncClient(timeout=20.0) as c:
        for u in (re.sub(r"/[a-z]*default\.jpg$", "/maxresdefault.jpg", thumb_url), thumb_url):
            try:
                r = await c.get(u)
                if r.status_code == 200 and len(r.content) > 3000:
                    data = r.content
                    break
            except httpx.HTTPError:
                pass
    if not data:
        return None, "", 0.0
    path = MEDIA / "images" / f"src_{hashlib.sha1(thumb_url.encode()).hexdigest()[:16]}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    try:
        look, cost = await google.generate_json("gemini-flash", [{"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(data).decode()}},
                                                                  {"text": THUMB_LOOK_PROMPT}], max_tokens=1024)
    except ProviderError:
        return None, media_url(path), 0.0
    if isinstance(look, list):
        look = look[0] if look and isinstance(look[0], dict) else {}
    col = lambda k: look.get(k).strip() if isinstance(look.get(k), str) and HEX.fullmatch(look.get(k).strip()) else ""  # noqa: E731
    look = {k: str(look.get(k) or "")[:300] for k in ("layout", "subject", "mood", "extras", "text")} | {
        "place": look.get("place") if look.get("place") in thumbs_mod.PLACES else "top",
        "text_color": col("text_color"), "highlight_color": col("highlight_color"), "box_color": col("box_color"),
        "outline": look.get("outline") is not False, "words": look.get("words") if isinstance(look.get("words"), int) else 0}
    storage.kv_put("thumb_look", thumb_url, {"look": look, "ref": media_url(path)})
    return look, media_url(path), cost


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
    idea: str = ""  # set when the creator starts from their own idea instead of a reference video
    script_type: str = "story"


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
        brief = (f"The creator's own idea for this video (there is no reference video): {body.idea[:4000]}\n"
                 "Build the story around that idea and keep the creator's details.\n" if body.idea.strip() else
                 f"Angle(s): {json.dumps([{k: a.get(k) for k in ('title', 'pitch', 'reason', 'hooks', 'type')} for a in body.angles])}\n"
                 f"Reference video it should sit next to: {body.source_title}. What that video did: {json.dumps(body.analysis)[:6000]}\n")
        data, cost, wn, warning = await write_json(body.writer, progress, (
            f"Write one continuous, original script for a {body.target_seconds}-second video (about {words} spoken words; stay within 10%).\n"
            f"{script_rule(body.script_type)}\n"
            f"{brief}"
            f"Tone: {body.tone or 'friendly, practical, no filler'}. Direction from the creator: {body.direction or 'none'}.\n"
            f"{fmt} Open with a strong hook, keep a new idea every few seconds, and end with a natural call to action.\n"
            'Return {"hook_options": [three different opening lines; the story starts with the first],'
            ' "story": "the full story",'
            ' "visuals": [{"sentence": "each sentence of the story, word for word, in order", "visual": "a short description of an original image that fits it, showing a real moment from the world of the story (never app icons, buttons, logos or on-screen text, even for a call to action)"}]}'))
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else {}
        story_text = str(data.get("story", "")).strip()
        if len(story_text.split()) < words * 0.4:
            raise ProviderError(f"{wn} returned a story that is far too short. Try again.")
        hooks = [str(h) for h in (data.get("hook_options") or [])][:3]
        while len(hooks) < 3:
            hooks.append(story_text.split(".")[0] + ".")
        visuals = [v for v in (data.get("visuals") or []) if isinstance(v, dict) and v.get("sentence")]
        return {"hook_options": hooks, "story": story_text, "visuals": visuals, "written_by": wn, "cost": round(cost, 4), "warning": warning}

    return _job("Writing the story", work)


# ---------- Projects saved on the server too, so clearing the browser or switching devices loses nothing

STATE_FILE = storage.ROOT / "app_state.json"


@router.get("/state")
def get_state():
    if not STATE_FILE.exists():
        return {"state": None, "saved_at": 0}
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {"state": None, "saved_at": 0}


class StateBody(BaseModel):
    state: dict
    saved_at: int


@router.put("/state")
def put_state(body: StateBody):
    data = json.dumps({"state": body.state, "saved_at": body.saved_at})
    if len(data) > 60_000_000:
        raise HTTPException(413, "Projects are too large to back up on the server (over 60 MB).")
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(data)
    if STATE_FILE.exists():  # keep the previous save too, in case a bad save ever overwrites good work
        STATE_FILE.replace(STATE_FILE.with_name("app_state.prev.json"))
    tmp.replace(STATE_FILE)  # atomic: a crash mid-write never leaves a broken file
    return {"ok": True, "saved_at": body.saved_at}


# ---------- Your music library (tracks you download, for example from the YouTube Audio Library)

MUSIC_TYPES = [(b"ID3", "mp3"), (b"\xff\xfb", "mp3"), (b"\xff\xf3", "mp3"), (b"\xff\xf2", "mp3"), (b"RIFF", "wav"),
               (b"OggS", "ogg"), (b"fLaC", "flac")]


@router.post("/music")
async def upload_music(request: Request, name: str = ""):
    """Saves one uploaded track (the raw file as the request body) and returns where it lives and how long it is."""
    import subprocess
    data = await request.body()
    if not data:
        raise HTTPException(422, "Choose a music file first.")
    if len(data) > 40_000_000:
        raise HTTPException(413, "That file is over 40 MB. Use an MP3 version of the track.")
    ext = next((e for magic, e in MUSIC_TYPES if data.startswith(magic)), "m4a" if data[4:8] == b"ftyp" else None)
    if not ext:
        raise HTTPException(422, "Use an MP3, WAV, M4A, OGG or FLAC file.")
    path = MEDIA / "music" / f"{uuid.uuid4().hex}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        seconds = float(p.stdout.strip())
    except ValueError:
        path.unlink(missing_ok=True)
        raise HTTPException(422, "That file could not be read as music.")
    return {"url": media_url(path), "duration": round(seconds, 1), "name": (name or "Track")[:80]}


# ---------- Text card preview: exactly what the render will draw (free)

@router.get("/card")
def card_preview(text: str = "", small: str = "", layout: str = "big", style: str = "", kind: str = "", palette: str = "", aspect: str = "9:16", emoji: str = ""):
    from fastapi.responses import Response
    from .cards import make_card
    import tempfile
    a, b = (int(x) for x in (aspect if aspect in ("9:16", "16:9", "1:1", "4:5") else "9:16").split(":"))
    w, h = (480, round(480 * b / a)) if a < b else (round(480 * a / b), 480)
    with tempfile.TemporaryDirectory() as d:
        out = make_card({"text": text[:80], "small": small[:40], "layout": layout, "emoji": emoji[:16]}, {"id": style, "kind": kind, "palette": [c for c in palette.split(",") if c][:4]},
                        w, h, Path(d) / "c.png")
        data = out.read_bytes()
    return Response(data, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


class MediaCheckBody(BaseModel):
    paths: list[str]


@router.post("/media-check")
async def media_check(body: MediaCheckBody):
    """Which of these saved files are gone (for example after a redeploy without a persistent disk)."""
    root = MEDIA.resolve()
    missing = []
    for ref in body.paths[:5000]:
        if not ref.startswith("/media/"):
            continue
        path = (MEDIA / ref.removeprefix("/media/")).resolve()
        if root not in path.parents or not path.is_file():
            missing.append(ref)
    return {"missing": missing}


# ---------- Image ideas that carry each scene's message

class SceneVisualsBody(BaseModel):
    writer: str = "gpt-5-5"
    story: str = ""
    scenes: list[dict]          # [{"n": 1, "text": "the narration this image sits under"}]
    style_name: str = ""
    idea: str = ""              # the creator's own rule for every image
    characters: list[dict] = []
    words_max: int = 3          # most words to print on an image (an idea like "one or two words" sets 2)
    cards_share: float = 0.0    # share of scenes to make free text cards (0.3 = about 30%)


def _no_words(visual: str) -> str:
    """Drops any sentence of a picture description that asks for printed words: the words are added once, separately."""
    parts = re.split(r"(?<=[.!?])\s+", visual.replace("**", ""))
    keep = [x for x in parts if not re.search(r"\b(words?|text|caption|label|lettering|reads|written|sign saying)\b", x, re.I)]
    return " ".join(keep or parts).strip()


@router.post("/scene-visuals")
async def scene_visuals(body: SceneVisualsBody):
    """For each scene: a picture that makes its narration obvious at a glance, plus 1-3 words that could be printed on it."""
    scenes = [s for s in body.scenes if str(s.get("text", "")).strip()][:400]
    if not scenes:
        raise HTTPException(422, "No scenes to picture.")

    async def work(progress):
        out, cost, wn, warn = {}, 0.0, WRITER_NAMES.get(body.writer, body.writer), None
        cast = "; ".join(f"{c.get('name')}: {c.get('description')}" for c in body.characters if c.get("name"))
        for k in range(0, len(scenes), 40):  # 40 scenes per call keeps each answer short and reliable
            batch = scenes[k:k + 40]
            progress(f"{wn} is picturing scenes {k + 1} to {k + len(batch)} of {len(scenes)}")
            data, c, wn, w = await write_json(body.writer, progress, (
                (f"The whole story, for context:\n---\n{body.story[:6000]}\n---\n" if body.story else "")
                + f"Visual style of every image: {body.style_name}.\n"
                + (f"Characters (describe them the same way every time they appear): {cast}\n" if cast else "")
                + (f"The creator's rule for every image (follow it exactly): {body.idea}\n" if body.idea.strip() else "")
                + "Each scene below is the narration heard while one image is on screen. For each, describe ONE image that "
                "makes the meaning of that narration obvious at a glance, even with the sound off and no text: who is in it, "
                "what they are doing, where, the key object or symbol, and the emotion on their faces. Turn abstract ideas "
                "into a concrete visual metaphor (for example 'saving money' becomes a hand dropping a coin into a jar). One clear "
                "focal point, no collage, no split screen, never app icons, buttons or logos. Do not put any words, signs, labels "
                "or captions in the visual itself. Separately, give " + ("1 word" if body.words_max <= 1 else f"1 to {min(3, body.words_max)} words") + " that sum up "
                "the scene's message, and mark about one scene in three as key (where printed words would help most).\n"
                + (f"Also mark about {round(body.cards_share * 100)}% of the scenes as TEXT CARDS: scenes whose message lands best as bold "
                   "words on a plain background (a key point, a number, a question, a strong statement, a turning point). Never the "
                   "first scene, never two text cards in a row, and keep scenes with people, places or actions as images. For a "
                   "card give card_text (1 to 3 words only: the most important words of that scene, nothing else; wrap the one "
                   "word to highlight in *stars*), card_emoji (exactly one emoji that fits the scene), card_small (only for a "
                   "countdown: 'Number 3'; otherwise empty) and card_layout: big, number (for a countdown or a numbered point) "
                   "or question (if it asks something).\n" if body.cards_share > 0 else "")
                + f"Scenes: {json.dumps([{'n': s.get('n'), 'text': str(s.get('text'))[:600]} for s in batch])}\n"
                + 'Return {"scenes": [{"n": scene number, "visual": "the image, under 45 words", "text": "the words", "key": true or false'
                + (', "card": true or false, "card_text": "1-3 words", "card_emoji": "one emoji", "card_small": "", "card_layout": "big|number|question"' if body.cards_share > 0 else "")
                + '}]}'))
            cost += c
            warn = warn or w
            if isinstance(data, list):
                data = {"scenes": data}
            for item in data.get("scenes") or []:
                if isinstance(item, dict) and item.get("visual"):
                    card = None
                    if body.cards_share > 0 and item.get("card") and str(item.get("card_text", "")).strip():
                        from .cards import first_emoji
                        card = {"text": " ".join(str(item["card_text"]).replace('"', "").split()[:3])[:40],
                                "emoji": first_emoji(str(item.get("card_emoji") or "")),
                                "small": str(item.get("card_small") or "")[:24],
                                "layout": item.get("card_layout") if item.get("card_layout") in ("big", "number", "question", "quote") else "big"}
                    out[str(item.get("n"))] = {"card": card, "visual": _no_words(str(item["visual"]))[:500], "text": " ".join(str(item.get("text", "")).replace('"', "").replace("*", "").split()[:max(1, min(3, body.words_max))])[:40],
                                               "key": bool(item.get("key"))}
        if not out:
            raise ProviderError(f"{wn} returned no image ideas. Try again.")
        # keep cards where they belong: not the first scene, never two in a row, about the share asked for
        if body.cards_share > 0:
            nums = sorted(int(k) for k in out if k.isdigit())
            cap = max(1, round(body.cards_share * len(scenes) + 0.5))
            used, prev = 0, None
            for n in nums:
                item = out[str(n)]
                if item.get("card") and (n == 1 or prev == n - 1 or used >= cap):
                    item["card"] = None
                if item.get("card"):
                    used, prev = used + 1, n
        return {"scenes": out, "written_by": wn, "cost": round(cost, 4), "warning": warn}

    return _job("Picturing the scenes", work)


class VisualsBody(BaseModel):
    writer: str = "gpt-5-5"
    story: str


@router.post("/visuals")
async def visuals(body: VisualsBody):
    """Image ideas for a story the creator pasted: one short visual per sentence. Text only, so it costs little."""
    if len(body.story.split()) < 5:
        raise HTTPException(422, "Paste a longer story first.")

    async def work(progress):
        progress(f"{WRITER_NAMES.get(body.writer, body.writer)} is picturing each sentence")
        data, cost, wn, warning = await write_json(body.writer, progress, (
            f"Here is a story a creator wrote for a narrated video:\n---\n{body.story[:40000]}\n---\n"
            "Do not change the story. For each sentence, describe one original image that fits it, showing a real moment "
            "from the world of the story (never app icons, buttons, logos or on-screen text). Keep each description under 25 words. "
            'Return {"visuals": [{"sentence": "the sentence, word for word, in order", "visual": "the image"}]}'))
        if isinstance(data, list):
            data = {"visuals": data}
        vis = [v for v in (data.get("visuals") or []) if isinstance(v, dict) and v.get("sentence") and v.get("visual")]
        if not vis:
            raise ProviderError(f"{wn} returned no image ideas. Try again.")
        return {"visuals": vis, "written_by": wn, "cost": round(cost, 4), "warning": warning}

    return _job("Picturing the story", work)


# ---------- 3. Voice, images

class SpeakBody(BaseModel):
    text: str
    voice: str = "v-aoede"
    model: str = "gemini-tts-flash"
    delivery: str = "warm, steady and conversational, like explaining something to a friend, at a medium pace"
    speed: float = 1.0   # exact tempo change, pitch kept (0.8 to 1.3)
    pitch: float = 0.0   # semitones (-4 to +4)


def tune_voice(path: Path, speed: float, pitch: float) -> float:
    """Applies an exact speed and pitch change to a voice clip in place; returns its new length in seconds."""
    import subprocess
    import wave
    speed = min(1.3, max(0.8, speed)); pitch = min(4.0, max(-4.0, pitch))
    with wave.open(str(path)) as w:
        rate = w.getframerate()
    if abs(speed - 1) > 1e-3 or abs(pitch) > 1e-3:
        f = 2 ** (pitch / 12)
        af = f"asetrate={round(rate * f)},aresample={rate},atempo={speed / f:.5f}"
        tmp = path.with_suffix(".tuned.wav")
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-af", af, "-ac", "1", "-c:a", "pcm_s16le", str(tmp)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise ProviderError("Could not change the voice speed or pitch: " + (p.stderr.strip()[-200:] or "FFmpeg failed"))
        tmp.replace(path)
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


@router.post("/tts")
async def tts(body: SpeakBody):
    if not body.text.strip():
        raise HTTPException(422, "Nothing to read.")
    out = MEDIA / "audio" / f"{uuid.uuid4().hex}.wav"
    try:
        seconds, cost = await google.speak(body.text, body.voice, body.model, out, body.delivery)
        seconds = await asyncio.to_thread(tune_voice, out, body.speed, body.pitch)
    except ProviderError as e:
        raise HTTPException(502, str(e))
    return {"url": media_url(out), "duration": round(seconds, 2), "cost": round(cost, 4)}


class ImageBody(BaseModel):
    prompt: str
    aspect: str = "9:16"
    model: str = "nano-banana-2"
    refs: list[str] = []  # style references: /media/... URLs or data: URLs (custom templates, the locked first scene)


def load_ref(ref: str) -> tuple[bytes, str]:
    """Reads a reference image from a data: URL, a template example picture, or this server's media folder."""
    if ref.startswith("data:"):
        head, b64 = ref.split(",", 1)
        mime = head[5:].split(";")[0] or "image/jpeg"
        data = base64.b64decode(b64)
    elif ref.lstrip("/").startswith("templates/"):  # a built-in template's example picture, used as a style reference
        root = (Path(__file__).resolve().parents[2] / "web" / "templates").resolve()
        path = (root / ref.lstrip("/").removeprefix("templates/")).resolve()
        if root not in path.parents or not path.exists():
            raise HTTPException(422, "A template example image is missing on the server.")
        data = path.read_bytes()
        mime = "image/jpeg"
    else:
        rel = ref.removeprefix("/media/")
        path = (MEDIA / rel).resolve()
        if MEDIA.resolve() not in path.parents or not path.exists():
            raise HTTPException(422, "A reference image is missing on the server. Regenerate scene 1 or turn off Lock look.")
        data = path.read_bytes()
        mime = {".png": "image/png", ".webp": "image/webp"}.get(path.suffix, "image/jpeg")
    if len(data) > 7_000_000:
        raise HTTPException(422, "A reference image is too large (over 7 MB).")
    return data, mime


def kontext_ref(ref: str) -> str:
    """FLUX Kontext copies the people in its reference, so a template example scene is swapped for that template's
    style swatch (one generic character and a few objects in the same style), when there is one."""
    m = re.fullmatch(r"/?templates/([a-z0-9]+)-\d+\.jpg", ref)
    if m and (Path(__file__).resolve().parents[2] / "web" / "templates" / f"{m.group(1)}-style.jpg").exists():
        return f"templates/{m.group(1)}-style.jpg"
    return ref


@router.post("/image")
async def image(body: ImageBody):
    if body.aspect not in ("9:16", "16:9", "1:1", "4:5"):
        raise HTTPException(422, "Unsupported frame.")
    try:
        if fal.is_fal(body.model):  # FLUX schnell and dev read text only; FLUX Kontext gets the first style reference
            ref = load_ref(kontext_ref(body.refs[0])) if body.model == "flux-kontext" and body.refs else None
            refs = [load_ref(r) for r in body.refs[:4] if r] if body.model == "seedream" else None
            path, cost = await fal.draw(body.prompt[:4000], body.aspect, body.model, MEDIA / "images" / uuid.uuid4().hex, ref, refs)
        else:
            refs = [load_ref(r) for r in body.refs[:4] if r]
            path, cost = await google.draw(body.prompt[:4000], body.aspect, body.model, MEDIA / "images" / uuid.uuid4().hex, refs)
    except ProviderError as e:
        raise HTTPException(502, str(e))
    return {"url": media_url(path), "cost": cost}


# ---------- Visual templates from your own images

class StyleFromImages(BaseModel):
    images: list[str]  # data: URLs, 1 to 3
    name: str = ""


@router.post("/style-from-images")
async def style_from_images(body: StyleFromImages):
    if not 1 <= len(body.images) <= 3:
        raise HTTPException(422, "Upload one to three images.")
    parts = [{"inlineData": {"mimeType": m, "data": base64.b64encode(d).decode()}} for d, m in (load_ref(i) for i in body.images)]
    parts.append({"text": (
        "These images show one visual style a video creator wants to reuse for every scene of their videos. Describe only "
        "the style, never the subjects. Return JSON: {\"name\": a short 2-3 word style name, \"mood\": 3-6 words, "
        "\"prefix\": one rich prompt sentence covering medium, rendering, lighting, color palette, texture and lens or brushwork "
        "that recreates this look for any subject, \"avoid\": what would break this look, comma separated, "
        "\"palette\": [four hex colors from the images, dark to light plus one accent]}")})
    try:
        data, cost = await google.generate_json("gemini-flash", parts, max_tokens=2048)
    except ProviderError as e:
        raise HTTPException(502, str(e))
    if isinstance(data, list):
        data = data[0] if data else {}
    pal = [c for c in (data.get("palette") or []) if isinstance(c, str) and c.startswith("#")][:4]
    while len(pal) < 4:
        pal.append(["#222222", "#777777", "#EEEEEE", "#C97B4B"][len(pal)])
    return {"name": body.name.strip() or str(data.get("name") or "My style")[:40], "mood": str(data.get("mood", ""))[:80],
            "prefix": str(data.get("prefix", ""))[:600], "avoid": str(data.get("avoid", ""))[:300], "palette": pal, "cost": round(cost, 4)}


# ---------- Rewrite the story with an idea

class ReviseBody(BaseModel):
    writer: str = "gpt-5-5"
    story: str
    idea: str
    target_seconds: int = 60
    video_type: str = "voiceover"
    characters: list[dict] = []
    script_type: str = ""  # set to rewrite the script as another type


@router.post("/revise")
async def revise(body: ReviseBody):
    if not body.idea.strip():
        raise HTTPException(422, "Type the idea you want added first.")

    async def work(progress):
        words = round(body.target_seconds * 2.5)
        progress(f"{WRITER_NAMES.get(body.writer, body.writer)} is rewriting the story with your idea")
        fmt = ("Keep the 'Name: what they say' dialogue format, one line per speaker, using only these characters: "
               + ", ".join(c.get("name", "") for c in body.characters) + "." if body.video_type == "character"
               else "Keep it as narration in short paragraphs separated by a blank line.")
        data, cost, wn, warning = await write_json(body.writer, progress, (
            f"Here is a story for a {body.target_seconds}-second video (about {words} spoken words):\n---\n{body.story}\n---\n"
            f"Rewrite it to include this idea from the creator: {body.idea}\n"
            "Work the idea in naturally where it fits best, keep what already works, keep the opening hook unless the idea "
            f"changes it, and stay within 10% of {words} words. {fmt}\n"
            + (f"Rewrite it in this script type, keeping its content: {script_rule(body.script_type)}\n" if body.script_type else "")
            + 
            'Return {"story": "the full rewritten story", "changed": "one sentence on what changed",'
            ' "visuals": [{"sentence": "each sentence of the story, word for word, in order", "visual": "a short description of an original image that fits it, showing a real moment from the world of the story (never app icons, buttons, logos or on-screen text, even for a call to action)"}]}'))
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else {}
        story_text = str(data.get("story", "")).strip()
        if len(story_text.split()) < words * 0.4:
            raise ProviderError(f"{wn} returned a story that is far too short. Try again.")
        return {"story": story_text, "changed": str(data.get("changed", "")), "written_by": wn, "cost": round(cost, 4), "warning": warning,
                "visuals": [v for v in (data.get("visuals") or []) if isinstance(v, dict) and v.get("sentence")]}

    return _job("Rewriting the story", work)


# ---------- After the video: titles, post captions and thumbnails to choose from

# Thumbnail layouts that make people click; each of the 3 thumbnails uses a different one.
THUMB_FORMULAS = ["Reaction and mystery: a huge shocked or scared face on one side, the thing causing it half hidden on the other",
                  "Moment before: frozen one second before the outcome, at peak tension",
                  "This or that: a split frame, before and after or two choices, with one clearly winning or failing",
                  "Hidden detail: an ordinary scene with one small strange thing circled in red",
                  "Stakes: one person against something huge, a countdown or a number that raises the stakes"]


class ThumbTextBody(BaseModel):
    image: str
    text: str
    place: str = "top"
    style: str = ""
    look: dict = {}  # the original thumbnail's words: text_color, highlight_color, box_color, outline


@router.post("/thumb-text")
def thumb_text(body: ThumbTextBody):
    """Draws the suspense words on a thumbnail picture (free, no AI model), so they read clearly and are spelled right."""
    path = (MEDIA / body.image.removeprefix("/media/")).resolve()
    if MEDIA.resolve() not in path.parents or not path.exists():
        raise HTTPException(422, "The thumbnail picture is missing on the server. Make it again.")
    out = MEDIA / "images" / f"{uuid.uuid4().hex}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    thumbs_mod.add_text(path, body.text[:60], body.place, body.style, out, body.look)
    return {"url": media_url(out)}

class KitBody(BaseModel):
    writer: str = "gpt-5-5"
    story: str
    angle: dict = {}
    source_title: str = ""
    tags: list[str] = []
    source: dict = {}    # the analyzed video: title, channel, tags, description, summary, hook, why it worked
    thumb: str = ""      # the analyzed video's thumbnail picture, to make thumbnails of the same type


@router.post("/publish-kit")
async def publish_kit(body: KitBody):
    async def work(progress):
        look, look_ref, look_cost = None, "", 0.0
        if body.thumb:
            progress("Gemini Flash is reading the original video's thumbnail")
            look, look_ref, look_cost = await source_thumb_look(body.thumb)
        progress("Writing titles, captions, descriptions and thumbnail ideas from your story")
        same_type = (
            f"THUMBNAILS: the original video's thumbnail is built like this: {json.dumps(look)}. Make all 3 thumbnails the SAME "
            "TYPE: the same layout, framing, subject size, kind of expression, extras (arrows, circles, emoji, split screen) and "
            f"the same kind of words ({look.get('words') or 'a few'} words, same tone, e.g. a claim, a question or a number), "
            "but with this video's own subject and moment and new words, so it looks like a sibling of the original, never a "
            "copy. Use formula \"Like the original\" for the first one, and for the other two the closest proven formulas. "
            f"Put the words where the original has them (place \"{look.get('place', 'top')}\").\n" if look else "")
        data, cost, wn, warning = await write_json(body.writer, progress, (
            f"A creator finished this video. Its story:\n---\n{body.story[:12000]}\n---\n"
            f"Angle: {json.dumps({k: body.angle.get(k) for k in ('title', 'pitch', 'reason', 'keeps', 'changes')})}. "
            f"Suggested tags so far: {body.tags}.\n"
            + (f"It recreates this viral video and must be recommended next to it: {json.dumps(body.source)[:5000]}\n"
               "Make the titles, captions, descriptions and thumbnails clearly related to that video: echo its title "
               "pattern, its main keywords and tags, its hook and its thumbnail style (subject, emotion, bold short text), "
               "so viewers and the platform connect the two. Do not copy its title word for word, and do not claim to be "
               "that creator or a part 2.\n"
               "POST CAPTIONS AND DESCRIPTIONS: write them in the same form as the original's description above: about the same "
               "length, the same structure (hook line, line breaks, lists, questions, calls to action), the same use of emoji and "
               "the same kind and number of hashtags, with this video's own content.\n" if body.source.get("title") else "")
            + same_type
            + "Write options the creator can choose from. Everything must honestly describe this video. Return "
            '{"titles": [4 different titles under 70 characters],'
            ' "captions": [3 different post captions/descriptions, each 2-4 sentences that hook, summarize and invite a comment, '
            'each ending with 3-5 hashtags],'
            ' "descriptions": [3 different full YouTube descriptions, each 2 short paragraphs that summarize the video and what '
            'the viewer will learn or feel, then one line inviting a comment, then a line of 3-5 hashtags],'
            ' "thumbnails": [3 objects, each built on a DIFFERENT proven click formula, ' + json.dumps(THUMB_FORMULAS) + ', each '
            '{"formula": "the formula name", "concept": "the picture: ONE main subject shown large and close (a face with a strong '
            'readable emotion when the story has people), one simple background, and the one detail that makes people wonder; '
            'stop at the moment of highest tension, never show the answer", "emotion": "the face or mood in 2-4 words", '
            '"focus": "the one object or detail to circle or point at with a red arrow, or empty", '
            '"text": "2-4 words of SUSPENSE in capitals that open a question the picture does not answer (an unfinished claim, a '
            'warning, a stake, a number, a twist); never repeat the title, never give away the ending; put *stars* around the '
            'one word to colour, e.g. \'IT WAS *ALIVE*\', \'DON\'T OPEN *IT*\', \'*3* DAYS LEFT\'", '
            '"place": "where the words go so they never cover the face: top, bottom, left or right"}],'
            ' "tags": [8 search tags]}'))
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else {}
        thumbs = [t for t in (data.get("thumbnails") or []) if isinstance(t, dict) and t.get("concept")][:3]
        return {"titles": [str(t) for t in (data.get("titles") or [])][:4], "captions": [str(c) for c in (data.get("captions") or [])][:3],
                "descriptions": [str(c) for c in (data.get("descriptions") or [])][:3],
                "thumbnails": [{"concept": str(t["concept"]), "text": str(t.get("text", ""))[:48], "formula": str(t.get("formula", ""))[:40],
                                "emotion": str(t.get("emotion", ""))[:60], "focus": str(t.get("focus", ""))[:120],
                                "place": t.get("place") if t.get("place") in thumbs_mod.PLACES else "top"} for t in thumbs],
                "tags": [str(t) for t in (data.get("tags") or [])][:10], "written_by": wn, "cost": round(cost + look_cost, 4), "warning": warning,
                "source_thumb": {"look": look, "ref": look_ref} if look else None}

    return _job("Writing the publish kit", work)


# ---------- Channel ideas: a channel built around this kind of story and look

class ChannelBody(BaseModel):
    writer: str = "gpt-5-5"
    story: str
    angle: dict = {}
    summary: str = ""
    style_name: str = ""
    style_recipe: str = ""
    video_type: str = "voiceover"
    platform: str = "YouTube"
    source: dict = {}    # the analyzed video
    idea: str = ""       # the creator's own idea of the channel they want


@router.post("/channel-kit")
async def channel_kit(body: ChannelBody):
    async def work(progress):
        progress("Reading the story type and the visual style")
        data, cost, wn, warning = await write_json(body.writer, progress, (
            f"A creator makes {body.video_type} videos like this one. Story:\n---\n{body.story[:8000]}\n---\n"
            + (f"The videos recreate the success of this viral video, and the channel should grow from it: {json.dumps(body.source)[:4000]}\n"
               "Base the channel's niche, name and promise on what made that video work, so it attracts the same viewers.\n" if body.source.get("title") else "")
            + (f"The creator's own idea of the channel they want (follow it closely): {body.idea[:1000]}\n" if body.idea.strip() else "")
            + 
            f"Angle: {json.dumps({k: body.angle.get(k) for k in ('title', 'type', 'pitch')})}. Reference video summary: {body.summary[:1500]}\n"
            f"Visual style: {body.style_name} ({body.style_recipe[:400]}).\n"
            f"First decide what type of story this is (its genre, tone and audience) and how the visual style feels. Then propose "
            f"3 different {body.platform} channels built for many videos like it. Names must be original, short, easy to say, and "
            "must not copy or imitate an existing well-known channel, brand or character. Return "
            '{"story_type": "one line: genre, tone and audience", "visual_type": "one line on the look",'
            ' "channels": [3 objects {"name": "channel name", "handle": "@handle without spaces", "tagline": "bio under 150 characters",'
            ' "about": "channel description of 3-4 sentences: what viewers get, how often, why subscribe", "niche": "2-4 words",'
            ' "avatar": "what the round profile picture shows: one simple, bold subject that reads at tiny size",'
            ' "banner": "what the wide banner shows, with room in the middle for the channel name"}]}'))
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else {}
        chans = []
        for c in (data.get("channels") or [])[:3]:
            if not isinstance(c, dict) or not c.get("name"):
                continue
            handle = "@" + "".join(ch for ch in str(c.get("handle") or c["name"]) if ch.isalnum() or ch in "._-")[:30]
            chans.append({"name": str(c["name"])[:50], "handle": handle, "tagline": str(c.get("tagline", ""))[:160],
                          "about": str(c.get("about", ""))[:1000], "niche": str(c.get("niche", ""))[:40],
                          "avatar": str(c.get("avatar", ""))[:400], "banner": str(c.get("banner", ""))[:400]})
        if not chans:
            raise ProviderError(f"{wn} returned no channel ideas. Try again.")
        return {"story_type": str(data.get("story_type", "")), "visual_type": str(data.get("visual_type", "")),
                "channels": chans, "written_by": wn, "cost": round(cost, 4), "warning": warning}

    return _job("Writing channel ideas", work)


# ---------- 4. Render

class RenderBody(BaseModel):
    final: bool = False
    res: int = 0  # short side in pixels: 480, 720 or 1080 (0 = 1080 for final, 480 for the draft)
    aspect: str = "9:16"
    caption_style: str = "bold"
    scenes: list[dict[str, Any]]
    lines: list[dict[str, Any]]
    music: dict[str, Any] = {}  # {url, volume 0-0.6, duck}
    card_style: dict[str, Any] = {}  # {id, kind, palette} of the template, for text cards


@router.post("/render")
async def render_video(body: RenderBody):
    async def work(progress):
        return await asyncio.to_thread(render.render, body.model_dump(), MEDIA, progress)

    return _job("Rendering final MP4" if body.final else "Building draft", work)
