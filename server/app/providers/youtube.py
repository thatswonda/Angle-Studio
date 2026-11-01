"""YouTube Data API v3: title, channel, stats and the top comments of a video (ING-1, ING-4)."""
from __future__ import annotations

import os
import re

import httpx

from .google import ProviderError

API = "https://www.googleapis.com/youtube/v3"
QUESTION = re.compile(r"\?")
REQUEST = re.compile(r"\b(please|pls|can you|could you|would you|do a|make a|part 2|part two|next video|would love|"
                     r"tutorial|how about|you should|waiting for|follow[- ]up|more of)\b", re.I)
CONFUSION = re.compile(r"\b(confus|don'?t understand|didn'?t understand|what do you mean|lost me|not sure)\w*", re.I)


def video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise ProviderError("That doesn't look like a YouTube video link.")
    return m.group(1)


def _seconds(iso: str) -> int:
    m = re.match(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    d, h, mi, s = (int(x) if x else 0 for x in (m.groups() if m else (0, 0, 0, 0)))
    return d * 86400 + h * 3600 + mi * 60 + s


def _views(n: int) -> str:
    return f"{n / 1e6:.1f}M views" if n >= 1e6 else f"{n / 1e3:.0f}K views" if n >= 1e3 else f"{n} views"


async def basic_info(url: str) -> dict:
    """Title, channel and thumbnail from YouTube's public oEmbed page info, which needs no key.
    Used when the Data API refuses, so the analysis can still run (without comments or view counts)."""
    vid = video_id(url)
    d = {}
    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.get("https://www.youtube.com/oembed", params={"url": f"https://www.youtube.com/watch?v={vid}", "format": "json"})
        if r.status_code in (401, 403, 404):
            raise ProviderError("YouTube found no public video at that link.")
        d = r.json() if r.status_code == 200 else {}
    except httpx.HTTPError:
        pass  # no page info either; Gemini still reads the title from the video itself
    return {"title": d.get("title", "Reference video"), "channel": d.get("author_name", ""), "duration": 0, "published": "",
            "views": "", "thumb": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg", "comments_scanned": 0, "comment_count": 0, "comments": []}


async def fetch(url: str) -> dict:
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise ProviderError("YOUTUBE_API_KEY is not set on the server.")
    vid = video_id(url)
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(f"{API}/videos", params={"part": "snippet,statistics,contentDetails", "id": vid, "key": key})
        if r.status_code != 200:
            msg = r.json().get("error", {}).get("message", "")
            if r.status_code == 403 and "blocked" in msg:
                msg = ("this API key is not allowed to use the YouTube Data API. In Google Cloud, enable YouTube Data API v3 "
                       "for the key's project and allow it under the key's API restrictions.")
            raise ProviderError(f"YouTube Data API error {r.status_code}: {msg}")
        items = r.json().get("items", [])
        if not items:
            raise ProviderError("YouTube found no public video at that link.")
        v = items[0]
        sn, st = v["snippet"], v.get("statistics", {})
        comments: list[dict] = []
        cr = await c.get(f"{API}/commentThreads", params={"part": "snippet", "videoId": vid, "order": "relevance",
                                                           "maxResults": 100, "textFormat": "plainText", "key": key})
        if cr.status_code == 200:
            for it in cr.json().get("items", []):
                top = it["snippet"]["topLevelComment"]["snippet"]
                comments.append({"text": top.get("textDisplay", "").strip(), "likes": int(top.get("likeCount", 0))})
        # Comments turned off (403) is normal; the analysis just has fewer audience signals.
    kept = []
    for cm in comments:
        t = cm["text"]
        kind = "question" if QUESTION.search(t) else "request" if REQUEST.search(t) else "confusion" if CONFUSION.search(t) else None
        if kind and 8 <= len(t) <= 400:
            kept.append({**cm, "kind": kind})
    kept.sort(key=lambda x: -x["likes"])
    for i, cm in enumerate(kept, 1):
        cm["id"] = f"c{i}"
    thumbs = sn.get("thumbnails", {})
    return {
        "title": sn.get("title", ""), "channel": sn.get("channelTitle", ""),
        "duration": _seconds(v.get("contentDetails", {}).get("duration", "")),
        "published": (sn.get("publishedAt") or "")[:10], "views": _views(int(st.get("viewCount", 0))),
        "thumb": (thumbs.get("maxres") or thumbs.get("high") or thumbs.get("medium") or {}).get("url", ""),
        "comments_scanned": len(comments), "comment_count": int(st.get("commentCount", 0)),
        "comments": kept[:40],
    }
