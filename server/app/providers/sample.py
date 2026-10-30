"""Canned data for the mock adapters, shared with the web prototype
(sample_data.json is exported from web/index.html)."""
from __future__ import annotations

import json
from pathlib import Path

_D = json.loads((Path(__file__).parent / "sample_data.json").read_text())
_S = _D["SAMPLE"]

VOICES = [{k: v[k] for k in ("id", "name", "desc", "speed")} | {"clone": v.get("clone", False)} for v in _D["VOICES"]]
STYLES = [{"template_id": s["id"], "name": s["name"], "image_prefix": s["prefix"], "mood": s["mood"], "palette": s["palette"]} for s in _D["STYLES"]]
DEFAULT_CHARACTERS = [{"name": c["name"], "description": c["description"], "notes": c["notes"], "voice_id": c["voice_id"]} for c in _D["DEFAULT_CHARS"]]

_likes = lambda s: int(float(s.rstrip("k")) * 1000) if s.endswith("k") else int(s)
VIDEO = {
    "title": _S["source"]["title"], "channel": _S["source"]["channel"], "duration": _S["source"]["duration"],
    "published": _S["source"]["published"], "views": 2_100_000, "comments_scanned": _S["source"]["comments_scanned"],
    "comments": [{"id": c["id"], "text": c["text"], "likes": _likes(c["likes"]), "kind": c["kind"]} for c in _S["source"]["comments"]],
}
TRANSCRIPT = _S["source"]["transcript"]
SCENES = [{"t": k["t"], "description": k["d"]} for k in _S["source"]["key_frames"]]
_a = _S["analysis"]
ANALYSIS = {
    "summary": _a["summary"], "hook": _a["hook"], "beats": [f"{t} {b}" for t, b in _a["beats"]], "pacing": _a["pacing"],
    "why_it_worked": _a["why_it_worked"], "gaps": [{"kind": g["kind"], "text": g["text"], "from_comment": g.get("from")} for g in _a["gaps"]],
}
ANGLES = [{
    "id": a["id"], "title": a["title"], "pitch": a["pitch"], "hook": a["hooks"][0], "type": a["type"],
    "suggested_style": a["style"], "relatedness_score": a["score"], "relatedness_reason": a["reason"],
    "suggested_title": a["suggested_title"], "suggested_tags": a["tags"], "effort": a["effort"],
    "potential": a["potential"], "origin_comment": a.get("origin"),
} for a in _S["angles"]]
_RAW = {a["id"]: a for a in _S["angles"]}


def script_for(prompt: str) -> dict:
    """The script prompt names the angle ids and target length on its first line: 'angles=a1,a4 seconds=60'."""
    head = dict(kv.split("=", 1) for kv in prompt.splitlines()[0].split())
    ids = head["angles"].split(",")
    count = max(3, round(int(head["seconds"]) / 6.5) - 1)
    beats: list[dict] = []
    if len(ids) > 1:
        a, b = _RAW[ids[0]]["beats"], _RAW[ids[1]]["beats"]
        for i in range(max(len(a), len(b))):
            beats += ([a[i]] if i < len(a) else []) + ([b[i]] if i < len(b) else [])
    else:
        beats = list(_RAW[ids[0]]["beats"])
    beats = beats[:count]
    missing = count - len(beats)  # demo only: pad so the simulated script reaches the chosen length
    if missing > 0:
        beats += _D["FILLERS"][-missing:]
    first = _RAW[ids[0]]
    lines = [{"text": first["hooks"][0], "visual": first["hook_visual"]}] + [{"text": b["t"], "visual": b["v"]} for b in beats]
    hooks = list(first["hooks"][:3])
    if len(ids) > 1:
        hooks[2] = _RAW[ids[1]]["hooks"][0]
    return {"hook_options": hooks, "lines": lines}
