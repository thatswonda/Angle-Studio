"""Assembly engine (ASM-1..6): FFmpeg pipeline for motion, transitions,
captions and audio mixing. In mock mode, or when FFmpeg or real media are
missing, it writes the render plan instead of a video so the flow still completes."""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from .models import Project

CAPTION_STYLES = {
    "bold": "FontName=Bricolage Grotesque,FontSize=18,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=3,Alignment=2,MarginV=180",
    "clean": "FontName=IBM Plex Sans,FontSize=12,BorderStyle=4,BackColour=&H80000000,Alignment=2,MarginV=90",
    "karaoke": "FontName=IBM Plex Sans,FontSize=14,Bold=1,SecondaryColour=&H0045E4FF,Alignment=2,MarginV=160",
}


def plan(p: Project, height: int, polish: bool) -> dict:
    # "1080p" means the short side: 9:16 -> 1080x1920, 16:9 -> 1920x1080, 1:1 -> 1080x1080, 4:5 -> 1080x1350.
    a, b = (int(x) for x in p.format.aspect_ratio.split(":"))
    w, h = (round(height * a / b / 2) * 2, height) if a >= b else (height, round(height * b / a / 2) * 2)
    scenes = []
    for s in sorted(p.scenes, key=lambda s: s.order):
        dur = s.duration or 3
        if s.animate and s.clip_path:
            vf = f"trim=duration={dur},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"  # ANM-5 trim/extend
        else:
            frames = int(dur * 30)
            vf = f"zoompan=z='min(zoom+0.0009,1.15)':d={frames}:s={w}x{h}:fps=30"  # zoom and pan
        scenes.append({"scene": s.id, "input": s.clip_path or s.image_path, "duration": dur, "filter": vf,
                       "audio": [p.line(i).audio_path for i in s.line_ids]})
    music = None
    if p.audio.music_path:
        duck = "sidechaincompress=threshold=0.05:ratio=8:attack=20:release=300" if p.audio.duck else None
        music = {"path": p.audio.music_path, "volume": p.audio.music_volume, "duck": duck}
    return {
        "size": [w, h], "fps": 30, "transition": "xfade=transition=fade:duration=0.25", "scenes": scenes, "music": music,
        "captions": {"style": CAPTION_STYLES[p.scenes[0].caption_style if p.scenes else "bold"], "source": "per-line audio timings"},
        "polish": {"loudnorm": "loudnorm=I=-14:TP=-1.5:LRA=11", "safe_area": "MarginV keeps captions out of platform UI"} if polish else None,
        "encode": ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"],
    }


async def render(p: Project, out: Path, height: int, polish: bool) -> None:
    spec = plan(p, height, polish)
    real_media = all(sc["input"] and Path(sc["input"]).suffix in (".png", ".jpg", ".mp4") and Path(sc["input"]).exists()
                     and not Path(sc["input"]).read_bytes()[:8].startswith(b'{"mock"') for sc in spec["scenes"])
    if not (shutil.which("ffmpeg") and real_media):
        out.with_suffix(".plan.json").write_text(json.dumps(spec, indent=1))
        out.write_text(json.dumps({"mock": "video", "plan": str(out.with_suffix(".plan.json"))}))
        await asyncio.sleep(0)
        return
    # Real path: render each scene segment, then concatenate with crossfades and mix audio.
    # Left as the next step once real image/voice adapters produce media files.
    raise NotImplementedError("Real FFmpeg rendering is wired up after the first real adapters land.")
