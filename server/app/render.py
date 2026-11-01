"""Real assembly with FFmpeg (ASM-1..4, ASM-6): one zoom-and-pan clip per scene image, the narration
track, burned-in captions timed to the narration, and loudness normalization on the final render."""
from __future__ import annotations

import base64
import re
import subprocess
import uuid
import wave
from pathlib import Path
from typing import Callable

from .providers.google import ProviderError

FPS = 30


def _run(args: list[str], cwd: Path) -> None:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise ProviderError("FFmpeg failed: " + (p.stderr.strip().splitlines() or ["unknown error"])[-1][:300])


def _size(aspect: str, short: int) -> tuple[int, int]:
    a, b = (int(x) for x in aspect.split(":"))
    w, h = (round(short * a / b / 2) * 2, short) if a >= b else (short, round(short * b / a / 2) * 2)
    return w, h


def _ts(t: float) -> str:
    h, rem = divmod(max(0.0, t), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _ass_text(t: str) -> str:
    return t.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def captions(scenes: list[dict], starts: list[float], durs: list[float], style: str, w: int, h: int) -> str:
    base = min(w, h)
    if style == "clean":
        st = f"Style: Default,DejaVu Sans,{int(base * 0.042)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,0,0,0,0,100,100,0,0,3,10,0,2,60,60,{int(h * 0.07)},1"
    elif style == "karaoke":
        st = f"Style: Default,DejaVu Sans,{int(base * 0.05)},&H005CE4FF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,4,1,2,60,60,{int(h * 0.16)},1"
    else:
        st = f"Style: Default,DejaVu Sans,{int(base * 0.075)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,6,2,2,60,60,{int(h * 0.18)},1"
    out = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 0", "",
           "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
           "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           st, "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    per = {"bold": 3, "clean": 12, "karaoke": 7}.get(style, 3)
    for sc, t0, d in zip(scenes, starts, durs):
        ws = (sc.get("text") or "").split()
        if not ws:
            continue
        chunks = [ws[i:i + per] for i in range(0, len(ws), per)]
        t = t0
        for ch in chunks:
            cd = d * len(ch) / len(ws)
            if style == "karaoke":
                k = max(1, int(cd * 100 / len(ch)))
                text = "".join("{\\k%d}%s " % (k, _ass_text(x)) for x in ch).strip()
            elif style == "bold":
                text = _ass_text(" ".join(ch).upper())
            else:
                text = _ass_text(" ".join(ch))
            out.append(f"Dialogue: 0,{_ts(t)},{_ts(t + cd)},Default,,0,0,0,,{text}")
            t += cd
    return "\n".join(out) + "\n"


def _wav_seconds(p: Path) -> float:
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


def render(payload: dict, media: Path, progress: Callable[[str], None]) -> str:
    """payload: {final, aspect, caption_style, scenes: [{image, duration, text}], lines: [{audio}]} -> media URL."""
    final = bool(payload.get("final"))
    w, h = _size(payload.get("aspect", "9:16"), 1080 if final else 480)
    scenes = payload.get("scenes") or []
    lines = payload.get("lines") or []
    if not scenes:
        raise ProviderError("There are no scenes to render.")
    work = media / "renders" / f"work_{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=True)

    def local(ref: str, name: str) -> Path:
        if ref.startswith("data:"):
            head, b64 = ref.split(",", 1)
            ext = ".png" if "png" in head else ".jpg"
            p = work / f"{name}{ext}"
            p.write_bytes(base64.b64decode(b64))
            return p
        rel = re.sub(r"^/media/", "", ref or "")
        p = (media / rel).resolve()
        if not rel or media.resolve() not in p.parents or not p.exists():
            raise ProviderError(f"A file the render needs is missing ({ref or 'no file'}). Regenerate it and try again.")
        return p

    # Narration: all sentence clips in story order. Its length is the video's length.
    progress("Joining the narration")
    if not lines or any(not l.get("audio") for l in lines):
        raise ProviderError("Every sentence needs its voiceover before rendering. Finish the Voice step first.")
    (work / "voice.txt").write_text("".join(f"file '{local(l['audio'], f'a{i}')}'\n" for i, l in enumerate(lines)))
    _run(["-f", "concat", "-safe", "0", "-i", "voice.txt", "-c:a", "pcm_s16le", "narration.wav"], work)
    total = _wav_seconds(work / "narration.wav")

    est = [max(0.5, float(s.get("duration") or 1)) for s in scenes]
    scale = total / sum(est)
    durs = [d * scale for d in est]
    starts = [sum(durs[:i]) for i in range(len(durs))]

    # One zoom-and-pan clip per scene; even scenes push in, odd scenes drift sideways.
    big_w, big_h = round(w * 1.25 / 2) * 2, round(h * 1.25 / 2) * 2
    seg_list = []
    for i, (sc, d) in enumerate(zip(scenes, durs)):
        progress(f"Scene {i + 1} of {len(scenes)}: zoom and pan")
        frames = max(2, round(d * FPS))
        if i % 2 == 0:
            zp = f"zoompan=z='min(1+0.12*on/{frames},1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        else:
            zp = f"zoompan=z='1.1':x='(iw-iw/zoom)*on/{frames}':y='ih/2-(ih/zoom/2)'"
        vf = f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,crop={big_w}:{big_h},{zp}:d={frames}:s={w}x{h}:fps={FPS},format=yuv420p"
        seg = f"seg{i:04d}.mp4"
        _run(["-i", str(local(sc.get("image", ""), f"img{i}")), "-vf", vf, "-frames:v", str(frames),
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "20" if final else "28", "-r", str(FPS), seg], work)
        seg_list.append(seg)
    (work / "video.txt").write_text("".join(f"file '{s}'\n" for s in seg_list))
    _run(["-f", "concat", "-safe", "0", "-i", "video.txt", "-c", "copy", "video.mp4"], work)

    progress("Burning in captions")
    (work / "captions.ass").write_text(captions(scenes, starts, durs, payload.get("caption_style", "bold"), w, h))
    progress("Normalizing loudness and encoding" if final else "Encoding the draft")
    name = f"{'final' if final else 'draft'}_{uuid.uuid4().hex[:8]}.mp4"
    af = ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"] if final else []
    _run(["-i", "video.mp4", "-i", "narration.wav", "-vf", "ass=captions.ass", *af,
          "-c:v", "libx264", "-preset", "medium" if final else "veryfast", "-crf", "20" if final else "28",
          "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", f"../{name}"], work)
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    return f"/media/renders/{name}"
