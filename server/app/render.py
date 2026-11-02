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

from .providers.google import ProviderError, clean_pcm, write_wav

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


def _srt_ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def caption_cues(scenes: list[dict], line_texts: list[str], line_times: list[tuple[float, float]],
                 starts: list[float], durs: list[float], per: int) -> list[tuple[float, float, list[str]]]:
    """Timed word groups. Untouched captions follow each sentence's own audio; edited ones spread over their scene."""
    cues: list[tuple[float, float, list[str]]] = []

    def spread(words: list[str], t0: float, d: float) -> None:
        if not words or d <= 0:
            return
        t = t0
        for i in range(0, len(words), per):
            ch = words[i:i + per]
            cd = d * len(ch) / len(words)
            cues.append((t, t + cd, ch))
            t += cd

    for sc, t0, d in zip(scenes, starts, durs):
        text = (sc.get("caption") or "").strip()
        lines = sc.get("lines") or []
        if sc.get("caption_auto", True) and lines and not sc.get("split") and all(0 <= i < len(line_texts) for i in lines):
            for i in lines:
                a, b = line_times[i]
                spread(line_texts[i].split(), a, b - a)
        else:
            spread(text.split(), t0, d)
    return cues


def captions_ass(cues: list[tuple[float, float, list[str]]], style: str, w: int, h: int) -> str:
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
    for a, b, ch in cues:
        if style == "karaoke":
            k = max(1, int((b - a) * 100 / len(ch)))
            text = "".join("{\\k%d}%s " % (k, _ass_text(x)) for x in ch).strip()
        elif style == "bold":
            text = _ass_text(" ".join(ch).upper())
        else:
            text = _ass_text(" ".join(ch))
        out.append(f"Dialogue: 0,{_ts(a)},{_ts(b)},Default,,0,0,0,,{text}")
    return "\n".join(out) + "\n"


def captions_srt(cues: list[tuple[float, float, list[str]]]) -> str:
    return "".join(f"{n}\n{_srt_ts(a)} --> {_srt_ts(b)}\n{' '.join(ch)}\n\n" for n, (a, b, ch) in enumerate(cues, 1))


def _clean_clip(src: Path, dst: Path) -> Path:
    """Voice clips saved before the fix carry a WAV header and metadata inside their samples; strip them."""
    with wave.open(str(src)) as w:
        rate, raw = w.getframerate(), w.readframes(w.getnframes())
    if raw[:4] != b"RIFF":
        return src
    pcm, rate = clean_pcm(raw, rate)
    write_wav(dst, pcm, rate)
    return dst


def _wav_seconds(p: Path) -> float:
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


def scene_timing(scenes: list[dict], line_secs: list[float]) -> tuple[list[float], list[float]] | None:
    """Exact scene lengths from the measured voice clips: a scene lasts as long as its own sentences
    (a split sentence is shared equally by its parts). None if the scenes do not cover every sentence once, in order."""
    durs, cover, order = [], [0.0] * len(line_secs), []
    for sc in scenes:
        idx = sc.get("lines") or []
        if not idx or any(not 0 <= i < len(line_secs) for i in idx):
            return None
        of = (sc.get("split") or {}).get("of") or 1
        durs.append(sum(line_secs[i] for i in idx) / of)
        for i in idx:
            cover[i] += 1 / of
            if not order or order[-1] != i:
                order.append(i)
    if any(abs(c - 1) > 1e-6 for c in cover) or order != sorted(order):
        return None
    starts = [sum(durs[:i]) for i in range(len(durs))]
    return starts, durs


def render(payload: dict, media: Path, progress: Callable[[str], None]) -> dict:
    """payload: {final, aspect, caption_style, scenes: [{image, lines: [sentence indexes], split, caption, caption_auto}],
    lines: [{audio, text}]} -> {"url": video, "srt": captions file or None}."""
    final = bool(payload.get("final"))
    w, h = _size(payload.get("aspect", "9:16"), 1080 if final else 480)
    scenes = payload.get("scenes") or []
    lines = payload.get("lines") or []
    style = payload.get("caption_style", "bold")
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

    # Narration: every sentence clip in story order, measured one by one so each image matches its own words.
    progress("Measuring the narration")
    if not lines or any(not l.get("audio") for l in lines):
        raise ProviderError("Every sentence needs its voiceover before rendering. Finish the Voice step first.")
    clips = [_clean_clip(local(l["audio"], f"a{i}"), work / f"clean{i}.wav") for i, l in enumerate(lines)]
    line_secs = [_wav_seconds(c) for c in clips]
    line_times, t = [], 0.0
    for d in line_secs:
        line_times.append((t, t + d))
        t += d
    (work / "voice.txt").write_text("".join(f"file '{c}'\n" for c in clips))
    _run(["-f", "concat", "-safe", "0", "-i", "voice.txt", "-c:a", "pcm_s16le", "narration.wav"], work)

    timing = scene_timing(scenes, line_secs)
    if timing:
        starts, durs = timing
    else:  # scenes edited out of story order: fall back to their estimated lengths, scaled to the narration
        est = [max(0.5, float(s.get("duration") or 1)) for s in scenes]
        durs = [d * t / sum(est) for d in est]
        starts = [sum(durs[:i]) for i in range(len(durs))]

    # One zoom-and-pan clip per scene. Frame counts come from rounded start and end times, so cuts never drift.
    big_w, big_h = round(w * 1.25 / 2) * 2, round(h * 1.25 / 2) * 2
    seg_list = []
    for i, (sc, a, d) in enumerate(zip(scenes, starts, durs)):
        progress(f"Scene {i + 1} of {len(scenes)}: zoom and pan")
        frames = max(1, round((a + d) * FPS) - round(a * FPS))
        if sc.get("variant"):  # second scene on a reused image: start close on another part, then pull back
            zp = f"zoompan=z='max(1.25-0.2*on/{frames},1.05)':x='(iw-iw/zoom)*(0.85-0.5*on/{frames})':y='(ih-ih/zoom)*0.35'"
        elif i % 2 == 0:
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

    stem = f"{'final' if final else 'draft'}_{uuid.uuid4().hex[:8]}"
    cues = caption_cues(scenes, [l.get("text", "") for l in lines], line_times, starts, durs,
                        {"bold": 3, "clean": 12, "karaoke": 7}.get(style, 6))
    srt_url = None
    if cues:
        (media / "renders" / f"{stem}.srt").write_text(captions_srt(cues))
        srt_url = f"/media/renders/{stem}.srt"
    vf_caps: list[str] = []
    if style != "none" and cues:
        progress("Burning in captions")
        (work / "captions.ass").write_text(captions_ass(cues, style, w, h))
        vf_caps = ["-vf", "ass=captions.ass"]
    progress("Normalizing loudness and encoding" if final else "Encoding the draft")
    af = ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"] if final else []
    _run(["-i", "video.mp4", "-i", "narration.wav", *vf_caps, *af,
          "-c:v", "libx264", "-preset", "medium" if final else "veryfast", "-crf", "20" if final else "28",
          "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", f"../{stem}.mp4"], work)
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    return {"url": f"/media/renders/{stem}.mp4", "srt": srt_url}
