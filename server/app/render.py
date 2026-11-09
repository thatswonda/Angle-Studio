"""Real assembly with FFmpeg (ASM-1..4, ASM-6): one zoom-and-pan clip per scene image, the narration
track, burned-in captions timed to the narration, and loudness normalization on the final render."""
from __future__ import annotations

import base64
import hashlib
import re
import shutil
import subprocess
import time
import uuid
import wave
from pathlib import Path
from typing import Callable

from .providers.google import ProviderError, clean_pcm, trim_silence, write_wav

FPS = 30
# Keeps each FFmpeg run well under 512 MB (Render Starter): few encoder threads and a short lookahead.
# Measured on a 1080x1920 render: about 270 MB peak (it was about 680 MB with default threads).
LOWMEM_CLIP = ["-threads", "2", "-filter_threads", "1", "-x264-params", "rc-lookahead=10:threads=2"]


# FFmpeg runs at low priority so the web server keeps answering health checks while a long render uses the CPU
# (a starved health check makes the host restart the app mid-render).
NICE = ["nice", "-n", "15"] if shutil.which("nice") else []


def _run(args: list[str], cwd: Path) -> None:
    p = subprocess.run([*NICE, "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], cwd=cwd, capture_output=True, text=True)
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


# Words per caption chunk for each style.
CAPTION_WORDS = {"bold": 3, "clean": 12, "karaoke": 7, "word": 1, "boxed": 3, "neon": 4, "comic": 2, "typewriter": 6, "minimal": 10}
CAPTION_STYLE_IDS = set(CAPTION_WORDS) | {"none"}


def _style_line(style: str, w: int, h: int) -> str:
    base = min(w, h)
    f = "Style: Default,{font},{size},{primary},{secondary},{outline},{back},{bold},0,0,0,100,100,{spacing},{angle},{border},{ol},{sh},{align},60,60,{mv},1"
    styles = {
        "clean": dict(font="DejaVu Sans", size=int(base * 0.042), primary="&H00FFFFFF", outline="&H00000000", back="&H99000000", bold=0, spacing=0, angle=0, border=3, ol=10, sh=0, align=2, mv=int(h * 0.07)),
        "karaoke": dict(font="DejaVu Sans", size=int(base * 0.05), primary="&H005CE4FF", outline="&H00000000", back="&H00000000", bold=1, spacing=0, angle=0, border=1, ol=4, sh=1, align=2, mv=int(h * 0.16)),
        "word": dict(font="DejaVu Sans", size=int(base * 0.11), primary="&H00FFFFFF", outline="&H00000000", back="&H80000000", bold=1, spacing=0, angle=0, border=1, ol=7, sh=3, align=5, mv=0),
        "boxed": dict(font="DejaVu Sans", size=int(base * 0.058), primary="&H00141414", outline="&H005CE4FF", back="&H005CE4FF", bold=1, spacing=0, angle=0, border=3, ol=14, sh=0, align=2, mv=int(h * 0.17)),
        "neon": dict(font="DejaVu Sans", size=int(base * 0.066), primary="&H00FFFFFF", outline="&H00FFF22E", back="&H00000000", bold=1, spacing=1, angle=0, border=1, ol=6, sh=0, align=2, mv=int(h * 0.16)),
        "comic": dict(font="DejaVu Sans", size=int(base * 0.085), primary="&H0000E5FF", outline="&H00000000", back="&H00000000", bold=1, spacing=1, angle=4, border=1, ol=8, sh=5, align=2, mv=int(h * 0.2)),
        "typewriter": dict(font="DejaVu Sans Mono", size=int(base * 0.046), primary="&H00FFFFFF", outline="&H40000000", back="&H40000000", bold=0, spacing=0, angle=0, border=3, ol=12, sh=0, align=2, mv=int(h * 0.1)),
        "minimal": dict(font="DejaVu Sans", size=int(base * 0.038), primary="&H00FFFFFF", outline="&H00000000", back="&H64000000", bold=0, spacing=1, angle=0, border=1, ol=0, sh=2, align=2, mv=int(h * 0.06)),
    }
    bold = dict(font="DejaVu Sans", size=int(base * 0.075), primary="&H00FFFFFF", outline="&H00000000", back="&H00000000", bold=1, spacing=0, angle=0, border=1, ol=6, sh=2, align=2, mv=int(h * 0.18))
    st = styles.get(style, bold)
    return f.format(secondary="&H00FFFFFF" if style == "karaoke" else st["primary"], **st)


def captions_ass(cues: list[tuple[float, float, list[str]]], style: str, w: int, h: int) -> str:
    out = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 0", "",
           "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
           "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           _style_line(style, w, h), "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]

    def ev(a: float, b: float, text: str) -> None:
        out.append(f"Dialogue: 0,{_ts(a)},{_ts(b)},Default,,0,0,0,,{text}")

    for a, b, ch in cues:
        words = [_ass_text(x) for x in ch]
        if style == "karaoke":
            k = max(1, int((b - a) * 100 / len(ch)))
            ev(a, b, "".join("{\\k%d}%s " % (k, x) for x in words).strip())
        elif style == "typewriter":  # words appear one by one
            step = (b - a) / len(words)
            for i in range(len(words)):
                ev(a + i * step, a + (i + 1) * step if i < len(words) - 1 else b, " ".join(words[:i + 1]))
        elif style == "word":
            ev(a, b, "{\\fscx135\\fscy135\\t(0,140,\\fscx100\\fscy100)}" + " ".join(words).upper())
        elif style == "neon":
            ev(a, b, "{\\blur9}" + " ".join(words))
        elif style in ("bold", "comic"):
            ev(a, b, " ".join(words).upper())
        elif style == "minimal":
            ev(a, b, " ".join(words).lower())
        else:
            ev(a, b, " ".join(words))
    return "\n".join(out) + "\n"


def captions_srt(cues: list[tuple[float, float, list[str]]]) -> str:
    return "".join(f"{n}\n{_srt_ts(a)} --> {_srt_ts(b)}\n{' '.join(ch)}\n\n" for n, (a, b, ch) in enumerate(cues, 1))


def _clean_clip(src: Path, dst: Path) -> Path:
    """Every voice clip is cleaned before joining: older clips carry a WAV header inside their samples (stripped),
    and the silence before and after the words is trimmed so one sentence follows the next with no waiting."""
    with wave.open(str(src)) as w:
        rate, raw, width, ch = w.getframerate(), w.readframes(w.getnframes()), w.getsampwidth(), w.getnchannels()
    if width != 2 or ch != 1:  # not a clip the app made: use it as it is
        return src
    if raw[:4] == b"RIFF":
        raw, rate = clean_pcm(raw, rate)
    write_wav(dst, trim_silence(raw, rate), rate)
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
    res = int(payload.get("res") or (1080 if final else 480))
    res = res if res in (480, 720, 1080) else 1080
    w, h = _size(payload.get("aspect", "9:16"), res)
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

    stem = f"{'final' if final else 'draft'}_{uuid.uuid4().hex[:8]}"
    cues = caption_cues(scenes, [l.get("text", "") for l in lines], line_times, starts, durs,
                        CAPTION_WORDS.get(style, 6))
    srt_url = None
    if cues:
        (media / "renders" / f"{stem}.srt").write_text(captions_srt(cues))
        srt_url = f"/media/renders/{stem}.srt"
    burn = style != "none" and bool(cues)

    # One zoom-and-pan clip per scene, with that scene's captions burned in, so each frame is encoded only once.
    # Finished clips are kept on disk by content, so a render that was interrupted continues where it stopped,
    # and scenes that did not change are reused by the next render.
    cache = media / "renders" / "clips"
    cache.mkdir(parents=True, exist_ok=True)
    for old in cache.glob("*.mp4"):
        if time.time() - old.stat().st_mtime > 3 * 86400:
            old.unlink(missing_ok=True)
    big_w, big_h = round(w * 1.25 / 2) * 2, round(h * 1.25 / 2) * 2
    crf = "20" if final else "28"
    seg_list, reused = [], 0
    for i, (sc, a, d) in enumerate(zip(scenes, starts, durs)):
        f0, f1 = round(a * FPS), round((a + d) * FPS)
        frames = max(1, f1 - f0)
        t0, t1 = f0 / FPS, f0 / FPS + frames / FPS
        if sc.get("variant"):  # second scene on a reused image: start close on another part, then pull back
            zp = f"zoompan=z='max(1.25-0.2*on/{frames},1.05)':x='(iw-iw/zoom)*(0.85-0.5*on/{frames})':y='(ih-ih/zoom)*0.35'"
        elif i % 2 == 0:
            zp = f"zoompan=z='min(1+0.12*on/{frames},1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        else:
            zp = f"zoompan=z='1.1':x='(iw-iw/zoom)*on/{frames}':y='ih/2-(ih/zoom/2)'"
        vf = f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,crop={big_w}:{big_h},{zp}:d={frames}:s={w}x{h}:fps={FPS},format=yuv420p"
        ass = ""
        if burn:  # this scene's slice of the captions, timed from the start of the clip
            part = [(max(0.0, ca - t0), min(cb, t1) - t0, ch) for ca, cb, ch in cues if cb > t0 + 1e-3 and ca < t1 - 1e-3]
            if part:
                ass = captions_ass(part, style, w, h)
                (work / f"cap{i}.ass").write_text(ass)
                vf += f",ass=cap{i}.ass"
        img = local(sc.get("image", ""), f"img{i}")
        key = hashlib.sha1(hashlib.sha1(img.read_bytes()).digest() + f"|{vf}|{frames}|{crf}|{ass}".encode()).hexdigest()[:24]
        clip = cache / f"{key}.mp4"
        if clip.exists():
            reused += 1
            progress(f"Scene {i + 1} of {len(scenes)}: already made")
        else:
            progress(f"Scene {i + 1} of {len(scenes)}: zoom, pan{' and captions' if ass else ''}")
            tmp = work / f"seg{i:04d}.mp4"
            _run(["-i", str(img), "-vf", vf, "-frames:v", str(frames),
                  "-c:v", "libx264", "-preset", "veryfast", "-crf", crf, "-r", str(FPS), *LOWMEM_CLIP, tmp.name], work)
            shutil.move(str(tmp), clip)  # only a finished clip lands in the cache
        seg_list.append(clip)
    (work / "video.txt").write_text("".join(f"file '{c}'\n" for c in seg_list))
    _run(["-f", "concat", "-safe", "0", "-i", "video.txt", "-c", "copy", "video.mp4"], work)

    voice = "narration.wav"
    if final:  # loudness in its own audio-only pass
        progress("Normalizing loudness")
        _run(["-i", "narration.wav", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "pcm_s16le", "voice_norm.wav"], work)
        voice = "voice_norm.wav"
    progress("Joining the scenes and the voice")
    _run(["-i", "video.mp4", "-i", voice, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
          "-movflags", "+faststart", "-shortest", f"../{stem}.mp4"], work)
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    return {"url": f"/media/renders/{stem}.mp4", "srt": srt_url, "res": res, "reused": reused,
            "line_secs": [round(x, 3) for x in line_secs]}  # the real length of each sentence, as heard in the video
