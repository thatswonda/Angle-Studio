"""Google Gemini API: watches the reference video, writes (optional), speaks the narration and draws scene images.

API keys come from the server environment (GEMINI_API_KEY) and are never sent to the browser.
"""
from __future__ import annotations

import array
import asyncio
import base64
import json
import os
import re
import sys
import wave
from pathlib import Path

import httpx

BASE = "https://generativelanguage.googleapis.com/v1beta"

# App model ids -> Gemini API model names available on the key (checked November 2026).
MODELS = {
    "gemini-pro": "gemini-pro-latest",
    # gemini-flash-latest was overloaded (503) and 3.7/3.8 Flash took ~2 minutes per call in testing; 3.5 Flash answers in seconds.
    "gemini-flash": os.getenv("GEMINI_FLASH_MODEL", "gemini-3.5-flash"),
    "gemini-pro-writer": "gemini-pro-latest",
    "gemini-flash-writer": os.getenv("GEMINI_FLASH_MODEL", "gemini-3.5-flash"),
    "gemini-tts-flash": "gemini-3.8-flash-tts",
    "gemini-tts-pro": "gemini-2.5-pro-preview-tts",
    "nano-banana-2": "gemini-3.1-flash-image",
    "nano-banana-pro": "gemini-3-pro-image",
    "nano-banana-2-lite": "gemini-3.1-flash-lite-image",
}
# Rough USD per 1M tokens (input, output), only for the app's spend counter. Confirm on Google's pricing page.
TOKEN_PRICES = {"pro": (2.0, 12.0), "flash": (0.30, 2.50)}
IMAGE_PRICES = {"nano-banana-2": 0.07, "nano-banana-pro": 0.14, "nano-banana-2-lite": 0.03}
TTS_PRICES_PER_1K_CHARS = {"gemini-tts-flash": 0.01, "gemini-tts-pro": 0.02}
# Gemini TTS prebuilt voices; the app's ids are "v-" + the lowercase name.
VOICE_NAMES = {"v-" + n.lower(): n for n in (
    "Aoede", "Kore", "Charon", "Puck", "Fenrir", "Zephyr", "Leda", "Orus", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus",
    "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar",
    "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat")}

_sem = asyncio.Semaphore(int(os.getenv("GEMINI_CONCURRENCY", "4")))


class ProviderError(Exception):
    """A provider call failed; the message is shown to the user as is."""


def _retry_after(r: httpx.Response) -> float | None:
    try:
        for d in r.json().get("error", {}).get("details", []):
            if "retryDelay" in d:
                return float(str(d["retryDelay"]).rstrip("s"))
    except Exception:
        pass
    return None


def _error_text(r: httpx.Response, model: str) -> str:
    try:
        msg = r.json().get("error", {}).get("message", "")
    except Exception:
        msg = r.text[:300]
    if r.status_code in (401, 403):
        return f"Gemini refused the request ({r.status_code}). Check that GEMINI_API_KEY is valid and has billing enabled. {msg}"
    if r.status_code == 402:
        return ("Your Gemini prepaid credits have run out. Top up in Google AI Studio (ai.studio/projects, then Billing) "
                "and try again.")
    if r.status_code == 429:
        return f"Gemini rate limit or quota reached for {model}. Wait a minute and retry. {msg}"
    return f"Gemini error {r.status_code} on {model}: {msg}"


def _quota_text(model: str, r: httpx.Response) -> str:
    if "free_tier" in r.text and "limit: 0" in r.text:
        hint = "" if ("image" in model or "tts" in model) else ", or pick a Flash model"
        return (f"Your Gemini key is on the free tier, which does not include {model}. Turn on billing for the key in "
                f"Google AI Studio (aistudio.google.com, then Billing){hint}.")
    return f"The daily Gemini quota for {model} is used up. It resets within 24 hours, or turn on billing in Google AI Studio for higher limits."


async def generate(model_id: str, body: dict, timeout: float = 600.0) -> dict:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY is not set on the server.")
    model = MODELS.get(model_id, model_id)
    url = f"{BASE}/models/{model}:generateContent"
    async with _sem:
        for attempt in range(6):
            try:
                async with httpx.AsyncClient(timeout=timeout) as c:
                    r = await c.post(url, json=body, headers={"x-goog-api-key": key})
            except httpx.HTTPError as e:
                if attempt == 5:
                    raise ProviderError(f"Could not reach Gemini: {e}") from e
                await asyncio.sleep(2 ** attempt * 2)
                continue
            wait = _retry_after(r) if r.status_code == 429 else None
            if r.status_code == 429 and ("limit: 0" in r.text or (wait or 0) > 120):
                raise ProviderError(_quota_text(model, r))  # a zero quota or an hours-long wait: retrying cannot help
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                await asyncio.sleep(min(90.0, wait or 2 ** attempt * 3))  # per-minute limits clear within a minute
                continue
            if r.status_code != 200:
                raise ProviderError(_error_text(r, model))
            data = r.json()
            if not data.get("candidates"):
                reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
                raise ProviderError(f"Gemini returned nothing ({reason}).")
            return data
    raise ProviderError("Gemini did not answer.")


def _parts(resp: dict) -> list[dict]:
    cand = resp["candidates"][0]
    parts = cand.get("content", {}).get("parts", [])
    if not parts:
        raise ProviderError(f"Gemini returned no content (finish reason: {cand.get('finishReason', 'unknown')}).")
    return parts


def text_of(resp: dict) -> str:
    return "".join(p.get("text", "") for p in _parts(resp) if not p.get("thought"))


def parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start:end + 1])
        raise


def token_cost(model_id: str, resp: dict) -> float:
    u = resp.get("usageMetadata", {})
    tier = "flash" if "flash" in MODELS.get(model_id, model_id) else "pro"
    pin, pout = TOKEN_PRICES[tier]
    out = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    return (u.get("promptTokenCount", 0) * pin + out * pout) / 1e6


async def generate_json(model_id: str, parts: list[dict], max_tokens: int = 32768, low_media: bool = False) -> tuple[dict, float]:
    """Asks for JSON. If the answer does not parse, a cheap text-only Flash call repairs it,
    so the video (or other large input) is never sent twice."""
    cfg = {"responseMimeType": "application/json", "maxOutputTokens": max_tokens}
    if low_media:
        cfg["mediaResolution"] = "MEDIA_RESOLUTION_LOW"  # about a third of the video tokens; plenty for analysis
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg}
    resp = await generate(model_id, body)
    cost = token_cost(model_id, resp)
    raw = text_of(resp)
    try:
        return parse_json(raw), cost
    except json.JSONDecodeError:
        pass
    fix = {"contents": [{"role": "user", "parts": [{"text": "Turn this into valid JSON. Keep every key and value, change "
                                                            "nothing else, and return only the JSON:\n" + raw[:200000]}]}],
           "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": max_tokens}}
    resp = await generate("gemini-flash", fix)
    cost += token_cost("gemini-flash", resp)
    try:
        return parse_json(text_of(resp)), cost
    except json.JSONDecodeError:
        raise ProviderError("Gemini's answer was not valid JSON, even after a repair. Try again.")


def youtube_part(url: str) -> dict:
    return {"fileData": {"fileUri": url, "mimeType": "video/*"}}


class GeminiWriter:
    """Writer adapter using Gemini, for when the main writer is unavailable."""

    def __init__(self, model_id: str = "gemini-pro-writer"):
        self.model_id = model_id

    async def complete_json(self, system: str, prompt: str) -> tuple[dict, float]:
        return await generate_json(self.model_id, [{"text": system + "\n\n" + prompt}])


# Google limits each voice model to a number of requests per day (separate from billing). When one model's
# daily limit is used up, the next one on the key reads instead; each has its own limit and the same voices.
TTS_CHAIN = {
    "gemini-tts-flash": ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts", "gemini-3.1-flash-tts-preview", "gemini-2.5-pro-preview-tts"],
    "gemini-tts-pro": ["gemini-2.5-pro-preview-tts", "gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts", "gemini-3.1-flash-tts-preview"],
}
_tts_out: dict[str, float] = {}  # API model -> time its daily limit was found used up (skipped for 3 hours)


def _voice_body(prompt: str, voice_id: str) -> dict:
    return {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE_NAMES.get(voice_id, "Aoede")}}}}}


async def _voice(prompt: str, voice_id: str, model_id: str) -> tuple[bytes, int]:
    """One TTS request, moving to the next voice model when one has used up its daily request limit."""
    import time
    chain = TTS_CHAIN.get(model_id, [MODELS.get(model_id, model_id)])
    last = None
    for api_model in chain:
        if time.time() - _tts_out.get(api_model, 0) < 3 * 3600:
            continue
        try:
            resp = await generate(api_model, _voice_body(prompt, voice_id), timeout=180.0)
        except ProviderError as e:
            if "quota" in str(e).lower():
                _tts_out[api_model] = time.time()
                last = e
                continue
            raise
        inline = next((p["inlineData"] for p in _parts(resp) if "inlineData" in p), None)
        if not inline:
            raise ProviderError("Gemini TTS returned no audio.")
        m = re.search(r"rate=(\d+)", inline.get("mimeType", ""))
        return clean_pcm(base64.b64decode(inline["data"]), int(m.group(1)) if m else 24000)
    raise ProviderError("Every Gemini voice model on this key has used today's request quota (Google limits voice requests "
                        "per day, separately from your credit). It resets within 24 hours.")


def _prompt(delivery: str, transcript: str, extra: str = "") -> str:
    # Director's-notes format: the style guides the voice without being read aloud (a plain "Say warmly:" prefix gets spoken).
    return f"# AUDIO PROFILE: Narrator\n### DIRECTOR'S NOTES\nStyle: {delivery}{extra}\n#### TRANSCRIPT\n{transcript}"


async def speak(text: str, voice_id: str, model_id: str, out: Path, delivery: str) -> tuple[float, float]:
    """Text to speech. Writes a 16-bit mono WAV and returns (seconds, cost)."""
    pcm, rate = await _voice(_prompt(delivery, text), voice_id, model_id)
    pcm = trim_silence(pcm, rate)
    write_wav(out, pcm, rate)
    return len(pcm) / 2 / rate, len(text) / 1000 * TTS_PRICES_PER_1K_CHARS.get(model_id, 0.01)


def split_at_pauses(pcm: bytes, rate: int, texts: list[str]) -> list[bytes] | None:
    """Cuts one recording of several sentences back into one clip per sentence, at the longest pauses. Returns None
    when the pauses do not line up with the sentences (then each sentence is read on its own instead)."""
    n = len(texts)
    if n == 1:
        return [pcm]
    samples = array.array("h", pcm[: len(pcm) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    win = max(1, rate // 100)  # 10 ms
    levels = [max((abs(x) for x in samples[i:i + win]), default=0) for i in range(0, len(samples), win)]
    peak = max(levels, default=0)
    if peak < 300:
        return None
    thr = max(300, peak * 0.04)
    loud = [v > thr for v in levels]
    first = loud.index(True)
    last = len(loud) - 1 - loud[::-1].index(True)
    gaps, k = [], first
    while k <= last:  # quiet runs inside the speech: (length, start, end) in 10 ms steps
        if not loud[k]:
            a = k
            while k <= last and not loud[k]:
                k += 1
            gaps.append((k - a, a, k))
        else:
            k += 1
    if len(gaps) < n - 1:
        return None
    chosen = sorted(sorted(gaps, reverse=True)[: n - 1], key=lambda g: g[1])
    if chosen[-1][0] < 22:  # the shortest cut is under 0.22 s: probably a comma, not the end of a sentence
        return None
    cuts = [first] + [(g[1] + g[2]) // 2 for g in chosen] + [last + 1]
    lens = [cuts[i + 1] - cuts[i] for i in range(n)]
    total_chars = sum(max(1, len(t)) for t in texts)
    for t, ln in zip(texts, lens):  # each piece must be about as long as its sentence
        want = (last + 1 - first) * max(1, len(t)) / total_chars
        if not 0.45 * want - 25 <= ln <= 2.2 * want + 40:
            return None
    out = []
    for i in range(n):
        piece = samples[cuts[i] * win: cuts[i + 1] * win]
        if sys.byteorder == "big":
            piece.byteswap()
        out.append(piece.tobytes())
    return out


async def speak_many(texts: list[str], voice_id: str, model_id: str, outs: list[Path], delivery: str) -> tuple[list[float], float]:
    """Reads several sentences in ONE request (so Google's daily request limit lasts several times longer), then cuts
    them apart at the pauses. If the cut does not line up, each sentence is read on its own. Returns (seconds, cost)."""
    cost = sum(len(t) for t in texts) / 1000 * TTS_PRICES_PER_1K_CHARS.get(model_id, 0.01)
    pieces = None
    if len(texts) > 1:
        transcript = "\n\n".join(t.strip() for t in texts)
        pcm, rate = await _voice(_prompt(delivery, transcript, "\nRead each paragraph as its own sentence, with a clear pause of about one "
                                                               "second between paragraphs and no pause longer than that inside a sentence."), voice_id, model_id)
        pieces = split_at_pauses(pcm, rate, texts)
    if pieces is None:
        secs = []
        for t, out in zip(texts, outs):
            d, _ = await speak(t, voice_id, model_id, out, delivery)
            secs.append(d)
        return secs, cost
    secs = []
    for piece, out in zip(pieces, outs):
        clip = trim_silence(piece, rate)
        write_wav(out, clip, rate)
        secs.append(len(clip) / 2 / rate)
    return secs, cost


def clean_pcm(raw: bytes, rate: int = 24000) -> tuple[bytes, int]:
    """Returns only the 16-bit mono samples. Some TTS models send a whole WAV file (a RIFF header in front and an
    AI-provenance metadata block at the end) instead of raw samples; played as sound, those bytes crackle at
    every sentence boundary. A raw clip passes through unchanged. Both ends get a 5 ms fade so joins never click."""
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        i = 12
        found = None
        while i + 8 <= len(raw):
            cid, size = raw[i:i + 4], int.from_bytes(raw[i + 4:i + 8], "little")
            if cid == b"fmt ":
                rate = int.from_bytes(raw[i + 12:i + 16], "little")
            if cid == b"data":
                found = raw[i + 8:i + 8 + size]
                break
            i += 8 + size + (size & 1)
        raw = found if found is not None else raw[44:]
    raw = raw[: len(raw) // 2 * 2]
    samples = array.array("h", raw)
    if sys.byteorder == "big":
        samples.byteswap()
    fade = min(len(samples) // 2, rate // 200)  # 5 ms
    for k in range(fade):
        g = k / fade
        samples[k] = int(samples[k] * g)
        samples[-1 - k] = int(samples[-1 - k] * g)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes(), rate


def trim_silence(pcm: bytes, rate: int, lead: float = 0.03, tail: float = 0.07) -> bytes:
    """Cuts the silence the TTS model leaves before and after the words (about 0.4 s and 0.5 s), keeping a short
    natural breath, so joined sentences follow each other with no waiting. Re-fades both ends so joins never click."""
    samples = array.array("h", pcm[: len(pcm) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    win = max(1, rate // 100)  # 10 ms
    levels = [max((abs(x) for x in samples[i:i + win]), default=0) for i in range(0, len(samples), win)]
    peak = max(levels, default=0)
    if peak < 300:  # nothing audible: leave it alone
        return pcm
    thr = max(300, peak * 0.03)
    first = next(i for i, v in enumerate(levels) if v > thr)
    last = len(levels) - 1 - next(i for i, v in enumerate(reversed(levels)) if v > thr)
    a = max(0, first * win - int(lead * rate))
    b = min(len(samples), (last + 1) * win + int(tail * rate))
    out = samples[a:b]
    fade = min(len(out) // 2, rate // 200)
    for k in range(fade):
        g = k / fade
        out[k] = int(out[k] * g)
        out[-1 - k] = int(out[-1 - k] * g)
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


STYLE_REF_NOTE = ("The reference image(s) show the exact visual style to match: medium, line work, rendering, color palette, "
                  "lighting and texture. Match that style closely. Do not copy their subjects, people, objects, composition or "
                  "any text in them. Create this new image:\n")


async def draw(prompt: str, aspect: str, model_id: str, out_stem: Path,
               refs: list[tuple[bytes, str]] | None = None) -> tuple[Path, float]:
    """Image generation with Nano Banana. refs are (bytes, mime) style references. Returns (file path, cost)."""
    parts = [{"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode()}} for data, mime in (refs or [])]
    parts.append({"text": (STYLE_REF_NOTE if refs else "") + prompt})
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": aspect}},
    }
    resp = await generate(model_id, body, timeout=240.0)
    inline = next((p["inlineData"] for p in _parts(resp) if "inlineData" in p), None)
    if not inline:
        why = text_of(resp)[:200] if any("text" in p for p in _parts(resp)) else "no image in the answer"
        raise ProviderError(f"No image came back ({why}). Rewording the prompt usually fixes this.")
    ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(inline.get("mimeType", ""), ".png")
    path = out_stem.with_suffix(ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(inline["data"]))
    return path, IMAGE_PRICES.get(model_id, 0.07)
