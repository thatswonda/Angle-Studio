"""FLUX image models on fal.ai (text prompt only). The key is FAL_KEY in server/.env or the host's environment."""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path

import httpx

from .google import ProviderError

# id -> (fal endpoint, extra inputs, price per image). fal bills per megapixel, rounded up. Every size below is
# just under 2 megapixels (2 units), so a frame is sharp at 1080p instead of being stretched up from ~1 megapixel.
MODELS = {
    "flux-schnell": ("fal-ai/flux/schnell", {"num_inference_steps": 4}, 0.006),
    "flux-dev": ("fal-ai/flux/dev", {"num_inference_steps": 28, "guidance_scale": 3.5}, 0.05),
    # Kontext sees a reference image: the template's example picture, so the result matches the template's look
    "flux-kontext": ("fal-ai/flux-pro/kontext", {"guidance_scale": 3.5}, 0.04),
    # Seedream 4 (ByteDance) sees up to 10 reference images: the template's example pictures, so the look matches
    "seedream": ("fal-ai/bytedance/seedream/v4/edit", {}, 0.03),
    # Qwen-Image reads text only but takes the template's avoid list as a negative prompt; drawn at about 1 megapixel
    "qwen-image": ("fal-ai/qwen-image", {"num_inference_steps": 30, "guidance_scale": 2.5}, 0.02),
}
TAKES_REFERENCE = {"flux-kontext", "seedream"}
SEEDREAM_NOTE = ("Use the reference images only for their visual style: line work, rendering, colours, background and texture. "
                 "Draw brand-new characters and a new scene; do not copy their people, objects, text or layout. ")
QWEN_SIZES = {"9:16": (736, 1312), "16:9": (1312, 736), "1:1": (992, 992), "4:5": (880, 1104)}  # just under 1 megapixel
KONTEXT_NOTE = ("Create a completely new image in exactly the same visual style as the reference image: the same line work, "
                "rendering, colours, background and texture. Only the style carries over: draw brand-new characters who do not "
                "look like anyone in the reference (different hair, face and clothes), and do not keep its objects, text, setting "
                "or layout. ")
SIZES = {"9:16": (1024, 1824), "16:9": (1824, 1024), "1:1": (1360, 1360), "4:5": (1216, 1520)}
PRESETS = {"9:16": "portrait_16_9", "16:9": "landscape_16_9", "1:1": "square_hd", "4:5": "portrait_4_3"}


def is_fal(model_id: str) -> bool:
    return model_id in MODELS


def flux_prompt(prompt: str) -> str:
    """FLUX has no 'avoid' input and tends to draw what an avoid list names, so that clause is dropped."""
    p = re.sub(r"[.]? Avoid: .*?(?=\. No (?:words|watermarks)|$)", "", prompt, flags=re.S)
    p = p.replace("No words, captions, watermarks or logos in the image.", "Clean image with no text, captions, signatures or watermarks.")
    return p.replace("No watermarks, logos or signatures.", "No watermarks or signatures.")


def _error(r: httpx.Response) -> str:
    try:
        detail = r.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):
        detail = "; ".join(str(d.get("msg", d)) for d in detail if isinstance(d, dict)) or str(detail)
    detail = str(detail or r.text[:200])
    if r.status_code in (401, 403):
        return "fal refused the API key. Check FAL_KEY (the full key from fal.ai > Settings > API Keys, with the API preset)."
    if r.status_code == 402 or "balance" in detail.lower() or "credit" in detail.lower():
        return "Your fal credit has run out. Add credit at fal.ai > Settings > Billing and try again."
    if r.status_code == 429:
        return "fal is limiting requests right now. Wait a minute and try again."
    return f"fal could not make the image ({r.status_code}): {detail}"


def avoid_of(prompt: str) -> str:
    m = re.search(r"Avoid: (.*?)(?:\. No (?:words|watermarks)|\.\s|$)", prompt, flags=re.S)
    return m.group(1).strip()[:500] if m else ""


def _data_url(ref: tuple[bytes, str]) -> str:
    return f"data:{ref[1]};base64,{base64.b64encode(ref[0]).decode()}"


async def draw(prompt: str, aspect: str, model_id: str, out_stem: Path, ref: tuple[bytes, str] | None = None,
               refs: list[tuple[bytes, str]] | None = None) -> tuple[Path, float]:
    key = os.getenv("FAL_KEY", "")
    if not key:
        raise ProviderError("FLUX needs a fal API key. Add FAL_KEY to the server's environment (on Render: Environment > Add variable).")
    endpoint, extra, price = MODELS[model_id]
    w, h = SIZES.get(aspect, SIZES["9:16"])
    if model_id == "seedream":
        refs = refs or ([ref] if ref else [])
        if not refs:
            raise ProviderError("Seedream needs a template with example pictures. Pick a built-in template, or another image model.")
        body = {"prompt": (SEEDREAM_NOTE + flux_prompt(prompt))[:3500], "image_urls": [_data_url(r) for r in refs[:4]],
                "image_size": {"width": w, "height": h}, "num_images": 1, "enable_safety_checker": True, "sync_mode": True}
    elif model_id == "qwen-image":
        qw, qh = QWEN_SIZES.get(aspect, QWEN_SIZES["9:16"])
        body = {"prompt": flux_prompt(prompt)[:3500], "negative_prompt": avoid_of(prompt), "image_size": {"width": qw, "height": qh},
                "num_images": 1, "enable_safety_checker": True, "sync_mode": True, "output_format": "jpeg", **extra}
    elif model_id in TAKES_REFERENCE:
        if not ref:
            raise ProviderError("FLUX Kontext needs a template with example pictures. Pick a built-in template, or use FLUX dev.")
        data, mime = ref
        body = {"prompt": (KONTEXT_NOTE + flux_prompt(prompt))[:3500], "image_url": f"data:{mime};base64,{base64.b64encode(data).decode()}",
                "aspect_ratio": aspect if aspect in ("9:16", "16:9", "1:1", "4:5") else "9:16", "num_images": 1,
                "enable_safety_checker": True, "sync_mode": True, "output_format": "jpeg", **extra}
    else:
        body = {"prompt": flux_prompt(prompt)[:3500], "image_size": {"width": w, "height": h}, "num_images": 1,
                "enable_safety_checker": True, "sync_mode": True, "output_format": "jpeg", **extra}
    headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=180.0) as c:
        r = await c.post(f"https://fal.run/{endpoint}", json=body, headers=headers)
        if r.status_code in (500, 502, 503, 504):  # a passing upstream failure: one more try
            r = await c.post(f"https://fal.run/{endpoint}", json=body, headers=headers)
        if r.status_code == 422 and model_id in ("flux-schnell", "flux-dev"):  # an older model version that only takes named sizes
            r = await c.post(f"https://fal.run/{endpoint}", json={**body, "image_size": PRESETS.get(aspect, "portrait_16_9")}, headers=headers)
        if r.status_code != 200:
            raise ProviderError(_error(r))
        data = r.json()
        if any(data.get("has_nsfw_concepts") or []):
            raise ProviderError("fal's safety filter blocked this image. Change the scene description and try again.")
        images = data.get("images") or []
        if not images or not images[0].get("url"):
            raise ProviderError("fal returned no image. Try again.")
        url = images[0]["url"]
        if url.startswith("data:"):
            raw = base64.b64decode(url.split(",", 1)[1])
        else:  # sync mode was not honored: fetch the hosted file
            g = await c.get(url)
            if g.status_code != 200:
                raise ProviderError("Could not download the image from fal.")
            raw = g.content
    path = out_stem.with_suffix(".png" if raw[:4] == b"\x89PNG" else ".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path, price
