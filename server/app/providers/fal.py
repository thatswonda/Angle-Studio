"""FLUX image models on fal.ai (text prompt only). The key is FAL_KEY in server/.env or the host's environment."""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path

import httpx

from .google import ProviderError

# id -> (fal endpoint, extra inputs, price per image). fal bills by megapixel, rounded up,
# so every size below stays just under 1 megapixel and costs one unit.
MODELS = {
    "flux-schnell": ("fal-ai/flux/schnell", {"num_inference_steps": 4}, 0.003),
    "flux-dev": ("fal-ai/flux/dev", {"num_inference_steps": 28, "guidance_scale": 3.5}, 0.025),
}
SIZES = {"9:16": (736, 1312), "16:9": (1312, 736), "1:1": (992, 992), "4:5": (880, 1104)}
PRESETS = {"9:16": "portrait_16_9", "16:9": "landscape_16_9", "1:1": "square_hd", "4:5": "portrait_4_3"}


def is_fal(model_id: str) -> bool:
    return model_id in MODELS


def flux_prompt(prompt: str) -> str:
    """FLUX has no 'avoid' input and tends to draw what an avoid list names, so that clause is dropped."""
    p = re.sub(r"\. Avoid: .*?(?=\. No words|$)", "", prompt, flags=re.S)
    return p.replace("No words, captions, watermarks or logos in the image.", "Clean image with no text, captions, signatures or watermarks.")


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


async def draw(prompt: str, aspect: str, model_id: str, out_stem: Path) -> tuple[Path, float]:
    key = os.getenv("FAL_KEY", "")
    if not key:
        raise ProviderError("FLUX needs a fal API key. Add FAL_KEY to the server's environment (on Render: Environment > Add variable).")
    endpoint, extra, price = MODELS[model_id]
    w, h = SIZES.get(aspect, SIZES["9:16"])
    body = {"prompt": flux_prompt(prompt)[:3500], "image_size": {"width": w, "height": h}, "num_images": 1,
            "enable_safety_checker": True, "sync_mode": True, "output_format": "jpeg", **extra}
    headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=180.0) as c:
        r = await c.post(f"https://fal.run/{endpoint}", json=body, headers=headers)
        if r.status_code == 422:  # an older model version that only takes named sizes
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
    path = out_stem.with_suffix(".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path, price
