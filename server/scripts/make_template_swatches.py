"""Makes one 'style swatch' per built-in template: web/templates/<id>-style.jpg.

FLUX Kontext copies the people in its reference image, so it gets this swatch instead of an example scene: one plain,
generic character and a few objects drawn in the template's style. The style carries over; no one's identity does.
Existing swatches are skipped. Run after adding a template:

    cd server && python scripts/make_template_swatches.py
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from app import config  # noqa: E402,F401  (loads the API key)
from app.providers import google  # noqa: E402

OUT = ROOT / "web" / "templates"
DATA = json.loads((OUT / "templates.json").read_text())
SCENE = ("Scene: one plain, generic character drawn exactly in this style, standing and facing forward with a calm friendly "
         "smile, simple neutral clothes, no distinctive features or accessories, next to a simple table with a teapot and a "
         "small potted plant. Plain background, nothing else.")


async def one(st: dict, sem: asyncio.Semaphore) -> str:
    out = OUT / f"{st['id']}-style.jpg"
    if out.exists():
        return f"skip {out.name}"
    refs = [((OUT / f"{st['id']}-{n}.jpg").read_bytes(), "image/jpeg") for n in (1, 2) if (OUT / f"{st['id']}-{n}.jpg").exists()]
    prompt = f"{st['prefix']}. {SCENE} Avoid: {st['avoid']}. No words, captions, watermarks or logos in the image."
    async with sem:
        for attempt in range(3):
            try:
                raw, _ = await google.draw(prompt, "1:1", "nano-banana-2", OUT / f".raw-{st['id']}-style", refs)
                break
            except google.ProviderError as e:
                if attempt == 2:
                    return f"FAIL {out.name}: {e}"
                await asyncio.sleep(5)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-vf", "scale=768:768", "-q:v", "3", str(out)], check=True)
    raw.unlink()
    return f"made {out.name}"


async def main() -> None:
    sem = asyncio.Semaphore(4)
    for line in await asyncio.gather(*[one(st, sem) for st in DATA["styles"]]):
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
