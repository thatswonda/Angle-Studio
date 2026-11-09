"""Generates the example images for the built-in visual templates (run once, after billing is on).

    cd server && python scripts/make_template_examples.py

Each template gets one image per sample subject, saved small as web/templates/<id>-<n>.jpg.
Existing images are skipped, so the script can be re-run to fill gaps.
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


async def one(style: dict, n: int, subject: str, sem: asyncio.Semaphore) -> str:
    out = OUT / f"{style['id']}-{n}.jpg"
    if out.exists():
        return f"skip {out.name}"
    prompt = f"{style['prefix']}. Scene: {subject}. Avoid: {style['avoid']}. No text, captions, watermarks or logos."
    async with sem:
        for attempt in range(3):
            try:
                raw, _ = await google.draw(prompt, "1:1", "nano-banana-2", OUT / f".raw-{style['id']}-{n}")
                break
            except google.ProviderError as e:
                if attempt == 2:
                    return f"FAIL {out.name}: {e}"
                await asyncio.sleep(5)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-vf", "scale=512:512", "-q:v", "4", str(out)], check=True)
    raw.unlink()
    return f"made {out.name}"


async def main() -> None:
    sem = asyncio.Semaphore(4)
    # a template can show its own sample scenes (for example ancient-world templates), otherwise the shared four
    jobs = [one(st, n + 1, subj, sem) for st in DATA["styles"] for n, subj in enumerate(st.get("subjects") or DATA["subjects"])]
    for line in await asyncio.gather(*jobs):
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
