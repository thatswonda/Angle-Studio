"""ChatGPT (OpenAI API) writer: angles, the story and metadata as JSON.

It implements the same complete_json(system, prompt) call as the other writers, so switching to Claude
later means adding a ClaudeWriter with the same method and choosing it in the Writing selector.
"""
from __future__ import annotations

import asyncio
import json
import os

import httpx

from .google import ProviderError, parse_json

URL = "https://api.openai.com/v1/chat/completions"
# App model ids -> OpenAI model names. Override with OPENAI_MODEL_<ID> if OpenAI renames them.
MODELS = {"gpt-5-5": os.getenv("OPENAI_MODEL_GPT_5_5", "gpt-5.5"), "gpt-5": os.getenv("OPENAI_MODEL_GPT_5", "gpt-5")}
# USD per 1M tokens (input, output); estimates from public trackers, confirm on OpenAI's pricing page.
PRICES = {"gpt-5-5": (5.0, 30.0), "gpt-5": (1.25, 10.0)}


class OpenAIWriter:
    def __init__(self, model_id: str = "gpt-5-5"):
        self.model_id = model_id

    async def complete_json(self, system: str, prompt: str) -> tuple[dict, float]:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set on the server.")
        body = {
            "model": MODELS.get(self.model_id, self.model_id),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        for attempt in range(4):
            try:
                async with httpx.AsyncClient(timeout=600.0) as c:
                    r = await c.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})
            except httpx.HTTPError as e:
                if attempt == 3:
                    raise ProviderError(f"Could not reach the OpenAI API: {e}. If this server blocks api.openai.com, "
                                        "pick Gemini Flash in the Writing selector.") from e
                await asyncio.sleep(2 ** attempt * 2)
                continue
            if r.status_code in (429, 500, 502, 503) and attempt < 3:
                await asyncio.sleep(2 ** attempt * 3)
                continue
            if r.status_code != 200:
                try:
                    msg = r.json().get("error", {}).get("message", "")
                except Exception:
                    msg = r.text[:300]
                raise ProviderError(f"OpenAI error {r.status_code}: {msg}")
            data = r.json()
            u = data.get("usage", {})
            pin, pout = PRICES.get(self.model_id, (5.0, 30.0))
            cost = (u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout) / 1e6
            try:
                return parse_json(data["choices"][0]["message"]["content"] or ""), cost
            except (json.JSONDecodeError, KeyError, IndexError):
                if attempt == 3:
                    raise ProviderError("ChatGPT's answer was not valid JSON. Try again.")
        raise ProviderError("The OpenAI API did not answer.")
