"""Claude (Anthropic API) writer: angles, scripts, rewrites, image ideas, titles and channel ideas, as JSON.

Same complete_json(system, prompt) call as the other writers. The key is ANTHROPIC_API_KEY on the server.
"""
from __future__ import annotations

import os

import anthropic

from .google import ProviderError, parse_json

# App model ids -> Anthropic model ids. Override with ANTHROPIC_MODEL_<ID> if needed.
MODELS = {"claude-opus-5-5": os.getenv("ANTHROPIC_MODEL_OPUS", "claude-opus-5-5"),
          "claude-sonnet-5-5": os.getenv("ANTHROPIC_MODEL_SONNET", "claude-sonnet-5-5")}
# USD per 1M tokens (input, output). Thinking tokens are billed as output.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0)}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeWriter:
    def __init__(self, model_id: str = "claude-opus-5-5", client: "anthropic.AsyncAnthropic | None" = None):
        self.model_id = model_id
        self.client = client

    async def complete_json(self, system: str, prompt: str) -> tuple[dict, float]:
        if not self.client and not os.getenv("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not set on the server.")
        client = self.client or anthropic.AsyncAnthropic(max_retries=3, timeout=600.0)
        try:
            # Streamed so long scripts never hit a request timeout. Thinking is adaptive (the default on these models);
            # effort stays at medium to keep the cost per video low. If Claude declines on a safety category,
            # the API reruns the request on a fallback model inside the same call.
            async with client.beta.messages.stream(
                model=MODELS.get(self.model_id, self.model_id),
                max_tokens=32000,
                system=system + " Reply with one JSON object and nothing else: no markdown fences, no comments.",
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "medium"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                msg = await stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise ProviderError("Anthropic refused the API key. Check ANTHROPIC_API_KEY on the server.") from e
        except anthropic.PermissionDeniedError as e:
            raise ProviderError("This Anthropic key cannot use " + self.model_id + ". Check the key's workspace and limits.") from e
        except anthropic.RateLimitError as e:
            raise ProviderError("Anthropic is limiting requests right now. Wait a minute and try again.") from e
        except anthropic.BadRequestError as e:
            text = str(getattr(e, "message", e))
            if "credit" in text.lower() or "balance" in text.lower():
                raise ProviderError("Your Anthropic credit has run out. Add credit at console.anthropic.com > Billing.") from e
            raise ProviderError("Anthropic rejected the request: " + text[:300]) from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"Anthropic error {e.status_code}. Try again in a minute.") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError("Could not reach the Anthropic API from this server.") from e
        if msg.stop_reason == "refusal":
            raise ProviderError("Claude declined to write this. Rephrase the idea or pick another writer.")
        if msg.stop_reason == "max_tokens":
            raise ProviderError("Claude's answer was too long and got cut off. Try a shorter length.")
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        u = msg.usage
        pin, pout = PRICES.get(self.model_id, (4.0, 20.0))
        cost = ((u.input_tokens or 0) * pin + (u.output_tokens or 0) * pout) / 1e6
        try:
            return parse_json(text), cost
        except ValueError as e:
            raise ProviderError("Claude's answer was not valid JSON. Try again.") from e
