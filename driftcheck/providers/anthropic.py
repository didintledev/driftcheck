"""Anthropic (Claude) provider. Requires prepaid API credits; a Pro or Max
subscription does not grant API access.

Install with `pip install "driftcheck[anthropic]"`.
"""

from functools import lru_cache

# ponytail: absolute import, so this module's own name does not shadow the SDK.
import anthropic as sdk

from . import require_key

__all__ = ["ENV_KEY", "DEFAULT_MODEL", "RPM", "PRICING", "FREE_NOTE",
           "count_input_tokens", "generate"]

ENV_KEY = "ANTHROPIC_API_KEY"
DEFAULT_MODEL = "claude-sonnet-5"
RPM = None  # tier-dependent; the SDK backs off on 429 instead
FREE_NOTE = None

# USD per million tokens (input, output), from the public price list.
PRICING = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
}


@lru_cache(maxsize=1)
def _client() -> "sdk.Anthropic":
    # SDK retries 429/5xx with backoff; anything left over is a real failure.
    return sdk.Anthropic(api_key=require_key(ENV_KEY), max_retries=5)


def count_input_tokens(model: str, prompt: str) -> int:
    return _client().messages.count_tokens(
        model=model, messages=[{"role": "user", "content": prompt}]
    ).input_tokens


def generate(model, prompt, max_tokens, effort="low", temperature=None) -> str:
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "output_config": {"effort": effort},
        "messages": [{"role": "user", "content": prompt}],
    }
    # Current Claude models reject temperature outright; only sent if asked for.
    if temperature is not None:
        kwargs["temperature"] = temperature

    message = _client().messages.create(**kwargs)
    if message.stop_reason == "refusal":
        raise RuntimeError("Model refused the prompt; the batch is not a valid measurement.")
    text = "".join(b.text for b in message.content if b.type == "text")
    if not text.strip():
        raise RuntimeError("Empty response; the batch is not a valid measurement.")
    return text
