"""Run one prompt N times against one model and collect the raw responses.

A batch of 30 that silently returned 24 is a corrupted measurement, not a
smaller one. Any failure that survives the SDK's retries aborts the whole batch.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

__all__ = ["PRICING", "estimate_cost", "run_batch"]

# USD per million tokens (input, output), from the public price list.
PRICING = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
}


def _load_dotenv() -> None:
    """Read KEY=value lines from ./.env into the environment, if present.

    ponytail: six lines of stdlib instead of a python-dotenv dependency.
    Real environment variables always win, so `.env` never shadows a real key.
    """
    path = Path.cwd() / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _client() -> anthropic.Anthropic:
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    # SDK retries 429/5xx with backoff; anything left over is a real failure.
    return anthropic.Anthropic(max_retries=5)


def _request(prompt: str, model: str, max_tokens: int, effort: str, temperature):
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "output_config": {"effort": effort},
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


def estimate_cost(prompt, model, n, max_tokens, effort, temperature) -> tuple[float, int]:
    """Worst-case USD estimate for the batch, plus the measured input token count.

    Output is priced at the full `max_tokens` because that is the ceiling the
    user is agreeing to. Actual spend is normally lower.
    """
    client = _client()
    counted = client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": prompt}]
    ).input_tokens
    in_rate, out_rate = PRICING.get(model, (10.0, 50.0))  # unknown model: assume dearest
    cost = n * (counted * in_rate + max_tokens * out_rate) / 1_000_000
    return cost, counted


def run_batch(
    prompt: str,
    model: str,
    n: int,
    max_tokens: int = 1024,
    effort: str = "low",
    temperature: float | None = None,
    workers: int = 4,
) -> list[str]:
    """Return N response texts, in request order. Raises if any run fails."""
    client = _client()
    kwargs = _request(prompt, model, max_tokens, effort, temperature)

    def once(_):
        message = client.messages.create(**kwargs)
        if message.stop_reason == "refusal":
            raise RuntimeError(
                "Model refused the prompt; the batch is not a valid measurement."
            )
        return "".join(b.text for b in message.content if b.type == "text")

    # ponytail: fixed small pool. Raise --workers if rate limits allow.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        texts = list(pool.map(once, range(n)))

    if len(texts) != n:
        raise RuntimeError(f"batch incomplete: got {len(texts)} of {n} runs")
    return texts
