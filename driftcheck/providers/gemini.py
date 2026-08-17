"""Google AI Studio (Gemini) — the default provider.

The free tier needs a Google account and no card, which is why it is the
default: anyone can reproduce the measurement in this repo for nothing.

ponytail: urllib against the REST API instead of the google-genai SDK. Two
POSTs and a retry loop is less code than a dependency, and it keeps the
default install dependency-free.
"""

import json
import time
import urllib.error
import urllib.request

from . import require_key

__all__ = ["ENV_KEY", "DEFAULT_MODEL", "RPM", "PRICING", "FREE_NOTE",
           "count_input_tokens", "generate"]

ENV_KEY = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Free-tier request limit for flash-lite. The batch runner paces to this, so a
# 30-run batch takes ~2 minutes and never trips a 429.
RPM = 15

FREE_NOTE = (
    f"Free tier covers this model at {RPM} requests/minute — no card required. "
    "The estimate below is what a paid key would cost."
)

# USD per million tokens (input, output), paid tier. Free tier bills nothing.
PRICING = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_RETRY_CODES = {429, 500, 502, 503, 504}
_ATTEMPTS = 5


def _post(model: str, method: str, body: dict) -> dict:
    """POST to one Gemini endpoint, retrying transient failures with backoff."""
    request = urllib.request.Request(
        f"{_BASE}/{model}:{method}",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "x-goog-api-key": require_key(ENV_KEY),
        },
    )
    for attempt in range(_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code not in _RETRY_CODES or attempt == _ATTEMPTS - 1:
                raise RuntimeError(
                    f"Gemini {method} failed: HTTP {exc.code} {detail}"
                ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == _ATTEMPTS - 1:
                raise RuntimeError(f"Gemini {method} failed: {exc}") from exc
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _contents(prompt: str) -> list[dict]:
    return [{"role": "user", "parts": [{"text": prompt}]}]


def count_input_tokens(model: str, prompt: str) -> int:
    return int(_post(model, "countTokens", {"contents": _contents(prompt)})["totalTokens"])


def generate(model, prompt, max_tokens, effort=None, temperature=None) -> str:
    """One completion. Raises on anything that would corrupt the measurement.

    `effort` is accepted for interface parity and ignored: Gemini has no
    equivalent knob, and flash-lite does not think by default.
    """
    config: dict = {"maxOutputTokens": max_tokens}
    if temperature is not None:
        config["temperature"] = temperature

    data = _post(
        model,
        "generateContent",
        {"contents": _contents(prompt), "generationConfig": config},
    )

    blocked = (data.get("promptFeedback") or {}).get("blockReason")
    if blocked:
        raise RuntimeError(
            f"Gemini blocked the prompt ({blocked}); the batch is not a valid measurement."
        )

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates; the batch is not a valid measurement.")

    reason = candidates[0].get("finishReason")
    if reason not in (None, "STOP", "MAX_TOKENS"):
        raise RuntimeError(
            f"Gemini stopped with finishReason={reason}; the batch is not a valid measurement."
        )

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    if not text.strip():
        raise RuntimeError(
            "Gemini returned an empty response; the batch is not a valid measurement. "
            "If finishReason was MAX_TOKENS, raise --max-tokens."
        )
    return text
