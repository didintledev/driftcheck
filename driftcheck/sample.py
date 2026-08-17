"""Run one prompt N times against one model and collect the raw responses.

Provider-agnostic: everything here talks to a provider module (see
`driftcheck.providers`) and knows nothing about any particular API.

A batch of 30 that silently returned 24 is a corrupted measurement, not a
smaller one. Any failure that survives the provider's retries aborts the batch.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

__all__ = ["estimate_cost", "run_batch"]


class RateLimiter:
    """Space request *starts* at least `60/rpm` seconds apart, across threads.

    ponytail: paces starts rather than tracking a sliding window. Slightly
    conservative, which is the right side to be wrong on for a free tier.
    Swap in a token bucket if bursting inside the window ever matters.
    """

    def __init__(self, rpm: float | None):
        self._gap = 0.0 if not rpm else 60.0 / rpm
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self._gap:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self._gap
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


def estimate_cost(provider, prompt, model, n, max_tokens) -> tuple[float, int]:
    """Worst-case USD estimate for the batch, plus the measured input token count.

    Output is priced at the full `max_tokens` because that is the ceiling the
    user is agreeing to. Actual spend is normally lower, and on a free tier it
    is zero regardless — see the provider's FREE_NOTE.
    """
    counted = provider.count_input_tokens(model, prompt)
    # Unknown model: assume the dearest one this provider lists, never cheaper.
    fallback = max(provider.PRICING.values(), default=(10.0, 50.0))
    in_rate, out_rate = provider.PRICING.get(model, fallback)
    cost = n * (counted * in_rate + max_tokens * out_rate) / 1_000_000
    return cost, counted


def run_batch(
    provider,
    prompt: str,
    model: str,
    n: int,
    max_tokens: int = 1024,
    effort: str = "low",
    temperature: float | None = None,
    workers: int = 4,
) -> list[str]:
    """Return N response texts, in request order. Raises if any run fails."""
    limiter = RateLimiter(getattr(provider, "RPM", None))

    def once(_):
        limiter.wait()
        return provider.generate(model, prompt, max_tokens, effort, temperature)

    # ponytail: fixed small pool. The limiter, not the pool size, is what keeps
    # a free tier happy, so raising --workers only helps if RPM is None.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # pool.map re-raises on iteration, so one bad run kills the batch here.
        texts = list(pool.map(once, range(n)))

    if len(texts) != n:
        raise RuntimeError(f"batch incomplete: got {len(texts)} of {n} runs")
    return texts


def demo() -> None:
    """Check the two things that are ours rather than the provider's: pacing,
    and that a single failed run fails the whole batch."""

    limiter = RateLimiter(rpm=600)  # 0.1s apart, so the check stays quick
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: limiter.wait(), range(8)))
    assert time.monotonic() - started >= 0.7, "limiter did not space 8 starts"

    assert RateLimiter(None)._gap == 0.0, "no RPM should mean no pacing"

    class Flaky:
        RPM = None
        PRICING = {"m": (1.0, 1.0)}
        calls = 0

        @staticmethod
        def generate(model, prompt, max_tokens, effort, temperature):
            Flaky.calls += 1
            if Flaky.calls == 3:
                raise RuntimeError("boom")
            return "text"

    try:
        run_batch(Flaky, "p", "m", 5, workers=1)
    except RuntimeError as exc:
        assert "boom" in str(exc), exc
    else:  # pragma: no cover
        raise AssertionError("a failed run must fail the batch, not shorten it")

    print("sample ok")


if __name__ == "__main__":
    demo()
