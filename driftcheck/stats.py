"""Variance, confidence intervals, and the noise floor.

The noise floor is the smallest difference between two batches that is
distinguishable from run-to-run sampling variance at 95% confidence. It is a
property of the batch size, not of the brand — which is why n must always be
reported alongside any result.
"""

import math
import statistics

__all__ = [
    "Z95",
    "MIN_N",
    "MIN_MENTIONS",
    "summarize",
    "noise_floor",
    "crit_diff_rate",
    "crit_diff_rank",
]

Z95 = 1.959963985  # two-sided 95%
MIN_N = 10  # below this, refuse to issue a signal/noise verdict
MIN_MENTIONS = 5  # below this, refuse to compare mean ranks


def summarize(run_ranks: list[dict[str, int]], brands: list[str]) -> list[dict]:
    """Per-brand mention rate, mean rank, rank stdev, and 95% CI on mean rank.

    `run_ranks` is one dict per run, as returned by extract.ranks. Mean rank is
    computed only over the runs where the brand appeared: a missing value is
    never averaged in as a rank.
    """
    n = len(run_ranks)
    if n == 0:
        raise ValueError("no runs to summarize")

    out = []
    for brand in brands:
        observed = [r[brand] for r in run_ranks if brand in r]
        mentions = len(observed)
        row = {
            "name": brand,
            "n": n,
            "mentions": mentions,
            "mention_rate": round(mentions / n, 4),
            "mean_rank": None,
            "rank_stdev": None,
            "rank_ci95": None,
        }
        if mentions:
            mean = statistics.fmean(observed)
            row["mean_rank"] = round(mean, 3)
            if mentions >= 2:
                sd = statistics.stdev(observed)
                half = Z95 * sd / math.sqrt(mentions)
                row["rank_stdev"] = round(sd, 3)
                row["rank_ci95"] = [round(mean - half, 3), round(mean + half, 3)]
        out.append(row)
    return out


def crit_diff_rate(n_a: int, n_b: int, p: float = 0.5) -> float:
    """Smallest mention-rate difference distinguishable from sampling noise.

    Two-proportion z-test critical difference. Defaults to p=0.5, the worst
    case, so the batch-level floor is never optimistic.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("batch sizes must be positive")
    return Z95 * math.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))


def crit_diff_rank(sd_a: float, m_a: int, sd_b: float, m_b: int) -> float | None:
    """Smallest mean-rank difference distinguishable from noise, or None if unknowable.

    Uses each batch's observed rank stdev over the runs where the brand actually
    appeared. Returns None when either side has too few mentions to estimate
    spread — that is an `underpowered` answer, not a zero.
    """
    if m_a < 2 or m_b < 2:
        return None
    return Z95 * math.sqrt(sd_a**2 / m_a + sd_b**2 / m_b)


def noise_floor(summary: list[dict], n: int) -> dict:
    """Batch-level floor: what a single batch of this size can and cannot resolve.

    The mention-rate floor is the worst-case (p=0.5) critical difference against
    a second batch of the same size. The mean-rank floor is the same computation
    using the median observed rank stdev across brands that were mentioned
    enough to have one; it is None when no brand qualifies.
    """
    rate = crit_diff_rate(n, n)
    spreads = [b["rank_stdev"] for b in summary if b["rank_stdev"] is not None]
    mentions = [b["mentions"] for b in summary if b["rank_stdev"] is not None]

    rank = None
    if spreads:
        sd = statistics.median(spreads)
        m = int(statistics.median(mentions))
        rank = crit_diff_rank(sd, m, sd, m)

    return {
        "mention_rate": round(rate, 4),
        "mean_rank": None if rank is None else round(rank, 3),
        "note": (
            f"Against a second batch of n={n}, changes smaller than these are "
            "indistinguishable from run-to-run variance at 95% confidence. "
            "The mention-rate floor is worst-case (p=0.5); the mean-rank floor "
            "uses the median observed rank spread."
        ),
    }


def demo() -> None:
    runs = [{"A": 1, "B": 2}, {"A": 1, "B": 2}, {"B": 1}, {"A": 2, "B": 1}]
    s = {b["name"]: b for b in summarize(runs, ["A", "B", "C"])}

    assert s["A"]["mentions"] == 3 and s["A"]["mention_rate"] == 0.75
    # 1, 1, 2 -> mean 1.333; the absent run is NOT averaged in as a 0 or a 3
    assert abs(s["A"]["mean_rank"] - 4 / 3) < 1e-3
    assert s["B"]["mention_rate"] == 1.0
    # never mentioned: rate 0, and no rank at all
    assert s["C"]["mention_rate"] == 0.0 and s["C"]["mean_rank"] is None

    # bigger batches resolve smaller differences
    assert crit_diff_rate(100, 100) < crit_diff_rate(10, 10)
    assert crit_diff_rank(1.0, 1, 1.0, 30) is None

    floor = noise_floor(list(s.values()), 4)
    assert floor["mention_rate"] > 0
    print("stats ok")


if __name__ == "__main__":
    demo()
