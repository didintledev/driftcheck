"""Batch A vs batch B: per-brand deltas plus a signal / noise / underpowered verdict.

`underpowered` is the honest third answer. A tool that only says "changed" or
"didn't change" is claiming a resolution it does not have.
"""

from .stats import MIN_MENTIONS, MIN_N, crit_diff_rank, crit_diff_rate

__all__ = ["compare"]


def _verdict(delta: float | None, crit: float | None, powered: bool) -> str:
    if not powered or delta is None or crit is None:
        return "underpowered"
    return "signal" if abs(delta) > crit else "noise"


def compare(result_a: dict, result_b: dict) -> dict:
    """Compare two run results. Both must be for the same prompt and brand list."""
    a_meta, b_meta = result_a["run"], result_b["run"]
    n_a, n_b = a_meta["n"], b_meta["n"]

    warnings = []
    if a_meta["prompt"] != b_meta["prompt"]:
        warnings.append("Prompts differ; a difference here is not drift.")
    if a_meta["model"] != b_meta["model"]:
        warnings.append("Models differ; cross-model comparison is out of scope.")

    a_by = {b["name"]: b for b in result_a["brands"]}
    b_by = {b["name"]: b for b in result_b["brands"]}
    if set(a_by) != set(b_by):
        warnings.append("Brand lists differ; only the shared brands are compared.")

    powered_n = n_a >= MIN_N and n_b >= MIN_N
    if not powered_n:
        warnings.append(
            f"n={n_a} and n={n_b}; a verdict needs at least {MIN_N} runs per batch."
        )

    rows = []
    for name in sorted(set(a_by) & set(b_by)):
        a, b = a_by[name], b_by[name]

        rate_delta = round(b["mention_rate"] - a["mention_rate"], 4)
        rate_crit = crit_diff_rate(n_a, n_b)

        enough_mentions = a["mentions"] >= MIN_MENTIONS and b["mentions"] >= MIN_MENTIONS
        rank_delta = rank_crit = None
        if a["mean_rank"] is not None and b["mean_rank"] is not None:
            rank_delta = round(b["mean_rank"] - a["mean_rank"], 3)
            rank_crit = crit_diff_rank(
                a["rank_stdev"] or 0.0, a["mentions"], b["rank_stdev"] or 0.0, b["mentions"]
            )

        rows.append(
            {
                "name": name,
                "mention_rate": {
                    "a": a["mention_rate"],
                    "b": b["mention_rate"],
                    "delta": rate_delta,
                    "noise_floor": round(rate_crit, 4),
                    "verdict": _verdict(rate_delta, rate_crit, powered_n),
                },
                "mean_rank": {
                    "a": a["mean_rank"],
                    "b": b["mean_rank"],
                    "delta": rank_delta,
                    "noise_floor": None if rank_crit is None else round(rank_crit, 3),
                    "verdict": _verdict(
                        rank_delta, rank_crit, powered_n and enough_mentions
                    ),
                },
            }
        )

    return {
        "a": a_meta,
        "b": b_meta,
        "brands": rows,
        "warnings": warnings,
    }


def demo() -> None:
    def batch(n, rate, mean, sd, mentions):
        return {
            "run": {"prompt": "p", "model": "m", "n": n},
            "brands": [
                {
                    "name": "Brooks",
                    "mentions": mentions,
                    "mention_rate": rate,
                    "mean_rank": mean,
                    "rank_stdev": sd,
                }
            ],
        }

    # tiny batches can never produce a verdict, however big the gap looks
    tiny = compare(batch(3, 0.0, 3.0, 0.5, 3), batch(3, 1.0, 1.0, 0.5, 3))
    assert tiny["brands"][0]["mention_rate"]["verdict"] == "underpowered", tiny

    # large batches, large gap -> signal
    big = compare(batch(200, 0.30, 3.0, 0.5, 60), batch(200, 0.90, 1.0, 0.5, 180))
    assert big["brands"][0]["mention_rate"]["verdict"] == "signal", big
    assert big["brands"][0]["mean_rank"]["verdict"] == "signal", big

    # large batches, tiny gap -> noise
    flat = compare(batch(200, 0.50, 2.00, 0.5, 100), batch(200, 0.51, 2.02, 0.5, 100))
    assert flat["brands"][0]["mention_rate"]["verdict"] == "noise", flat
    assert flat["brands"][0]["mean_rank"]["verdict"] == "noise", flat

    print("compare ok")


if __name__ == "__main__":
    demo()
