import math

import pytest

from driftcheck.compare import compare
from driftcheck.stats import (
    MIN_N,
    crit_diff_rank,
    crit_diff_rate,
    noise_floor,
    summarize,
)

RUNS = [
    {"A": 1, "B": 2},
    {"A": 1, "B": 2},
    {"B": 1},
    {"A": 2, "B": 1},
]


def _by_name(runs, brands):
    return {row["name"]: row for row in summarize(runs, brands)}


def test_mention_rate_counts_runs_not_mentions():
    s = _by_name(RUNS, ["A", "B", "C"])
    assert s["A"]["mention_rate"] == 0.75
    assert s["B"]["mention_rate"] == 1.0
    assert s["C"]["mention_rate"] == 0.0


def test_absent_runs_are_not_averaged_into_mean_rank():
    # A ranked 1, 1, 2 and was absent once. Mean is over the three mentions only.
    s = _by_name(RUNS, ["A"])
    assert s["A"]["mean_rank"] == pytest.approx(4 / 3, abs=1e-3)


def test_never_mentioned_brand_has_no_rank_at_all():
    s = _by_name(RUNS, ["C"])
    assert s["C"]["mean_rank"] is None
    assert s["C"]["rank_stdev"] is None
    assert s["C"]["rank_ci95"] is None


def test_single_mention_gives_a_mean_but_no_spread():
    s = _by_name([{"A": 1}, {}, {}], ["A"])
    assert s["A"]["mean_rank"] == 1.0
    assert s["A"]["rank_stdev"] is None
    assert s["A"]["rank_ci95"] is None


def test_ci_brackets_the_mean_and_widens_with_spread():
    tight = _by_name([{"A": 2}] * 10 + [{"A": 3}], ["A"])["A"]
    loose = _by_name([{"A": 1}] * 5 + [{"A": 5}] * 6, ["A"])["A"]
    lo, hi = tight["rank_ci95"]
    assert lo < tight["mean_rank"] < hi
    assert (hi - lo) < (loose["rank_ci95"][1] - loose["rank_ci95"][0])


def test_summarize_rejects_an_empty_batch():
    with pytest.raises(ValueError):
        summarize([], ["A"])


def test_noise_floor_shrinks_as_n_grows():
    assert crit_diff_rate(10, 10) > crit_diff_rate(100, 100) > crit_diff_rate(1000, 1000)


def test_noise_floor_matches_the_two_proportion_formula():
    expected = 1.959963985 * math.sqrt(0.25 * (1 / 30 + 1 / 30))
    assert crit_diff_rate(30, 30) == pytest.approx(expected, rel=1e-9)


def test_rank_floor_is_none_when_spread_cannot_be_estimated():
    assert crit_diff_rank(1.0, 1, 1.0, 50) is None
    assert crit_diff_rank(1.0, 50, 1.0, 1) is None
    assert crit_diff_rank(1.0, 50, 1.0, 50) is not None


def test_batch_noise_floor_reports_n_in_its_note():
    floor = noise_floor(summarize(RUNS, ["A", "B"]), len(RUNS))
    assert "n=4" in floor["note"]
    assert floor["mention_rate"] > 0


# --- compare: the three verdicts -------------------------------------------


def _batch(n, rate, mean, sd, mentions, prompt="p", model="m"):
    return {
        "run": {"prompt": prompt, "model": model, "n": n},
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


def test_small_batches_are_underpowered_however_large_the_gap():
    out = compare(_batch(3, 0.0, 3.0, 0.5, 3), _batch(3, 1.0, 1.0, 0.5, 3))
    assert out["brands"][0]["mention_rate"]["verdict"] == "underpowered"
    assert any(str(MIN_N) in w for w in out["warnings"])


def test_large_gap_in_large_batches_is_signal():
    out = compare(_batch(200, 0.30, 3.0, 0.5, 60), _batch(200, 0.90, 1.0, 0.5, 180))
    assert out["brands"][0]["mention_rate"]["verdict"] == "signal"
    assert out["brands"][0]["mean_rank"]["verdict"] == "signal"


def test_tiny_gap_in_large_batches_is_noise():
    out = compare(_batch(200, 0.50, 2.00, 0.5, 100), _batch(200, 0.51, 2.02, 0.5, 100))
    assert out["brands"][0]["mention_rate"]["verdict"] == "noise"
    assert out["brands"][0]["mean_rank"]["verdict"] == "noise"


def test_rank_verdict_is_underpowered_when_mentions_are_too_few():
    out = compare(_batch(50, 0.04, 3.0, 0.5, 2), _batch(50, 0.04, 1.0, 0.5, 2))
    assert out["brands"][0]["mean_rank"]["verdict"] == "underpowered"
    # the rate verdict is still available at this n
    assert out["brands"][0]["mention_rate"]["verdict"] in {"signal", "noise"}


def test_differing_prompt_or_model_is_warned_about():
    out = compare(
        _batch(30, 0.5, 2.0, 0.5, 15, prompt="one"),
        _batch(30, 0.5, 2.0, 0.5, 15, prompt="two", model="other"),
    )
    assert any("Prompt" in w for w in out["warnings"])
    assert any("Model" in w for w in out["warnings"])
