# driftcheck

**Is that change real, or is the model just noisy?**

Ask an LLM the same question twice and you'll get two different answers. That's expected. The problem is that tools tracking brand visibility in AI answers report the difference as a change, when most of the time it's just variance.

Before you can say a brand's position dropped, you need to know how much it moves when nothing has changed. `driftcheck` measures that noise floor and tells you whether an observed change clears it.

It has three verdicts: **signal**, **noise**, and **underpowered**. The third one is the interesting one, and most tools don't have it.

---

## The worked example

Two batches of 30 runs, same prompt, same model, nothing changed between them except time:

```
brand         rate A -> B    delta   noise floor   verdict
ASICS         0.50 -> 0.70   +0.20      0.25       noise
Brooks        0.90 -> 0.87   -0.03      0.25       noise
Hoka          0.73 -> 0.77   +0.03      0.25       noise
New Balance   0.27 -> 0.60   +0.33      0.25       signal
Saucony       0.10 -> 0.07   -0.03      0.25       noise   (rank: underpowered)
```

Read the ASICS row twice. **A twenty-point swing in mention rate is not a finding at n=30.** It is inside what the same prompt does to itself on a quiet afternoon. A dashboard that charts that as a line going up is selling you a story about your brand that the data cannot support.

New Balance moved 33 points and clears the floor, so that one is worth a meeting. Saucony was mentioned three times across thirty runs — enough to estimate a rate, nowhere near enough to say anything about its average position, so the rank verdict refuses to answer instead of guessing.

The floor is a property of your sample size, not your brand. At n=30 you cannot resolve anything smaller than ~25 points. Want to detect a 10-point shift? You need roughly 200 runs. The tool tells you this before you spend the money, not after you've briefed the client.

*(Numbers above are from `examples/`, produced by running synthetic responses through the real extraction and statistics code — no API calls. The pipeline is real; the responses are not. Marked `_synthetic` in the files.)*

---

## Usage

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...        # or copy .env.example to .env

# brands.json: {"Brooks": ["Brooks Running"], "Hoka": ["Hoka One One"], "ASICS": []}
driftcheck run "best running shoes for flat feet" --brands brands.json -n 30 -o week1.json

# ...later, same prompt, same model
driftcheck run "best running shoes for flat feet" --brands brands.json -n 30 -o week2.json

driftcheck compare week1.json week2.json
```

`run` prints a cost estimate and waits for confirmation before spending anything (`-y` to skip). Options: `--model`, `--max-tokens`, `--effort`, `--workers`, `--save-responses`.

---

## The design decisions, and why

The code is secondary. These are the calls that determine whether the number means anything.

**1. Extraction is deterministic string matching, never an LLM.** This is the single most important decision in the tool. If a model extracted the brand mentions, extraction variance would be folded into the measurement and you'd be measuring two noise sources while reporting one. `extract.py` is pure functions over (text, brand list): casefold, strip accents, collapse punctuation, match on word boundaries against a supplied alias list. It is auditable and it returns the same answer every time.

**2. Rank means the position of a brand's first mention, ordered against the other brands' first mentions in the same response.** Rank 1 is mentioned earliest. It is dense over the brands that actually appeared, so it behaves the same for prose and for numbered lists. Repeat mentions don't count; only the first does.

**3. Absent brands are not rank zero.** Absence and being-mentioned-last are different events, and averaging a missing value into a rank silently corrupts the mean. Mention rate is computed over all N runs; mean rank is computed only over the runs where the brand appeared. A brand that never appears has a mention rate of 0 and no rank at all — not a rank of 0, not a rank of "last".

**4. Every point estimate ships with its interval.** A mean rank of 3.2 is useless without knowing it swings between 1 and 7. Means come with a standard deviation and a 95% CI, and a brand mentioned once gets a mean with no spread rather than a fake precision.

**5. N is part of the result, and there is a floor below which the tool refuses to answer.** Ten runs and a hundred runs are not comparable claims, so `n` is recorded in every output and echoed in the noise-floor note. Below 10 runs per batch, or 5 mentions for a rank comparison, `compare` returns `underpowered` and says why. It does not fall back to a verdict it can't support.

**6. Model parameters are recorded — including the ones you cannot set.** Current Claude models (Opus 5, Sonnet 5, Opus 4.7+) **reject `temperature` outright with a 400**, and the API has no seed parameter at all. The flag exists and its value is recorded, but it defaults to unset, and the output records `temperature: null, seed: null` truthfully rather than implying a determinism knob that isn't there.

   This is worth sitting with, because it is the strongest version of the argument. On current frontier models **you cannot turn the sampling noise off.** There is no `temperature=0` escape hatch and no seed to pin. The variance is not a configuration mistake you can fix — it is a permanent property of the measurement, and the only honest response is to quantify it and report against it. That is what this tool does.

### How the floor is computed

The noise floor is the smallest difference between two batches that a two-sample test can distinguish at 95% confidence.

- **Mention rate:** the two-proportion critical difference, `1.96 × sqrt(p(1−p)(1/nA + 1/nB))`. The batch-level floor uses the worst case `p = 0.5`, so it is never optimistic. Per-brand comparisons in `compare` use the same formula against the actual batch sizes.
- **Mean rank:** `1.96 × sqrt(sA²/mA + sB²/mB)` over the observed rank spread, where `m` is the number of runs in which the brand actually appeared. When either side has fewer than two mentions the spread is unknowable and the floor is `None` — which surfaces as `underpowered`, not as zero.

A difference larger than the floor is `signal`. Smaller is `noise`. Not enough data to run the test is `underpowered`.

---

## What this is not

Naming the non-goals is part of the point. Out of scope, deliberately:

- Multi-model comparison
- Scheduling, storage, dashboards, alerting
- Sentiment, citation tracking, share-of-voice
- Anything resembling a product

This is a demonstration artifact: one measurement, done carefully, with its limits stated. A small correct thing beats a broad half-finished one.

---

## Development

```bash
pip install -e ".[dev]"
pytest                              # 25 tests over extract.py and stats.py
python -m driftcheck.extract        # module self-checks, no API key needed
python -m driftcheck.stats
python -m driftcheck.compare
```

Tests cover the two modules that are pure functions with no API calls — extraction and statistics — because those are where a silent error would invalidate every number the tool prints. They pin the decisions above: absence is never rank zero, missing values never enter a mean, the floor shrinks as `sqrt(n)`, and small batches return `underpowered` no matter how large the apparent gap.

Batch failures are loud. The SDK retries transient errors; anything that survives that aborts the whole batch. A batch of 30 that quietly returned 24 is a corrupted measurement, not a smaller one.

## Layout

```
driftcheck/
  sample.py    run prompt N times, collect raw responses, estimate cost
  extract.py   deterministic brand + rank extraction
  stats.py     variance, confidence intervals, noise floor
  compare.py   batch A vs batch B, three-way verdict
  cli.py
```

MIT.
