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

**The default setup costs nothing.** No card, no credits, no paid dependency. Get a free Google AI Studio key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and you can reproduce every number below.

```bash
pip install -e .                           # no dependencies
export GEMINI_API_KEY=...                  # or copy .env.example to .env

# brands.json: {"Brooks": ["Brooks Running"], "Hoka": ["Hoka One One"], "ASICS": []}
driftcheck run "best running shoes for flat feet" --brands brands.json -n 30 -o week1.json

# ...later, same prompt, same model
driftcheck run "best running shoes for flat feet" --brands brands.json -n 30 -o week2.json

driftcheck compare week1.json week2.json
```

Default provider is Gemini on `gemini-2.5-flash-lite`, which the free tier covers at 15 requests per minute. The runner paces itself to that limit, so a 30-run batch takes about two minutes and bills nothing. `run` still prints what a paid key would cost and waits for confirmation (`-y` to skip), because "free" is a property of your key, not of the tool.

Claude is the other provider:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...        # prepaid API credits; Pro/Max is not API access
driftcheck run "..." --brands brands.json --provider anthropic --model claude-sonnet-5
```

Options: `--provider`, `--model`, `--max-tokens`, `--effort`, `--workers`, `--save-responses`.

---

## The design decisions, and why

The code is secondary. These are the calls that determine whether the number means anything.

**1. Extraction is deterministic string matching, never an LLM.** This is the single most important decision in the tool. If a model extracted the brand mentions, extraction variance would be folded into the measurement and you'd be measuring two noise sources while reporting one. `extract.py` is pure functions over (text, brand list): casefold, strip accents, collapse punctuation, match on word boundaries against a supplied alias list. It is auditable and it returns the same answer every time.

**2. Rank means the position of a brand's first mention, ordered against the other brands' first mentions in the same response.** Rank 1 is mentioned earliest. It is dense over the brands that actually appeared, so it behaves the same for prose and for numbered lists. Repeat mentions don't count; only the first does.

**3. Absent brands are not rank zero.** Absence and being-mentioned-last are different events, and averaging a missing value into a rank silently corrupts the mean. Mention rate is computed over all N runs; mean rank is computed only over the runs where the brand appeared. A brand that never appears has a mention rate of 0 and no rank at all — not a rank of 0, not a rank of "last".

**4. Every point estimate ships with its interval.** A mean rank of 3.2 is useless without knowing it swings between 1 and 7. Means come with a standard deviation and a 95% CI, and a brand mentioned once gets a mean with no spread rather than a fake precision.

**5. N is part of the result, and there is a floor below which the tool refuses to answer.** Ten runs and a hundred runs are not comparable claims, so `n` is recorded in every output and echoed in the noise-floor note. Below 10 runs per batch, or 5 mentions for a rank comparison, `compare` returns `underpowered` and says why. It does not fall back to a verdict it can't support.

**6. Model parameters are recorded — including the ones you cannot set.** `--temperature` defaults to unset and is only sent when you ask for it, because whether it can be sent at all depends on the provider. Current Claude models (Opus 5, Sonnet 5, Opus 4.7+) **reject `temperature` outright with a 400**, and the Anthropic API has no seed parameter. The output records `temperature: null, seed: null` truthfully rather than implying a determinism knob that isn't there.

   This is worth sitting with, because it is the strongest version of the argument. On a frontier model you may not have a determinism knob at all — and where you do have one, turning it down narrows the sampling distribution without collapsing it. The variance is not a configuration mistake you can fix. It is a property of the measurement, and the only honest response is to quantify it and report against it. That is what this tool does.

**7. Providers are interchangeable; a measurement is not.** `driftcheck/providers/` holds one module per provider, each exposing five names: the env var for its key, a default model, its price list, its rate limit, a token counter, and a generate call. Nothing outside that package knows which one is in use, so adding Groq is one new file and one line in the registry. The default is Gemini on the free tier specifically so the claim in this README is checkable by anyone, not just by someone willing to spend money to audit it.

   Interchangeable providers are not comparable batches. `compare` warns if the provider, model, or prompt differs between two runs — cross-provider comparison is a different question with a different noise structure, and this tool doesn't answer it.

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
python -m driftcheck.sample         # checks rate-limit pacing and loud batch failure
```

Tests cover the two modules that are pure functions with no API calls — extraction and statistics — because those are where a silent error would invalidate every number the tool prints. They pin the decisions above: absence is never rank zero, missing values never enter a mean, the floor shrinks as `sqrt(n)`, and small batches return `underpowered` no matter how large the apparent gap.

Batch failures are loud. Each provider retries transient errors (429s and 5xx) with backoff; anything that survives that aborts the whole batch, as does a refusal, a safety block, or an empty response. A batch of 30 that quietly returned 24 is a corrupted measurement, not a smaller one.

## Layout

```
driftcheck/
  providers/
    __init__.py  registry, key loading — the whole provider contract
    gemini.py    default; REST over stdlib urllib, free tier
    anthropic.py optional extra
  sample.py    run prompt N times, rate-limit, collect raw responses, estimate cost
  extract.py   deterministic brand + rank extraction
  stats.py     variance, confidence intervals, noise floor
  compare.py   batch A vs batch B, three-way verdict
  cli.py
```

MIT.
