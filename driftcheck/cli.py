"""driftcheck CLI: `driftcheck run` and `driftcheck compare`."""

import argparse
import json
import sys
from datetime import datetime, timezone

from . import __version__, providers
from .compare import compare
from .extract import ranks
from .sample import estimate_cost, run_batch
from .stats import noise_floor, summarize


def _load_brands(path: str) -> dict[str, list[str]]:
    """Brand file: {"Brooks": ["Brooks Running"], "Hoka": []} or a plain list."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, list):
        return {name: [] for name in raw}
    if isinstance(raw, dict):
        return {name: list(aliases or []) for name, aliases in raw.items()}
    raise SystemExit(f"{path}: expected a JSON object or array of brand names")


def _cmd_run(args) -> int:
    brands = _load_brands(args.brands)
    prompt = args.prompt
    provider = providers.get(args.provider)
    model = args.model or provider.DEFAULT_MODEL

    rpm = provider.RPM if args.rpm is None else args.rpm
    cost, input_tokens = estimate_cost(provider, prompt, model, args.n, args.max_tokens)
    lines = [
        f"{args.n} runs of {model} via {args.provider} "
        f"({input_tokens} input tokens, up to {args.max_tokens} output tokens each)",
        f"Estimated worst-case cost: ${cost:.2f}",
    ]
    if provider.FREE_NOTE:
        lines.append(provider.FREE_NOTE)
    if rpm:
        lines.append(
            f"Paced to {rpm:g} requests/minute: "
            f"about {max(1, round(args.n / rpm))} minute(s) to run."
        )
    print("\n".join(lines), file=sys.stderr)

    if not args.yes:
        if input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Aborted.", file=sys.stderr)
            return 1

    texts = run_batch(
        provider,
        prompt,
        model,
        args.n,
        max_tokens=args.max_tokens,
        effort=args.effort,
        temperature=args.temperature,
        workers=args.workers,
        rpm=rpm,
    )

    run_ranks = [ranks(t, brands) for t in texts]
    summary = summarize(run_ranks, list(brands))

    result = {
        "driftcheck_version": __version__,
        "run": {
            "prompt": prompt,
            "provider": args.provider,
            "model": model,
            "n": args.n,
            "max_tokens": args.max_tokens,
            "effort": args.effort,
            # None means the parameter was not sent. Current Claude models reject
            # temperature outright; see the README. There is no seed parameter.
            "temperature": args.temperature,
            "seed": None,
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "brands": summary,
        "noise_floor": noise_floor(summary, args.n),
    }
    if args.save_responses:
        result["responses"] = texts

    _emit(result, args.out)
    return 0


def _cmd_compare(args) -> int:
    with open(args.a, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(args.b, encoding="utf-8") as fh:
        b = json.load(fh)
    _emit(compare(a, b), args.out)
    return 0


def _emit(payload: dict, out: str | None) -> None:
    text = json.dumps(payload, indent=2)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"Wrote {out}", file=sys.stderr)
    else:
        print(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="driftcheck",
        description="Measure the noise floor of LLM brand-visibility metrics.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one prompt N times and summarize")
    run.add_argument("prompt")
    run.add_argument("--brands", required=True, help="path to a JSON brand list")
    run.add_argument("-n", type=int, default=30, help="runs in the batch (default 30)")
    run.add_argument(
        "--provider",
        default=providers.DEFAULT,
        choices=providers.names(),
        help=f"model provider (default {providers.DEFAULT})",
    )
    run.add_argument("--model", default=None, help="default: the provider's own")
    # 2048, not 1024: a real "best X for Y" answer runs ~900-1000 tokens, and a
    # response cut off at the ceiling is a failed run, not a shorter one.
    run.add_argument("--max-tokens", type=int, default=2048)
    run.add_argument(
        "--effort",
        default="low",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Claude only; ignored by providers without an effort knob",
    )
    run.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="not accepted by current Claude models; omitted unless set",
    )
    run.add_argument("--workers", type=int, default=4)
    run.add_argument(
        "--rpm",
        type=float,
        default=None,
        help="override the provider's request pacing; 0 disables it",
    )
    run.add_argument("--save-responses", action="store_true")
    run.add_argument("-o", "--out", help="write JSON here instead of stdout")
    run.add_argument("-y", "--yes", action="store_true", help="skip the cost prompt")
    run.set_defaults(func=_cmd_run)

    cmp_ = sub.add_parser("compare", help="compare two run results")
    cmp_.add_argument("a", help="baseline result JSON")
    cmp_.add_argument("b", help="later result JSON")
    cmp_.add_argument("-o", "--out", help="write JSON here instead of stdout")
    cmp_.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
