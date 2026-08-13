"""Deterministic brand extraction. No LLM here, deliberately.

If an LLM extracted the mentions, extraction variance would contaminate the
measurement and we would be measuring two noise sources at once. Everything in
this module is a pure function of (text, brand list).

Rank is defined as: position of the brand's FIRST mention, ordered against the
first mentions of the other brands in the same response. Rank 1 = mentioned
earliest. Rank is over *distinct brands*, not list index, so it is well-defined
for prose and for numbered lists alike.

Offsets are measured in the normalized text (see `normalize`), not the raw text.
Only their ordering is ever used, so the two agree.
"""

import unicodedata

__all__ = ["normalize", "first_offsets", "ranks"]


def normalize(text: str) -> str:
    """Casefold, strip accents, and reduce every non-alphanumeric run to one space.

    "Hoka One One!" and "hoka  one one" both become "hoka one one".
    """
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    # Drop combining marks rather than spacing them: NFKD turns "sénse" into
    # "se" + accent + "nse", and spacing the accent would split the word.
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    flattened = "".join(c if c.isalnum() else " " for c in stripped)
    return " ".join(flattened.split())


def first_offsets(text: str, brands: dict[str, list[str]]) -> dict[str, int]:
    """Offset of each brand's earliest mention. Absent brands are omitted, not zero.

    `brands` maps a canonical name to a list of accepted variants. The canonical
    name is always matched too, so `{"Brooks": []}` is valid.
    """
    haystack = f" {normalize(text)} "
    found: dict[str, int] = {}
    for brand, aliases in brands.items():
        best = None
        for variant in (brand, *aliases):
            needle = f" {normalize(variant)} "
            if not needle.strip():
                continue
            at = haystack.find(needle)
            if at != -1 and (best is None or at < best):
                best = at
        if best is not None:
            found[brand] = best
    return found


def ranks(text: str, brands: dict[str, list[str]]) -> dict[str, int]:
    """Brand -> 1-based rank by first mention. Unmentioned brands are absent.

    Absence is a different event from ranking last, so it is never encoded as a
    rank value. Callers track mention rate separately.
    """
    ordered = sorted(first_offsets(text, brands).items(), key=lambda kv: kv[1])
    return {brand: i + 1 for i, (brand, _) in enumerate(ordered)}


def demo() -> None:
    brands = {"Brooks": [], "Hoka": ["Hoka One One"], "New Balance": ["NB"]}
    text = "For flat feet, Hoka One One is the usual pick, then Brooks."
    assert ranks(text, brands) == {"Hoka": 1, "Brooks": 2}, ranks(text, brands)
    assert "New Balance" not in ranks(text, brands)
    print("extract ok")


if __name__ == "__main__":
    demo()
