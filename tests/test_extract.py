from driftcheck.extract import first_offsets, normalize, ranks

BRANDS = {
    "Brooks": ["Brooks Running"],
    "Hoka": ["Hoka One One"],
    "New Balance": ["NB"],
    "ASICS": [],
}


def test_normalize_collapses_case_punctuation_and_accents():
    assert normalize("Hoka One One!") == "hoka one one"
    assert normalize("  ASICS   Gel-Kayano ") == "asics gel kayano"
    assert normalize("Salomon Sénse") == "salomon sense"


def test_rank_is_order_of_first_mention():
    text = "Start with Hoka, then Brooks, then ASICS."
    assert ranks(text, BRANDS) == {"Hoka": 1, "Brooks": 2, "ASICS": 3}


def test_rank_uses_first_mention_not_last():
    text = "Brooks is solid. Hoka is popular. Brooks again, and Brooks once more."
    assert ranks(text, BRANDS)["Brooks"] == 1


def test_absent_brands_are_omitted_not_ranked_zero():
    result = ranks("Only ASICS here.", BRANDS)
    assert result == {"ASICS": 1}
    assert "Brooks" not in result
    assert 0 not in result.values()


def test_aliases_match_and_map_to_the_canonical_name():
    text = "Hoka One One and Brooks Running are the picks."
    assert ranks(text, BRANDS) == {"Hoka": 1, "Brooks": 2}


def test_matching_is_word_bounded():
    # "NB" must not fire on "NBA"; "Hoka" must not fire on "Hokas"
    assert first_offsets("The NBA season", BRANDS) == {}
    assert first_offsets("nice Hokas though", BRANDS) == {}
    assert "New Balance" in first_offsets("I wear NB shoes", BRANDS)


def test_numbered_list_ranks_by_position():
    text = "1. ASICS\n2. Brooks\n3. Hoka"
    assert ranks(text, BRANDS) == {"ASICS": 1, "Brooks": 2, "Hoka": 3}


def test_ranks_are_dense_over_mentioned_brands_only():
    # Two of four brands appear -> ranks 1 and 2, never 1 and 3.
    assert sorted(ranks("Hoka beats ASICS.", BRANDS).values()) == [1, 2]


def test_extraction_is_deterministic():
    text = "Brooks, Hoka, and NB are all fine."
    assert ranks(text, BRANDS) == ranks(text, BRANDS)


def test_empty_text_yields_no_mentions():
    assert ranks("", BRANDS) == {}
