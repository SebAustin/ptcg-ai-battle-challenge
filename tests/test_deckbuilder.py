"""Tests for the deckbuilder.

Unit tests use small synthetic pools (no competition data needed). The full
build is exercised as an integration test against the real pool, which skips
when the dataset is absent (same policy as tests/test_cards.py).
"""

from __future__ import annotations

import pytest

from deckbuilder import build_deck, classify, score, validate
from deckbuilder.deck import Deck
from deckbuilder.roles import Role
from ptcg_bot.cards import DEFAULT_CSV, Card, CardPool, Move, load_pool
from ptcg_bot.rules import EnergyType


def mk_card(
    card_id: int,
    name: str,
    subtype: str,
    *,
    rule: str = "",
    hp: int | None = None,
    poke_type: EnergyType | None = None,
    moves: tuple[Move, ...] = (),
    text: str = "",
    previous_stage: str = "",
) -> Card:
    return Card(
        card_id=card_id,
        name=name,
        expansion="TST",
        collection_no=str(card_id),
        subtype=subtype,
        category="",
        rule=rule,
        previous_stage=previous_stage,
        hp=hp,
        poke_type=poke_type,
        weakness=None,
        resistance=None,
        retreat=None,
        moves=moves,
        text=text,
    )


@pytest.fixture
def mini_pool() -> CardPool:
    return CardPool(
        [
            mk_card(1, "Testmon", "Basic Pokémon", hp=70, poke_type=EnergyType.WATER),
            mk_card(2, "Basic {W} Energy", "Basic Energy", poke_type=EnergyType.WATER),
            mk_card(3, "Draw Item", "Item", text="Draw 3 cards."),
            mk_card(4, "Big Rod", "Item", rule="ACE SPEC", text="Shuffle cards back."),
            mk_card(
                5, "Master Ball", "Item", rule="ACE SPEC", text="Search your deck."
            ),
        ]
    )


# --- constraints ------------------------------------------------------------


@pytest.mark.unit
def test_legal_deck_passes(mini_pool):
    deck = Deck.from_counts(
        {1: 1, 3: 4, 2: 55}
    )  # 60, 1 basic, energy exempt from 4-cap
    assert validate(deck, mini_pool) == []


@pytest.mark.unit
def test_wrong_size_fails(mini_pool):
    deck = Deck.from_counts({1: 1, 2: 58})  # 59 cards
    assert any("must be exactly" in p for p in validate(deck, mini_pool))


@pytest.mark.unit
def test_copy_limit_fails_for_non_energy(mini_pool):
    deck = Deck.from_counts({1: 1, 3: 5, 2: 54})  # 5 copies of an Item
    assert any("per-name limit" in p for p in validate(deck, mini_pool))


@pytest.mark.unit
def test_basic_energy_is_exempt_from_copy_limit(mini_pool):
    deck = Deck.from_counts({1: 1, 2: 59})  # 59 basic energy is fine
    assert validate(deck, mini_pool) == []


@pytest.mark.unit
def test_ace_spec_total_capped_across_different_cards(mini_pool):
    deck = Deck.from_counts({1: 1, 4: 1, 5: 1, 2: 57})  # two distinct ACE SPECs
    assert any("ACE SPEC" in p for p in validate(deck, mini_pool))


@pytest.mark.unit
def test_requires_a_basic_pokemon(mini_pool):
    deck = Deck.from_counts({3: 4, 2: 56})  # no Basic Pokémon
    assert any("Basic Pokémon" in p for p in validate(deck, mini_pool))


# --- roles ------------------------------------------------------------------


@pytest.mark.unit
def test_role_switch_and_search_and_draw():
    switch = mk_card(10, "Boss", "Supporter", text="Switch in the opponent's Pokémon.")
    search = mk_card(11, "Ball", "Item", text="Search your deck for a Pokémon.")
    draw = mk_card(12, "Prof", "Supporter", text="Draw 7 cards.")
    assert Role.SWITCH in classify(switch)
    assert Role.SEARCH in classify(search)
    assert Role.DRAW in classify(draw)


@pytest.mark.unit
def test_stadium_and_pokemon_roles():
    stadium = mk_card(13, "Arena", "Stadium", text="Both players' Pokémon...")
    poke = mk_card(14, "Mon", "Basic Pokémon", hp=60)
    assert Role.STADIUM in classify(stadium)
    assert classify(poke) == set()  # Pokémon have no trainer role


# --- deck model -------------------------------------------------------------


@pytest.mark.unit
def test_deck_from_counts_drops_zero_and_sorts():
    deck = Deck.from_counts({3: 0, 1: 2, 2: 1})
    assert deck.counts == ((1, 2), (2, 1))
    assert deck.total == 3


@pytest.mark.unit
def test_deck_csv_round_trip(mini_pool):
    deck = Deck.from_counts({1: 2, 3: 4, 2: 54})
    restored = Deck.from_csv(deck.to_csv(mini_pool))
    assert restored.counts == deck.counts


@pytest.mark.unit
def test_deck_csv_quotes_names_with_commas():
    pool = CardPool([mk_card(20, "Bill, the Great", "Item", text="Draw 2.")])
    csv_text = Deck.from_counts({20: 1}).to_csv(pool)
    assert '"Bill, the Great"' in csv_text


@pytest.mark.unit
def test_score_returns_bounded_total(mini_pool):
    deck = Deck.from_counts({1: 1, 3: 4, 2: 55})
    card = score(deck, mini_pool)
    assert 0.0 <= card.total <= 100.0
    assert dict(card.breakdown).keys() >= {"consistency", "energy", "basics"}


# --- full build (integration with the real pool) ----------------------------


@pytest.fixture(scope="module")
def pool() -> CardPool:
    if not DEFAULT_CSV.exists():
        pytest.skip(
            "competition dataset not present (data/EN_Card_Data.csv); run `make data`"
        )
    return load_pool()


@pytest.mark.integration
def test_build_deck_is_legal_and_full(pool):
    deck = build_deck(pool)
    assert deck.total == 60
    assert validate(deck, pool) == []


@pytest.mark.integration
def test_build_deck_is_deterministic(pool):
    assert build_deck(pool).counts == build_deck(pool).counts


@pytest.mark.integration
def test_build_deck_csv_round_trips(pool):
    deck = build_deck(pool)
    assert Deck.from_csv(deck.to_csv(pool)).counts == deck.counts


@pytest.mark.integration
def test_build_deck_scores_reasonably(pool):
    assert score(build_deck(pool), pool).total >= 80.0
