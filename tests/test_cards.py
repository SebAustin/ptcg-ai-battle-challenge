"""Unit tests for the card model + loader (ptcg_bot.cards) and rules helpers.

These guard the data layer everything else is built on: cost/damage parsing,
evolution-line resolution, supertype flags, and the dirty-data edge cases we
saw in the export (Dragon-as-kanji, multi-move cards, null sentinels).
"""

from __future__ import annotations

import pytest

from ptcg_bot.cards import EnergyCost, parse_cost, parse_damage, load_pool
from ptcg_bot.rules import EnergyType, normalize_type, prizes_for_knockout


# --- cost parsing -----------------------------------------------------------

def test_parse_cost_mixes_typed_and_colorless():
    # Arrange / Act
    cost = parse_cost("{D}●●")
    # Assert
    assert cost.colorless == 2
    assert cost.as_dict()[EnergyType.DARKNESS] == 1
    assert cost.total == 3


def test_parse_cost_explicit_colorless_brace_counts_as_colorless():
    cost = parse_cost("{C}{C}")
    assert cost.colorless == 2
    assert cost.total == 2
    assert EnergyType.COLORLESS not in dict(cost.typed)  # colorless never in typed


def test_parse_cost_pure_pips():
    assert parse_cost("●●●").total == 3
    assert parse_cost("●●●").colorless == 3


def test_parse_cost_empty_and_na_are_free():
    assert parse_cost("").total == 0
    assert parse_cost("n/a").total == 0
    assert not parse_cost("")  # __bool__ is False for a free cost


def test_parse_cost_unknown_token_degrades_to_colorless():
    with pytest.warns(UserWarning):
        cost = parse_cost("{Team Rocket}")
    assert cost.total == 1
    assert cost.colorless == 1


# --- damage parsing ---------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("120", (120, "")),
        ("30×", (30, "×")),
        ("100+", (100, "+")),
        ("n/a", (0, "")),
        ("", (0, "")),
    ],
)
def test_parse_damage(raw, expected):
    assert parse_damage(raw) == expected


# --- type normalization -----------------------------------------------------

def test_normalize_dragon_kanji():
    # The EN export leaks Dragon as the Japanese kanji.
    assert normalize_type("竜") is EnergyType.DRAGON


def test_normalize_braced_and_null():
    assert normalize_type("{G}") is EnergyType.GRASS
    assert normalize_type("n/a") is None
    assert normalize_type("") is None


# --- prize rules ------------------------------------------------------------

def test_prizes_for_knockout_by_rule():
    assert prizes_for_knockout("Mega Pokémon ex") == 3
    assert prizes_for_knockout("Pokémon ex") == 2
    assert prizes_for_knockout("") == 1
    assert prizes_for_knockout(None) == 1


# --- pool loading (integration with the real dataset) -----------------------

@pytest.fixture(scope="module")
def pool():
    return load_pool()


def test_pool_loads_expected_size(pool):
    # 1267 distinct Card IDs confirmed from the 2026-06 export.
    assert len(pool) == 1267


def test_multi_move_card_keeps_all_attacks(pool):
    # Team Rocket's Kangaskhan ex (id 24) has two attacks across two CSV rows.
    kanga = pool.by_id(24)
    assert kanga is not None
    assert kanga.is_ex
    assert kanga.is_basic_pokemon
    assert len(kanga.attacks) == 2
    assert any("Comet Punch" in m.name for m in kanga.attacks)


def test_evolution_line_resolves_by_name(pool):
    # Greninja ex (id 40) evolves from Frogadier.
    greninja = pool.by_id(40)
    assert greninja.previous_stage == "Frogadier"
    assert greninja.stage == 2
    # And the pool can find what evolves from a given name.
    assert greninja in pool.evolutions_of("Frogadier")


def test_supertype_partition_covers_pool(pool):
    # Every card is exactly one of Pokémon / Trainer / Energy.
    for card in pool:
        flags = (card.is_pokemon, card.is_trainer, card.is_energy)
        assert sum(flags) == 1, f"{card.name} has ambiguous supertype {card.subtype!r}"


def test_basic_energy_present(pool):
    # In-deck Basic Energy is the hallmark of classic TCG (vs Pocket).
    basics = [c for c in pool.energy if c.is_basic_energy]
    assert len(basics) >= 8
