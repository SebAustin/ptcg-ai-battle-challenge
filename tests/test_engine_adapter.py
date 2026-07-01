"""Tests for the engine adapter.

Unit tests parse a synthetic observation dict against a tiny synthetic pool — no
engine or competition data needed. One integration test drives the real libcg
engine and skips when ``engine/`` is absent (fetch it with ``make engine``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ptcg_bot.cards import Card, CardPool
from ptcg_bot.engine_adapter import (
    EngineNotWiredError,
    encode_action,
    is_deck_request,
    parse_observation,
)
from ptcg_bot.rules import EnergyType

_ENGINE = (
    Path(__file__).resolve().parent.parent
    / "engine"
    / "sample_submission"
    / "sample_submission"
)


def mk_card(card_id: int, subtype: str = "Basic Pokémon", hp: int | None = 70) -> Card:
    return Card(
        card_id=card_id,
        name=f"C{card_id}",
        expansion="TST",
        collection_no="1",
        subtype=subtype,
        category="",
        rule="",
        previous_stage="",
        hp=hp,
        poke_type=None,
        weakness=None,
        resistance=None,
        retreat=None,
        moves=(),
    )


@pytest.fixture
def pool() -> CardPool:
    return CardPool([mk_card(100), mk_card(101), mk_card(102, "Stage 1 Pokémon")])


def _obs(your_index: int = 0) -> dict:
    return {
        "select": {"type": 0, "option": [{}, {}], "minCount": 1, "maxCount": 1},
        "current": {
            "yourIndex": your_index,
            "turn": 3,
            "players": [
                {
                    "active": [{"id": 100, "hp": 50, "maxHp": 70, "energies": [2, 3]}],
                    "bench": [{"id": 101, "hp": 60, "maxHp": 60, "energies": []}],
                    "handCount": 5,
                    "deckCount": 40,
                    "prize": [None, None, None, None],
                },
                {
                    "active": [{"id": 102, "hp": 90, "maxHp": 90, "energies": []}],
                    "bench": [],
                    "handCount": 6,
                    "deckCount": 44,
                    "prize": [None] * 6,
                },
            ],
        },
    }


@pytest.mark.unit
def test_parse_observation_maps_us_and_them(pool):
    state = parse_observation(_obs(your_index=0), pool)
    assert state.turn == 3
    assert state.us.prizes_remaining == 4
    assert state.them.prizes_remaining == 6
    assert state.us.hand_size == 5
    assert state.us.active is not None
    assert state.us.active.remaining_hp == 50  # 70 maxHp - 20 damage
    assert state.us.active.attached == (EnergyType.FIRE, EnergyType.WATER)
    assert state.us.bench_basics == 1


@pytest.mark.unit
def test_your_index_selects_the_right_side(pool):
    # With yourIndex=1, "us" is the second player (6 prizes, id 102 active).
    state = parse_observation(_obs(your_index=1), pool)
    assert state.us.prizes_remaining == 6
    assert state.them.prizes_remaining == 4


@pytest.mark.unit
def test_is_deck_request():
    assert is_deck_request({"select": None, "current": None})
    assert not is_deck_request({"select": {"option": []}})


@pytest.mark.unit
def test_parse_observation_without_state_raises(pool):
    with pytest.raises(EngineNotWiredError):
        parse_observation({"select": None, "current": None}, pool)


@pytest.mark.unit
def test_encode_action_validates():
    assert encode_action([0, 2, 5]) == [0, 2, 5]
    with pytest.raises(ValueError):
        encode_action([0, 0])  # duplicate
    with pytest.raises(ValueError):
        encode_action([-1])  # negative


# --- live engine (skips without engine/) ------------------------------------


@pytest.mark.integration
def test_engine_accepts_deck_and_obs_parses():
    if not (_ENGINE / "cg" / "api.py").exists():
        pytest.skip("engine not present (run `make engine`)")

    from ptcg_bot.cards import DEFAULT_CSV, load_pool

    if not DEFAULT_CSV.exists():
        pytest.skip("competition dataset not present (run `make data`)")
    sys.path.insert(0, str(_ENGINE))
    from cg import game  # type: ignore[import-not-found]

    from deckbuilder import build_deck

    real_pool = load_pool()
    deck_ids = [cid for cid, n in build_deck(real_pool).counts for _ in range(n)]
    obs, start = game.battle_start(deck_ids, list(deck_ids))
    try:
        assert (
            obs is not None
        ), f"engine rejected our deck (errorType={start.errorType})"
        state = parse_observation(obs, real_pool)
        assert 0 <= state.us.prizes_remaining <= 6
    finally:
        game.battle_finish()
