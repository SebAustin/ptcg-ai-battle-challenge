"""Tests for the internal state model (ptcg_bot.state) and heuristic evaluator.

All synthetic — no competition data or engine needed.
"""

from __future__ import annotations

import pytest

from ptcg_bot import config as cfg
from ptcg_bot.cards import Card, EnergyCost, Move
from ptcg_bot.evaluate import explain, state_value
from ptcg_bot.rules import EnergyType
from ptcg_bot.state import GameState, PlayerState, PokemonInPlay, can_pay


def mk_card(
    name: str,
    *,
    subtype: str = "Basic Pokémon",
    hp: int | None = 70,
    poke_type: EnergyType | None = None,
    weakness: EnergyType | None = None,
    moves: tuple[Move, ...] = (),
) -> Card:
    return Card(
        card_id=abs(hash(name)) % 100000,
        name=name,
        expansion="TST",
        collection_no="1",
        subtype=subtype,
        category="",
        rule="",
        previous_stage="",
        hp=hp,
        poke_type=poke_type,
        weakness=weakness,
        resistance=None,
        retreat=None,
        moves=moves,
    )


JAB = Move(
    name="Jab",
    cost=EnergyCost(colorless=1),
    damage_base=20,
    damage_modifier="",
    effect="",
)


# --- energy payment ---------------------------------------------------------


@pytest.mark.unit
def test_can_pay_colorless_from_any():
    assert can_pay(EnergyCost(colorless=2), (EnergyType.WATER, EnergyType.FIRE))


@pytest.mark.unit
def test_can_pay_requires_typed():
    cost = EnergyCost(typed=((EnergyType.FIRE, 1),))
    assert not can_pay(cost, (EnergyType.WATER,))
    assert can_pay(cost, (EnergyType.FIRE,))


@pytest.mark.unit
def test_can_pay_typed_plus_colorless():
    cost = EnergyCost(colorless=1, typed=((EnergyType.FIRE, 1),))
    assert can_pay(cost, (EnergyType.FIRE, EnergyType.WATER))
    assert not can_pay(cost, (EnergyType.FIRE,))  # missing the colorless


# --- pokemon in play --------------------------------------------------------


@pytest.mark.unit
def test_remaining_hp_and_ko():
    poke = PokemonInPlay(mk_card("Mon", hp=70), damage=20)
    assert poke.remaining_hp == 50
    assert not poke.is_knocked_out
    assert PokemonInPlay(mk_card("Mon", hp=70), damage=70).is_knocked_out


@pytest.mark.unit
def test_can_attack_needs_energy():
    attacker = mk_card("Striker", moves=(JAB,))
    assert PokemonInPlay(attacker, attached=(EnergyType.WATER,)).can_attack
    assert not PokemonInPlay(attacker, attached=()).can_attack


# --- player / game state ----------------------------------------------------


@pytest.mark.unit
def test_bench_basics_and_energy_on_board():
    basic = PokemonInPlay(mk_card("Basic", subtype="Basic Pokémon"))
    stage1 = PokemonInPlay(mk_card("Evo", subtype="Stage 1 Pokémon"))
    active = PokemonInPlay(
        mk_card("Act"), attached=(EnergyType.WATER, EnergyType.WATER)
    )
    player = PlayerState(active=active, bench=(basic, stage1))
    assert player.bench_basics == 1
    assert player.energy_on_board == 2
    assert len(player.in_play) == 3


@pytest.mark.unit
def test_winner_by_prizes_and_wipeout():
    live = PlayerState(active=PokemonInPlay(mk_card("A")))
    won = PlayerState(active=PokemonInPlay(mk_card("A")), prizes_remaining=0)
    empty = PlayerState(active=None, bench=())
    assert GameState(us=live, them=live, turn=1).winner == 0
    assert GameState(us=won, them=live).winner == 1
    assert GameState(us=live, them=empty).winner == 1


# --- evaluator --------------------------------------------------------------


@pytest.mark.unit
def test_terminal_clamps_to_win_loss():
    live = PlayerState(active=PokemonInPlay(mk_card("A")))
    empty = PlayerState(active=None)
    assert state_value(GameState(us=live, them=empty)) == cfg.VALUE_WIN
    assert state_value(GameState(us=empty, them=live)) == cfg.VALUE_LOSS


@pytest.mark.unit
def test_prize_lead_dominates():
    a = PlayerState(active=PokemonInPlay(mk_card("A")), prizes_remaining=2)
    b = PlayerState(active=PokemonInPlay(mk_card("B")), prizes_remaining=5)
    ahead = state_value(GameState(us=a, them=b))
    behind = state_value(GameState(us=b, them=a))
    assert ahead > behind
    assert ahead - behind == pytest.approx(2 * 3 * cfg.W_PRIZE_LEAD)


@pytest.mark.unit
def test_type_advantage_bonus():
    ours = PokemonInPlay(mk_card(" Curr", poke_type=EnergyType.WATER))
    weak = PokemonInPlay(mk_card("Weak", weakness=EnergyType.WATER))
    strong = PokemonInPlay(mk_card("Strong", weakness=EnergyType.FIRE))
    with_adv = GameState(us=PlayerState(active=ours), them=PlayerState(active=weak))
    without = GameState(us=PlayerState(active=ours), them=PlayerState(active=strong))
    assert "type_advantage" in explain(with_adv)
    assert "type_advantage" not in explain(without)
