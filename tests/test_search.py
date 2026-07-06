"""Tests for the search layer (offline — no engine needed).

Without the ``cg`` engine on the path, search + belief degrade gracefully
(return ``None``), so ``search_agent`` behaves like the heuristic. The evaluator
is tested against a duck-typed observation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ptcg_bot import belief, search
from ptcg_bot.main import search_agent


def _player(
    prize_n: int, active_hp: int | None, bench=(), hand: int = 0
) -> SimpleNamespace:
    active = (
        [SimpleNamespace(hp=active_hp, energies=[])]
        if active_hp is not None
        else [None]
    )
    return SimpleNamespace(
        active=active, bench=list(bench), prize=[None] * prize_n, handCount=hand
    )


def _obs(me_prize: int, them_prize: int) -> SimpleNamespace:
    current = SimpleNamespace(
        yourIndex=0,
        result=-1,
        players=[_player(me_prize, 100), _player(them_prize, 100)],
    )
    return SimpleNamespace(current=current)


@pytest.mark.unit
def test_evaluate_observation_prefers_prize_lead():
    ahead = search.evaluate_observation(_obs(me_prize=2, them_prize=5), 0)
    behind = search.evaluate_observation(_obs(me_prize=5, them_prize=2), 0)
    assert ahead > behind


@pytest.mark.unit
def test_evaluate_terminal_win_loss():
    win = SimpleNamespace(
        current=SimpleNamespace(
            yourIndex=0, result=0, players=[_player(1, 100), _player(1, 100)]
        )
    )
    loss = SimpleNamespace(
        current=SimpleNamespace(
            yourIndex=0, result=1, players=[_player(1, 100), _player(1, 100)]
        )
    )
    draw = SimpleNamespace(
        current=SimpleNamespace(
            yourIndex=0, result=2, players=[_player(1, 100), _player(1, 100)]
        )
    )
    # Win-probability scale, from player 0's fixed perspective.
    assert search.evaluate_observation(win, 0) == 1.0
    assert search.evaluate_observation(loss, 0) == 0.0
    assert search.evaluate_observation(draw, 0) == 0.5


@pytest.mark.unit
def test_learned_leaf_used_and_guarded(monkeypatch):
    from ptcg_bot import eval_weights, features

    obs = _obs(me_prize=2, them_prize=5)
    # Learned path: a valid-length vector goes through eval_weights.predict.
    value = search.evaluate_observation(obs, 0)
    assert 0.0 < value < 1.0
    # Version-skew guard: FEATURE_COUNT mismatch -> heuristic fallback, no raise.
    monkeypatch.setattr(features, "extract", lambda o, m: [0.0])
    fallback = search.evaluate_observation(obs, 0)
    assert 0.0 < fallback < 1.0
    assert eval_weights.FEATURE_COUNT != 1


@pytest.mark.unit
def test_choose_by_search_returns_none_without_engine():
    obs = {
        "search_begin_input": "x",
        "select": {"maxCount": 1, "option": [{"type": 13}, {"type": 14}]},
    }
    assert search.choose_by_search(obs) is None  # cg not importable offline


@pytest.mark.unit
def test_determinize_shapes_and_our_side(monkeypatch):
    import random

    from ptcg_bot import metadata

    monkeypatch.setattr(metadata, "valid_card_ids", lambda: tuple(range(1, 11)))
    monkeypatch.setattr(metadata, "basic_pokemon_ids", lambda: (7,))

    def card(i: int) -> SimpleNamespace:
        return SimpleNamespace(id=i)

    us = SimpleNamespace(
        deckCount=3,
        prize=[None, None],
        handCount=2,
        hand=[card(1), card(2)],
        active=[SimpleNamespace(id=3, energyCards=[], preEvolution=[])],
        bench=[],
        discard=[],
    )
    them = SimpleNamespace(
        deckCount=4, prize=[None, None], handCount=3, active=[card(9)], bench=[]
    )
    obs = SimpleNamespace(current=SimpleNamespace(yourIndex=0, players=[us, them]))

    your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active = (
        belief.determinize(obs, list(range(1, 11)), random.Random(0))
    )
    assert (len(your_deck), len(your_prize)) == (3, 2)
    assert (len(opp_deck), len(opp_prize), len(opp_hand)) == (4, 2, 3)
    # Our unseen cards come from the decklist minus what's visible (1, 2, 3).
    assert set(your_deck + your_prize) <= set(range(4, 11))
    assert opp_active == []  # opponent Active is face-up (id 9), so not determinized


@pytest.mark.unit
def test_belief_determinize_none_without_metadata(monkeypatch):
    # Force the no-metadata path (isolation-safe: a prior integration test may
    # have put the engine on sys.path, making cg importable session-wide).
    from ptcg_bot import metadata

    monkeypatch.setattr(metadata, "valid_card_ids", lambda: ())
    monkeypatch.setattr(metadata, "basic_pokemon_ids", lambda: ())
    obs = SimpleNamespace(
        current=SimpleNamespace(yourIndex=0, players=[_player(6, 100), _player(6, 100)])
    )
    assert belief.determinize(obs) is None


@pytest.mark.unit
def test_search_agent_falls_back_to_heuristic_offline():
    # Search returns None offline -> attack-first heuristic picks the ATTACK option.
    obs = {
        "search_begin_input": "x",
        "select": {"maxCount": 1, "option": [{"type": 14}, {"type": 13}, {"type": 8}]},
    }
    assert search_agent(obs) == [1]
