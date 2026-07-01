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
    ahead = search.evaluate_observation(_obs(me_prize=2, them_prize=5))
    behind = search.evaluate_observation(_obs(me_prize=5, them_prize=2))
    assert ahead > behind


@pytest.mark.unit
def test_evaluate_terminal_win_loss():
    from ptcg_bot import config as cfg

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
    assert search.evaluate_observation(win) == cfg.VALUE_WIN
    assert search.evaluate_observation(loss) == cfg.VALUE_LOSS


@pytest.mark.unit
def test_choose_by_search_returns_none_without_engine():
    obs = {
        "search_begin_input": "x",
        "select": {"maxCount": 1, "option": [{"type": 13}, {"type": 14}]},
    }
    assert search.choose_by_search(obs) is None  # cg not importable offline


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
