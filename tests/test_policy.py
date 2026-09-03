"""Tests for the behavior-cloned policy wrapper (fake weights, offline)."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from ptcg_bot import config as cfg
from ptcg_bot import features, option_features, policy


def _obs(n_options: int = 3, sel_type: int = 0, max_count: int = 1) -> dict:
    return {
        "current": {"yourIndex": 0, "turn": 3, "players": [{}, {}]},
        "select": {
            "type": sel_type,
            "maxCount": max_count,
            "option": [{"type": 7, "area": 2, "index": i} for i in range(n_options)],
        },
    }


def _install_fake_weights(monkeypatch, favourite: int) -> None:
    """A policy_weights stand-in that scores option ``favourite`` highest via r_index."""
    fake = ModuleType("ptcg_bot.policy_weights")
    fake.STATE_COUNT = features.FEATURE_COUNT + option_features.DECISION_COUNT
    fake.OPTION_COUNT = option_features.OPTION_COUNT
    idx = option_features.OPTION_FEATURE_NAMES.index("r_index")

    def prepare(state):
        return tuple(state)

    def score(ctx, option):
        # r_index is index/(n-1): closest to the favourite's relative position wins.
        return -abs(option[idx] - favourite / 2.0)

    fake.prepare, fake.score = prepare, score
    monkeypatch.setitem(sys.modules, "ptcg_bot.policy_weights", fake)


@pytest.mark.unit
def test_scores_none_without_weights(monkeypatch):
    monkeypatch.setitem(sys.modules, "ptcg_bot.policy_weights", None)
    assert policy.scores(_obs()) is None
    assert policy.choose(_obs()) is None


@pytest.mark.unit
def test_choose_is_legal_and_prefers_favourite(monkeypatch):
    _install_fake_weights(monkeypatch, favourite=2)
    picked = policy.choose(_obs(3))
    assert picked == [2]
    multi = policy.choose(_obs(3, max_count=2))
    assert multi is not None and len(multi) == 2 and multi == sorted(set(multi))


@pytest.mark.unit
def test_declines_unsupported_and_wide_prompts(monkeypatch):
    _install_fake_weights(monkeypatch, favourite=0)
    assert policy.scores(_obs(3, sel_type=9)) is None  # YES/NO not supported
    monkeypatch.setattr(cfg, "BC_MAX_OPTIONS", 2)
    assert policy.scores(_obs(3)) is None
    assert policy.scores({"select": None}) is None
    assert policy.scores(None) is None


@pytest.mark.unit
def test_flag_routes_heuristic_to_policy(monkeypatch):
    from ptcg_bot import main as m

    _install_fake_weights(monkeypatch, favourite=1)
    obs = _obs(3)
    obs["select"]["option"][0]["type"] = 8  # v7 would attach first (index 0)
    monkeypatch.setattr(cfg, "BC_POLICY", 0)
    assert m._heuristic_selection(obs) == [0]
    monkeypatch.setattr(cfg, "BC_POLICY", 1)
    assert m._heuristic_selection(obs) == [1]
