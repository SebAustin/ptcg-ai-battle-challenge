"""Tests for the feature extractor and the generated evaluator weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ptcg_bot import eval_weights, features


def _pokemon(pid: int = 100, hp: int = 80, max_hp: int = 100, energies: int = 2):
    return {"id": pid, "hp": hp, "maxHp": max_hp, "energies": [3] * energies}


def _player(prizes: int = 6, hand: int = 5, deck: int = 40, active_hp: int = 80):
    return {
        "prize": [None] * prizes,
        "handCount": hand,
        "deckCount": deck,
        "discard": [],
        "active": [_pokemon(hp=active_hp)],
        "bench": [_pokemon(pid=101, hp=60, max_hp=60, energies=1)],
    }


def _obs(us_prizes: int = 6, them_prizes: int = 6) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "turn": 5,
            "players": [_player(prizes=us_prizes), _player(prizes=them_prizes)],
        }
    }


@pytest.mark.unit
def test_extract_length_and_bounds():
    feats = features.extract(_obs(), 0)
    assert len(feats) == features.FEATURE_COUNT
    assert all(-1.0 <= f <= 1.0 for f in feats)


@pytest.mark.unit
def test_extract_dict_and_namespace_agree():
    obs_dict = _obs()
    # Same state as attribute-style objects (engine dataclass shape).
    ns_players = []
    for p in obs_dict["current"]["players"]:
        ns_players.append(
            SimpleNamespace(
                prize=p["prize"],
                handCount=p["handCount"],
                deckCount=p["deckCount"],
                discard=p["discard"],
                active=[SimpleNamespace(**p["active"][0])],
                bench=[SimpleNamespace(**p["bench"][0])],
            )
        )
    obs_ns = SimpleNamespace(
        current=SimpleNamespace(yourIndex=0, turn=5, players=ns_players)
    )
    assert features.extract(obs_dict, 0) == features.extract(obs_ns, 0)


@pytest.mark.unit
def test_extract_perspective_swaps_sides():
    obs = _obs(us_prizes=2, them_prizes=5)
    us_view = features.extract(obs, 0)
    them_view = features.extract(obs, 1)
    names = features.FEATURE_NAMES
    assert us_view[names.index("us_prizes")] == them_view[names.index("them_prizes")]
    assert us_view[names.index("prize_lead")] == -them_view[names.index("prize_lead")]


@pytest.mark.unit
def test_extract_garbage_returns_zero_vector():
    assert features.extract(None, 0) == [0.0] * features.FEATURE_COUNT
    assert features.extract({"current": {}}, 0) == [0.0] * features.FEATURE_COUNT
    assert features.extract(object(), 1) == [0.0] * features.FEATURE_COUNT


@pytest.mark.unit
def test_eval_weights_pure_and_bounded():
    assert eval_weights.FEATURE_COUNT == features.FEATURE_COUNT
    zero = eval_weights.predict([0.0] * eval_weights.FEATURE_COUNT)
    ones = eval_weights.predict([1.0] * eval_weights.FEATURE_COUNT)
    huge = eval_weights.predict([1e9] * eval_weights.FEATURE_COUNT)
    for value in (zero, ones, huge):
        assert 0.0 < value < 1.0  # sigmoid output, overflow-clamped
    # Deterministic.
    assert zero == eval_weights.predict([0.0] * eval_weights.FEATURE_COUNT)


@pytest.mark.unit
def test_model_prefers_prize_lead():
    ahead = features.extract(_obs(us_prizes=1, them_prizes=5), 0)
    behind = features.extract(_obs(us_prizes=5, them_prizes=1), 0)
    assert eval_weights.predict(ahead) > eval_weights.predict(behind)
