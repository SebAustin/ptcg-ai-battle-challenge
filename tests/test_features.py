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
def test_feature_names_unique_and_counted():
    assert len(features.FEATURE_NAMES) == features.FEATURE_COUNT == 61
    assert len(set(features.FEATURE_NAMES)) == features.FEATURE_COUNT


def _fake_metadata(monkeypatch):
    """Deterministic metadata: active id=100 (stage1 ex, cost 3, retreat 2),
    bench id=101 (basic megaEx, evolvable, cost 1)."""
    from ptcg_bot import metadata

    monkeypatch.setattr(metadata, "stage_info", lambda: {100: (1, 2), 101: (0, 3)})
    monkeypatch.setattr(metadata, "evolvable_ids", lambda: frozenset({101}))
    monkeypatch.setattr(metadata, "retreat_cost", lambda: {100: 2})
    monkeypatch.setattr(metadata, "card_kind", lambda: {900: 5, 901: 0, 902: 3})
    monkeypatch.setattr(metadata, "attack_cost", lambda: {100: 3, 101: 1})
    monkeypatch.setattr(metadata, "card_power", lambda: {})
    monkeypatch.setattr(metadata, "type_info", lambda: {})


@pytest.mark.unit
def test_v2_features_from_metadata(monkeypatch):
    _fake_metadata(monkeypatch)
    obs = _obs()
    obs["current"]["players"][0]["discard"] = [
        {"id": 900},
        {"id": 900},
        {"id": 901},
        {"id": 902},
    ]
    feats = features.extract(obs, 0)
    names = features.FEATURE_NAMES

    def val(name):
        return feats[names.index(name)]

    assert val("us_active_stage") == 0.5  # stage 1 of 2
    assert val("us_active_prize_risk") == 0.5  # ex -> opponent takes 2
    assert val("us_bench_prize_risk") == 1.0  # megaEx on bench -> takes 3
    assert val("us_active_evolvable") == 0.0
    assert val("us_bench_evolvable") == pytest.approx(1 / 5)
    assert val("us_active_deficit") == pytest.approx(1 / 3)  # cost 3, 2 attached
    assert val("us_bench_ready_deficit") == 0.0  # bench cost 1, 1 attached
    assert val("us_active_retreat") == 0.5  # retreat 2 of 4
    assert val("us_discard_energy") == pytest.approx(2 / 15)
    assert val("us_discard_pokemon") == pytest.approx(1 / 10)
    assert val("them_discard_energy") == 0.0  # their discard is empty


@pytest.mark.unit
def test_v2_features_zero_offline(monkeypatch):
    """Without cg metadata every v2 feature must be exactly 0.0 (the offline
    convention: structural v1 features carry the load, metadata adds on top)."""
    from ptcg_bot import metadata

    for fn in (
        "stage_info",
        "retreat_cost",
        "card_kind",
        "attack_cost",
        "card_power",
        "type_info",
    ):
        monkeypatch.setattr(metadata, fn, dict)
    monkeypatch.setattr(metadata, "evolvable_ids", frozenset)

    feats = features.extract(_obs(), 0)
    names = features.FEATURE_NAMES
    for name in names[41:]:
        assert feats[names.index(name)] == 0.0, name


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
