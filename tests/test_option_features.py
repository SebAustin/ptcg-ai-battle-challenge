"""Tests for the option/decision feature extractor (synthetic, offline)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ptcg_bot import metadata
from ptcg_bot import option_features as of


def _obs() -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "turn": 6,
            "players": [
                {
                    "hand": [{"id": 300}, {"id": 301}],
                    "active": [{"id": 100, "hp": 80, "maxHp": 100, "energies": [3, 3]}],
                    "bench": [{"id": 101, "hp": 60, "maxHp": 60, "energies": []}],
                },
                {"active": [{"id": 200, "hp": 40, "maxHp": 120, "energies": [1]}]},
            ],
        },
        "select": {
            "type": 0,
            "maxCount": 1,
            "option": [
                {"type": 13, "attackId": 901},
                {"type": 7, "area": 2, "index": 1},
                {"type": 8, "area": 2, "index": 0, "inPlayArea": 5, "inPlayIndex": 0},
                {"type": 14},
            ],
        },
    }


def _fake_meta(monkeypatch) -> None:
    monkeypatch.setattr(metadata, "card_kind", lambda: {300: 5, 301: 0, 101: 0, 100: 0})
    monkeypatch.setattr(metadata, "card_power", lambda: {301: (90, 120), 101: (30, 60)})
    monkeypatch.setattr(metadata, "card_traits", lambda: {301: (1, 1, 0, 2, 1)})
    monkeypatch.setattr(metadata, "attack_cost", lambda: {101: 1, 100: 2})
    monkeypatch.setattr(metadata, "trainer_value", dict)
    monkeypatch.setattr(metadata, "type_info", lambda: {301: (2, 1), 100: (2, 3)})
    monkeypatch.setattr(metadata, "attack_info", lambda: {901: (50, 2)})


def _to_ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_ns(v) for v in value]
    return value


@pytest.mark.unit
def test_shapes_bounds_and_names_unique():
    assert len(set(of.DECISION_FEATURE_NAMES)) == of.DECISION_COUNT
    assert len(set(of.OPTION_FEATURE_NAMES)) == of.OPTION_COUNT
    obs = _obs()
    dec = of.extract_decision(obs, obs["select"])
    rows = of.extract_options(obs, obs["select"])
    assert len(dec) == of.DECISION_COUNT
    assert len(rows) == 4 and all(len(r) == of.OPTION_COUNT for r in rows)
    assert all(0.0 <= v <= 1.0 for v in dec)
    assert all(0.0 <= v <= 1.0 for r in rows for v in r)


@pytest.mark.unit
def test_semantics_with_metadata(monkeypatch):
    _fake_meta(monkeypatch)
    obs = _obs()
    names = of.OPTION_FEATURE_NAMES
    dnames = of.DECISION_FEATURE_NAMES
    dec = of.extract_decision(obs, obs["select"])
    rows = of.extract_options(obs, obs["select"])
    assert dec[dnames.index("any_lethal")] == 1.0  # 50 dmg >= 40 hp
    assert dec[dnames.index("sel_main")] == 1.0
    attack, play, attach, end = rows
    assert attack[names.index("t_attack")] == 1.0
    assert attack[names.index("a_lethal")] == 1.0
    assert attack[names.index("a_is_max")] == 1.0
    assert play[names.index("c_pokemon")] == 1.0  # card 301 is a Pokémon
    assert play[names.index("c_type_matches_active")] == 1.0  # type 2 == active type 2
    assert play[names.index("r_strongest_card")] == 1.0
    assert attach[names.index("c_energy")] == 1.0  # card 300 is energy
    assert attach[names.index("g_bench")] == 1.0  # targets bench slot 0
    assert attach[names.index("g_deficit")] == pytest.approx(
        1 / 3
    )  # cost 1, 0 attached
    assert end[names.index("t_end")] == 1.0


@pytest.mark.unit
def test_dict_and_namespace_agree(monkeypatch):
    _fake_meta(monkeypatch)
    obs = _obs()
    ns = _to_ns(obs)
    assert of.extract_decision(obs, obs["select"]) == of.extract_decision(ns, ns.select)
    assert of.extract_options(obs, obs["select"]) == of.extract_options(ns, ns.select)


@pytest.mark.unit
def test_garbage_never_raises():
    assert of.extract_decision(None, None) == [0.0] * of.DECISION_COUNT
    assert of.extract_options(None, None) == []
    rows = of.extract_options({}, {"option": [{}, {"type": "x", "area": 99}]})
    assert len(rows) == 2 and all(len(r) == of.OPTION_COUNT for r in rows)
