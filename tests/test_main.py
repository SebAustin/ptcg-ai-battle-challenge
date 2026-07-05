"""Tests for the agent entrypoint (ptcg_bot.main).

Unit tests use synthetic observation dicts (no engine/data). One integration
test drives a full battle with OUR agent on both sides against the live engine,
proving every selection it returns is legal; it skips without engine/ + data/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ptcg_bot.main import agent

_ENGINE = (
    Path(__file__).resolve().parent.parent
    / "engine"
    / "sample_submission"
    / "sample_submission"
)


# --- deck request -----------------------------------------------------------


@pytest.mark.unit
def test_deck_request_returns_60_ids(tmp_path, monkeypatch):
    (tmp_path / "deck.csv").write_text("\n".join(str(i) for i in range(60)) + "\n")
    monkeypatch.chdir(tmp_path)
    result = agent({"select": None, "current": None})
    assert result == list(range(60))


# --- MAIN-phase priority ----------------------------------------------------


@pytest.mark.unit
def test_main_attacks_when_able_over_developing():
    # options: END(14), ATTACK(13), ATTACH(8). Attack-first: ATTACK wins.
    obs = {
        "select": {"maxCount": 1, "option": [{"type": 14}, {"type": 13}, {"type": 8}]}
    }
    assert agent(obs) == [1]  # the ATTACK option (index 1)


@pytest.mark.unit
def test_main_develops_when_no_attack_available():
    # No ATTACK option -> develop: ATTACH(8) over PLAY(7) over END(14).
    obs = {
        "select": {"maxCount": 1, "option": [{"type": 14}, {"type": 7}, {"type": 8}]}
    }
    assert agent(obs) == [2]  # ATTACH (index 2)


@pytest.mark.unit
def test_main_attacks_when_only_attack_or_end():
    obs = {"select": {"maxCount": 1, "option": [{"type": 14}, {"type": 13}]}}
    assert agent(obs) == [1]  # ATTACK preferred over END


@pytest.mark.unit
def test_main_ends_when_nothing_else():
    obs = {"select": {"maxCount": 1, "option": [{"type": 14}]}}
    assert agent(obs) == [0]  # END


# --- generic selection default ----------------------------------------------


@pytest.mark.unit
def test_generic_select_takes_maxcount_options():
    # A non-MAIN multi-select (option type CARD=3): take the first maxCount.
    obs = {"select": {"maxCount": 2, "option": [{"type": 3}] * 4}}
    assert agent(obs) == [0, 1]


@pytest.mark.unit
def test_empty_options_returns_empty():
    assert agent({"select": {"maxCount": 1, "option": []}}) == []


@pytest.mark.unit
def test_agent_never_raises_on_malformed_input():
    # Malformed maxCount must not raise — the agent falls back to a legal (empty) pick.
    result = agent({"select": {"maxCount": "oops", "option": [{}, {}]}})
    assert isinstance(result, list)


# --- live full game (skips without engine/ + data/) -------------------------


@pytest.mark.integration
def test_agent_plays_a_full_legal_game():
    if not (_ENGINE / "cg" / "api.py").exists():
        pytest.skip("engine not present (run `make engine`)")

    from ptcg_bot.cards import DEFAULT_CSV, load_pool

    if not DEFAULT_CSV.exists():
        pytest.skip("competition dataset not present (run `make data`)")
    sys.path.insert(0, str(_ENGINE))
    from cg import game  # type: ignore[import-not-found]

    from deckbuilder import build_deck

    deck_ids = [cid for cid, n in build_deck(load_pool()).counts for _ in range(n)]
    obs, start = game.battle_start(deck_ids, list(deck_ids))
    assert obs is not None, f"engine rejected our deck (errorType={start.errorType})"
    try:
        steps = 0
        result = -1
        while steps < 3000:
            if obs.get("select") is None:
                break
            obs = game.battle_select(agent(obs))  # engine raises if selection illegal
            steps += 1
            result = (obs.get("current") or {}).get("result", -1)
            if result != -1:
                break
        assert result in (
            0,
            1,
            2,
        ), f"game did not resolve (result={result}, steps={steps})"
    finally:
        game.battle_finish()


# --- v4: smart attach + gated retreat (synthetic, metadata monkeypatched) ----


def _v4_obs(active_energy: int, bench_energy: int) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "active": [{"id": 100, "energies": [3] * active_energy}],
                    "bench": [{"id": 101, "energies": [3] * bench_energy}],
                },
                {"active": [{"id": 200, "hp": 90}], "bench": []},
            ],
        }
    }


@pytest.mark.unit
def test_should_retreat_only_when_wall_stuck_and_bench_ready(monkeypatch):
    from ptcg_bot import main as m
    from ptcg_bot import metadata

    monkeypatch.setattr(metadata, "attack_cost", lambda: {100: 2, 101: 1})
    assert m._should_retreat(_v4_obs(active_energy=0, bench_energy=1))  # stuck + ready
    assert not m._should_retreat(_v4_obs(active_energy=2, bench_energy=1))  # active ok
    assert not m._should_retreat(_v4_obs(active_energy=0, bench_energy=0))  # no rescuer


@pytest.mark.unit
def test_should_retreat_false_without_metadata(monkeypatch):
    from ptcg_bot import main as m
    from ptcg_bot import metadata

    monkeypatch.setattr(metadata, "attack_cost", lambda: {})
    assert not m._should_retreat(_v4_obs(0, 1))  # offline -> never retreat


@pytest.mark.unit
def test_attach_feeds_bench_once_active_charged(monkeypatch):
    from ptcg_bot import main as m
    from ptcg_bot import metadata

    monkeypatch.setattr(metadata, "attack_cost", lambda: {100: 1, 101: 2})
    monkeypatch.setattr(metadata, "card_power", lambda: {100: (50, 60), 101: (80, 70)})
    options = [
        {"type": 8, "inPlayArea": 4, "inPlayIndex": 0},  # attach to active
        {"type": 8, "inPlayArea": 5, "inPlayIndex": 0},  # attach to bench
    ]
    charged = _v4_obs(active_energy=1, bench_energy=0)
    assert m._attach_target(charged, options) == 1  # active charged -> feed bench
    hungry = _v4_obs(active_energy=0, bench_energy=0)
    assert m._attach_target(hungry, options) == 0  # active first while uncharged
