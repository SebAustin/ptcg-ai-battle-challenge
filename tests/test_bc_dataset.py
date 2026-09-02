"""Tests for the BC dataset builder's pure parts (synthetic replay, offline)."""

from __future__ import annotations

import pytest

from tools.bc_dataset import agent_ratings, deck_key, encode, iter_decisions


def _select(n: int, sel_type: int = 0) -> dict:
    return {
        "type": sel_type,
        "maxCount": 1,
        "option": [{"type": 7, "area": 2, "index": i} for i in range(n)],
    }


def _step(agent0: dict, agent1: dict) -> list[dict]:
    return [agent0, agent1]


def _agent(obs_select, your_index, action=None, status="ACTIVE") -> dict:
    return {
        "status": status,
        "observation": {
            "current": {"yourIndex": your_index, "players": [{"hand": []}, {}]},
            "select": obs_select,
        },
        "action": action or [],
    }


def _replay_fixture() -> dict:
    deck_a = list(range(60))
    deck_b = list(range(100, 160))
    return {
        "id": 42,
        "steps": [
            _step(_agent(None, 0), _agent(None, 1)),
            _step(
                _agent(None, 0, deck_a), _agent(None, 1, deck_b)
            ),  # deck registration
            # t=2: agent 0 decides among 3 options; its action lands at t=3.
            _step(_agent(_select(3), 0), _agent(_select(3), 1, status="INACTIVE")),
            _step(_agent(None, 0, [2]), _agent(_select(2), 1)),
            # t=4: agent 1 decides (2 options); action at t=5 is out of range -> skipped.
            _step(_agent(None, 0), _agent(None, 1, [5])),
        ],
    }


@pytest.mark.unit
def test_iter_decisions_uses_next_step_action_and_filters():
    replay = _replay_fixture()
    ratings = {0: 1300.0, 1: 1300.0}
    found = list(iter_decisions(replay, ratings, 1150.0, frozenset({0})))
    assert (
        len(found) == 1
    )  # agent 1's out-of-range action (beyond the window) is rejected
    agent_idx, obs, select, chosen, offset = found[0]
    assert agent_idx == 0 and chosen == [2] and len(select["option"]) == 3
    assert offset == 1  # the response landed one step after the select was shown
    # Low-rated agent contributes nothing.
    assert (
        list(iter_decisions(replay, {0: 900.0, 1: 900.0}, 1150.0, frozenset({0}))) == []
    )


@pytest.mark.unit
def test_iter_decisions_finds_same_step_action():
    """Regression: ~72% of real decisions record the response at OFFSET 0 (the
    same step as the select), not only at t+1 — verified against real replays
    (see tools/bc_dataset.py docstring). A fixed t+1 lookup would silently
    drop these, so the window scan must accept offset 0 too."""
    replay = {
        "id": 7,
        "steps": [_step(_agent(_select(3), 0, action=[1]), _agent(None, 1))],
    }
    found = list(iter_decisions(replay, {0: 1300.0, 1: 0.0}, 1150.0, frozenset({0})))
    assert len(found) == 1
    _idx, _obs, _sel, chosen, offset = found[0]
    assert chosen == [1] and offset == 0


@pytest.mark.unit
def test_agent_ratings_and_deck_key():
    meta = {"agents": [{"submissionId": 1, "updatedScore": 1250.5}, {"index": 1}]}
    assert agent_ratings(meta) == {0: 1250.5, 1: 0.0}
    replay = _replay_fixture()
    reordered = dict(replay)
    reordered["steps"] = [list(s) for s in replay["steps"]]
    reordered["steps"][1][0]["action"] = list(reversed(replay["steps"][1][0]["action"]))
    assert deck_key(replay, 0) == deck_key(reordered, 0) != deck_key(replay, 1)


@pytest.mark.unit
def test_encode_dedupes_identical_options(monkeypatch):
    from ptcg_bot import metadata

    # Offline (no metadata): the three PLAY options resolve to card id 0 and
    # differ only by position -> genuinely the same choice -> ONE class.
    replay = _replay_fixture()
    _idx, obs, select, chosen, _offset = next(
        iter_decisions(replay, {0: 1300.0, 1: 0.0}, 1150.0, frozenset({0}))
    )
    state, unique, weights, first_class, v7_class = encode(obs, select, chosen)
    assert len(unique) == 1 and weights == [1.0] and first_class == 0

    # With metadata resolving each hand slot to a DIFFERENT card, otherwise-
    # identical PLAY options stay distinct classes.
    obs["current"]["players"][0]["hand"] = [{"id": 10}, {"id": 11}, {"id": 12}]
    monkeypatch.setattr(
        metadata, "card_power", lambda: {10: (0, 0), 11: (50, 60), 12: (90, 100)}
    )
    monkeypatch.setattr(metadata, "card_kind", lambda: {10: 0, 11: 0, 12: 0})
    state, unique, weights, first_class, v7_class = encode(obs, select, chosen)
    assert len(unique) == 3 and sum(weights) == pytest.approx(1.0)
    assert weights[2] == 1.0  # chosen == [2] resolves to hand slot 2 -> its own class

    # Two hand slots with the SAME card id collapse into one class alongside
    # the third, distinct one.
    obs["current"]["players"][0]["hand"] = [{"id": 10}, {"id": 10}, {"id": 12}]
    _s, unique2, weights2, _f, _v = encode(obs, select, [0, 1])
    assert len(unique2) == 2 and sum(weights2) == pytest.approx(1.0)
