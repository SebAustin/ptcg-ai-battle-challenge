"""Tests for the replay harvester's pure functions (synthetic fixtures only)."""

from __future__ import annotations

import pytest

from tools.harvest import (
    canonical,
    decks_from_replay,
    episode_agent_scores,
    merge_unique,
)

_DECK_A = [7] * 8 + list(range(100, 152))  # 60 ids
_DECK_B = [344] * 4 + list(range(200, 256))  # 60 ids


def _replay(deck0: list[int], deck1: list[int]) -> dict:
    return {
        "steps": [
            [{"action": []}, {"action": []}],
            [{"action": deck0}, {"action": deck1}],
        ]
    }


@pytest.mark.unit
def test_decks_from_replay_extracts_both_sides():
    decks = decks_from_replay(_replay(_DECK_A, _DECK_B))
    assert decks == [_DECK_A, _DECK_B]


@pytest.mark.unit
def test_decks_from_replay_rejects_malformed():
    assert decks_from_replay({}) == []
    assert decks_from_replay({"steps": [[]]}) == []
    # Wrong length action is skipped, valid one kept.
    decks = decks_from_replay(_replay([1, 2, 3], _DECK_B))
    assert decks == [_DECK_B]


@pytest.mark.unit
def test_merge_unique_dedupes_canonically():
    reordered_a = list(reversed(_DECK_A))
    merged, added = merge_unique([_DECK_A], [reordered_a, _DECK_B, _DECK_B])
    assert added == 1  # reordered A is the same deck; B added once
    assert len(merged) == 2
    assert canonical(merged[1]) == canonical(_DECK_B)


@pytest.mark.unit
def test_episode_agent_scores_reads_submission_and_score():
    episode = {
        "agents": [
            {"submissionId": 111, "updatedScore": 1234.5},
            {"submissionId": 222},  # missing score -> 0.0
            {"reward": 1},  # missing submissionId -> skipped
        ]
    }
    assert episode_agent_scores(episode) == [(111, 1234.5), (222, 0.0)]
