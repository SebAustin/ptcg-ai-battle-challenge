"""Agent entrypoint — ``agent(obs_dict) -> list[int]``.

The engine calls this each decision point; the return is a list of indices into
``obs["select"]["option"]`` (length within ``minCount..maxCount``, distinct). On
the very first call ``obs["select"]`` is ``None`` and we must return the 60-card
deck (card IDs).

This is the **v1 "Heuristic" phase** policy (README: Heuristic -> Tuned ->
Search). It is pure standard library and reads the raw observation dict directly
(no ``cg`` import), so it drops straight into the submitted bundle:

- deck request        -> the 60 IDs from deck.csv
- MAIN phase          -> a develop-then-attack greedy over option *types*
- any other selection -> take the allowed options (legal; mirrors the baseline)

A catch-all guarantees a legal selection is always returned — an illegal or
raised response would forfeit the game. Smarter option scoring (using
``evaluate.state_value`` and the engine's search hooks) is the next phase.
"""

from __future__ import annotations

import os
from typing import Any

from . import engine_adapter

# OptionType ids from the engine's cg/api.py.
_ABILITY = 10
_PLAY = 7
_ATTACH = 8
_EVOLVE = 9
_ATTACK = 13
_END = 14

# MAIN-phase action preference, earlier = higher priority: develop the board
# (abilities, plays, energy, evolutions) before attacking; end only as a last
# resort. RETREAT is intentionally absent (never retreat voluntarily in v1).
_MAIN_PRIORITY: tuple[int, ...] = (_ABILITY, _PLAY, _ATTACH, _EVOLVE, _ATTACK, _END)

_DECK_PATHS = ("deck.csv", "/kaggle_simulations/agent/deck.csv")
_DECK_SIZE = 60


def _read_deck() -> list[int]:
    """Read the 60 card IDs from deck.csv (one integer per line)."""
    for path in _DECK_PATHS:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                ids = [int(x) for x in handle.read().splitlines() if x.strip()]
            if len(ids) >= _DECK_SIZE:
                return ids[:_DECK_SIZE]
    raise FileNotFoundError("deck.csv not found or has fewer than 60 card IDs")


def _choose_main(options: list[dict[str, Any]]) -> list[int] | None:
    """Pick the single highest-priority MAIN action; ``None`` if not a MAIN prompt."""
    best_rank: int | None = None
    best_index: int | None = None
    for index, option in enumerate(options):
        otype = option.get("type")
        if otype in _MAIN_PRIORITY:
            rank = _MAIN_PRIORITY.index(otype)
            if best_rank is None or rank < best_rank:
                best_rank, best_index = rank, index
    return None if best_index is None else [best_index]


def _default_selection(select: dict[str, Any]) -> list[int]:
    """A always-legal fallback: the first ``maxCount`` option indices."""
    option_count = len(select.get("option") or ())
    max_count = int(select.get("maxCount") or 0)
    return list(range(min(max_count, option_count)))


def choose(obs: dict[str, Any]) -> list[int]:
    """Decide the selection for one observation (raises are caught by :func:`agent`)."""
    if engine_adapter.is_deck_request(obs):
        return _read_deck()

    select = obs.get("select") or {}
    options = select.get("option") or []
    if not options:
        return []

    if int(select.get("maxCount") or 0) == 1:
        picked = _choose_main(options)
        if picked is not None:
            return picked

    return _default_selection(select)


def _fallback(obs_dict: dict[str, Any]) -> list[int]:
    """A legal selection derived only from the raw select, or ``[]`` as last resort."""
    try:
        select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        return _default_selection(select) if isinstance(select, dict) else []
    except Exception:
        return []


def agent(obs_dict: dict[str, Any]) -> list[int]:
    """Engine entrypoint. Never raises: a bad decision forfeits, so we always
    fall back to a legal selection (and, in the worst case, to an empty one)."""
    try:
        return choose(obs_dict)
    except Exception:
        return _fallback(obs_dict)
