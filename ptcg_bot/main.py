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

from . import engine_adapter, metadata

# OptionType ids from the engine's cg/api.py.
_ABILITY = 10
_PLAY = 7
_ATTACH = 8
_EVOLVE = 9
_ATTACK = 13
_END = 14

# MAIN-phase policy: ATTACK as soon as able (with the highest-damage attack),
# otherwise develop the board (ability, energy, evolution, play), and end only
# as a last resort. Attacking-when-able rather than over-developing first is the
# big lever — it lifts win-rate from ~21% to ~100% vs a random baseline (measured
# by tools/tournament). RETREAT is intentionally absent (never retreat).
_DEVELOP_PRIORITY: tuple[int, ...] = (_ABILITY, _ATTACH, _EVOLVE, _PLAY)

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


def _first_of_type(options: list[dict[str, Any]], otype: int) -> int | None:
    for index, option in enumerate(options):
        if option.get("type") == otype:
            return index
    return None


def _best_attack(options: list[dict[str, Any]]) -> int | None:
    """Index of the highest-damage ATTACK option (by engine metadata), or None."""
    attacks = [i for i, o in enumerate(options) if o.get("type") == _ATTACK]
    if not attacks:
        return None
    damage = metadata.attack_damage()

    def _dmg(index: int) -> int:
        attack_id = options[index].get("attackId")
        return damage.get(attack_id, 0) if isinstance(attack_id, int) else 0

    return max(attacks, key=_dmg)


def _choose_main(options: list[dict[str, Any]]) -> list[int] | None:
    """Attack when able (best attack), else develop the board, else end.

    Returns ``None`` when the prompt has no MAIN-type option (so the caller
    falls back to the generic selector).
    """
    attack = _best_attack(options)
    if attack is not None:
        return [attack]
    for otype in _DEVELOP_PRIORITY:
        index = _first_of_type(options, otype)
        if index is not None:
            return [index]
    return None if (end := _first_of_type(options, _END)) is None else [end]


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
