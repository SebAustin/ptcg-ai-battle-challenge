"""Determinized one-ply lookahead over the engine's search API.

At a single-choice decision, we clone the battle for each option
(``search_begin`` + ``search_step`` — fully isolated from the real game), score
the resulting observation with a config-weighted positional heuristic, and pick
the best option. Hidden information is filled in by :mod:`ptcg_bot.belief`.

This is the first, deliberately small IS-MCTS step: one ply, one determinized
world. It reads the engine's ``Observation`` dataclass directly (no card pool
needed) and is entirely best-effort — any missing ``cg``, bad determinization,
or engine error makes :func:`choose_by_search` return ``None`` so the caller
falls back to the heuristic policy. Extending to K worlds and deeper rollouts is
the next step toward full PIMC.
"""

from __future__ import annotations

import contextlib
from typing import Any

from . import belief
from . import config as cfg

# Only search genuine single-choice decisions with a handful of options — enough
# to cover MAIN and most target/card sub-selections while bounding the cost.
_MAX_OPTIONS = 12


def evaluate_observation(observation: Any) -> float:
    """Positional value of an engine ``Observation`` from our perspective."""
    current = getattr(observation, "current", None)
    if current is None:
        return 0.0

    me = int(getattr(current, "yourIndex", 0))
    result = getattr(current, "result", -1)
    if result != -1:
        if result == me:
            return cfg.VALUE_WIN
        return 0.0 if result == 2 else cfg.VALUE_LOSS

    players = getattr(current, "players", None) or []
    if len(players) < 2:
        return 0.0
    us, them = players[me], players[1 - me]

    def _hp(pokemon: Any) -> int:
        return int(getattr(pokemon, "hp", 0) or 0)

    def _energy(pokemon: Any) -> int:
        return len(getattr(pokemon, "energies", None) or ())

    us_active = (getattr(us, "active", None) or [None])[0]
    them_active = (getattr(them, "active", None) or [None])[0]
    us_bench = getattr(us, "bench", None) or []

    score = (
        len(getattr(them, "prize", []) or []) - len(getattr(us, "prize", []) or [])
    ) * cfg.W_PRIZE_LEAD
    if us_active is not None:
        score += _hp(us_active) * cfg.W_OWN_ACTIVE_HP
    if them_active is not None:
        score -= _hp(them_active) * cfg.W_OPP_ACTIVE_HP
    score += sum(_hp(p) for p in us_bench) * cfg.W_OWN_BENCH_HP
    score += ((1 if us_active is not None else 0) + len(us_bench)) * cfg.W_BOARD_POKEMON
    board_energy = (_energy(us_active) if us_active is not None else 0) + sum(
        _energy(p) for p in us_bench
    )
    score += board_energy * cfg.W_ENERGY_ON_BOARD
    score += int(getattr(us, "handCount", 0) or 0) * cfg.W_HAND_SIZE
    return score


def choose_by_search(obs_dict: dict[str, Any]) -> list[int] | None:
    """Best option index for a single-choice decision, or ``None`` to fall back."""
    try:
        from cg.api import (  # type: ignore[import-not-found]
            search_begin,
            search_end,
            search_release,
            search_step,
            to_observation_class,
        )
    except Exception:
        return None

    try:
        select = obs_dict.get("select")
        if not isinstance(select, dict):
            return None
        options = select.get("option") or []
        if not (2 <= len(options) <= _MAX_OPTIONS):
            return None
        if int(select.get("maxCount") or 0) != 1:
            return None
        if not obs_dict.get("search_begin_input"):
            return None

        observation = to_observation_class(obs_dict)
        hidden = belief.determinize(observation)
        if hidden is None:
            return None

        best_index: int | None = None
        best_score = float("-inf")
        search_ids: list[int] = []
        try:
            for index in range(len(options)):
                state = search_begin(observation, *hidden)
                search_ids.append(state.searchId)
                nxt = search_step(state.searchId, [index])
                score = evaluate_observation(nxt.observation)
                if score > best_score:
                    best_score, best_index = score, index
        finally:
            for sid in search_ids:
                with contextlib.suppress(Exception):
                    search_release(sid)
            with contextlib.suppress(Exception):
                search_end()

        return None if best_index is None else [best_index]
    except Exception:
        return None
