"""Determinized rollout search (PIMC) over the engine's search API.

At a MAIN decision, for each option we run ``K`` determinized rollouts: clone the
battle (``search_begin`` — isolated from the real game), take the option, then
play the branch forward with an attack-first base policy for up to ``depth``
plies (``search_step``), and score the leaf with a config-weighted positional
evaluator taken **from our fixed root perspective**. The option's value is the
mean leaf value across worlds (PIMC); we pick the argmax.

Why this can beat the base policy: one step of lookahead + rollout with a base
policy yields a policy at least as good as the base in expectation (policy
improvement). Worlds differ via the engine's own (un-seeded) coin/shuffle RNG.

Only MAIN decisions are searched (sub-selections use the fast heuristic), and the
whole thing is wall-clock bounded (``config.TURN_DEADLINE_S``) and best-effort:
any missing ``cg``, bad determinization, engine error, or time-out returns the
best option found so far, or ``None`` to fall back to the heuristic.
"""

from __future__ import annotations

import contextlib
import random
import time
from typing import Any

from . import belief
from . import config as cfg

_MAX_OPTIONS = 12
_SELECT_TYPE_MAIN = 0

# Worlds/rollout-depth come from config (env-tunable via PTCG_SEARCH_WORLDS /
# PTCG_SEARCH_ROLLOUT_DEPTH). The real cap is the wall-clock deadline below, so
# these can be set high — the search does as many rollouts as fit the turn budget.
_WORLDS = max(1, cfg.SEARCH_WORLDS)
_DEPTH = max(1, cfg.SEARCH_ROLLOUT_DEPTH)

# OptionType ids (engine cg/api.py). Rollout base policy = attack-first.
_ATTACK = 13
_END = 14
_DEVELOP = (10, 8, 9, 7)  # ability, attach, evolve, play


def evaluate_observation(observation: Any, me: int) -> float:
    """Positional value of an ``Observation`` from player ``me``'s perspective."""
    current = getattr(observation, "current", None)
    if current is None:
        return 0.0

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
    them_bench = getattr(them, "bench", None) or []

    score = (
        len(getattr(them, "prize", []) or []) - len(getattr(us, "prize", []) or [])
    ) * cfg.W_PRIZE_LEAD
    if us_active is not None:
        score += _hp(us_active) * cfg.W_OWN_ACTIVE_HP
    if them_active is not None:
        score -= _hp(them_active) * cfg.W_OPP_ACTIVE_HP
    score += sum(_hp(p) for p in us_bench) * cfg.W_OWN_BENCH_HP
    score -= sum(_hp(p) for p in them_bench) * cfg.W_OWN_BENCH_HP
    score += ((1 if us_active is not None else 0) + len(us_bench)) * cfg.W_BOARD_POKEMON
    board_energy = (_energy(us_active) if us_active is not None else 0) + sum(
        _energy(p) for p in us_bench
    )
    score += board_energy * cfg.W_ENERGY_ON_BOARD
    score += int(getattr(us, "handCount", 0) or 0) * cfg.W_HAND_SIZE
    return score


def _rollout_choice(observation: Any) -> list[int]:
    """Attack-first base policy over an engine ``Observation`` (for rollouts)."""
    select = getattr(observation, "select", None)
    if select is None:
        return []
    options = getattr(select, "option", None) or []
    if not options:
        return []
    max_count = int(getattr(select, "maxCount", 1) or 1)
    if max_count == 1:
        for i, o in enumerate(options):
            if getattr(o, "type", None) == _ATTACK:
                return [i]
        for otype in _DEVELOP:
            for i, o in enumerate(options):
                if getattr(o, "type", None) == otype:
                    return [i]
        for i, o in enumerate(options):
            if getattr(o, "type", None) == _END:
                return [i]
    return list(range(min(max_count, len(options))))


def _rollout(
    search_step: Any, search_id: int, observation: Any, depth: int, me: int
) -> float:
    """Play the branch forward with the base policy, then score the leaf as ``me``."""
    obs = observation
    sid = search_id
    for _ in range(depth):
        current = getattr(obs, "current", None)
        if current is not None and getattr(current, "result", -1) != -1:
            break
        if getattr(obs, "select", None) is None:
            break
        state = search_step(sid, _rollout_choice(obs))
        obs, sid = state.observation, state.searchId
    return evaluate_observation(obs, me)


def choose_by_search(obs_dict: dict[str, Any]) -> list[int] | None:
    """Best MAIN option by K-world rollout search, or ``None`` to fall back."""
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
        if select.get("type") != _SELECT_TYPE_MAIN:
            return None  # only search MAIN; sub-selects use the fast heuristic
        options = select.get("option") or []
        if not (2 <= len(options) <= _MAX_OPTIONS):
            return None
        if int(select.get("maxCount") or 0) != 1:
            return None
        if not obs_dict.get("search_begin_input"):
            return None

        observation = to_observation_class(obs_dict)
        current = getattr(observation, "current", None)
        if current is None:
            return None
        me = int(getattr(current, "yourIndex", 0))
        my_deck = list(belief.my_deck_ids()) or None

        deadline = time.monotonic() + max(0.1, cfg.TURN_DEADLINE_S * 0.8)
        totals = [0.0] * len(options)
        visited = [0] * len(options)
        search_ids: list[int] = []
        try:
            for world in range(_WORLDS):
                if time.monotonic() > deadline:
                    break
                # One determinization per world (seeded → diverse but reproducible);
                # all options are compared under the same world.
                hidden = belief.determinize(observation, my_deck, random.Random(world))
                if hidden is None:
                    break
                for i in range(len(options)):
                    if time.monotonic() > deadline:
                        break
                    state = search_begin(observation, *hidden)
                    search_ids.append(state.searchId)
                    stepped = search_step(state.searchId, [i])
                    totals[i] += _rollout(
                        search_step, stepped.searchId, stepped.observation, _DEPTH, me
                    )
                    visited[i] += 1
        finally:
            for sid in search_ids:
                with contextlib.suppress(Exception):
                    search_release(sid)
            with contextlib.suppress(Exception):
                search_end()

        scored = [
            (totals[i] / visited[i], i) for i in range(len(options)) if visited[i]
        ]
        return [max(scored)[1]] if scored else None
    except Exception:
        return None
