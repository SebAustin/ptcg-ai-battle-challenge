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
import math
import random
import time
from typing import Any

from . import belief
from . import config as cfg
from . import features

_MAX_OPTIONS = 12
_SELECT_TYPE_MAIN = 0

# Worlds/rollout-depth come from config (env-tunable via PTCG_SEARCH_WORLDS /
# PTCG_SEARCH_ROLLOUT_DEPTH). The real cap is the wall-clock deadline below, so
# these can be set high. Depth 0 = direct 1-ply: evaluate the post-option
# observation immediately (more worlds per deadline; no rollout-policy bias).
_WORLDS = max(1, cfg.SEARCH_WORLDS)
_DEPTH = max(0, cfg.SEARCH_ROLLOUT_DEPTH)

# OptionType ids (engine cg/api.py). Rollout base policy = attack-first.
_ATTACK = 13
_END = 14
_DEVELOP = (10, 8, 9, 7)  # ability, attach, evolve, play


def _learned_value(observation: Any, me: int) -> float | None:
    """Win probability from the trained evaluator, or ``None`` to fall back."""
    if not cfg.LEARNED_EVAL:
        return None
    try:
        from . import eval_weights  # noqa: PLC0415 - optional generated module

        feats = features.extract(observation, me)
        if len(feats) != eval_weights.FEATURE_COUNT:
            return None  # features/weights version skew -> heuristic fallback
        return eval_weights.predict(feats)
    except Exception:
        return None


def evaluate_observation(observation: Any, me: int) -> float:
    """Value of an ``Observation`` for player ``me`` on a WIN-PROBABILITY scale.

    Terminal states are exact (1.0 win / 0.0 loss / 0.5 draw). Non-terminal
    states use the learned evaluator (``eval_weights.predict`` over
    ``features.extract``); without it, the legacy hand-crafted positional score
    squashed through a sigmoid keeps both paths on the same scale.
    """
    current = getattr(observation, "current", None)
    if current is None:
        return 0.5

    result = getattr(current, "result", -1)
    if result != -1:
        if result == me:
            return 1.0
        return 0.5 if result == 2 else 0.0

    learned = _learned_value(observation, me)
    if learned is not None:
        return learned

    players = getattr(current, "players", None) or []
    if len(players) < 2:
        return 0.5
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
    # Same scale as the learned path: squash the positional score to (0, 1).
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score / 300.0))))


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


def _is_our_main(observation: Any, me: int) -> bool:
    """True while the clone still offers US a MAIN decision (turn continues)."""
    current = getattr(observation, "current", None)
    select = getattr(observation, "select", None)
    if current is None or select is None:
        return False
    if getattr(current, "result", -1) != -1:
        return False
    if int(getattr(current, "yourIndex", -1)) != me:
        return False
    return getattr(select, "type", None) == _SELECT_TYPE_MAIN


def _sequence_value(
    search_step: Any,
    search_id: int,
    observation: Any,
    me: int,
    deadline: float,
    track: list[int],
) -> float:
    """Value of a node after OPTIMIZING the rest of our turn with a small beam.

    The engine's search tree is persistent (branching from a parent id is
    supported — validated live), so at each of OUR MAIN decisions we expand all
    options, score children with the leaf evaluator, and keep the best
    ``SEQ_BEAM_WIDTH``. Sub-selections and everything after our turn ends fall
    back to the base-policy rollout. Returns max over beam leaves (we control
    our own actions). ``track`` collects created searchIds for release.
    """
    beam: list[tuple[int, Any]] = [(search_id, observation)]
    best_final = None
    for _ in range(max(1, cfg.SEQ_ACTION_CAP)):
        if time.monotonic() > deadline or not beam:
            break
        scored_children: list[tuple[float, int, Any]] = []
        next_beam: list[tuple[int, Any]] = []
        for sid, obs in beam:
            if not _is_our_main(obs, me):
                value = _rollout(search_step, sid, obs, _DEPTH, me)
                best_final = value if best_final is None else max(best_final, value)
                continue
            options = getattr(getattr(obs, "select", None), "option", None) or []
            if not options or int(getattr(obs.select, "maxCount", 1) or 1) != 1:
                # Multi-selects inside our turn: single base-policy path.
                state = search_step(sid, _rollout_choice(obs))
                track.append(state.searchId)
                next_beam.append((state.searchId, state.observation))
                continue
            for i in range(len(options)):
                if time.monotonic() > deadline:
                    break
                state = search_step(sid, [i])
                track.append(state.searchId)
                scored_children.append(
                    (
                        evaluate_observation(state.observation, me),
                        state.searchId,
                        state.observation,
                    )
                )
        scored_children.sort(key=lambda t: -t[0])
        keep = scored_children[: max(1, cfg.SEQ_BEAM_WIDTH)]
        beam = [(sid, obs) for _v, sid, obs in keep] + next_beam
    for sid, obs in beam:  # beam exhausted by cap/deadline: rollout remainders
        if time.monotonic() > deadline + 0.2:
            value = evaluate_observation(obs, me)
        else:
            value = _rollout(search_step, sid, obs, _DEPTH, me)
        best_final = value if best_final is None else max(best_final, value)
    return best_final if best_final is not None else 0.5


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
                    search_ids.append(stepped.searchId)
                    if cfg.SEQ_SEARCH:
                        totals[i] += _sequence_value(
                            search_step,
                            stepped.searchId,
                            stepped.observation,
                            me,
                            deadline,
                            search_ids,
                        )
                    else:
                        totals[i] += _rollout(
                            search_step,
                            stepped.searchId,
                            stepped.observation,
                            _DEPTH,
                            me,
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
