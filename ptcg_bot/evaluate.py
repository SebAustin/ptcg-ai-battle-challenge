"""Hand-crafted, explainable state-value heuristic.

Scores a :class:`~ptcg_bot.state.GameState` from the acting player's (``us``)
perspective using the tunable weights in :mod:`ptcg_bot.config`. This is the
narrative the Strategy writeup rests on — every term is inspectable, and prize
tempo dominates because taking (and denying) prizes is how the game is won.

The same function is the leaf evaluator for the future search, and the 1-ply
policy fallback when search runs out of time.
"""

from __future__ import annotations

from . import config as cfg
from .state import GameState, PlayerState


def _own_bench_hp(player: PlayerState) -> int:
    return sum(p.remaining_hp for p in player.bench)


def explain(state: GameState) -> dict[str, float]:
    """Return each weighted contribution to the score (for tuning / the writeup)."""
    if state.is_terminal:
        terminal = cfg.VALUE_WIN if state.winner == 1 else cfg.VALUE_LOSS
        return {"terminal": terminal}

    us, them = state.us, state.them
    parts: dict[str, float] = {}

    parts["prize_lead"] = (
        them.prizes_remaining - us.prizes_remaining
    ) * cfg.W_PRIZE_LEAD

    if us.active is not None:
        parts["own_active_hp"] = us.active.remaining_hp * cfg.W_OWN_ACTIVE_HP
        parts["attack_ready"] = cfg.W_ATTACK_READY if us.active.can_attack else 0.0
    if them.active is not None:
        parts["opp_active_hp"] = -them.active.remaining_hp * cfg.W_OPP_ACTIVE_HP

    parts["own_bench_hp"] = _own_bench_hp(us) * cfg.W_OWN_BENCH_HP
    parts["board_pokemon"] = len(us.in_play) * cfg.W_BOARD_POKEMON
    parts["energy_on_board"] = us.energy_on_board * cfg.W_ENERGY_ON_BOARD
    parts["hand_size"] = us.hand_size * cfg.W_HAND_SIZE
    parts["bench_basics"] = us.bench_basics * cfg.W_BENCH_BASICS

    if (
        us.active is not None
        and them.active is not None
        and us.active.card.poke_type is not None
        and us.active.card.poke_type is them.active.card.weakness
    ):
        parts["type_advantage"] = cfg.W_TYPE_ADVANTAGE

    return parts


def state_value(state: GameState) -> float:
    """Scalar heuristic value of ``state`` from ``us``'s perspective."""
    return sum(explain(state).values())
