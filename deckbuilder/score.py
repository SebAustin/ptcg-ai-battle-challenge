"""Offline deck-quality heuristic — a proxy fitness until the engine harness lands.

Decks are ultimately meant to be scored by self-play in the tournament harness
(shared with the agent), but that is engine-blocked. This transparent heuristic
stands in for now: consistency, energy balance, opening reliability, attacker
power, and prize safety. It is isolated so it can be swapped for real win-rate
later without touching archetype selection or constraints (see ASSUMPTIONS.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from ptcg_bot.cards import CardPool

from . import roles
from .deck import Deck

# Rough targets for a consistent Standard-format deck.
_TARGET_ENERGY = 12
_MIN_BASICS = 8
_TARGET_CONSISTENCY = 12  # draw + search count combined
_EXCELLENT_DPE = 40.0  # damage-per-energy considered "excellent"
_PRIZE_RISK_CAP = 12.0  # ex/mega copies beyond which prize risk saturates


@dataclass(frozen=True)
class Scorecard:
    """A deck's heuristic score (0..100) plus the per-component breakdown."""

    total: float
    breakdown: tuple[tuple[str, float], ...]


def _best_fixed_dpe(pool: CardPool, deck: Deck) -> float:
    best = 0.0
    for card, _ in deck.cards(pool):
        for move in card.attacks:
            if move.damage_base and not move.is_variable_damage:
                best = max(best, move.damage_base / max(1, move.cost.total))
    return best


def score(deck: Deck, pool: CardPool) -> Scorecard:
    """Score a deck in ``[0, 100]`` with an explainable component breakdown."""
    resolved = list(deck.cards(pool))
    energy = sum(n for card, n in resolved if card.is_energy)
    basics = sum(n for card, n in resolved if card.is_basic_pokemon)
    draw = sum(n for card, n in resolved if roles.has_role(card, roles.Role.DRAW))
    search = sum(n for card, n in resolved if roles.has_role(card, roles.Role.SEARCH))
    prize_risk = sum(n for card, n in resolved if card.is_ex)

    consistency = draw + search
    power = _best_fixed_dpe(pool, deck)

    s_consistency = min(1.0, consistency / _TARGET_CONSISTENCY)
    s_energy = max(0.0, 1.0 - abs(energy - _TARGET_ENERGY) / _TARGET_ENERGY)
    s_basics = min(1.0, basics / _MIN_BASICS)
    s_power = min(1.0, power / _EXCELLENT_DPE)
    s_prize = max(0.0, 1.0 - prize_risk / _PRIZE_RISK_CAP)

    weighted = (
        0.30 * s_consistency
        + 0.20 * s_energy
        + 0.20 * s_basics
        + 0.20 * s_power
        + 0.10 * s_prize
    ) * 100.0

    return Scorecard(
        total=round(weighted, 1),
        breakdown=(
            ("consistency", round(s_consistency, 3)),
            ("energy", round(s_energy, 3)),
            ("basics", round(s_basics, 3)),
            ("attacker_power", round(s_power, 3)),
            ("prize_safety", round(s_prize, 3)),
        ),
    )
