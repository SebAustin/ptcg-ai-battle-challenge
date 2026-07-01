"""Internal game-state model — our OWN representation of a battle position.

This is deliberately independent of the simulator's wire format: the (still
rules-gated) engine schema is translated INTO these types by
:mod:`ptcg_bot.engine_adapter`, and everything downstream — the heuristic
evaluator, and later legal-move generation / search — reads only this model.

The model captures exactly the features the heuristic in :mod:`ptcg_bot.evaluate`
scores (prizes, active/bench HP, attached energy, attack readiness, type
matchup), plus what a forward model will need. It is immutable: a move produces a
new state, never mutating the old one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .cards import Card, EnergyCost, Move
from .rules import PRIZE_COUNT, EnergyType


def can_pay(cost: EnergyCost, attached: tuple[EnergyType, ...]) -> bool:
    """True if ``attached`` energy can pay ``cost`` (Colorless = any energy)."""
    pool: Counter[EnergyType] = Counter(attached)
    for etype, need in cost.typed:
        if pool[etype] < need:
            return False
        pool[etype] -= need
    return sum(pool.values()) >= cost.colorless


@dataclass(frozen=True)
class PokemonInPlay:
    """A Pokémon on the board: its card, damage taken, and attached energy."""

    card: Card
    damage: int = 0
    attached: tuple[EnergyType, ...] = ()

    @property
    def max_hp(self) -> int:
        return self.card.hp or 0

    @property
    def remaining_hp(self) -> int:
        return max(0, self.max_hp - self.damage)

    @property
    def is_knocked_out(self) -> bool:
        return self.max_hp > 0 and self.remaining_hp == 0

    @property
    def energy_count(self) -> int:
        return len(self.attached)

    def can_use(self, move: Move) -> bool:
        """True if this Pokémon currently has the energy to use ``move``."""
        return move.is_attack and can_pay(move.cost, self.attached)

    @property
    def can_attack(self) -> bool:
        return any(self.can_use(m) for m in self.card.attacks)


@dataclass(frozen=True)
class PlayerState:
    """One player's side of the board."""

    active: PokemonInPlay | None = None
    bench: tuple[PokemonInPlay, ...] = ()
    hand_size: int = 0
    deck_size: int = 0
    prizes_remaining: int = PRIZE_COUNT

    @property
    def in_play(self) -> tuple[PokemonInPlay, ...]:
        """Active + bench (excluding an empty active slot)."""
        return ((self.active,) if self.active is not None else ()) + self.bench

    @property
    def has_pokemon(self) -> bool:
        return bool(self.in_play)

    @property
    def bench_basics(self) -> int:
        return sum(1 for p in self.bench if p.card.is_basic_pokemon)

    @property
    def energy_on_board(self) -> int:
        return sum(p.energy_count for p in self.in_play)


@dataclass(frozen=True)
class GameState:
    """A full battle position, viewed from ``us`` (the player to act)."""

    us: PlayerState = field(default_factory=PlayerState)
    them: PlayerState = field(default_factory=PlayerState)
    turn: int = 1

    @property
    def is_terminal(self) -> bool:
        return self.winner != 0

    @property
    def winner(self) -> int:
        """+1 if ``us`` has won, -1 if ``them`` has won, 0 otherwise.

        A side wins by taking all its prizes, or when the opponent has no
        Pokémon left in play (nothing to promote to Active).
        """
        us_won = self.us.prizes_remaining <= 0 or not self.them.has_pokemon
        them_won = self.them.prizes_remaining <= 0 or not self.us.has_pokemon
        if us_won and not them_won:
            return 1
        if them_won and not us_won:
            return -1
        return 0
