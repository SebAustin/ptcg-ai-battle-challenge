"""Assemble a legal 60-card deck from a chosen archetype.

Strategy: lock in the attacker line and an energy base, then fill the remaining
slots with Trainers by role priority (draw -> search -> switch -> accel ->
other), deduped by card name so per-name limits hold. Deterministic: same pool
in, same deck out.
"""

from __future__ import annotations

from ptcg_bot import rules
from ptcg_bot.cards import Card, CardPool
from ptcg_bot.rules import EnergyType

from . import constraints
from .archetype import Archetype, choose_archetype, offense_score
from .deck import Deck
from .roles import Role, has_role

_TARGET_ENERGY = 12
_TARGET_BASICS = 8
_COPIES_PER_SECONDARY_BASIC = 2
_COPIES_PER_TRAINER = 4
_TRAINER_PRIORITY: tuple[Role, ...] = (
    Role.DRAW,
    Role.SEARCH,
    Role.SWITCH,
    Role.ENERGY_ACCEL,
    Role.OTHER,
)


class DeckError(RuntimeError):
    """Raised when construction cannot produce a legal deck."""


def _basic_energy_of(pool: CardPool, etype: object) -> Card | None:
    basics = sorted(
        (c for c in pool.energy if c.is_basic_energy), key=lambda c: c.card_id
    )
    if etype is not None:
        for card in basics:
            if card.poke_type is etype:
                return card
    return basics[0] if basics else None


def _affordable_on(card: Card, etype: EnergyType | None) -> bool:
    """True if the card has a fixed attack payable with only ``etype`` + Colorless."""
    for move in card.attacks:
        if (
            move.damage_base
            and not move.is_variable_damage
            and all(t is etype or t is EnergyType.COLORLESS for t, _ in move.cost.typed)
        ):
            return True
    return False


def build_deck(pool: CardPool, *, target_energy: int = _TARGET_ENERGY) -> Deck:
    """Construct a legal 60-card deck around the pool's strongest attacker line."""
    arch: Archetype = choose_archetype(pool)

    counts: dict[int, int] = {}
    for cid, n in zip(arch.line, arch.counts, strict=True):
        counts[cid] = counts.get(cid, 0) + n

    energy_card = _basic_energy_of(pool, arch.energy_type)
    energy_n = min(target_energy, rules.DECK_SIZE - sum(counts.values()))
    if energy_card is not None and energy_n > 0:
        counts[energy_card.card_id] = counts.get(energy_card.card_id, 0) + energy_n

    used_names: set[str] = set()
    for cid in counts:
        card = pool.by_id(cid)
        if card is not None:
            used_names.add(card.name)

    # Reduce mulligan risk: add secondary Basic attackers that run on the same
    # energy, until we have a reliable count of Basics to open with.
    basics_now = sum(
        n
        for cid, n in counts.items()
        if (c := pool.by_id(cid)) is not None and c.is_basic_pokemon
    )
    secondaries = sorted(
        (
            c
            for c in pool.basics
            if c.name not in used_names and _affordable_on(c, arch.energy_type)
        ),
        # Prefer single-prize Basics (not ex) for resilience, then raw offense.
        key=lambda c: (0 if c.is_ex else 1, offense_score(c), -c.card_id),
        reverse=True,
    )
    for card in secondaries:
        if basics_now >= _TARGET_BASICS:
            break
        counts[card.card_id] = counts.get(card.card_id, 0) + _COPIES_PER_SECONDARY_BASIC
        used_names.add(card.name)
        basics_now += _COPIES_PER_SECONDARY_BASIC

    trainers = [c for c in pool.trainers if c.text and not c.is_ace_spec]
    remaining = rules.DECK_SIZE - sum(counts.values())
    for role in _TRAINER_PRIORITY:
        if remaining <= 0:
            break
        candidates = sorted(
            (
                c
                for c in trainers
                if c.name not in used_names
                and (role is Role.OTHER or has_role(c, role))
            ),
            key=lambda c: c.card_id,
        )
        for card in candidates:
            if remaining <= 0:
                break
            add = min(_COPIES_PER_TRAINER, remaining)
            counts[card.card_id] = counts.get(card.card_id, 0) + add
            used_names.add(card.name)
            remaining -= add

    if remaining > 0 and energy_card is not None:  # ran out of trainers (unlikely)
        counts[energy_card.card_id] = counts.get(energy_card.card_id, 0) + remaining

    deck = Deck.from_counts(counts)
    problems = constraints.validate(deck, pool)
    if problems:
        raise DeckError("built an illegal deck: " + "; ".join(problems))
    return deck
