"""Pick a primary attacker line + energy type from the pool (deterministic).

We build around a single, explainable attacker line — the focused shape the
Strategy rubric rewards — chosen by an offensive heuristic: fixed damage per
energy, durability (HP), minus prize liability (ex = 2 prizes, Mega ex = 3).
"""

from __future__ import annotations

from dataclasses import dataclass

from ptcg_bot.cards import Card, CardPool
from ptcg_bot.rules import EnergyType, prizes_for_knockout

# Copies per stage, indexed by evolution-line length (basic -> final).
_LINE_COUNTS: dict[int, tuple[int, ...]] = {
    1: (4,),
    2: (3, 3),
    3: (3, 2, 3),
}


@dataclass(frozen=True)
class Archetype:
    """The chosen attacker line and the energy type it runs on."""

    line: tuple[int, ...]  # card ids, basic -> final stage
    counts: tuple[int, ...]  # copies per stage, aligned to ``line``
    energy_type: EnergyType | None
    attacker_id: int


def best_fixed_dpe(card: Card) -> float:
    """Highest fixed (non-variable) damage-per-energy across the card's attacks."""
    best = 0.0
    for move in card.attacks:
        if move.damage_base and not move.is_variable_damage:
            best = max(best, move.damage_base / max(1, move.cost.total))
    return best


def offense_score(card: Card) -> float:
    """Rank an attacker: damage efficiency + durability - prize liability."""
    dpe = best_fixed_dpe(card)
    hp = (card.hp or 0) / 100.0
    prize_penalty = 0.5 * (prizes_for_knockout(card.rule) - 1)
    return dpe + hp - prize_penalty


def evolution_chain(pool: CardPool, attacker: Card) -> list[Card]:
    """Resolve ``attacker`` back to its Basic, returning cards basic -> final."""
    chain = [attacker]
    seen = {attacker.name}
    current = attacker
    while current.previous_stage and current.previous_stage not in seen:
        printings = pool.by_name(current.previous_stage)
        if not printings:
            break
        prev = min(printings, key=lambda c: c.card_id)  # deterministic printing
        chain.append(prev)
        seen.add(prev.name)
        current = prev
    chain.reverse()
    return chain


def primary_energy_type(attacker: Card) -> EnergyType | None:
    """The dominant typed energy the attacker's best attack needs."""
    best_move = None
    best_dpe = 0.0
    for move in attacker.attacks:
        if move.damage_base and not move.is_variable_damage:
            dpe = move.damage_base / max(1, move.cost.total)
            if dpe > best_dpe:
                best_dpe, best_move = dpe, move
    if best_move is not None and best_move.cost.typed:
        etype, _ = max(best_move.cost.typed, key=lambda tc: tc[1])
        return etype
    if attacker.poke_type and attacker.poke_type is not EnergyType.COLORLESS:
        return attacker.poke_type
    return None


def choose_archetype(pool: CardPool, rank: int = 0) -> Archetype:
    """Select an attacker whose line bottoms out at a Basic Pokémon.

    ``rank`` picks among the Basic-anchored attackers ordered by offense score:
    0 = strongest (the default), 1 = next, etc. Used by the tuner to generate
    candidate decks. Out-of-range ranks clamp to the last acceptable attacker.
    """
    candidates = [c for c in pool.pokemon if c.hp and best_fixed_dpe(c) > 0]
    candidates.sort(
        key=lambda c: (offense_score(c), c.hp or 0, -c.card_id), reverse=True
    )

    acceptable: list[tuple[Card, list[Card]]] = []
    for attacker in candidates:
        chain = evolution_chain(pool, attacker)
        if chain[0].is_basic_pokemon:
            acceptable.append((attacker, chain))
            if len(acceptable) > rank:
                break  # have enough to satisfy this rank

    if not acceptable:
        raise ValueError(
            "no attacker with a Basic-anchored evolution line found in pool"
        )

    attacker, chain = acceptable[min(rank, len(acceptable) - 1)]
    counts = _LINE_COUNTS.get(len(chain), (3,) * len(chain))
    return Archetype(
        line=tuple(c.card_id for c in chain),
        counts=counts,
        energy_type=primary_energy_type(attacker),
        attacker_id=attacker.card_id,
    )
