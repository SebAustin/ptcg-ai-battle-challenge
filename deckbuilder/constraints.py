"""Deck legality — the deck-construction rules, read from :mod:`ptcg_bot.rules`.

Never hard-code a limit here; every constant comes from the rules source of
truth so the deckbuilder and the agent agree on what "legal" means.
"""

from __future__ import annotations

from ptcg_bot import rules
from ptcg_bot.cards import Card, CardPool

from .deck import Deck


def copy_limit(card: Card) -> int | None:
    """Max copies allowed by card name. ``None`` means unlimited (Basic Energy)."""
    if card.is_basic_energy:
        return None
    if card.is_ace_spec:
        return rules.ACE_SPEC_PER_DECK
    return rules.MAX_COPIES_BY_NAME


def validate(deck: Deck, pool: CardPool) -> list[str]:
    """Return a list of legality violations; an empty list means the deck is legal."""
    problems: list[str] = []
    resolved: list[tuple[Card, int]] = []
    for cid, n in deck.counts:
        card = pool.by_id(cid)
        if card is None:
            problems.append(f"unknown card id {cid}")
        else:
            resolved.append((card, n))

    total = deck.total
    if total != rules.DECK_SIZE:
        problems.append(f"deck has {total} cards, must be exactly {rules.DECK_SIZE}")

    # Copy limits are per card NAME, summed across reprints.
    by_name: dict[str, int] = {}
    name_card: dict[str, Card] = {}
    for card, n in resolved:
        by_name[card.name] = by_name.get(card.name, 0) + n
        name_card[card.name] = card
    for name, n in by_name.items():
        limit = copy_limit(name_card[name])
        if limit is not None and n > limit:
            problems.append(f"{name}: {n} copies exceeds the {limit}-per-name limit")

    ace = sum(n for card, n in resolved if card.is_ace_spec)
    if ace > rules.ACE_SPEC_PER_DECK:
        problems.append(
            f"{ace} ACE SPEC cards; at most {rules.ACE_SPEC_PER_DECK} allowed per deck"
        )

    basics = sum(n for card, n in resolved if card.is_basic_pokemon)
    if basics < rules.MIN_BASIC_POKEMON:
        problems.append(
            f"{basics} Basic Pokémon; need at least {rules.MIN_BASIC_POKEMON}"
        )

    return problems


def is_legal(deck: Deck, pool: CardPool) -> bool:
    """True when the deck passes every legality check in :func:`validate`."""
    return not validate(deck, pool)
