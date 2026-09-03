"""Heuristic role classification of Trainer / Special-Energy cards.

Keyword matching over ``Card.text`` — a transparent approximation of what a
Trainer *does*, good enough to assemble a balanced package and to explain the
deck in the writeup. It is deliberately not a full effect parser.
"""

from __future__ import annotations

from enum import Enum

from ptcg_bot.cards import Card


class Role(str, Enum):
    """A functional role a Trainer (or Special Energy) can fill in a deck."""

    DRAW = "draw"
    SEARCH = "search"
    SWITCH = "switch"
    DISRUPTION = "disruption"
    HEAL = "heal"
    ENERGY_ACCEL = "energy_accel"
    STADIUM = "stadium"
    TOOL = "tool"
    OTHER = "other"


def classify(card: Card) -> set[Role]:
    """Return the set of roles a card fills (empty for Pokémon / Basic Energy)."""
    is_special_energy = card.is_energy and not card.is_basic_energy
    if not card.is_trainer and not is_special_energy:
        return set()

    roles: set[Role] = set()
    if card.subtype == "Stadium":
        roles.add(Role.STADIUM)
    if card.subtype == "Pokémon Tool":
        roles.add(Role.TOOL)

    text = card.text.lower()
    if not text:
        return roles or {Role.OTHER}

    if "draw" in text:
        roles.add(Role.DRAW)
    if "search your deck" in text:
        roles.add(Role.SEARCH)
    if "switch" in text or "to the active spot" in text:
        roles.add(Role.SWITCH)
    if "heal" in text or ("remove" in text and "damage" in text):
        roles.add(Role.HEAL)
    if "attach" in text and "energy" in text:
        roles.add(Role.ENERGY_ACCEL)
    if "opponent" in text and (
        "discard" in text or "shuffle" in text or "reveal their hand" in text
    ):
        roles.add(Role.DISRUPTION)

    return roles or {Role.OTHER}


def has_role(card: Card, role: Role) -> bool:
    return role in classify(card)
