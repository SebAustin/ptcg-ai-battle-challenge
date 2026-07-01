"""Determinize the hidden information for a search rollout.

The engine's ``search_begin`` needs concrete guesses for everything we cannot
see — our own deck/prize order and the opponent's deck/prize/hand/active — as
lists of valid card IDs whose *lengths* match the public counts, with at least
one Basic Pokémon in the opponent's deck.

This v1 fills those with valid, distinct card IDs (from the engine's card list)
of the correct lengths — a crude-but-legal determinization. For evaluating our
*own* next move one ply deep, the exact hidden cards matter little; correctness
of lengths/validity is what the engine enforces. Sampling from a realistic deck
prior is a future refinement (toward true IS-MCTS over many worlds).

Works on the engine's ``Observation`` dataclass. Returns ``None`` when card
metadata is unavailable (offline), so the caller falls back to the heuristic.
"""

from __future__ import annotations

from typing import Any

from . import metadata

# The six hidden-info arrays search_begin expects, in order.
Determinization = tuple[
    list[int], list[int], list[int], list[int], list[int], list[int]
]


def _fill(ids: tuple[int, ...], n: int) -> list[int]:
    """``n`` distinct valid IDs (counts are small vs the ~1.3k-card pool)."""
    return list(ids[:n])


def determinize(observation: Any) -> Determinization | None:
    """Return ``(your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active)``.

    ``None`` if the game state or card metadata is unavailable.
    """
    current = getattr(observation, "current", None)
    if current is None:
        return None
    valid = metadata.valid_card_ids()
    basics = metadata.basic_pokemon_ids()
    if not valid or not basics:
        return None

    me = int(getattr(current, "yourIndex", 0))
    players = getattr(current, "players", None) or []
    if len(players) < 2:
        return None
    us, them = players[me], players[1 - me]

    your_deck = _fill(valid, int(getattr(us, "deckCount", 0)))
    your_prize = _fill(valid, len(getattr(us, "prize", []) or []))
    opp_deck = _fill(valid, int(getattr(them, "deckCount", 0)))
    opp_prize = _fill(valid, len(getattr(them, "prize", []) or []))
    opp_hand = _fill(valid, int(getattr(them, "handCount", 0)))

    # The opponent's deck must contain at least one Basic Pokémon.
    basic_set = set(basics)
    if opp_deck and not any(cid in basic_set for cid in opp_deck):
        opp_deck[0] = basics[0]

    # opponent_active: exactly one Pokémon ID only when the Active is face-down.
    opp_active: list[int] = []
    active = getattr(them, "active", None) or []
    if active and active[0] is None:
        opp_active = [basics[0]]

    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active
