"""Determinize the hidden information for a search rollout.

The engine's ``search_begin`` needs concrete guesses for everything we cannot
see — our own deck/prize order and the opponent's deck/prize/hand/active — as
lists of valid card IDs whose *lengths* match the public counts, with at least
one Basic Pokémon in the opponent's deck.

Two halves, with very different certainty:

- **Our side is known.** The observation reveals our hand, board, and discard,
  and we know our own 60-card decklist (``deck.csv``). So our unseen cards are
  exactly ``decklist − visible``; we shuffle them per world and split into deck
  and prizes. This is an accurate determinization (only the *order* is unknown).
- **The opponent's hidden cards are not known.** We fill them with a *varied
  legal sample* of real card IDs (≥1 Basic in the deck), re-sampled per world —
  a plausible stand-in, not a learned prior. The opponent's Active is used
  exactly when it is face-up.

Per-world randomness (a caller-supplied ``rng``) makes the K worlds genuinely
different, which is what gives PIMC something to average over. Returns ``None``
when card metadata is unavailable (offline), so the caller falls back to the
heuristic.
"""

from __future__ import annotations

import random
from collections import Counter
from functools import cache
from typing import Any

from . import metadata

# (your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active)
Determinization = tuple[
    list[int], list[int], list[int], list[int], list[int], list[int]
]

_DECK_PATHS = ("deck.csv", "/kaggle_simulations/agent/deck.csv")
_DECK_SIZE = 60


def _read_ids(path: str) -> list[int]:
    try:
        with open(path, encoding="utf-8") as handle:
            return [int(x) for x in handle.read().splitlines() if x.strip()]
    except Exception:
        return []


@cache
def my_deck_ids() -> tuple[int, ...]:
    """Our own decklist (60 card IDs) from deck.csv, or ``()`` if unavailable."""
    for path in _DECK_PATHS:
        ids = _read_ids(path)
        if len(ids) >= _DECK_SIZE:
            return tuple(ids[:_DECK_SIZE])
    return ()


def _sample(valid: tuple[int, ...], n: int, rng: random.Random) -> list[int]:
    if n <= 0:
        return []
    if n <= len(valid):
        return rng.sample(valid, n)
    return [rng.choice(valid) for _ in range(n)]


def _card_id(obj: Any) -> int | None:
    cid = getattr(obj, "id", None)
    return int(cid) if cid is not None else None


def _our_visible_ids(player: Any) -> list[int]:
    """Card IDs we can see on our own side (hand + board + attached + discard)."""
    ids: list[int] = []
    for card in getattr(player, "hand", None) or []:
        cid = _card_id(card)
        if cid is not None:
            ids.append(cid)
    for zone in ("active", "bench"):
        for pokemon in getattr(player, zone, None) or []:
            if pokemon is None:
                continue
            for obj in (
                pokemon,
                *(getattr(pokemon, "energyCards", None) or []),
                *(getattr(pokemon, "preEvolution", None) or []),
            ):
                cid = _card_id(obj)
                if cid is not None:
                    ids.append(cid)
    for card in getattr(player, "discard", None) or []:
        cid = _card_id(card)
        if cid is not None:
            ids.append(cid)
    return ids


def _remaining(deck: list[int], seen: list[int]) -> list[int]:
    """Multiset ``deck − seen`` (each seen card removes one copy)."""
    counts = Counter(deck)
    for cid in seen:
        if counts.get(cid, 0) > 0:
            counts[cid] -= 1
    out: list[int] = []
    for cid, n in counts.items():
        out.extend([cid] * n)
    return out


def determinize(
    observation: Any,
    my_deck: list[int] | None = None,
    rng: random.Random | None = None,
) -> Determinization | None:
    """Return one sampled ``(your_deck, your_prize, opp_deck, opp_prize, opp_hand,
    opp_active)``, or ``None`` if card metadata is unavailable."""
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
    r = rng or random.Random(0)

    us_deck_n = int(getattr(us, "deckCount", 0) or 0)
    us_prize_n = len(getattr(us, "prize", []) or [])

    # Our side: exact set (decklist − visible), shuffled; else a legal sample.
    if my_deck:
        remaining = _remaining(my_deck, _our_visible_ids(us))
        r.shuffle(remaining)
        need = us_deck_n + us_prize_n
        if len(remaining) < need:  # accounting gaps (tools/lost zone) → pad
            remaining.extend(_sample(valid, need - len(remaining), r))
        your_deck = remaining[:us_deck_n]
        your_prize = remaining[us_deck_n : us_deck_n + us_prize_n]
    else:
        your_deck = _sample(valid, us_deck_n, r)
        your_prize = _sample(valid, us_prize_n, r)

    # Opponent side: varied legal sample; ensure a Basic in the deck.
    opp_deck = _sample(valid, int(getattr(them, "deckCount", 0) or 0), r)
    if opp_deck and not any(cid in set(basics) for cid in opp_deck):
        opp_deck[0] = r.choice(basics)
    opp_prize = _sample(valid, len(getattr(them, "prize", []) or []), r)
    opp_hand = _sample(valid, int(getattr(them, "handCount", 0) or 0), r)

    opp_active: list[int] = []
    active = getattr(them, "active", None) or []
    if active and active[0] is None:
        opp_active = [r.choice(basics)]

    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active
