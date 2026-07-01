"""Immutable decklist model + CSV serialization.

A :class:`Deck` is just sorted ``(card_id, count)`` pairs — hashable, cheap to
compare, and safe to share. Resolution to full :class:`~ptcg_bot.cards.Card`
objects is done lazily against a :class:`~ptcg_bot.cards.CardPool`.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from dataclasses import dataclass

from ptcg_bot.cards import Card, CardPool

# The submission's deck.csv columns. SUBMIT.md says to confirm the exact schema
# on the competition page; this is the one place it is defined (see ASSUMPTIONS).
_CSV_HEADER = "Card ID,Card Name,Count"


@dataclass(frozen=True)
class Deck:
    """A decklist as an immutable, sorted tuple of ``(card_id, count)`` pairs."""

    counts: tuple[tuple[int, int], ...] = ()

    @classmethod
    def from_counts(cls, counts: dict[int, int]) -> Deck:
        """Build from a ``{card_id: count}`` mapping, dropping non-positive counts."""
        items = tuple(sorted((cid, n) for cid, n in counts.items() if n > 0))
        return cls(items)

    @property
    def total(self) -> int:
        return sum(n for _, n in self.counts)

    def as_dict(self) -> dict[int, int]:
        return dict(self.counts)

    def cards(self, pool: CardPool) -> Iterator[tuple[Card, int]]:
        """Yield ``(Card, count)`` for every resolvable entry, in id order."""
        for cid, n in self.counts:
            card = pool.by_id(cid)
            if card is not None:
                yield card, n

    def to_csv(self, pool: CardPool) -> str:
        """Serialize to the submission deck.csv (``Card ID,Card Name,Count``)."""
        lines = [_CSV_HEADER]
        for card, n in self.cards(pool):
            name = card.name
            if "," in name or '"' in name:
                name = '"' + name.replace('"', '""') + '"'
            lines.append(f"{card.card_id},{name},{n}")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_csv(cls, text: str) -> Deck:
        """Parse a deck.csv back into a :class:`Deck` (inverse of :meth:`to_csv`)."""
        counts: dict[int, int] = {}
        for row in csv.DictReader(io.StringIO(text)):
            cid = int(row["Card ID"])
            counts[cid] = counts.get(cid, 0) + int(row["Count"])
        return cls.from_counts(counts)
