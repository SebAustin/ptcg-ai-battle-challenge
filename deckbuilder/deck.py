"""Immutable decklist model + CSV serialization.

A :class:`Deck` is just sorted ``(card_id, count)`` pairs — hashable, cheap to
compare, and safe to share. Resolution to full :class:`~ptcg_bot.cards.Card`
objects is done lazily against a :class:`~ptcg_bot.cards.CardPool`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ptcg_bot.cards import Card, CardPool


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

    def to_csv(self) -> str:
        """Serialize to the engine's deck.csv: one card ID per line, 60 lines.

        The engine reads exactly 60 integer lines with copies repeated and no
        header (see engine sample_submission/deck.csv and its main.py, which does
        ``int(csv[i])`` for ``i`` in ``range(60)``).
        """
        lines = [str(cid) for cid, n in self.counts for _ in range(n)]
        return "\n".join(lines) + "\n"

    @classmethod
    def from_csv(cls, text: str) -> Deck:
        """Parse the engine deck.csv (one card ID per line) into a :class:`Deck`."""
        counts: dict[int, int] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                cid = int(stripped)
                counts[cid] = counts.get(cid, 0) + 1
        return cls.from_counts(counts)
