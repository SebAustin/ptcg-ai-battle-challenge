"""Offline deck construction & optimization (the Strategy 20% deliverable).

Not submitted with the agent. Candidate decklists are ultimately scored by the
SAME tournament harness used to evaluate the agent, so deck strength and agent
strength share one fitness function; until that engine harness is wired,
:mod:`deckbuilder.score` supplies an offline proxy (see ASSUMPTIONS.md).

Modules: deck, constraints, roles, score, archetype, optimize.

Typical use::

    from ptcg_bot.cards import load_pool
    from deckbuilder import build_deck, score, validate

    pool = load_pool()
    deck = build_deck(pool)
    assert validate(deck, pool) == []
    print(score(deck, pool).total)
    print(deck.to_csv(pool))
"""

from __future__ import annotations

from .archetype import Archetype, choose_archetype
from .constraints import copy_limit, is_legal, validate
from .deck import Deck
from .optimize import DeckError, build_deck
from .roles import Role, classify
from .score import Scorecard, score

__all__ = [
    "Archetype",
    "Deck",
    "DeckError",
    "Role",
    "Scorecard",
    "build_deck",
    "choose_archetype",
    "classify",
    "copy_limit",
    "is_legal",
    "score",
    "validate",
]
