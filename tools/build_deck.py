"""Build the submission deck.csv from the card pool (Strategy 20% deliverable).

Runs the offline deckbuilder and writes ``dist/deck.csv`` — the decklist that
ships alongside the agent. Dev-only tool; not part of the bundled agent.

    make deck        # or: .venv/bin/python -m tools.build_deck
"""

from __future__ import annotations

from pathlib import Path

from deckbuilder import build_deck, score, validate
from ptcg_bot.cards import load_pool

OUT_PATH = Path(__file__).resolve().parent.parent / "dist" / "deck.csv"


def main() -> None:
    pool = load_pool()
    deck = build_deck(pool)

    problems = validate(deck, pool)
    if problems:
        raise SystemExit("refusing to write an illegal deck: " + "; ".join(problems))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(deck.to_csv(), encoding="utf-8")

    card = score(deck, pool)
    print(f"wrote {OUT_PATH}  ({deck.total} cards, score {card.total}/100)")
    for name, value in card.breakdown:
        print(f"  {name:<16} {value}")
    print("\nPokémon / Energy lines:")
    for resolved, count in deck.cards(pool):
        if not resolved.is_trainer:
            print(f"  {count}x {resolved.name}")


if __name__ == "__main__":
    main()
