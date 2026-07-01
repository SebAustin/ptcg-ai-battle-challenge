"""Generate the writeup figures — our own charts only (no card art).

Produces PNGs in ``writeup/figures/``:
  1. deck composition (Pokémon / Trainer / Energy counts)
  2. the deckbuilder's heuristic deck-quality breakdown
  3. agent win-rate vs a random baseline (develop-first vs attack-first)

Dev-only; needs the competition data (``make data``). Run: ``make figures``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402  (must follow use())

_OUT = Path(__file__).resolve().parent / "figures"

# Representative single-run win-rates vs the random baseline (tools/tournament).
# The engine RNG is un-seeded, so runs vary — attack-first ~75-100%, develop-first
# ~5-25%. These illustrative values show the qualitative gap, not fixed constants.
_WINRATE_DEVELOP_FIRST = 20.8
_WINRATE_ATTACK_FIRST = 100.0


def _save(fig: plt.Figure, name: str) -> Path:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def composition_figure(pool, deck) -> Path:
    counts: Counter[str] = Counter()
    for card, n in deck.cards(pool):
        kind = (
            "Pokémon" if card.is_pokemon else "Energy" if card.is_energy else "Trainer"
        )
        counts[kind] += n
    order = ["Pokémon", "Trainer", "Energy"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(
        order,
        [counts.get(k, 0) for k in order],
        color=["#d1495b", "#2e86ab", "#e3b505"],
    )
    ax.set_ylabel("cards")
    ax.set_title(f"Deck composition ({deck.total} cards)")
    for i, k in enumerate(order):
        ax.text(i, counts.get(k, 0) + 0.4, str(counts.get(k, 0)), ha="center")
    return _save(fig, "deck_composition.png")


def score_figure(scorecard) -> Path:
    labels = [k for k, _ in scorecard.breakdown]
    values = [v for _, v in scorecard.breakdown]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(labels, values, color="#2e86ab")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("component score (0–1)")
    ax.set_title(f"Deck-quality heuristic — total {scorecard.total}/100")
    fig.gca().invert_yaxis()
    return _save(fig, "deck_score.png")


def winrate_figure() -> Path:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ["develop-first\n(v1)", "attack-first\n(v2)"]
    values = [_WINRATE_DEVELOP_FIRST, _WINRATE_ATTACK_FIRST]
    ax.bar(bars, values, color=["#9aa0a6", "#d1495b"])
    ax.axhline(50, ls="--", lw=1, color="#666")
    ax.set_ylim(0, 105)
    ax.set_ylabel("win-rate vs random (%)")
    ax.set_title("Policy win-rate vs random (representative run; engine RNG un-seeded)")
    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center")
    return _save(fig, "winrate.png")


def main() -> None:
    from deckbuilder import build_deck, score
    from ptcg_bot.cards import DEFAULT_CSV, load_pool

    if not DEFAULT_CSV.exists():
        raise SystemExit("competition dataset not present — run `make data` first")

    pool = load_pool()
    deck = build_deck(pool)
    paths = [
        composition_figure(pool, deck),
        score_figure(score(deck, pool)),
        winrate_figure(),
    ]
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
