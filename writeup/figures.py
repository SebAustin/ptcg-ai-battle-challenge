"""Generate the writeup figures — our own charts only (no card art).

Produces PNGs in ``writeup/figures/`` (each a deterministic 1008x588, clearing
Kaggle's 640x360 Media Gallery minimum) plus ``writeup/CAPTIONS.md``:
  1. deck composition (Pokémon / Trainer / Energy counts)
  2. the deckbuilder's heuristic deck-quality breakdown
  3. agent win-rate vs a random baseline (develop-first vs attack-first)
  4. search-vs-heuristic progress across the build passes

Dev-only; needs the competition data (``make data``). Run: ``make figures``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402  (must follow use())

_OUT = Path(__file__).resolve().parent / "figures"

# Kaggle's Media Gallery requires >= 640x360 px. We fix the canvas size and DPI
# and do NOT crop with bbox_inches="tight" (that trimmed output to 635 wide, just
# under the minimum), so every PNG is a deterministic 1008x588.
_FIG_W, _FIG_H, _DPI = 7.2, 4.2, 140

# One caption per figure for the Kaggle Media Gallery (written to CAPTIONS.md).
_CAPTIONS = {
    "deck_composition.png": (
        "Deck composition — the built 60-card deck: a focused attacker line, "
        "single-prize support Basics, ~12 Basic Energy, and a role-balanced "
        "Trainer package."
    ),
    "deck_score.png": (
        "Offline deck-quality heuristic (0-100) — the transparent proxy used "
        "during construction: consistency, energy balance, opening reliability, "
        "attacker power, and prize safety."
    ),
    "winrate.png": (
        "Why the policy fix mattered: switching the agent from develop-first to "
        "attack-first lifted its win-rate vs a random baseline from ~21% to ~100% "
        "(representative run; engine RNG un-seeded)."
    ),
    "search_progress.png": (
        "Honest search progress: determinized rollout-PIMC climbed from ~10% to "
        "~52% vs the heuristic (perspective-bug fix + K worlds + realistic "
        "determinization) — parity, not a decisive win, so the heuristic still ships."
    ),
}

# Representative single-run win-rates vs the random baseline (tools/tournament).
# The engine RNG is un-seeded, so runs vary — attack-first ~75-100%, develop-first
# ~5-25%. These illustrative values show the qualitative gap, not fixed constants.
_WINRATE_DEVELOP_FIRST = 20.8
_WINRATE_ATTACK_FIRST = 100.0

# Search vs the heuristic (mirror A/B) across the build passes — representative
# runs; the point is the trend from "much worse" to "parity", not exact values.
_SEARCH_PROGRESS = (
    ("1-ply\n(buggy)", 10.0),
    ("PIMC +\nperspective fix", 47.0),
    ("+ realistic\ndeterminization", 52.0),
)


def _save(fig: plt.Figure, name: str) -> Path:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    fig.set_size_inches(_FIG_W, _FIG_H)  # enforce a min-resolution canvas
    fig.tight_layout()  # arrange within the canvas (does NOT crop it)
    fig.savefig(path, dpi=_DPI)  # deterministic 1008x588 px, no tight-bbox crop
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


def search_progress_figure() -> Path:
    labels = [label for label, _ in _SEARCH_PROGRESS]
    values = [value for _, value in _SEARCH_PROGRESS]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(labels, values, color=["#9aa0a6", "#2e86ab", "#d1495b"])
    ax.axhline(50, ls="--", lw=1, color="#666")
    ax.set_ylim(0, 60)
    ax.set_ylabel("win-rate vs heuristic (%)")
    ax.set_title("Search vs heuristic — honest progress (mirror A/B; RNG un-seeded)")
    for i, value in enumerate(values):
        ax.text(i, value + 1, f"{value:.0f}%", ha="center")
    return _save(fig, "search_progress.png")


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
        search_progress_figure(),
    ]

    captions_md = ["# Media Gallery captions\n"]
    for path in paths:
        caption = _CAPTIONS.get(path.name, "")
        print(f"wrote {path}")
        print(f"  caption: {caption}\n")
        captions_md.append(f"**{path.name}**\n\n{caption}\n")
    (_OUT.parent / "CAPTIONS.md").write_text("\n".join(captions_md), encoding="utf-8")
    print(
        f"wrote {_OUT.parent / 'CAPTIONS.md'} (copy captions into the Kaggle gallery)"
    )


if __name__ == "__main__":
    main()
