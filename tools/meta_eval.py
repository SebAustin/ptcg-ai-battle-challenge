"""Evaluate candidate decks against REAL ladder meta decks (dev-only).

``data/meta_decks.json`` (gitignored; rebuilt from episode replays) holds the
opponents' 60-card decklists from games we lost on Kaggle. This harness plays
each candidate deck against every meta deck — the SAME strong pilot on both
sides, first player alternated — so the score isolates the DECK against the
distribution that actually beats us, not a mirror.

Run: ``.venv/bin/python -m tools.meta_eval --games-per-deck 4``
(Reduce search budget via env for throughput, e.g. PTCG_TURN_DEADLINE_S=0.5.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ENGINE = _ROOT / "engine" / "sample_submission" / "sample_submission"
_META = _ROOT / "data" / "meta_decks.json"

# Candidate axes: (label, attacker_rank, target_energy).
_CANDIDATES = (
    ("default r0/e12", 0, 12),
    ("r1/e12", 1, 12),
    ("r2/e12", 2, 12),
    ("r3/e10", 3, 10),
    ("r0/e14", 0, 14),
    ("r1/e10", 1, 10),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="deck evaluation vs real meta decks")
    parser.add_argument("--games-per-deck", type=int, default=4)
    parser.add_argument(
        "--candidates", type=int, default=len(_CANDIDATES), help="how many to test"
    )
    args = parser.parse_args(argv)

    if not _META.exists():
        raise SystemExit("data/meta_decks.json missing — extract it from replays first")
    meta_decks: list[list[int]] = json.loads(_META.read_text())

    sys.path.insert(0, str(_ENGINE))
    from cg import game  # noqa: PLC0415

    from deckbuilder import build_deck, validate  # noqa: PLC0415
    from ptcg_bot.cards import load_pool  # noqa: PLC0415
    from ptcg_bot.main import agent  # noqa: PLC0415
    from tools.tournament import play_game  # noqa: PLC0415

    pool = load_pool()
    results = []
    for label, rank, energy in _CANDIDATES[: args.candidates]:
        try:
            deck = build_deck(pool, attacker_rank=rank, target_energy=energy)
        except Exception:
            continue
        if validate(deck, pool):
            continue
        ids = [c for c, n in deck.counts for _ in range(n)]
        wins = losses = 0
        for meta in meta_decks:
            for i in range(args.games_per_deck):
                if i % 2 == 0:
                    result = play_game(game, ids, list(meta), agent, agent)
                    outcome = result
                else:
                    result = play_game(game, list(meta), ids, agent, agent)
                    outcome = 1 - result if result in (0, 1) else result
                wins += outcome == 0
                losses += outcome == 1
        total = wins + losses
        rate = 100.0 * wins / total if total else 0.0
        results.append((rate, label, wins, losses))
        print(
            f"[meta] {label:>14}: {wins}W/{losses}L = {rate:.1f}% vs {len(meta_decks)} meta decks"
        )

    results.sort(reverse=True)
    best = results[0]
    print(f"\n[meta] best: {best[1]} at {best[0]:.1f}%  (default is the shipped deck)")


if __name__ == "__main__":
    main()
