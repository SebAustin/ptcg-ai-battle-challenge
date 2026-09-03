"""Pilot gate: our shipped deck vs every meta deck, SAME pilot both sides.

The flywheel's standing evaluator-gate harness (ASSUMPTIONS §25 pre-registered
the gen4 bar against this protocol): plays ``--deck`` (default the shipped
``data/best_meta_deck.csv``) against every deck in ``data/meta_decks.json``,
``--games-per-deck`` games with first player alternated, BOTH sides piloted by
``ptcg_bot.main.agent`` — whatever evaluator weights/features are currently
installed. An evaluator change upgrades both pilots, so OUR side's pooled win
rate over the recorded baseline measures how much better the new brain
exploits OUR deck against the field's decks (same philosophy as
``tools/meta_eval.py``, which isolates the DECK the same way).

Turn 3 ran this protocol from a scratch script (85.1% baseline over 204
games); this tool makes the harness reproducible.

Run: ``PTCG_TURN_DEADLINE_S=0.5 .venv/bin/python -m tools.pilot_gate --workers 8``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ENGINE = _ROOT / "engine" / "sample_submission" / "sample_submission"
_META = _ROOT / "data" / "meta_decks.json"
_DEFAULT_DECK = _ROOT / "data" / "best_meta_deck.csv"


def _load_deck(path: Path) -> list[int]:
    ids = [
        int(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(ids) != 60:
        raise SystemExit(f"{path}: expected 60 card ids, got {len(ids)}")
    return ids


def _worker(args: tuple[int, list[int], list[list[int]], int]) -> tuple[int, int]:
    """Play our deck vs this worker's slice of meta decks; return (wins, losses)."""
    worker_id, ours, metas, games_per_deck = args
    sys.path.insert(0, str(_ENGINE))
    from cg import game  # noqa: PLC0415

    from ptcg_bot.main import agent  # noqa: PLC0415
    from tools.tournament import play_game  # noqa: PLC0415

    wins = losses = 0
    for meta in metas:
        for i in range(games_per_deck):
            if i % 2 == 0:
                outcome = play_game(game, list(ours), list(meta), agent, agent)
            else:
                result = play_game(game, list(meta), list(ours), agent, agent)
                outcome = 1 - result if result in (0, 1) else result
            wins += outcome == 0
            losses += outcome == 1
    print(f"[pilot-gate] worker {worker_id}: {wins}W/{losses}L")
    return wins, losses


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="our deck vs every meta deck, same pilot both sides"
    )
    parser.add_argument("--deck", default=str(_DEFAULT_DECK))
    parser.add_argument("--games-per-deck", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--label", default="pilot", help="tag for the report line")
    parser.add_argument("--meta", default=str(_META), help="opponent deck set JSON")
    args = parser.parse_args(argv)

    if not (_ENGINE / "cg" / "api.py").exists():
        raise SystemExit("engine not found — run `make engine` first")
    meta_path = Path(args.meta)
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} missing — extract it from replays")

    ours = _load_deck(Path(args.deck))
    metas: list[list[int]] = json.loads(meta_path.read_text(encoding="utf-8"))

    workers = max(1, min(args.workers, len(metas)))
    slices: list[list[list[int]]] = [metas[w::workers] for w in range(workers)]
    jobs = [(w, ours, slices[w], args.games_per_deck) for w in range(workers)]
    if workers == 1:
        outcomes = [_worker(jobs[0])]
    else:
        import multiprocessing as mp  # noqa: PLC0415

        with mp.get_context("spawn").Pool(workers) as pool:
            outcomes = pool.map(_worker, jobs)

    wins = sum(won for won, _ in outcomes)
    losses = sum(lost for _, lost in outcomes)
    total = wins + losses
    rate = 100.0 * wins / total if total else 0.0
    print(
        f"[pilot-gate] {args.label}: {wins}W/{losses}L = {rate:.1f}% "
        f"pooled over {total} games vs {len(metas)} meta decks"
    )


if __name__ == "__main__":
    main()
