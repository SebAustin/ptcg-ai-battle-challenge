"""Deck fitness tuning by real self-play (plan §W6).

Generates candidate decks along two axes — attacker-line **rank** (0 = strongest)
× **energy count** — and scores each by win-rate vs the current default deck.
Both sides are piloted by the shipped heuristic agent (fast, ~0.15s/game), sides
alternated to cancel first-player advantage, so the number measures the *deck*,
not the pilot. The offline ``score.py`` is only a proxy; this is the real signal.

Prints a ranked table (real win-rate vs the offline score) and writes the best
deck to ``dist/best_deck.csv``. Adopt a candidate only if it *decisively* beats
the default.

Run: ``make tune ARGS="--games 40"``  (needs engine + data)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ENGINE = (
    Path(__file__).resolve().parent.parent
    / "engine"
    / "sample_submission"
    / "sample_submission"
)
_OUT = Path(__file__).resolve().parent.parent / "dist" / "best_deck.csv"
_RANKS = (0, 1, 2, 3)
_ENERGIES = (10, 12, 14)


def _import_game():
    if not (_ENGINE / "cg" / "api.py").exists():
        raise SystemExit("engine not found — run `make engine` first")
    sys.path.insert(0, str(_ENGINE))
    from cg import game  # type: ignore[import-not-found]

    return game


def _ids(deck) -> list[int]:
    return [cid for cid, n in deck.counts for _ in range(n)]


def _winrate_vs(game, play_game, agent, deck_ids, ref_ids, games) -> float:
    wins = losses = 0
    for i in range(games):
        if i % 2 == 0:
            result = play_game(game, deck_ids, list(ref_ids), agent, agent)
            outcome = result  # candidate is P0
        else:
            result = play_game(game, list(ref_ids), deck_ids, agent, agent)
            outcome = 1 - result if result in (0, 1) else result  # candidate is P1
        wins += outcome == 0
        losses += outcome == 1
    return (wins / (wins + losses) * 100.0) if (wins + losses) else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PTCG deck fitness tuner")
    parser.add_argument("--games", type=int, default=60, help="games per candidate")
    args = parser.parse_args(argv)

    game = _import_game()
    from deckbuilder import build_deck, score, validate
    from ptcg_bot.cards import load_pool
    from ptcg_bot.main import agent
    from tools.tournament import play_game

    pool = load_pool()
    default = build_deck(pool)
    default_ids = _ids(default)

    seen: set = {default.counts}
    candidates = [(0, 12, default, True)]  # (rank, energy, deck, is_default)
    for rank in _RANKS:
        for energy in _ENERGIES:
            try:
                deck = build_deck(pool, attacker_rank=rank, target_energy=energy)
            except Exception:
                continue
            if validate(deck, pool) or deck.counts in seen:
                continue
            # Only keep decks the live engine accepts.
            obs, start = game.battle_start(_ids(deck), list(default_ids))
            game.battle_finish()
            if obs is None:
                continue
            seen.add(deck.counts)
            candidates.append((rank, energy, deck, False))

    print(f"[tune] {len(candidates)} candidates × {args.games} games vs default\n")
    rows = []
    for rank, energy, deck, is_default in candidates:
        offline = score(deck, pool).total
        if is_default:
            win_rate = 50.0
        else:
            win_rate = _winrate_vs(
                game, play_game, agent, _ids(deck), default_ids, args.games
            )
        rows.append((win_rate, rank, energy, offline, is_default, deck))

    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"{'winrate% vs default':>20} {'rank':>5} {'energy':>7} {'offline':>8}  note")
    for win_rate, rank, energy, offline, is_default, _ in rows:
        note = "DEFAULT" if is_default else ""
        print(f"{win_rate:>20.1f} {rank:>5} {energy:>7} {offline:>8.1f}  {note}")

    best = rows[0]
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(best[5].to_csv(), encoding="utf-8")
    print(f"\n[tune] best: rank={best[1]} energy={best[2]} winrate={best[0]:.1f}%")
    print(f"[tune] wrote {_OUT}")
    # The engine RNG is un-seeded, so a single small sample is noisy; only treat a
    # LARGE, clear margin as a real win, and always say to re-check the leader.
    if best[4] or best[0] < 57.0:
        print(
            "[tune] no candidate clearly beats the default at this sample — keep the "
            "default deck. (Small samples are noisy; the offline score also proved a "
            "weak predictor of real win-rate.)"
        )
    else:
        print(
            f"[tune] rank={best[1]} energy={best[2]} leads at this sample — CONFIRM with a "
            "large `--games` run before adopting; results are noisy under the un-seeded RNG."
        )


if __name__ == "__main__":
    main()
