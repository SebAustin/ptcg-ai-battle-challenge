"""Self-play data generation for the learned evaluator (dev-only).

Plays engine games with a mix of pilots and records, at every MAIN decision,
``features.extract(obs, deciding_player)`` labeled with that player's final
result (1 win / 0 loss / 0.5 draw). Both players contribute rows, so each game
yields ~20-60 positions.

Coverage comes from the pilot mix (heuristic mirror / epsilon-noised heuristic /
random opponent) and from opponent-deck variants off the ``build_deck`` grid —
mitigating "trained only on the mirror".

The engine is a per-process singleton, so parallelism is one engine per worker
process (spawn). Output: CSV shards in ``data/selfplay/<gen>/`` (gitignored),
header ``game_id,label,<FEATURE_NAMES...>``.

Run: ``.venv/bin/python -m tools.selfplay --games 30000 --workers 8 --out data/selfplay/gen0``
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections.abc import Callable
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ENGINE = _ROOT / "engine" / "sample_submission" / "sample_submission"

Policy = Callable[[dict], list[int]]

# Deck variants for opponent diversity (rank, energy) — the tune.py grid.
_DECK_VARIANTS = ((0, 12), (1, 12), (2, 12), (3, 10), (0, 14))


def epsilon_policy(base: Policy, eps: float, rng: random.Random) -> Policy:
    """``base`` but with probability ``eps`` a random legal selection."""

    def choose(obs: dict) -> list[int]:
        select = obs.get("select") or {}
        options = select.get("option") or []
        if options and rng.random() < eps:
            want = min(int(select.get("maxCount") or 1), len(options))
            return rng.sample(range(len(options)), want)
        return base(obs)

    return choose


def _worker(args: tuple[int, int, str, int, str | None]) -> str:
    """Play ``n_games`` and write one CSV shard; returns the shard path."""
    worker_id, n_games, out_dir, id_offset, meta_path = args
    sys.path.insert(0, str(_ENGINE))
    import json  # noqa: PLC0415

    from cg import game  # noqa: PLC0415

    from deckbuilder import build_deck  # noqa: PLC0415
    from ptcg_bot.cards import load_pool  # noqa: PLC0415
    from ptcg_bot.features import FEATURE_NAMES, extract  # noqa: PLC0415
    from ptcg_bot.main import agent  # noqa: PLC0415

    pool = load_pool()
    our_deck = [c for c, n in build_deck(pool).counts for _ in range(n)]
    variant_decks = []
    for rank, energy in _DECK_VARIANTS:
        try:
            deck = build_deck(pool, attacker_rank=rank, target_energy=energy)
            variant_decks.append([c for c, n in deck.counts for _ in range(n)])
        except Exception:
            continue

    # Meta mode: sample BOTH sides' decks from real ladder decklists, so the
    # training distribution matches the games the agent actually plays.
    meta_decks: list[list[int]] = []
    if meta_path:
        meta_decks = json.loads(Path(meta_path).read_text(encoding="utf-8"))

    rng = random.Random(worker_id * 7919 + 13)

    def random_policy(obs: dict) -> list[int]:
        select = obs.get("select") or {}
        options = select.get("option") or []
        if not options:
            return []
        want = min(int(select.get("maxCount") or 1), len(options))
        return rng.sample(range(len(options)), want)

    shard = Path(out_dir) / f"shard_{worker_id}.csv"
    shard.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with shard.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["game_id", "label", *FEATURE_NAMES])
        for i in range(n_games):
            game_id = id_offset + worker_id * 10_000_000 + i
            # Pilot mix: 50% mirror, 35% vs eps-heuristic, 15% vs random.
            roll = rng.random()
            if roll < 0.50:
                opponent: Policy = agent
            elif roll < 0.85:
                opponent = epsilon_policy(agent, rng.choice((0.1, 0.25)), rng)
            else:
                opponent = random_policy
            if meta_decks:
                my_deck = rng.choice(meta_decks)
                opp_deck = rng.choice(meta_decks)
            else:
                my_deck = our_deck
                opp_deck = (
                    rng.choice(variant_decks)
                    if (variant_decks and rng.random() < 0.3)
                    else our_deck
                )
            we_start = i % 2 == 0
            deck0 = list(my_deck) if we_start else list(opp_deck)
            deck1 = list(opp_deck) if we_start else list(my_deck)
            p0: Policy = agent if we_start else opponent
            p1: Policy = opponent if we_start else agent

            obs, start = game.battle_start(deck0, deck1)
            if obs is None:
                continue
            rows: list[tuple[int, list[float]]] = []
            result = -1
            try:
                for _ in range(5000):
                    select = obs.get("select")
                    if select is None:
                        break
                    current = obs.get("current") or {}
                    deciding = int(current.get("yourIndex") or 0)
                    if select.get("type") == 0:  # MAIN decision
                        rows.append((deciding, extract(obs, deciding)))
                    policy = p0 if deciding == 0 else p1
                    obs = game.battle_select(policy(obs))
                    result = (obs.get("current") or {}).get("result", -1)
                    if result != -1:
                        break
            finally:
                game.battle_finish()

            if result not in (0, 1, 2):
                continue  # unfinished — drop
            for perspective, feats in rows:
                label = 0.5 if result == 2 else (1.0 if result == perspective else 0.0)
                writer.writerow([game_id, label, *feats])
                written += 1
    print(f"[selfplay] worker {worker_id}: {written} rows -> {shard}")
    return str(shard)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="self-play data generation")
    parser.add_argument("--games", type=int, default=2000, help="total games")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="data/selfplay/gen0")
    parser.add_argument(
        "--id-offset",
        type=int,
        default=0,
        help="added to every game_id (avoids collisions when combining generations)",
    )
    parser.add_argument(
        "--meta-decks",
        default=None,
        help="JSON file of real decklists; both sides sample from it (meta-vs-meta)",
    )
    args = parser.parse_args(argv)

    if not (_ENGINE / "cg" / "api.py").exists():
        raise SystemExit("engine not found — run `make engine` first")

    per_worker = max(1, args.games // args.workers)
    jobs = [
        (w, per_worker, args.out, args.id_offset, args.meta_decks)
        for w in range(args.workers)
    ]
    if args.workers <= 1:
        shards = [_worker(jobs[0])]
    else:
        import multiprocessing as mp  # noqa: PLC0415

        with mp.get_context("spawn").Pool(args.workers) as pool:
            shards = pool.map(_worker, jobs)
    print(f"[selfplay] wrote {len(shards)} shards under {args.out}")


if __name__ == "__main__":
    main()
