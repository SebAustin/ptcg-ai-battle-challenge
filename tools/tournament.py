"""Self-play tournament harness — measure agent/deck strength (plan §W4).

Drives the live ``libcg`` engine for N games between two policies, routing each
decision to the *deciding* player (``obs["current"]["yourIndex"]``), and reports
the win-rate. By default: our heuristic ``main.agent`` (player 0) vs a seeded
random baseline (player 1). This is the fitness signal for tuning the agent and
the deckbuilder.

Engine-gated (needs ``make engine`` + ``make data``).

Run: ``make tournament ARGS="--games 30"``  (or ``.venv/bin/python -m tools.tournament``)
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable
from pathlib import Path

_ENGINE = (
    Path(__file__).resolve().parent.parent
    / "engine"
    / "sample_submission"
    / "sample_submission"
)

Policy = Callable[[dict], list[int]]


def _import_game():
    if not (_ENGINE / "cg" / "api.py").exists():
        raise SystemExit("engine not found — run `make engine` first")
    sys.path.insert(0, str(_ENGINE))
    from cg import game  # type: ignore[import-not-found]

    return game


def random_policy(rng: random.Random) -> Policy:
    """A legal random selector (the sanctioned baseline: choose maxCount options)."""

    def choose(obs: dict) -> list[int]:
        select = obs.get("select") or {}
        options = select.get("option") or []
        if not options:
            return []
        want = min(int(select.get("maxCount") or 1), len(options))
        return rng.sample(range(len(options)), want)

    return choose


def play_game(
    game,
    deck0: list[int],
    deck1: list[int],
    policy0: Policy,
    policy1: Policy,
    max_steps: int = 5000,
) -> int:
    """Play one game; return the engine result (0=p0, 1=p1, 2=draw, -1=unfinished)."""
    obs, start = game.battle_start(deck0, deck1)
    if obs is None:
        raise SystemExit(f"engine rejected a deck (errorType={start.errorType})")
    try:
        for _ in range(max_steps):
            if obs.get("select") is None:
                return -1
            deciding = int((obs.get("current") or {}).get("yourIndex", 0))
            policy = policy0 if deciding == 0 else policy1
            obs = game.battle_select(policy(obs))
            result = (obs.get("current") or {}).get("result", -1)
            if result != -1:
                return result
        return -1
    finally:
        game.battle_finish()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PTCG self-play tournament")
    parser.add_argument("--games", type=int, default=20, help="number of games")
    args = parser.parse_args(argv)

    game = _import_game()
    from deckbuilder import build_deck
    from ptcg_bot.cards import load_pool
    from ptcg_bot.main import agent

    deck = [cid for cid, n in build_deck(load_pool()).counts for _ in range(n)]

    wins = losses = draws = unfinished = 0
    for i in range(args.games):
        result = play_game(
            game, deck, list(deck), agent, random_policy(random.Random(i))
        )
        if result == 0:
            wins += 1
        elif result == 1:
            losses += 1
        elif result == 2:
            draws += 1
        else:
            unfinished += 1

    decided = wins + losses
    win_rate = (wins / decided * 100.0) if decided else 0.0
    print(
        f"[tournament] {args.games} games — agent(P0) vs random(P1): "
        f"{wins}W / {losses}L / {draws}D / {unfinished}U"
    )
    print(f"[tournament] win-rate vs random (decided games): {win_rate:.1f}%")


if __name__ == "__main__":
    main()
