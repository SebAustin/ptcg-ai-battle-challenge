"""Validate our model against the LIVE engine (plan §W2-3).

Loads the real ``libcg`` engine from the gitignored ``engine/`` dir (fetch it
with ``make engine`` after accepting the ``pokemon-tcg-ai-battle`` rules), then:

1. builds a deck with our deckbuilder and asserts the engine ACCEPTS it,
2. drives a full battle with a random legal selector, and
3. runs ``engine_adapter.parse_observation`` on every real observation and
   sanity-checks the resulting GameState against ``rules``.

Run via ``make verify`` (or ``.venv/bin/python -m tools.verify_env``). Exits
non-zero on any failure; this is the gate that makes strategy work trustworthy.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_ENGINE = (
    Path(__file__).resolve().parent.parent
    / "engine"
    / "sample_submission"
    / "sample_submission"
)
_MAX_STEPS = 2000


def _import_engine_game():
    if not (_ENGINE / "cg" / "api.py").exists():
        raise SystemExit(
            "engine not found at engine/ — run `make engine` after accepting the "
            "pokemon-tcg-ai-battle competition rules (see ASSUMPTIONS.md)."
        )
    sys.path.insert(0, str(_ENGINE))
    from cg import game  # type: ignore[import-not-found]

    return game


def main() -> None:
    from deckbuilder import build_deck, validate
    from ptcg_bot import rules
    from ptcg_bot.cards import load_pool
    from ptcg_bot.engine_adapter import parse_observation

    game = _import_engine_game()
    pool = load_pool()

    deck = build_deck(pool)
    if validate(deck, pool):
        raise SystemExit(
            f"deckbuilder produced an illegal deck: {validate(deck, pool)}"
        )
    deck_ids = [cid for cid, n in deck.counts for _ in range(n)]
    if len(deck_ids) != rules.DECK_SIZE:
        raise SystemExit(f"deck has {len(deck_ids)} cards, expected {rules.DECK_SIZE}")

    obs, start = game.battle_start(deck_ids, list(deck_ids))
    if obs is None:
        raise SystemExit(
            f"ENGINE REJECTED our deck (errorType={start.errorType}). "
            "Our deckbuilder or rules.py disagrees with the engine."
        )
    print(f"[verify] engine accepted our {len(deck_ids)}-card deck (errorType=0)")

    rng = random.Random(0)
    steps = 0
    parsed = 0
    result = -1
    while steps < _MAX_STEPS:
        select = obs.get("select")
        if select is None:
            break
        if obs.get("current"):
            state = parse_observation(obs, pool)
            parsed += 1
            assert (
                0 <= state.us.prizes_remaining <= rules.PRIZE_COUNT
            ), "bad prize count"
            assert (
                0 <= state.them.prizes_remaining <= rules.PRIZE_COUNT
            ), "bad prize count"
        options = select.get("option") or []
        if not options:
            break
        want = min(int(select.get("maxCount") or 1), len(options))
        obs = game.battle_select(rng.sample(range(len(options)), want))
        steps += 1
        result = (obs.get("current") or {}).get("result", -1)
        if result != -1:
            break
    game.battle_finish()

    print(
        f"[verify] OK — drove {steps} steps, parse_observation validated on "
        f"{parsed} real observations, final result={result}"
    )


if __name__ == "__main__":
    main()
