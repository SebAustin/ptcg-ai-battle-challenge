"""Isolation boundary for the (not-yet-local) official simulator.

THE ONLY MODULE THAT CHANGES WHEN THE SIMULATOR API LANDS.

The engine is now published (2026-07-01) on the ``pokemon-tcg-ai-battle``
competition as a vendored Python ``cg/`` package (api.py/game.py/sim.py +
native ``libcg``), but downloading it is gated on accepting that competition's
rules, so its observation/action schema is not yet readable here (see
ASSUMPTIONS.md). Everything else in ``ptcg_bot`` is written against our OWN
internal model (:mod:`ptcg_bot.state`), which already exists and is exercised by
:mod:`ptcg_bot.evaluate`. This adapter is the single translation layer:

    sim observation  --parse_observation-->  internal GameState
    internal Action  --encode_action------>  sim action

Once the sample_submission is downloaded, implement the two functions below
against ``cg/api.py`` and wire ``tools/verify_env.py`` to assert round-trip
fidelity and the rule-variant constants in :mod:`ptcg_bot.rules` (plan §W2).

Keeping this stub explicit (rather than absent) means imports resolve and the
heuristic/deckbuilder work can proceed in parallel against synthetic states.
"""

from __future__ import annotations

from typing import Any


class EngineNotWiredError(NotImplementedError):
    """Raised when adapter functions are called before the sim schema is known."""


def parse_observation(obs: Any) -> Any:
    """Translate a raw simulator observation into our internal ``GameState``.

    Args:
        obs: whatever the official engine hands the agent each decision point.

    Returns:
        A validated internal ``GameState`` (see :mod:`ptcg_bot.state`).

    Raises:
        EngineNotWiredError: until implemented against the real schema (W2).
    """
    raise EngineNotWiredError(
        "parse_observation is a stub: wire it to the official simulator schema "
        "in Week 2 (see plan §W2 and tools/verify_env.py)."
    )


def encode_action(action: Any) -> Any:
    """Translate an internal ``Action`` into the simulator's action format.

    Raises:
        EngineNotWiredError: until implemented against the real schema (W2).
    """
    raise EngineNotWiredError(
        "encode_action is a stub: wire it to the official simulator schema "
        "in Week 2 (see plan §W2 and tools/verify_env.py)."
    )
