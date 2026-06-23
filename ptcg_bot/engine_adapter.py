"""Isolation boundary for the (not-yet-local) official simulator.

THE ONLY MODULE THAT CHANGES WHEN THE SIMULATOR API LANDS.

We do not yet have the competition's battle engine on this machine, and its
observation/action schema is unknown. Everything else in ``ptcg_bot`` is written
against our OWN internal model (:mod:`ptcg_bot.state`, once it exists). This
adapter is the single translation layer:

    sim observation  --parse_observation-->  internal GameState
    internal Action  --encode_action------>  sim action

When the engine arrives (Week 2 of the plan), implement the two functions below
against its real schema and wire ``tools/verify_env.py`` to assert round-trip
fidelity and the rule-variant constants in :mod:`ptcg_bot.rules`.

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
