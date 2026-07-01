"""Optional runtime attack metadata from the engine.

An observation gives an ATTACK option's ``attackId`` but not its damage. The
engine's ``cg.api.all_attack()`` provides ``{attackId -> damage}``. We load it
**lazily and gracefully**: if the ``cg`` package is importable (it is inside the
submission bundle and whenever the engine is on the path) we build the table
once; otherwise we return an empty table and callers fall back to structural
heuristics.

Importing ``cg`` is the sanctioned engine interface (the competition's own
sample agent imports it) and performs no network or subprocess I/O, so the
no-egress rule (SECURITY.md) still holds.
"""

from __future__ import annotations

from functools import cache


@cache
def attack_damage() -> dict[int, int]:
    """Map ``attackId -> base damage`` from the engine, or ``{}`` if unavailable."""
    try:
        from cg.api import all_attack  # type: ignore[import-not-found]
    except Exception:
        return {}

    table: dict[int, int] = {}
    try:
        for attack in all_attack():
            attack_id = getattr(attack, "attackId", None)
            if attack_id is not None:
                table[int(attack_id)] = int(getattr(attack, "damage", 0) or 0)
    except Exception:
        return {}
    return table
