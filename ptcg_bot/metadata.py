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


@cache
def _all_cards() -> tuple:
    """The engine's full CardData list, or an empty tuple if cg is unavailable."""
    try:
        from cg.api import all_card_data
    except Exception:
        return ()
    try:
        return tuple(all_card_data())
    except Exception:
        return ()


@cache
def basic_pokemon_ids() -> tuple[int, ...]:
    """Card IDs of Basic Pokémon (needed to seed a legal opponent deck for search)."""
    return tuple(int(c.cardId) for c in _all_cards() if getattr(c, "basic", False))


@cache
def card_power() -> dict[int, tuple[int, int]]:
    """Map ``cardId -> (best attack damage, hp)`` for Pokémon, or ``{}`` offline.

    The cheap "how strong is this card" signal used to rank card selections
    (setup active, promotion, search targets, discards).
    """
    damage = attack_damage()
    table: dict[int, tuple[int, int]] = {}
    try:
        for c in _all_cards():
            attacks = getattr(c, "attacks", None) or ()
            best = max((damage.get(int(a), 0) for a in attacks), default=0)
            table[int(c.cardId)] = (best, int(getattr(c, "hp", 0) or 0))
    except Exception:
        return {}
    return table


@cache
def type_info() -> dict[int, tuple[int, int]]:
    """Map ``cardId -> (energyType id, weakness id or -1)`` for matchup features.

    ``{}`` offline (no ``cg``); callers treat missing entries as "unknown".
    """
    table: dict[int, tuple[int, int]] = {}
    try:
        for c in _all_cards():
            etype = getattr(c, "energyType", None)
            weak = getattr(c, "weakness", None)
            table[int(c.cardId)] = (
                int(etype) if etype is not None else -1,
                int(weak) if weak is not None else -1,
            )
    except Exception:
        return {}
    return table


@cache
def attack_cost() -> dict[int, int]:
    """Map ``cardId -> energy units its best (highest-damage) attack needs``.

    Used to decide when a Pokémon is "charged" (attach elsewhere) and whether a
    bench Pokémon is a ready attacker (retreat logic). ``{}`` offline.
    """
    try:
        from cg.api import all_attack
    except Exception:
        return {}
    try:
        per_attack = {
            int(a.attackId): (
                int(getattr(a, "damage", 0) or 0),
                len(getattr(a, "energies", None) or ()),
            )
            for a in all_attack()
        }
        table: dict[int, int] = {}
        for c in _all_cards():
            best_damage, best_cost = 0, 0
            for aid in getattr(c, "attacks", None) or ():
                dmg, cost = per_attack.get(int(aid), (0, 0))
                if dmg > best_damage:
                    best_damage, best_cost = dmg, cost
            table[int(c.cardId)] = best_cost
        return table
    except Exception:
        return {}


@cache
def valid_card_ids() -> tuple[int, ...]:
    """All valid card IDs (used to fill determinized hidden-info arrays)."""
    return tuple(int(c.cardId) for c in _all_cards())


@cache
def stage_info() -> dict[int, tuple[int, int]]:
    """Map ``cardId -> (evolution stage 0/1/2, KO prize value 1/2/3)`` for Pokémon.

    Prize value is what the OPPONENT takes when this Pokémon is Knocked Out:
    3 for Mega Evolution ex, 2 for ex, 1 otherwise. ``{}`` offline.
    """
    table: dict[int, tuple[int, int]] = {}
    try:
        for c in _all_cards():
            kind = getattr(c, "cardType", None)
            if kind is None or int(kind) != 0:  # Pokémon only
                continue
            stage = (
                2
                if getattr(c, "stage2", False)
                else (1 if getattr(c, "stage1", False) else 0)
            )
            prize = (
                3
                if getattr(c, "megaEx", False)
                else (2 if getattr(c, "ex", False) else 1)
            )
            table[int(c.cardId)] = (stage, prize)
    except Exception:
        return {}
    return table


@cache
def evolvable_ids() -> frozenset[int]:
    """Card IDs for which a NEXT evolution stage exists in the card pool.

    Built from ``evolvesFrom`` names: a Pokémon is "evolvable" if any card in
    the pool lists its name as pre-evolution. Empty offline.
    """
    try:
        cards = _all_cards()
        next_stage_sources = {getattr(c, "evolvesFrom", None) for c in cards} - {None}
        return frozenset(
            int(c.cardId)
            for c in cards
            if getattr(c, "name", None) in next_stage_sources
        )
    except Exception:
        return frozenset()


@cache
def retreat_cost() -> dict[int, int]:
    """Map ``cardId -> retreat cost in energies`` (mobility). ``{}`` offline."""
    table: dict[int, int] = {}
    try:
        for c in _all_cards():
            table[int(c.cardId)] = int(getattr(c, "retreatCost", 0) or 0)
    except Exception:
        return {}
    return table


@cache
def card_kind() -> dict[int, int]:
    """Map ``cardId -> CardType int`` (0 Pokémon ... 5/6 Energy). ``{}`` offline."""
    table: dict[int, int] = {}
    try:
        for c in _all_cards():
            kind = getattr(c, "cardType", None)
            table[int(c.cardId)] = int(kind) if kind is not None else -1
    except Exception:
        return {}
    return table
