"""Translate the live engine's observation <-> our internal model.

The engine hands the agent a plain ``dict`` (see the engine's
``cg/api.py``: ``agent(obs_dict) -> list[int]``). We parse that dict **directly**
— deliberately with NO dependency on the ``cg`` package — so this module stays
pure-stdlib and importable even without the (rules-gated, gitignored) engine
present, and so the bundled agent has no import egress (SECURITY.md).

Observation shape we rely on (from cg/api.py)::

    obs["select"]                      # None only at the initial deck request
    obs["current"]["yourIndex"]        # 0/1 — which player is us
    obs["current"]["turn"]
    obs["current"]["players"][i]       # {active, bench, handCount, deckCount, prize, ...}
    pokemon = {"id", "hp", "maxHp", "energies": [engineEnergyTypeInt, ...], ...}

Actions are just indices into ``obs["select"]["option"]``, so encoding is a
validated pass-through, not a translation.
"""

from __future__ import annotations

from typing import Any

from .cards import CardPool
from .rules import EnergyType
from .state import GameState, PlayerState, PokemonInPlay

# Engine EnergyType enum id -> our EnergyType. RAINBOW (10, "any") and
# TEAM_ROCKET (11, Psychic+Darkness) are treated as Colorless for affordability
# (safe: they satisfy Colorless costs, never over-claim a typed requirement).
_ENERGY_BY_ENGINE_ID: dict[int, EnergyType] = {
    0: EnergyType.COLORLESS,
    1: EnergyType.GRASS,
    2: EnergyType.FIRE,
    3: EnergyType.WATER,
    4: EnergyType.LIGHTNING,
    5: EnergyType.PSYCHIC,
    6: EnergyType.FIGHTING,
    7: EnergyType.DARKNESS,
    8: EnergyType.METAL,
    9: EnergyType.DRAGON,
    10: EnergyType.COLORLESS,
    11: EnergyType.COLORLESS,
}


class EngineNotWiredError(RuntimeError):
    """Raised when an observation has no game state to parse (e.g. deck request)."""


def is_deck_request(obs: Any) -> bool:
    """True on the initial observation, where the engine wants the 60-card deck."""
    return isinstance(obs, dict) and obs.get("select") is None


def _energy_types(units: list[int] | None) -> tuple[EnergyType, ...]:
    return tuple(
        _ENERGY_BY_ENGINE_ID.get(int(u), EnergyType.COLORLESS) for u in (units or ())
    )


def _pokemon(entry: dict[str, Any] | None, pool: CardPool) -> PokemonInPlay | None:
    if not entry:
        return None
    card = pool.by_id(int(entry["id"]))
    if card is None:
        return None
    max_hp = int(entry.get("maxHp") or entry.get("hp") or 0)
    current_hp = int(entry.get("hp") or 0)
    damage = max(0, max_hp - current_hp)
    return PokemonInPlay(
        card=card, damage=damage, attached=_energy_types(entry.get("energies"))
    )


def _player(player: dict[str, Any], pool: CardPool) -> PlayerState:
    active_slot = player.get("active") or []
    active = _pokemon(active_slot[0], pool) if active_slot else None
    bench = tuple(
        p for p in (_pokemon(b, pool) for b in player.get("bench") or ()) if p
    )
    return PlayerState(
        active=active,
        bench=bench,
        hand_size=int(player.get("handCount") or 0),
        deck_size=int(player.get("deckCount") or 0),
        prizes_remaining=len(player.get("prize") or ()),
    )


def parse_observation(obs: Any, pool: CardPool) -> GameState:
    """Translate a raw engine observation dict into our internal ``GameState``.

    Args:
        obs: the ``obs_dict`` the engine passes the agent.
        pool: card metadata source (offline: ``load_pool()``; at runtime, a pool
            built from the engine's ``all_card_data`` — see the module docs).

    Raises:
        EngineNotWiredError: if ``obs`` carries no game state (e.g. the initial
            deck-selection request; use :func:`is_deck_request` first).
    """
    if not isinstance(obs, dict):
        raise EngineNotWiredError(f"expected a dict observation, got {type(obs)!r}")
    current = obs.get("current")
    if not current:
        raise EngineNotWiredError(
            "observation has no game state (initial deck request?); "
            "check is_deck_request(obs) first"
        )
    players = current.get("players") or []
    if len(players) < 2:
        raise EngineNotWiredError("observation is missing both players")
    ours = int(current.get("yourIndex") or 0)
    return GameState(
        us=_player(players[ours], pool),
        them=_player(players[1 - ours], pool),
        turn=int(current.get("turn") or 1),
    )


def encode_action(selection: list[int]) -> list[int]:
    """Validate a chosen set of option indices for the engine's ``Select`` call.

    The engine expects a ``list[int]`` of distinct, non-negative indices into
    ``obs["select"]["option"]``. This is the whole "action encoding".
    """
    if not isinstance(selection, list) or not all(
        isinstance(i, int) and i >= 0 for i in selection
    ):
        raise ValueError("selection must be a list of non-negative option indices")
    if len(set(selection)) != len(selection):
        raise ValueError("selection must not contain duplicate indices")
    return selection
