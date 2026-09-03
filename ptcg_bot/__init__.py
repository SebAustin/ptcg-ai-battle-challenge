"""PTCG AI Battle Challenge — battle agent package.

Public surface stable enough to import today:
    from ptcg_bot.cards import load_pool, Card, CardPool
    from ptcg_bot import rules, config
    from ptcg_bot.state import GameState
    from ptcg_bot.evaluate import state_value
    from ptcg_bot.main import agent          # the engine entrypoint

The engine adapter is wired against the live simulator (fetch it with
``make engine``). Still to come (per the plan): legal, belief, search — the
lookahead layer that will replace ``main``'s v1 heuristic option policy.
"""

from . import config, engine_adapter, rules
from .cards import Card, CardPool, EnergyCost, Move, load_pool
from .evaluate import state_value
from .main import agent
from .state import GameState, PlayerState, PokemonInPlay

__all__ = [
    "config",
    "engine_adapter",
    "rules",
    "Card",
    "CardPool",
    "EnergyCost",
    "Move",
    "load_pool",
    "GameState",
    "PlayerState",
    "PokemonInPlay",
    "state_value",
    "agent",
]
