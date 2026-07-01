"""PTCG AI Battle Challenge — battle agent package.

Public surface stable enough to import today:
    from ptcg_bot.cards import load_pool, Card, CardPool
    from ptcg_bot import rules, config
    from ptcg_bot.state import GameState
    from ptcg_bot.evaluate import state_value

Modules still to come (per the plan): sim, effects, legal, belief, search, main.
Those and ``engine_adapter`` are gated on the simulator schema (accept the
``pokemon-tcg-ai-battle`` competition rules to download it — see ASSUMPTIONS.md).
The agent entrypoint will be ``ptcg_bot.main.agent``.
"""

from . import config, rules
from .cards import Card, CardPool, EnergyCost, Move, load_pool
from .evaluate import state_value
from .state import GameState, PlayerState, PokemonInPlay

__all__ = [
    "config",
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
]
