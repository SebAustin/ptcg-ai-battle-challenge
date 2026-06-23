"""PTCG AI Battle Challenge — battle agent package.

Public surface stable enough to import today:
    from ptcg_bot.cards import load_pool, Card, CardPool
    from ptcg_bot import rules, config

Modules still to come (per the plan): state, sim, effects, legal, belief,
search, evaluate, main. The agent entrypoint will be ``ptcg_bot.main.agent``.
"""

from . import config, rules
from .cards import Card, CardPool, EnergyCost, Move, load_pool

__all__ = ["config", "rules", "Card", "CardPool", "EnergyCost", "Move", "load_pool"]
