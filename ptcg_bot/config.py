"""All tunable WEIGHTS for the PTCG agent in one place.

This is the surface ``tools/tune.py`` mutates (coordinate descent). It imports
only ``os`` so it stays dependency-free and safe to freeze into the bundled
single-file submission. RULE CONSTANTS do NOT live here — those are in
:mod:`ptcg_bot.rules` and are verified against the engine, never tuned.

Conventions:
- Weights are dimensionless; only relative magnitudes matter.
- Each is overridable via env var ``PTCG_<NAME>`` so the tuner can evaluate a
  candidate in a fresh subprocess (the override propagates to workers).
"""

import os


def _f(name: str, default: float) -> float:
    """Tunable float overridable via env var ``PTCG_<NAME>``."""
    try:
        return float(os.environ.get("PTCG_" + name, default))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    """Tunable int overridable via env var ``PTCG_<NAME>``."""
    try:
        return int(os.environ.get("PTCG_" + name, default))
    except (TypeError, ValueError):
        return default


# --- Heuristic state-value weights (evaluate.state_value) --------------------
# The position is scored from the acting player's perspective. Prize tempo
# dominates: taking prizes / denying them is how you win, so it carries the
# largest coefficient. The rest shape *how* we get there.
W_PRIZE_LEAD = _f("W_PRIZE_LEAD", 100.0)  # per net prize we are ahead
W_OWN_ACTIVE_HP = _f("W_OWN_ACTIVE_HP", 0.20)  # remaining HP of our active
W_OPP_ACTIVE_HP = _f("W_OPP_ACTIVE_HP", 0.15)  # (negated) opponent active HP
W_OWN_BENCH_HP = _f("W_OWN_BENCH_HP", 0.05)  # bench durability
W_BOARD_POKEMON = _f("W_BOARD_POKEMON", 6.0)  # per Pokémon in play (board state)
W_ENERGY_ON_BOARD = _f("W_ENERGY_ON_BOARD", 4.0)  # attached energy (tempo)
W_ATTACK_READY = _f("W_ATTACK_READY", 25.0)  # our active can attack this turn
W_TYPE_ADVANTAGE = _f("W_TYPE_ADVANTAGE", 18.0)  # weakness matchup vs opp active
W_HAND_SIZE = _f("W_HAND_SIZE", 2.0)  # card advantage (mild)
W_BENCH_BASICS = _f("W_BENCH_BASICS", 3.0)  # evolution / recovery potential

# Terminal sentinels — the eval clamps to these for won/lost positions so search
# always prefers a real win over any heuristic accumulation.
VALUE_WIN = _f("VALUE_WIN", 1.0e6)
VALUE_LOSS = _f("VALUE_LOSS", -1.0e6)

# --- Search parameters (search.py — determinized IS-MCTS) -------------------
SEARCH_WORLDS = _i("SEARCH_WORLDS", 12)  # K sampled hidden-state worlds
SEARCH_ITERS_PER_WORLD = _i("SEARCH_ITERS_PER_WORLD", 200)
SEARCH_ROLLOUT_DEPTH = _i("SEARCH_ROLLOUT_DEPTH", 8)
UCT_C = _f("UCT_C", 1.4)  # exploration constant

# --- Time budget (main.py guard) --------------------------------------------
# Hard per-decision deadline; below the engine's turn timeout with cushion. The
# heuristic 1-ply policy is the never-crash fallback if search runs long.
TURN_DEADLINE_S = _f("TURN_DEADLINE_S", 2.5)
