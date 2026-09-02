"""Behavior-cloned option policy — "play like the 1200+ agents".

``scores(observation)`` returns one score per legal option of the current
selection using the imitation model in :mod:`ptcg_bot.policy_weights`
(trained by ``tools/train_policy.py`` on top-bracket replay decisions), or
``None`` whenever the policy must not be trusted: weights missing, feature
contract skew, unsupported prompt type, too many options, or any exception.
Callers fall back to the v7 heuristics on ``None``.

Late-fusion contract (see policy_weights): the state tower runs ONCE per
decision (``prepare``), then each option costs one dot product (``score``) —
cheap enough for search rollouts. Pure stdlib; works on observation dicts and
engine dataclasses alike.
"""

from __future__ import annotations

import importlib
from typing import Any

from . import config as cfg
from . import features, option_features
from .features import _get, _num

_SUPPORTED_SELECT_TYPES = frozenset({0, 1})  # MAIN, CARD


def scores(observation: Any) -> list[float] | None:
    """Per-option imitation scores for the current selection, or ``None``."""
    try:
        # Optional generated module (tools/train_policy.py); absent -> decline.
        policy_weights = importlib.import_module("ptcg_bot.policy_weights")
    except Exception:
        return None
    try:
        select = _get(observation, "select")
        if select is None:
            return None
        if _get(select, "type") not in _SUPPORTED_SELECT_TYPES:
            return None
        options = _get(select, "option") or []
        if len(options) < 2 or len(options) > cfg.BC_MAX_OPTIONS:
            return None
        current = _get(observation, "current")
        me = int(_num(_get(current, "yourIndex")))
        state = features.extract(observation, me) + option_features.extract_decision(
            observation, select
        )
        if len(state) != policy_weights.STATE_COUNT:
            return None
        rows = option_features.extract_options(observation, select)
        if not rows or len(rows[0]) != policy_weights.OPTION_COUNT:
            return None
        ctx = policy_weights.prepare(state)
        return [policy_weights.score(ctx, row) for row in rows]
    except Exception:
        return None


def choose(observation: Any) -> list[int] | None:
    """Legal selection by imitation: argmax (maxCount 1) or the top ``maxCount``
    indices in ascending order; ``None`` when the policy declines."""
    values = scores(observation)
    if values is None:
        return None
    try:
        select = _get(observation, "select")
        count = min(int(_num(_get(select, "maxCount")) or 1), len(values))
        if count <= 0:
            return None
        order = sorted(range(len(values)), key=lambda i: -values[i])
        return sorted(order[:count])
    except Exception:
        return None
