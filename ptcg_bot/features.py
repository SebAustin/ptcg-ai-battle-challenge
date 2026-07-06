"""State features for the learned win-probability evaluator.

``extract(observation, me)`` returns ``FEATURE_COUNT`` floats, each normalized
to roughly [0, 1] (leads in [-1, 1]), computed from player ``me``'s fixed
perspective. It is the SINGLE source of features for both training
(``tools/selfplay.py``) and inference (``search.evaluate_observation``), so the
two can never skew.

Works on the engine's ``Observation`` dataclasses (attribute access) AND on raw
observation dicts (key access) — rollout leaves see dataclasses, self-play
recording sees dicts. Uses only information PUBLIC to both sides (no hand
contents: the opponent's hand is absent at their decision points). Metadata
features (damage/cost/type) come from :mod:`ptcg_bot.metadata` and are 0.0
offline. Never raises; anything unresolvable contributes 0.0.
"""

from __future__ import annotations

from typing import Any

from . import metadata

# Normalizers (rough scale caps; clamping keeps everything bounded).
_MAX_HP = 340.0
_MAX_HAND = 10.0
_MAX_DECK = 60.0
_MAX_ENERGY = 5.0
_MAX_BENCH = 5.0
_MAX_DMG = 300.0
_MAX_TURN = 50.0

FEATURE_NAMES: tuple[str, ...] = (
    # prizes
    "us_prizes",
    "them_prizes",
    "prize_lead",
    # counts
    "us_hand",
    "them_hand",
    "us_deck",
    "them_deck",
    "us_discard",
    "them_discard",
    # active
    "us_active_present",
    "them_active_present",
    "us_active_hp",
    "them_active_hp",
    "us_active_hp_frac",
    "them_active_hp_frac",
    "us_active_energy",
    "them_active_energy",
    # active power (metadata)
    "us_active_dmg",
    "them_active_dmg",
    "us_active_charged",
    "them_active_charged",
    # threat (metadata)
    "us_can_ko",
    "them_can_ko",
    "us_hits_weakness",
    "them_hits_weakness",
    # bench
    "us_bench_n",
    "them_bench_n",
    "us_bench_hp",
    "them_bench_hp",
    "us_bench_energy",
    "them_bench_energy",
    "us_bench_charged",
    "them_bench_charged",
    "us_bench_dmg",
    "them_bench_dmg",
    # board / status / tempo
    "us_in_play",
    "them_in_play",
    "us_status",
    "them_status",
    "turn",
    "our_selection",
)
FEATURE_COUNT = len(FEATURE_NAMES)

_STATUS_FLAGS = ("poisoned", "burned", "asleep", "paralyzed", "confused")


def _get(obj: Any, name: str) -> Any:
    """Field access that works for dicts and dataclasses alike."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _num(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def _pokemon_stats(pokemon: Any) -> tuple[float, float, float, int, float, int]:
    """(hp, max_hp, energy_n, card_id, best_dmg, charged) for one board Pokémon."""
    hp = _num(_get(pokemon, "hp"))
    max_hp = _num(_get(pokemon, "maxHp")) or hp
    energy_n = float(len(_get(pokemon, "energies") or ()))
    try:
        card_id = int(_get(pokemon, "id") or 0)
    except Exception:
        card_id = 0
    dmg, _hp_meta = metadata.card_power().get(card_id, (0, 0))
    cost = metadata.attack_cost().get(card_id)
    charged = 1 if (cost is not None and energy_n >= cost) else 0
    return hp, max_hp, energy_n, card_id, float(dmg), charged


def _side(player: Any) -> dict[str, float]:
    """Aggregate one player's public state into named raw values."""
    out: dict[str, float] = {}
    out["prizes"] = float(len(_get(player, "prize") or ()))
    out["hand"] = _num(_get(player, "handCount"))
    out["deck"] = _num(_get(player, "deckCount"))
    out["discard"] = float(len(_get(player, "discard") or ()))

    active_list = _get(player, "active") or []
    active = active_list[0] if active_list else None
    if active is not None:
        hp, max_hp, energy_n, card_id, dmg, charged = _pokemon_stats(active)
        out.update(
            active_present=1.0,
            active_hp=hp,
            active_hp_frac=(hp / max_hp) if max_hp > 0 else 0.0,
            active_energy=energy_n,
            active_dmg=dmg,
            active_charged=float(charged),
            active_id=float(card_id),
        )
    else:
        out.update(
            active_present=0.0,
            active_hp=0.0,
            active_hp_frac=0.0,
            active_energy=0.0,
            active_dmg=0.0,
            active_charged=0.0,
            active_id=0.0,
        )

    bench = [p for p in (_get(player, "bench") or ()) if p is not None]
    bench_hp = bench_energy = bench_dmg = 0.0
    bench_charged = 0
    for pokemon in bench:
        hp, _mx, energy_n, _cid, dmg, charged = _pokemon_stats(pokemon)
        bench_hp += hp
        bench_energy += energy_n
        bench_dmg = max(bench_dmg, dmg)
        bench_charged += charged
    out.update(
        bench_n=float(len(bench)),
        bench_hp=bench_hp,
        bench_energy=bench_energy,
        bench_charged=float(bench_charged),
        bench_dmg=bench_dmg,
    )
    out["status"] = sum(1.0 for f in _STATUS_FLAGS if _get(player, f))
    return out


def _threats(
    us: dict[str, float], them: dict[str, float]
) -> tuple[float, float, float, float]:
    """(us_can_ko, them_can_ko, us_hits_weakness, them_hits_weakness)."""
    types = metadata.type_info()
    us_type, _us_weak = types.get(int(us["active_id"]), (-1, -1))
    them_type, them_weak = types.get(int(them["active_id"]), (-1, -1))
    _, us_weak = types.get(int(us["active_id"]), (-1, -1))

    us_hits_weak = 1.0 if (us_type >= 0 and us_type == them_weak) else 0.0
    them_hits_weak = 1.0 if (them_type >= 0 and them_type == us_weak) else 0.0
    us_dmg = us["active_dmg"] * (2.0 if us_hits_weak else 1.0)
    them_dmg = them["active_dmg"] * (2.0 if them_hits_weak else 1.0)
    us_can_ko = 1.0 if (them["active_hp"] > 0 and us_dmg >= them["active_hp"]) else 0.0
    them_can_ko = 1.0 if (us["active_hp"] > 0 and them_dmg >= us["active_hp"]) else 0.0
    return us_can_ko, them_can_ko, us_hits_weak, them_hits_weak


def extract(observation: Any, me: int) -> list[float]:
    """``FEATURE_COUNT`` bounded floats from player ``me``'s fixed perspective."""
    try:
        current = _get(observation, "current")
        players = _get(current, "players") or []
        if current is None or len(players) < 2:
            return [0.0] * FEATURE_COUNT
        us = _side(players[me])
        them = _side(players[1 - me])
        us_ko, them_ko, us_weak, them_weak = _threats(us, them)

        turn = _num(_get(current, "turn"))
        ours = 1.0 if int(_num(_get(current, "yourIndex"))) == me else 0.0

        return [
            _clamp(us["prizes"] / 6.0),
            _clamp(them["prizes"] / 6.0),
            _clamp((them["prizes"] - us["prizes"]) / 6.0, -1.0, 1.0),
            _clamp(us["hand"] / _MAX_HAND),
            _clamp(them["hand"] / _MAX_HAND),
            _clamp(us["deck"] / _MAX_DECK),
            _clamp(them["deck"] / _MAX_DECK),
            _clamp(us["discard"] / _MAX_DECK),
            _clamp(them["discard"] / _MAX_DECK),
            us["active_present"],
            them["active_present"],
            _clamp(us["active_hp"] / _MAX_HP),
            _clamp(them["active_hp"] / _MAX_HP),
            _clamp(us["active_hp_frac"]),
            _clamp(them["active_hp_frac"]),
            _clamp(us["active_energy"] / _MAX_ENERGY),
            _clamp(them["active_energy"] / _MAX_ENERGY),
            _clamp(us["active_dmg"] / _MAX_DMG),
            _clamp(them["active_dmg"] / _MAX_DMG),
            us["active_charged"],
            them["active_charged"],
            us_ko,
            them_ko,
            us_weak,
            them_weak,
            _clamp(us["bench_n"] / _MAX_BENCH),
            _clamp(them["bench_n"] / _MAX_BENCH),
            _clamp(us["bench_hp"] / 1000.0),
            _clamp(them["bench_hp"] / 1000.0),
            _clamp(us["bench_energy"] / 10.0),
            _clamp(them["bench_energy"] / 10.0),
            _clamp(us["bench_charged"] / _MAX_BENCH),
            _clamp(them["bench_charged"] / _MAX_BENCH),
            _clamp(us["bench_dmg"] / _MAX_DMG),
            _clamp(them["bench_dmg"] / _MAX_DMG),
            _clamp((us["active_present"] + us["bench_n"]) / 6.0),
            _clamp((them["active_present"] + them["bench_n"]) / 6.0),
            _clamp(us["status"] / 5.0),
            _clamp(them["status"] / 5.0),
            _clamp(turn / _MAX_TURN),
            ours,
        ]
    except Exception:
        return [0.0] * FEATURE_COUNT
