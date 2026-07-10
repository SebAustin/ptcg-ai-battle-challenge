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

v2 (gen4) appends 20 features after the original 41: evolution stage and
headroom (a next stage exists in the pool), KO prize liability (ex/megaEx —
multi-prize risk the v1 evaluator could not see), energy-tempo deficits
(turns-to-ready for active and best bench backup), retreat cost, and discard
composition (energies spent, Pokémon lost). All public-board metadata.
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
_MAX_DEFICIT = 3.0
_MAX_RETREAT = 4.0
_MAX_DISCARD_ENERGY = 15.0
_MAX_DISCARD_POKEMON = 10.0

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
    # v2 — evolution / prize-liability / energy-tempo / discard (metadata; 0 offline)
    "us_active_stage",
    "them_active_stage",
    "us_active_prize_risk",
    "them_active_prize_risk",
    "us_bench_prize_risk",
    "them_bench_prize_risk",
    "us_active_evolvable",
    "them_active_evolvable",
    "us_bench_evolvable",
    "them_bench_evolvable",
    "us_active_deficit",
    "them_active_deficit",
    "us_bench_ready_deficit",
    "them_bench_ready_deficit",
    "us_active_retreat",
    "them_active_retreat",
    "us_discard_energy",
    "them_discard_energy",
    "us_discard_pokemon",
    "them_discard_pokemon",
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

    stages = metadata.stage_info()
    evolvable = metadata.evolvable_ids()
    costs = metadata.attack_cost()
    has_meta = bool(costs)

    active_id = int(out["active_id"])
    stage, prize = stages.get(active_id, (0, 1 if active_id else 0))
    active_cost = costs.get(active_id)
    out.update(
        active_stage=float(stage) if active_id else 0.0,
        active_prize=float(prize) if active_id else 0.0,
        active_evolvable=1.0 if active_id in evolvable else 0.0,
        active_deficit=(
            max(0.0, active_cost - out["active_energy"])
            if active_cost is not None
            else 0.0
        ),
        active_retreat=float(metadata.retreat_cost().get(active_id, 0)),
    )

    bench = [p for p in (_get(player, "bench") or ()) if p is not None]
    bench_hp = bench_energy = bench_dmg = 0.0
    bench_charged = 0
    bench_prize = bench_evolvable = 0.0
    # "No ready backup attacker" reads as the max deficit; 0.0 offline (no cg).
    bench_ready_deficit = _MAX_DEFICIT if (has_meta and not bench) else 0.0
    for pokemon in bench:
        hp, _mx, energy_n, cid, dmg, charged = _pokemon_stats(pokemon)
        bench_hp += hp
        bench_energy += energy_n
        bench_dmg = max(bench_dmg, dmg)
        bench_charged += charged
        _stage, b_prize = stages.get(cid, (0, 1))
        bench_prize = max(bench_prize, float(b_prize))
        bench_evolvable += 1.0 if cid in evolvable else 0.0
        if has_meta:
            cost = costs.get(cid)
            deficit = max(0.0, cost - energy_n) if cost is not None else 0.0
            bench_ready_deficit = min(bench_ready_deficit, deficit)
    out.update(
        bench_n=float(len(bench)),
        bench_hp=bench_hp,
        bench_energy=bench_energy,
        bench_charged=float(bench_charged),
        bench_dmg=bench_dmg,
        bench_prize=bench_prize if has_meta else 0.0,
        bench_evolvable=bench_evolvable,
        bench_ready_deficit=bench_ready_deficit,
    )

    kinds = metadata.card_kind()
    discard_energy = discard_pokemon = 0.0
    for card in _get(player, "discard") or ():
        try:
            kind = kinds.get(int(_get(card, "id") or 0), -1)
        except Exception:
            kind = -1
        if kind in (5, 6):  # BASIC_ENERGY / SPECIAL_ENERGY
            discard_energy += 1.0
        elif kind == 0:  # POKEMON
            discard_pokemon += 1.0
    out["discard_energy"] = discard_energy
    out["discard_pokemon"] = discard_pokemon

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
            # v2 (order matches the FEATURE_NAMES v2 block)
            _clamp(us["active_stage"] / 2.0),
            _clamp(them["active_stage"] / 2.0),
            _clamp((us["active_prize"] - 1.0) / 2.0),
            _clamp((them["active_prize"] - 1.0) / 2.0),
            _clamp((us["bench_prize"] - 1.0) / 2.0),
            _clamp((them["bench_prize"] - 1.0) / 2.0),
            us["active_evolvable"],
            them["active_evolvable"],
            _clamp(us["bench_evolvable"] / _MAX_BENCH),
            _clamp(them["bench_evolvable"] / _MAX_BENCH),
            _clamp(us["active_deficit"] / _MAX_DEFICIT),
            _clamp(them["active_deficit"] / _MAX_DEFICIT),
            _clamp(us["bench_ready_deficit"] / _MAX_DEFICIT),
            _clamp(them["bench_ready_deficit"] / _MAX_DEFICIT),
            _clamp(us["active_retreat"] / _MAX_RETREAT),
            _clamp(them["active_retreat"] / _MAX_RETREAT),
            _clamp(us["discard_energy"] / _MAX_DISCARD_ENERGY),
            _clamp(them["discard_energy"] / _MAX_DISCARD_ENERGY),
            _clamp(us["discard_pokemon"] / _MAX_DISCARD_POKEMON),
            _clamp(them["discard_pokemon"] / _MAX_DISCARD_POKEMON),
        ]
    except Exception:
        return [0.0] * FEATURE_COUNT
