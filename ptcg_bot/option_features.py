"""Decision- and option-level features for the behavior-cloned policy.

The imitation policy (``ptcg_bot.policy``) scores each legal option of a
selection prompt. Its inputs are:

* the 41 public-board features (``features.extract`` — frozen contract, shared
  with the value evaluator) **plus** ``DECISION_FEATURE_NAMES`` (this module):
  what kind of prompt this is and what the option set contains;
* per option, ``OPTION_FEATURE_NAMES``: option type, the ATTRIBUTES of the card
  behind it (never its id — top-bracket decks are not ours, so only attributes
  transfer), the state of the Pokémon it targets, attack damage/lethality, and
  a few relative flags.

Works on raw observation dicts (replay data, live agent) and on the engine's
dataclasses (search rollouts). Metadata comes from :mod:`ptcg_bot.metadata`
and is 0.0 offline. Never raises; anything unresolvable contributes zeros.
"""

from __future__ import annotations

from typing import Any

from . import metadata
from .features import _clamp, _get, _num

# OptionType / AreaType / SelectType ids (engine cg/api.py).
_PLAY, _ATTACH, _EVOLVE, _ABILITY, _RETREAT, _ATTACK, _END = 7, 8, 9, 10, 12, 13, 14
_CARD_OPTION = 3
_AREA_DECK, _AREA_HAND, _AREA_DISCARD, _AREA_ACTIVE, _AREA_BENCH = 1, 2, 3, 4, 5
_SELECT_MAIN, _SELECT_CARD, _SELECT_YESNO = 0, 1, 9
_CTX_STRONG = frozenset({1, 2, 3, 4, 6, 7})
_CTX_WEAK = frozenset({8, 9, 10, 11})

DECISION_FEATURE_NAMES: tuple[str, ...] = (
    "sel_main",
    "sel_card",
    "sel_yesno",
    "sel_other",
    "ctx_strong",
    "ctx_weak",
    "ctx_other",
    "max_count",
    "n_options",
    "any_attack",
    "any_lethal",
    "any_play",
    "any_attach",
    "any_evolve",
    "any_ability",
    "turn",
)
DECISION_COUNT = len(DECISION_FEATURE_NAMES)

OPTION_FEATURE_NAMES: tuple[str, ...] = (
    # option type
    "t_play",
    "t_attach",
    "t_evolve",
    "t_ability",
    "t_retreat",
    "t_attack",
    "t_end",
    "t_card",
    "t_other",
    # resolved card attributes (never the id)
    "c_pokemon",
    "c_item",
    "c_tool",
    "c_supporter",
    "c_stadium",
    "c_energy",
    "c_hp",
    "c_dmg",
    "c_stage",
    "c_ex",
    "c_mega",
    "c_retreat",
    "c_cost",
    "c_trainer_value",
    "c_evolvable",
    "c_type_matches_active",
    # target Pokémon (inPlayArea/inPlayIndex)
    "g_present",
    "g_active",
    "g_bench",
    "g_hp_frac",
    "g_energy",
    "g_charged",
    "g_deficit",
    "g_dmg",
    # attack
    "a_dmg",
    "a_lethal",
    "a_ratio",
    "a_cost",
    "a_is_max",
    # relative
    "r_index",
    "r_first_of_type",
    "r_strongest_card",
)
OPTION_COUNT = len(OPTION_FEATURE_NAMES)

# Position-dependent features excluded when testing two options for identity
# (e.g. four copies of the same Energy card in different hand slots ARE the
# same choice content-wise; only their index/first-of-type flags differ).
_POSITIONAL = frozenset({"r_index", "r_first_of_type"})
DEDUP_MASK: tuple[bool, ...] = tuple(n not in _POSITIONAL for n in OPTION_FEATURE_NAMES)

_TYPE_SLOT = {
    _PLAY: 0,
    _ATTACH: 1,
    _EVOLVE: 2,
    _ABILITY: 3,
    _RETREAT: 4,
    _ATTACK: 5,
    _END: 6,
    _CARD_OPTION: 7,
}
_KIND_SLOT = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5}


def _players(observation: Any) -> tuple[Any, int]:
    current = _get(observation, "current")
    players = _get(current, "players") or []
    me = int(_num(_get(current, "yourIndex")))
    return players, me


def _opp_active(observation: Any) -> Any:
    try:
        players, me = _players(observation)
        active = _get(players[1 - me], "active") or []
        return active[0] if active else None
    except Exception:
        return None


def resolve_card(observation: Any, select: Any, option: Any) -> int:
    """Card id behind ``option`` (hand/deck/discard/board via area+index), 0 if unknown."""
    try:
        area = _get(option, "area")
        index = int(_num(_get(option, "index")))
        players, me = _players(observation)
        owner = _get(option, "playerIndex")
        player = players[owner if isinstance(owner, int) and owner in (0, 1) else me]
        if area == _AREA_DECK:
            source = _get(select, "deck") or []
        elif area == _AREA_HAND:
            source = _get(player, "hand") or []
        elif area == _AREA_DISCARD:
            source = _get(player, "discard") or []
        elif area == _AREA_ACTIVE:
            source = _get(player, "active") or []
        elif area == _AREA_BENCH:
            source = _get(player, "bench") or []
        else:
            return 0
        card = source[index]
        return int(_num(_get(card, "id")))
    except Exception:
        return 0


def resolve_target(observation: Any, option: Any) -> tuple[Any, int]:
    """(Pokémon dict/obj, area) targeted via inPlayArea/inPlayIndex; (None, 0) if none."""
    try:
        area = _get(option, "inPlayArea")
        if area not in (_AREA_ACTIVE, _AREA_BENCH):
            return None, 0
        players, me = _players(observation)
        zone = "active" if area == _AREA_ACTIVE else "bench"
        pokemon = (_get(players[me], zone) or [])[
            int(_num(_get(option, "inPlayIndex")))
        ]
        return pokemon, int(area)
    except Exception:
        return None, 0


def _attack_damage(option: Any) -> tuple[int, int]:
    attack_id = _get(option, "attackId")
    if not isinstance(attack_id, int):
        return 0, 0
    return metadata.attack_info().get(attack_id, (0, 0))


def extract_decision(observation: Any, select: Any) -> list[float]:
    """``DECISION_COUNT`` bounded floats describing the prompt and option set."""
    if select is None:
        return [0.0] * DECISION_COUNT
    try:
        options = _get(select, "option") or []
        sel_type = _get(select, "type")
        context = _get(select, "context")
        types = [_get(o, "type") for o in options]
        opp = _opp_active(observation)
        opp_hp = _num(_get(opp, "hp")) if opp is not None else 0.0
        any_lethal = 0.0
        if opp_hp > 0:
            for o in options:
                if _get(o, "type") == _ATTACK and _attack_damage(o)[0] >= opp_hp:
                    any_lethal = 1.0
                    break
        turn = _num(_get(_get(observation, "current"), "turn"))
        return [
            1.0 if sel_type == _SELECT_MAIN else 0.0,
            1.0 if sel_type == _SELECT_CARD else 0.0,
            1.0 if sel_type == _SELECT_YESNO else 0.0,
            1.0 if sel_type not in (_SELECT_MAIN, _SELECT_CARD, _SELECT_YESNO) else 0.0,
            1.0 if context in _CTX_STRONG else 0.0,
            1.0 if context in _CTX_WEAK else 0.0,
            1.0 if (context not in _CTX_STRONG and context not in _CTX_WEAK) else 0.0,
            _clamp(_num(_get(select, "maxCount")) / 10.0),
            _clamp(len(options) / 20.0),
            1.0 if _ATTACK in types else 0.0,
            any_lethal,
            1.0 if _PLAY in types else 0.0,
            1.0 if _ATTACH in types else 0.0,
            1.0 if _EVOLVE in types else 0.0,
            1.0 if _ABILITY in types else 0.0,
            _clamp(turn / 50.0),
        ]
    except Exception:
        return [0.0] * DECISION_COUNT


def extract_options(observation: Any, select: Any) -> list[list[float]]:
    """One ``OPTION_COUNT`` vector per option of ``select`` (zeros when unresolvable)."""
    options = _get(select, "option") or []
    try:
        kinds = metadata.card_kind()
        power = metadata.card_power()
        traits = metadata.card_traits()
        costs = metadata.attack_cost()
        values = metadata.trainer_value()
        types_meta = metadata.type_info()
        opp = _opp_active(observation)
        opp_hp = _num(_get(opp, "hp")) if opp is not None else 0.0
        players, me = _players(observation)
        my_active = (_get(players[me], "active") or [None])[0]
        my_type = types_meta.get(int(_num(_get(my_active, "id"))), (-1, -1))[0]

        card_ids = [resolve_card(observation, select, o) for o in options]
        dmgs = [_attack_damage(o)[0] for o in options]
        max_dmg = max(dmgs, default=0)
        strengths = [power.get(cid, (0, 0)) for cid in card_ids]
        best_strength = max(strengths, default=(0, 0))
        seen_types: set[Any] = set()
        rows: list[list[float]] = []
        for i, option in enumerate(options):
            otype = _get(option, "type")
            first_of_type = otype not in seen_types
            seen_types.add(otype)
            try:
                rows.append(
                    _option_row(
                        option,
                        otype,
                        card_ids[i],
                        i,
                        len(options),
                        first_of_type,
                        strengths[i] == best_strength and best_strength != (0, 0),
                        opp_hp,
                        max_dmg,
                        my_type,
                        observation,
                        kinds,
                        power,
                        traits,
                        costs,
                        values,
                        types_meta,
                    )
                )
            except Exception:
                rows.append([0.0] * OPTION_COUNT)
        return rows
    except Exception:
        return [[0.0] * OPTION_COUNT for _ in options]


def _option_row(  # noqa: PLR0913 - a feature row is inherently wide
    option: Any,
    otype: Any,
    card_id: int,
    index: int,
    n_options: int,
    first_of_type: bool,
    strongest: bool,
    opp_hp: float,
    max_dmg: int,
    my_type: int,
    observation: Any,
    kinds: dict[int, int],
    power: dict[int, tuple[int, int]],
    traits: dict[int, tuple[int, int, int, int, int]],
    costs: dict[int, int],
    values: dict[int, int],
    types_meta: dict[int, tuple[int, int]],
) -> list[float]:
    type_slot = [0.0] * 9
    type_slot[_TYPE_SLOT.get(otype, 8)] = 1.0

    kind_slot = [0.0] * 6
    kind = kinds.get(card_id, -1)
    if kind in _KIND_SLOT:
        kind_slot[_KIND_SLOT[kind]] = 1.0
    dmg, hp = power.get(card_id, (0, 0))
    stage, ex, mega, retreat, evolvable = traits.get(card_id, (0, 0, 0, 0, 0))
    card_type = types_meta.get(card_id, (-1, -1))[0]
    card_feats = [
        *kind_slot,
        _clamp(hp / 340.0),
        _clamp(dmg / 300.0),
        _clamp(stage / 2.0),
        float(ex),
        float(mega),
        _clamp(retreat / 4.0),
        _clamp(costs.get(card_id, 0) / 5.0),
        _clamp(values.get(card_id, 0) / 5.0),
        float(evolvable),
        1.0 if (card_type >= 0 and card_type == my_type) else 0.0,
    ]

    target, area = resolve_target(observation, option)
    if target is not None:
        t_hp = _num(_get(target, "hp"))
        t_max = _num(_get(target, "maxHp")) or t_hp
        t_energy = float(len(_get(target, "energies") or ()))
        t_id = int(_num(_get(target, "id")))
        t_cost = costs.get(t_id)
        t_dmg = power.get(t_id, (0, 0))[0]
        target_feats = [
            1.0,
            1.0 if area == _AREA_ACTIVE else 0.0,
            1.0 if area == _AREA_BENCH else 0.0,
            _clamp(t_hp / t_max) if t_max > 0 else 0.0,
            _clamp(t_energy / 5.0),
            1.0 if (t_cost is not None and t_energy >= t_cost) else 0.0,
            _clamp(max(0.0, (t_cost or 0) - t_energy) / 3.0),
            _clamp(t_dmg / 300.0),
        ]
    else:
        target_feats = [0.0] * 8

    a_dmg, a_cost = _attack_damage(option) if otype == _ATTACK else (0, 0)
    attack_feats = [
        _clamp(a_dmg / 300.0),
        1.0 if (otype == _ATTACK and opp_hp > 0 and a_dmg >= opp_hp) else 0.0,
        _clamp(a_dmg / opp_hp / 2.0) if (otype == _ATTACK and opp_hp > 0) else 0.0,
        _clamp(a_cost / 5.0),
        1.0 if (otype == _ATTACK and a_dmg == max_dmg and max_dmg > 0) else 0.0,
    ]

    relative = [
        _clamp(index / max(1, n_options - 1)) if n_options > 1 else 0.0,
        1.0 if first_of_type else 0.0,
        1.0 if strongest else 0.0,
    ]
    return [*type_slot, *card_feats, *target_feats, *attack_feats, *relative]
