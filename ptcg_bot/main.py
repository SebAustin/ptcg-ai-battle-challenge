"""Agent entrypoint — ``agent(obs_dict) -> list[int]``.

The engine calls this each decision point; the return is a list of indices into
``obs["select"]["option"]`` (length within ``minCount..maxCount``, distinct). On
the very first call ``obs["select"]`` is ``None`` and we must return the 60-card
deck (card IDs).

This is the **v3 card-aware heuristic** (live-ladder informed; see ASSUMPTIONS):

- deck request        -> the 60 IDs from deck.csv
- MAIN phase          -> attack when able — preferring a LETHAL attack (enough
  damage to KO the opponent's Active) over raw max damage — else develop
  (ability, energy-to-Active, evolve-Active, play), else end
- card selections     -> ranked by what the card IS: keep/field/promote the
  strongest (setup active, switch, promotion, search-to-hand), discard the
  weakest (discard, to-deck, to-prize); resolved via each option's area/index
  against the observation, valued via the engine's card metadata (lazy, and
  offline everything degrades to the legacy first-option behavior)
- everything else     -> take the allowed options (legal baseline)

Pure standard library; reads the raw observation dict (no ``cg`` import at
module level — card metadata loads lazily and is absent-safe). A catch-all
guarantees a legal selection is always returned — an illegal or raised response
would forfeit the game. ``legacy_agent`` preserves the previous (v2) policy for
A/B measurement by ``tools/tournament``.
"""

from __future__ import annotations

import os
from typing import Any

from . import engine_adapter, metadata, search

# OptionType ids from the engine's cg/api.py.
_ABILITY = 10
_PLAY = 7
_ATTACH = 8
_EVOLVE = 9
_ATTACK = 13
_END = 14

# AreaType ids (cg/api.py): where an option's card lives.
_AREA_DECK = 1
_AREA_HAND = 2
_AREA_DISCARD = 3
_AREA_ACTIVE = 4
_AREA_BENCH = 5

# SelectContext ids: what a CARD selection is for. "Strong" contexts should get
# our best cards (start/promote/fetch); "weak" contexts should give up our worst
# (discard/shuffle back/prize).
_CTX_PREFER_STRONG = frozenset(
    {1, 2, 3, 4, 6, 7}
)  # setup A/B, switch, to-active/field/hand
_CTX_PREFER_WEAK = frozenset({8, 9, 10, 11})  # discard, to-deck(+bottom), to-prize

# MAIN-phase policy: ATTACK as soon as able (with the highest-damage attack),
# otherwise develop the board (ability, energy, evolution, play), and end only
# as a last resort. Attacking-when-able rather than over-developing first is the
# big lever — it lifts win-rate from ~21% to ~100% vs a random baseline (measured
# by tools/tournament). RETREAT is intentionally absent (never retreat).
_DEVELOP_PRIORITY: tuple[int, ...] = (_ABILITY, _ATTACH, _EVOLVE, _PLAY)

_DECK_PATHS = ("deck.csv", "/kaggle_simulations/agent/deck.csv")
_DECK_SIZE = 60


def _read_deck() -> list[int]:
    """Read the 60 card IDs from deck.csv (one integer per line)."""
    for path in _DECK_PATHS:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                ids = [int(x) for x in handle.read().splitlines() if x.strip()]
            if len(ids) >= _DECK_SIZE:
                return ids[:_DECK_SIZE]
    raise FileNotFoundError("deck.csv not found or has fewer than 60 card IDs")


def _first_of_type(options: list[dict[str, Any]], otype: int) -> int | None:
    for index, option in enumerate(options):
        if option.get("type") == otype:
            return index
    return None


def _opp_active_hp(obs: dict[str, Any]) -> int:
    """Remaining HP of the opponent's Active Pokémon, or 0 if unknown."""
    try:
        current = obs.get("current") or {}
        me = int(current.get("yourIndex") or 0)
        active = (current.get("players") or [])[1 - me].get("active") or []
        return int((active[0] or {}).get("hp") or 0) if active else 0
    except Exception:
        return 0


def _best_attack(options: list[dict[str, Any]], opp_hp: int = 0) -> int | None:
    """Best ATTACK option index: the cheapest LETHAL attack when one exists
    (damage >= the opponent Active's remaining HP), else max damage. ``None``
    when there is no attack option."""
    attacks = [i for i, o in enumerate(options) if o.get("type") == _ATTACK]
    if not attacks:
        return None
    damage = metadata.attack_damage()

    def _dmg(index: int) -> int:
        attack_id = options[index].get("attackId")
        return damage.get(attack_id, 0) if isinstance(attack_id, int) else 0

    if opp_hp > 0:
        lethal = [i for i in attacks if _dmg(i) >= opp_hp]
        if lethal:
            return min(lethal, key=_dmg)  # KO with the least overkill
    return max(attacks, key=_dmg)


def _prefer_active_target(options: list[dict[str, Any]], otype: int) -> int | None:
    """First option of ``otype`` targeting the ACTIVE Pokémon, else the first."""
    indices = [i for i, o in enumerate(options) if o.get("type") == otype]
    if not indices:
        return None
    for i in indices:
        if options[i].get("inPlayArea") == _AREA_ACTIVE:
            return i
    return indices[0]


def _choose_main(
    obs: dict[str, Any], options: list[dict[str, Any]]
) -> list[int] | None:
    """Attack when able (lethal first), else develop, else end.

    Development targets the Active where the option says which Pokémon it
    touches (energy attachment, evolution). Returns ``None`` when the prompt has
    no MAIN-type option (so the caller falls back to the generic selector).
    """
    attack = _best_attack(options, _opp_active_hp(obs))
    if attack is not None:
        return [attack]
    for otype in _DEVELOP_PRIORITY:
        index = (
            _prefer_active_target(options, otype)
            if otype in (_ATTACH, _EVOLVE)
            else _first_of_type(options, otype)
        )
        if index is not None:
            return [index]
    return None if (end := _first_of_type(options, _END)) is None else [end]


def _default_selection(select: dict[str, Any]) -> list[int]:
    """A always-legal fallback: the first ``maxCount`` option indices."""
    option_count = len(select.get("option") or ())
    max_count = int(select.get("maxCount") or 0)
    return list(range(min(max_count, option_count)))


def _option_strength(
    obs: dict[str, Any], select: dict[str, Any], option: dict[str, Any]
) -> tuple[int, int, int]:
    """How good the card behind ``option`` is: (attached energy, damage, hp).

    Resolves the option's area/index against the observation (hand, board,
    discard, or the revealed deck list) and values the card via the engine
    metadata. Unresolvable options score (0, 0, 0) — neutral.
    """
    try:
        area = option.get("area")
        index = int(option.get("index") or 0)
        current = obs.get("current") or {}
        me = int(current.get("yourIndex") or 0)
        owner = option.get("playerIndex")
        player = (current.get("players") or [{}, {}])[
            owner if isinstance(owner, int) and owner in (0, 1) else me
        ]

        energy = 0
        raw_id: Any = None
        if area == _AREA_DECK and select.get("deck"):
            raw_id = ((select.get("deck") or [])[index] or {}).get("id")
        elif area == _AREA_HAND:
            raw_id = ((player.get("hand") or [])[index] or {}).get("id")
        elif area == _AREA_DISCARD:
            raw_id = ((player.get("discard") or [])[index] or {}).get("id")
        elif area in (_AREA_ACTIVE, _AREA_BENCH):
            zone = "active" if area == _AREA_ACTIVE else "bench"
            pokemon = (player.get(zone) or [])[index] or {}
            raw_id = pokemon.get("id")
            energy = len(pokemon.get("energies") or ())
        if raw_id is None:
            return (0, 0, 0)
        dmg, hp = metadata.card_power().get(int(raw_id), (0, 0))
        return (energy, dmg, hp)
    except Exception:
        return (0, 0, 0)


def _ranked_selection(obs: dict[str, Any], select: dict[str, Any]) -> list[int]:
    """Context-aware card selection: same count as the legacy default, but the
    options are ORDERED by card value — strongest first for keep/field/promote
    contexts, weakest first for discard/give-up contexts. Falls back to index
    order when the context is unknown or metadata is unavailable."""
    options = select.get("option") or []
    count = min(int(select.get("maxCount") or 0), len(options))
    context = select.get("context")
    if count <= 0:
        return []
    if not metadata.card_power() or (
        context not in _CTX_PREFER_STRONG and context not in _CTX_PREFER_WEAK
    ):
        return list(range(count))
    strengths = [_option_strength(obs, select, o) for o in options]
    reverse = context in _CTX_PREFER_STRONG
    order = sorted(range(len(options)), key=lambda i: strengths[i], reverse=reverse)
    return sorted(order[:count])  # sorted indices: stable, duplicate-free, legal


def _heuristic_selection(obs: dict[str, Any]) -> list[int]:
    """Search-free v3 policy for one (non-deck-request) selection."""
    select = obs.get("select") or {}
    options = select.get("option") or []
    if not options:
        return []
    if int(select.get("maxCount") or 0) == 1:
        picked = _choose_main(obs, options)
        if picked is not None:
            return picked
    return _ranked_selection(obs, select)


# --- legacy (v2) policy — kept ONLY for A/B measurement by tools/tournament ---


def _legacy_choose_main(options: list[dict[str, Any]]) -> list[int] | None:
    attacks = [i for i, o in enumerate(options) if o.get("type") == _ATTACK]
    if attacks:
        damage = metadata.attack_damage()

        def _dmg(i: int) -> int:
            attack_id = options[i].get("attackId")
            return damage.get(attack_id, 0) if isinstance(attack_id, int) else 0

        return [max(attacks, key=_dmg)]
    for otype in _DEVELOP_PRIORITY:
        index = _first_of_type(options, otype)
        if index is not None:
            return [index]
    return None if (end := _first_of_type(options, _END)) is None else [end]


def legacy_agent(obs_dict: dict[str, Any]) -> list[int]:
    """The previous (v2) shipped policy, unchanged — the A/B baseline."""
    try:
        if engine_adapter.is_deck_request(obs_dict):
            return _read_deck()
        select = obs_dict.get("select") or {}
        options = select.get("option") or []
        if not options:
            return []
        if int(select.get("maxCount") or 0) == 1:
            picked = _legacy_choose_main(options)
            if picked is not None:
                return picked
        return _default_selection(select)
    except Exception:
        return _fallback(obs_dict)


def _fallback(obs_dict: dict[str, Any]) -> list[int]:
    """A legal selection derived only from the raw select, or ``[]`` as last resort."""
    try:
        select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        return _default_selection(select) if isinstance(select, dict) else []
    except Exception:
        return []


def agent(obs_dict: dict[str, Any]) -> list[int]:
    """Engine entrypoint — the SHIPPED policy: attack-first heuristic (100% vs
    random). Never raises: a bad/raised selection would forfeit the game.

    A determinized rollout-PIMC search (:func:`search_agent`) exists and reaches
    ~parity with this heuristic (measured by ``tools/tournament --opponent
    heuristic``) but not a decisive win, so the heuristic stays the default. See
    ASSUMPTIONS.md.
    """
    try:
        if engine_adapter.is_deck_request(obs_dict):
            return _read_deck()
        return _heuristic_selection(obs_dict)
    except Exception:
        return _fallback(obs_dict)


def search_agent(obs_dict: dict[str, Any]) -> list[int]:
    """EXPERIMENTAL entrypoint: one-ply determinized lookahead (:mod:`ptcg_bot.search`)
    with a heuristic fallback. Not yet stronger than :func:`agent` — retained for
    the in-progress IS-MCTS work and A/B'd by ``tools/tournament``.
    """
    try:
        if engine_adapter.is_deck_request(obs_dict):
            return _read_deck()
        searched = search.choose_by_search(obs_dict)
        return searched if searched is not None else _heuristic_selection(obs_dict)
    except Exception:
        return _fallback(obs_dict)
