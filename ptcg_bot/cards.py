"""Typed card model + loader for the competition card pool.

The Kaggle ``EN_Card_Data.csv`` has ONE ROW PER MOVE, so a card with three
attacks spans three rows that share a ``Card ID``. :func:`load_pool` groups by
id and produces one immutable :class:`Card` per id with a tuple of :class:`Move`.

Design choices:
- **Immutable** (frozen dataclasses) per the project coding style — the pool is
  read-only reference data shared across search worlds; nothing should mutate it.
- **Defensive parsing** — the export is dirty (``n/a`` sentinels, leading
  newlines in move names, a Dragon type leaking through as kanji, stray
  ``{Team Rocket}`` energy tokens). Parsers degrade gracefully instead of
  raising on a single bad cell; unknown tokens are logged via ``warnings``.
"""

from __future__ import annotations

import csv
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from .rules import EnergyType, normalize_type

# Sentinels the dataset uses for "no value".
_NULLS = frozenset({"", "n/a", "N/A", "na"})

# One Colorless-energy requirement is drawn as a filled dot in the Cost column.
_COLORLESS_PIP = "●"
_BRACE_TOKEN = re.compile(r"\{([^}]*)\}")
_LEADING_INT = re.compile(r"^\s*(\d+)")


def _clean(value: str | None) -> str:
    """Strip whitespace and collapse the dataset's null sentinels to ``""``."""
    if value is None:
        return ""
    stripped = value.strip()
    return "" if stripped in _NULLS else stripped


@dataclass(frozen=True)
class EnergyCost:
    """An attack's energy requirement: typed energy + generic Colorless pips."""

    colorless: int = 0
    # Sorted tuple of (type, count) pairs — a dict can't live on a frozen,
    # hashable dataclass. Colorless requirements live in ``colorless``, not here.
    typed: tuple[tuple[EnergyType, int], ...] = ()

    @property
    def total(self) -> int:
        return self.colorless + sum(count for _, count in self.typed)

    def as_dict(self) -> dict[EnergyType, int]:
        out = dict(self.typed)
        if self.colorless:
            out[EnergyType.COLORLESS] = (
                out.get(EnergyType.COLORLESS, 0) + self.colorless
            )
        return out

    def __bool__(self) -> bool:
        return self.total > 0


def parse_cost(raw: str | None) -> EnergyCost:
    """Parse a Cost cell like ``"{D}●●"`` -> 1 Darkness + 2 Colorless.

    ``●`` and an explicit ``{C}`` both count as Colorless. Unrecognized brace
    tokens (e.g. ``{Team Rocket}``) are treated as one Colorless requirement —
    the safest approximation for cost-affordability checks.
    """
    text = _clean(raw)
    if not text:
        return EnergyCost()
    colorless = text.count(_COLORLESS_PIP)
    typed: dict[EnergyType, int] = defaultdict(int)
    for token in _BRACE_TOKEN.findall(text):
        etype = normalize_type(token)
        if etype is None:
            warnings.warn(
                f"unknown energy token {token!r} in cost {raw!r}; counting as Colorless",
                stacklevel=2,
            )
            colorless += 1
        elif etype is EnergyType.COLORLESS:
            colorless += 1
        else:
            typed[etype] += 1
    return EnergyCost(
        colorless=colorless,
        typed=tuple(sorted(typed.items(), key=lambda kv: kv[0].value)),
    )


def parse_damage(raw: str | None) -> tuple[int, str]:
    """Parse a Damage cell -> (base, modifier).

    ``"120"`` -> (120, ""), ``"30×"`` -> (30, "×"), ``"100+"`` -> (100, "+").
    A modifier means the printed number is conditional (coin flips, bonuses); the
    forward model resolves the real number at resolution time.
    """
    text = _clean(raw)
    if not text:
        return 0, ""
    match = _LEADING_INT.match(text)
    if not match:
        return 0, ""
    base = int(match.group(1))
    modifier = text[match.end() :].strip()
    return base, modifier


@dataclass(frozen=True)
class Move:
    """An attack or ability printed on a card."""

    name: str
    cost: EnergyCost
    damage_base: int
    damage_modifier: str  # "", "×", "+", "-"
    effect: str

    @property
    def is_attack(self) -> bool:
        """Attacks have an energy cost or deal damage; abilities/rules have neither."""
        return bool(self.cost) or self.damage_base > 0

    @property
    def is_variable_damage(self) -> bool:
        return self.damage_modifier in {"×", "x", "+"}


# The Stage/Type field values, grouped into supertypes.
_POKEMON_SUBTYPES = frozenset({"Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"})
_TRAINER_SUBTYPES = frozenset({"Item", "Pokémon Tool", "Supporter", "Stadium"})
_ENERGY_SUBTYPES = frozenset({"Basic Energy", "Special Energy"})


@dataclass(frozen=True)
class Card:
    """One card in the legal pool (immutable reference data)."""

    card_id: int
    name: str
    expansion: str
    collection_no: str
    subtype: str  # raw Stage/Type field, e.g. "Basic Pokémon", "Item"
    category: str  # Category column: Tera(...)/Ancient/Trainer's Pokémon/...
    rule: str  # "Pokémon ex" / "Mega Pokémon ex" / "ACE SPEC" / ""
    previous_stage: str  # pre-evolution name (evolution is matched by name)
    hp: int | None
    poke_type: EnergyType | None
    weakness: EnergyType | None
    resistance: EnergyType | None
    retreat: int | None
    moves: tuple[Move, ...]

    # --- supertype helpers ---------------------------------------------------
    @property
    def is_pokemon(self) -> bool:
        return self.subtype in _POKEMON_SUBTYPES

    @property
    def is_trainer(self) -> bool:
        return self.subtype in _TRAINER_SUBTYPES

    @property
    def is_energy(self) -> bool:
        return self.subtype in _ENERGY_SUBTYPES

    @property
    def is_basic_pokemon(self) -> bool:
        return self.subtype == "Basic Pokémon"

    @property
    def is_basic_energy(self) -> bool:
        return self.subtype == "Basic Energy"

    @property
    def stage(self) -> int | None:
        """0 for Basic, 1 for Stage 1, 2 for Stage 2; ``None`` for non-Pokémon."""
        return {"Basic Pokémon": 0, "Stage 1 Pokémon": 1, "Stage 2 Pokémon": 2}.get(
            self.subtype
        )

    @property
    def is_ex(self) -> bool:
        return self.rule in {"Pokémon ex", "Mega Pokémon ex"}

    @property
    def is_ace_spec(self) -> bool:
        return self.rule == "ACE SPEC"

    @property
    def attacks(self) -> tuple[Move, ...]:
        return tuple(m for m in self.moves if m.is_attack)

    @property
    def abilities(self) -> tuple[Move, ...]:
        return tuple(m for m in self.moves if not m.is_attack)


class CardPool:
    """An indexed, read-only collection of every legal :class:`Card`."""

    def __init__(self, cards: Iterable[Card]) -> None:
        self._cards: tuple[Card, ...] = tuple(cards)
        self._by_id: dict[int, Card] = {c.card_id: c for c in self._cards}
        self._by_name: dict[str, list[Card]] = defaultdict(list)
        # name -> cards that evolve FROM that name (their pre-evolution).
        self._evolves_from: dict[str, list[Card]] = defaultdict(list)
        for card in self._cards:
            self._by_name[card.name].append(card)
            if card.previous_stage:
                self._evolves_from[card.previous_stage].append(card)

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self) -> Iterator[Card]:
        return iter(self._cards)

    def by_id(self, card_id: int) -> Card | None:
        return self._by_id.get(card_id)

    def by_name(self, name: str) -> list[Card]:
        """All printings of a card name (reprints span sets)."""
        return list(self._by_name.get(name, ()))

    def evolutions_of(self, name: str) -> list[Card]:
        """Cards that list ``name`` as their previous stage (i.e. evolve from it)."""
        return list(self._evolves_from.get(name, ()))

    @cached_property
    def pokemon(self) -> tuple[Card, ...]:
        return tuple(c for c in self._cards if c.is_pokemon)

    @cached_property
    def basics(self) -> tuple[Card, ...]:
        return tuple(c for c in self._cards if c.is_basic_pokemon)

    @cached_property
    def trainers(self) -> tuple[Card, ...]:
        return tuple(c for c in self._cards if c.is_trainer)

    @cached_property
    def energy(self) -> tuple[Card, ...]:
        return tuple(c for c in self._cards if c.is_energy)


def _build_card(rows: list[dict[str, str]]) -> Card:
    """Assemble one Card from its (one-or-more) CSV rows."""
    head = rows[0]
    hp_raw = _clean(head.get("HP"))
    retreat_raw = _clean(head.get("Retreat"))
    moves = tuple(
        Move(
            name=_clean(r.get("Move Name")),
            cost=parse_cost(r.get("Cost")),
            damage_base=parse_damage(r.get("Damage"))[0],
            damage_modifier=parse_damage(r.get("Damage"))[1],
            effect=_clean(r.get("Effect Explanation")),
        )
        for r in rows
        if _clean(r.get("Move Name"))
    )
    return Card(
        card_id=int(head["Card ID"]),
        name=_clean(head.get("Card Name")),
        expansion=_clean(head.get("Expansion")),
        collection_no=_clean(head.get("Collection No.")),
        subtype=_clean(head.get("Stage (Pokémon)/Type (Energy and Trainer)")),
        category=_clean(head.get("Category")),
        rule=_clean(head.get("Rule")),
        previous_stage=_clean(head.get("Previous stage")),
        hp=int(hp_raw) if hp_raw.isdigit() else None,
        poke_type=normalize_type(head.get("Type")),
        weakness=normalize_type(head.get("Weakness")),
        resistance=normalize_type(head.get("Resistance (Type)")),
        retreat=int(retreat_raw) if retreat_raw.isdigit() else None,
        moves=moves,
    )


# Default dataset location (overridable by callers / the engine adapter).
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "EN_Card_Data.csv"


def load_pool(csv_path: str | Path = DEFAULT_CSV) -> CardPool:
    """Load the full legal card pool from the competition CSV.

    Rows are grouped by ``Card ID`` (multi-move cards span multiple rows) and
    parsed into immutable :class:`Card` objects. Raises ``FileNotFoundError`` with
    a clear message if the dataset has not been downloaded yet.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Card data not found at {path}. Download it with:\n"
            "  .venv/bin/kaggle competitions download "
            "-c pokemon-tcg-ai-battle-challenge-strategy -f EN_Card_Data.csv -p data/"
        )
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw_id = (row.get("Card ID") or "").strip()
            if raw_id.isdigit():
                grouped[int(raw_id)].append(row)
    return CardPool(_build_card(rows) for _, rows in sorted(grouped.items()))
