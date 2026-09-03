"""Source of truth for the PTCG rule variant.

CONFIRMED from the competition dataset (data/EN_Card_Data.csv, 2026-06):
classic **Pokémon TCG, Standard format, Scarlet & Violet + Mega era** — NOT
TCG Pocket. Evidence: in-deck Basic/Special Energy cards, the full classic type
taxonomy (Basic/Stage 1/Stage 2 Pokémon, Item/Tool/Supporter/Stadium), SV-era
rule keywords (``Pokémon ex``, ``Mega Pokémon ex``, ``ACE SPEC``), HP up to 380,
and SV-era set codes (TWM, SSP, TEF, SCR, MEG, DRI, ...).

Everything rule-dependent in the codebase reads from THIS module. The constants
are hard literals (not tuned). ``tools/verify_env.py`` asserts they match the
live simulator before any strategy work is trusted — do not edit a constant to
make a test pass; edit it only to match the engine.
"""

from __future__ import annotations

from enum import Enum

# --- Deck construction --------------------------------------------------------
DECK_SIZE: int = 60
MAX_COPIES_BY_NAME: int = 4  # except Basic Energy (unlimited) and ACE SPEC (1)
ACE_SPEC_PER_DECK: int = 1
MIN_BASIC_POKEMON: int = 1  # a legal deck must contain >=1 Basic Pokémon

# --- Match / board ------------------------------------------------------------
PRIZE_COUNT: int = 6
OPENING_HAND_SIZE: int = 7
MAX_BENCH: int = 5
MAX_HAND_BEFORE_DISCARD: int | None = None  # no hand-size limit in this format

# Prizes taken per knockout, by the KO'd Pokémon's rule.
PRIZES_FOR_KO_DEFAULT: int = 1
PRIZES_FOR_KO_EX: int = 2  # Pokémon ex
PRIZES_FOR_KO_MEGA_EX: int = 3  # Mega Pokémon ex (KO is worth 3 prizes)

# --- Combat math --------------------------------------------------------------
WEAKNESS_MULTIPLIER: int = 2  # SV era: weakness is ×2 (additive ×2, applied
# before resistance). Resistance, when present,
# subtracts a flat amount (usually 30).
RESISTANCE_REDUCTION_DEFAULT: int = 30

# --- Turn structure (per-turn once-only actions) ------------------------------
ENERGY_ATTACHMENTS_PER_TURN: int = 1  # one manual energy from hand per turn
SUPPORTERS_PER_TURN: int = 1
STADIUMS_PER_TURN: int = 1
RETREATS_PER_TURN: int = 1
FIRST_PLAYER_ATTACKS_TURN: int = 2  # player going first may not attack on T1

# A coin flip is the only RNG primitive the rules expose directly (besides
# shuffles / random draws). Heads probability is fair.
COIN_HEADS_PROB: float = 0.5


class EnergyType(str, Enum):
    """The nine Standard-format energy/Pokémon types, plus Dragon.

    The element symbol used by the dataset is the value (so ``EnergyType('G')``
    works after stripping braces). Dragon has no Basic Energy and leaks through
    the EN CSV as the kanji ``竜`` — :func:`normalize_type` maps it here.
    """

    GRASS = "G"
    FIRE = "R"
    WATER = "W"
    LIGHTNING = "L"
    PSYCHIC = "P"
    FIGHTING = "F"
    DARKNESS = "D"
    METAL = "M"
    COLORLESS = "C"
    DRAGON = "N"


# Raw type tokens seen in the dataset that need remapping onto EnergyType values.
_TYPE_ALIASES = {
    "竜": "N",  # Dragon leaked as Japanese kanji in the EN export
    "A": "N",  # rare alternate Dragon token observed in a few rows
}

# Basic Energy provides exactly its own type; these have in-deck cards (SVE set).
BASIC_ENERGY_TYPES = frozenset(
    {
        EnergyType.GRASS,
        EnergyType.FIRE,
        EnergyType.WATER,
        EnergyType.LIGHTNING,
        EnergyType.PSYCHIC,
        EnergyType.FIGHTING,
        EnergyType.DARKNESS,
        EnergyType.METAL,
    }
)


def normalize_type(token: str | None) -> EnergyType | None:
    """Map a raw dataset type token (e.g. ``"{G}"``, ``"竜"``) to an EnergyType.

    Returns ``None`` for empty / ``"n/a"`` / unrecognized tokens so callers can
    decide how to treat unknowns rather than crashing on dirty data.
    """
    if not token:
        return None
    cleaned = token.strip().strip("{}").strip()
    if not cleaned or cleaned.lower() == "n/a":
        return None
    cleaned = _TYPE_ALIASES.get(cleaned, cleaned)
    try:
        return EnergyType(cleaned)
    except ValueError:
        return None


def prizes_for_knockout(rule: str | None) -> int:
    """Prizes the attacker takes for knocking out a Pokémon with this ``rule``."""
    if rule == "Mega Pokémon ex":
        return PRIZES_FOR_KO_MEGA_EX
    if rule == "Pokémon ex":
        return PRIZES_FOR_KO_EX
    return PRIZES_FOR_KO_DEFAULT
