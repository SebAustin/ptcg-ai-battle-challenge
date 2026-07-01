# PTCG AI Battle Challenge — Battle Agent + Strategy

An AI agent for Kaggle's **[Pokémon Company — PTCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy)**,
plus the accompanying **Strategy** writeup. Two linked submissions:

- **Simulation** — the agent that plays battles in the official simulator (scored on win rate).
- **Strategy** (our primary score) — a ≤2000-word writeup + figures, scored **70%** model/approach,
  **20%** deck construction, **10%** report.

> Status: card data layer, rule spine, **deckbuilder**, internal **GameState**, the
> **explainable heuristic evaluator**, the **engine adapter**, and a first **playable agent**
> (`main.agent`) are in place and tested behind an enforced quality/CI + security gate.
> `make verify` runs the **live engine**; the agent plays full legal games against it. A K-world
> rollout **PIMC** `search`/`belief` layer is built and A/B-tested — now roughly on par with the
> shipped heuristic (~40–47% in mirror A/B) but not a clear win, so the heuristic still ships.
> Realistic determinization + deeper rollouts are the next lever (see ASSUMPTIONS.md).
>
> **Engine:** the simulator (`pokemon-tcg-ai-battle`: `sample_submission/cg/` + native `libcg`)
> is fetched locally into the gitignored `engine/` via `make engine` (after accepting that
> competition's rules — a one-time manual step). `engine_adapter` parses the observation dict
> directly (pure-stdlib, no `cg` import). See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## The game we're playing (confirmed from the data)

`ptcg_bot/cards.py` loads the **1,267-card** legal pool from `data/EN_Card_Data.csv`. The pool
unambiguously identifies the variant as **classic Pokémon TCG, Standard format, Scarlet & Violet +
Mega era** — *not* TCG Pocket:

- In-deck **Basic & Special Energy** cards, the full classic type taxonomy (Basic / Stage 1 /
  Stage 2 Pokémon, Item / Tool / Supporter / Stadium), and SV-era rules (`Pokémon ex`,
  `Mega Pokémon ex`, `ACE SPEC`).
- → **60-card deck, 6 prizes**, manual energy attachment from hand, standard turn structure.
  All of this lives in **`ptcg_bot/rules.py`**, the single source of truth, to be verified against
  the live engine by `tools/verify_env.py`.

## How it will play (architecture)

The defining challenge vs a perfect-information game is **imperfect information + stochasticity**:
hidden hand, deck order, and prizes; coin flips; shuffles. The decision core is therefore
**determinized Information-Set MCTS (PIMC + MCTS)**:

1. **Sample `K` hidden-state worlds** (`belief.py`) — each a rules-consistent guess at the opponent's
   hand / deck order / prizes, drawn from public info + a deck prior.
2. **Search each world** (`search.py`) — bounded MCTS with sampled chance nodes, rollouts evaluated
   by a fast, **hand-crafted, tunable heuristic** (`evaluate.py`).
3. **Aggregate at the root action** across worlds (PIMC voting) — the move that is best *in
   expectation over hidden states*, which is exactly the robustness the Strategy rubric rewards.

Shipped in phases, each a valid submission: **Heuristic → Tuned → Search → (optional) Learned.**
The heuristic stays the *explainable* narrative even if a learned evaluator is added later.

## Why this design scores

| Rubric ask (70% Model) | How the design answers it |
|---|---|
| Consistency under repeated matches | Averaging over `K` sampled worlds = marginalizing over openings/prizes |
| No over-reliance on initial states/matchups | Search adapts online; no per-matchup hardcoding |
| Technical soundness & originality | IS-MCTS over a determinized belief (Cowling et al.) |
| Clarity of rationale | Inspectable heuristic value (prize tempo, board HP, energy, type advantage) |

## Layout

```
ptcg_bot/        the submitted agent (pure stdlib once bundled)
  rules.py       SOURCE OF TRUTH for the variant (verified vs the engine)
  cards.py       typed Card model + CSV loader  [done, tested]
  config.py      tunable heuristic/search weights (_f env-overridable)
  state.py       internal GameState model  [done, tested]
  evaluate.py    explainable state-value heuristic  [done, tested]
  engine_adapter.py  obs dict -> GameState + action encoding  [wired, tested vs live engine]
  main.py        agent entrypoint — attack-first heuristic (shipped; consistently beats random)  [done]
  search.py/belief.py  K-world rollout PIMC search  [built; ~on par w/ heuristic, not shipped — see ASSUMPTIONS.md]
  legal  [next — typed option semantics; realistic determinization + deeper rollouts]
deckbuilder/     offline deck construction & optimization (the 20% deliverable)
tools/           eval harness: verify_env/tournament/soak/tune/bundle/run_match
tests/           pytest  [test_cards.py passing]
writeup/         WRITEUP.md (<=2000w) + figures.py + ACCEPTANCE.md
data/            competition card files (gitignored — see data/README.md)
```

## Quickstart

```bash
make setup    # .venv + dependencies
make data     # download the card files (needs Kaggle auth — see SUBMIT.md)
make test     # run the unit suite
make help     # list all targets
```

## Development / quality gate

The dev tooling (ruff · isort · black · mypy) is config-driven from `pyproject.toml` and is
**dev-only** — none of it ships in the bundled agent. GitHub Actions runs the same gate on every
push (`.github/workflows/ci.yml`).

```bash
make format      # auto-fix: ruff --fix + isort + black
make lint        # check-only: ruff + isort + black --check
make typecheck   # mypy ptcg_bot
make audit       # SECURITY.md guard: no socket/urllib/requests/subprocess imports
make ci          # the full engine-free gate: lint + typecheck + audit + test

pre-commit install   # optional: run the gate automatically on every commit
```

> The 5 dataset-backed tests in `tests/test_cards.py` are marked `integration` and **skip**
> when `data/EN_Card_Data.csv` is absent (e.g. in CI, where the card data can't be
> redistributed). Run `make data` to exercise them locally.

## Building a deck

`deckbuilder/` constructs a legal 60-card deck from the pool (the Strategy 20% deliverable).
Decks are ultimately meant to be scored by the same tournament harness as the agent; until that
engine is wired, construction optimizes an **offline deck-quality heuristic**
(`deckbuilder/score.py`: consistency, energy balance, opening reliability, attacker power, prize
safety). See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the decisions behind it.

```bash
make deck        # writes dist/deck.csv + prints the score breakdown and rationale
```

```python
from ptcg_bot.cards import load_pool
from deckbuilder import build_deck, score, validate

pool = load_pool()
deck = build_deck(pool)
assert validate(deck, pool) == []        # legal 60
print(score(deck, pool).total)           # heuristic quality, 0..100
print(deck.to_csv())                       # submission deck.csv (60 bare card-ID lines)
```

The full plan (architecture rationale, deckbuilder, writeup→rubric mapping, 11-week milestones,
risks) lives in the approved plan file referenced from the project notes.
