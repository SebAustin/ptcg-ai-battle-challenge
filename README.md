# PTCG AI Battle Challenge — Battle Agent + Strategy

An AI agent for Kaggle's **[Pokémon Company — PTCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy)**,
plus the accompanying **Strategy** writeup. Two linked submissions:

- **Simulation** — the agent that plays battles in the official simulator (scored on win rate).
- **Strategy** (our primary score) — a ≤2000-word writeup + figures, scored **70%** model/approach,
  **20%** deck construction, **10%** report.

> Status: **scaffolding** (plan §W1). The card data layer and the rule spine are in place and
> tested; the forward model, search, deckbuilder, and harness land in subsequent weeks.

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
  engine_adapter.py  isolates the unknown simulator schema  [stub, W2]
  state/sim/effects/legal/belief/search/evaluate/main  [W3-W8]
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

The full plan (architecture rationale, deckbuilder, writeup→rubric mapping, 11-week milestones,
risks) lives in the approved plan file referenced from the project notes.
