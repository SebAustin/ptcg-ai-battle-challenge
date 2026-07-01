# PTCG AI Battle Challenge — Strategy Writeup

*Word budget: ≤2000 words. Rubric weighting: 70% model/approach, 20% deck construction,
10% report clarity — this writeup is structured to match.*

## 1. The game we're building for

The competition's card pool (`ptcg_bot/cards.py` loads 1,267 legal cards from
`EN_Card_Data.csv`) unambiguously identifies the variant as classic **Pokémon TCG,
Standard format, Scarlet & Violet + Mega era** — not TCG Pocket. The evidence is in the
data itself: in-deck Basic and Special Energy cards, the full classic supertype
taxonomy (Basic/Stage 1/Stage 2 Pokémon, Item/Tool/Supporter/Stadium), and SV-era rule
keywords (`Pokémon ex`, `Mega Pokémon ex`, `ACE SPEC`). That fixes the rule spine in
`ptcg_bot/rules.py`, our single source of truth: 60-card decks, 6 prizes, a knocked-out
`Pokémon ex` gives up 2 prizes and a `Mega Pokémon ex` gives up 3, and weakness is a
flat ×2 multiplier. Every downstream module — the deckbuilder, the state model, the
evaluator — reads these constants rather than hard-coding them, and `tools/verify_env.py`
checks them against the live simulator before anything else is trusted.

## 2. Model / approach (70%)

### 2.1 Design target: determinized Information-Set MCTS

The defining difficulty versus a perfect-information game is hidden information and
chance: the opponent's hand, deck order, and prizes are unknown, and coin flips and
shuffles inject randomness. Our target architecture is a determinized Information-Set
MCTS (PIMC + MCTS): sample `K` rules-consistent "worlds" for the hidden state
(`belief.py`), run bounded search independently within each sampled world
(`search.py`), and vote across worlds at the root so the move chosen is the one that is
best *in expectation over hidden states* — exactly the robustness the rubric's "no
over-reliance on initial states/matchups" criterion rewards. Rollouts inside each
search are scored by a fast, hand-crafted, explainable heuristic rather than a
black-box network, so the rationale for every decision stays inspectable. This is a
staged plan — Heuristic → Tuned → Search → (optional) Learned — where every stage is a
valid, submittable agent in its own right.

### 2.2 What is actually built today: the Heuristic phase

Search and belief sampling are not implemented yet (see §5, Honesty). What exists,
tested, and verified against the live engine is the full **Heuristic phase**:

**Internal state model (`ptcg_bot/state.py`).** An immutable `GameState` — our own
representation of a battle position, independent of the simulator's wire format. It
tracks both players' active and bench Pokémon, attached energy, hand/deck sizes, and
remaining prizes, plus a same-module energy-payment check (`can_pay`) that resolves
whether a Pokémon's attached energy can cover an attack's typed + Colorless cost. Being
engine-independent, this model — and the evaluator built on it — could be designed and
unit-tested well before the (rules-gated) simulator was available.

**Explainable evaluator (`ptcg_bot/evaluate.py`).** A hand-crafted state-value
heuristic, `state_value(state) -> float`, scored from the acting player's perspective.
Every term is a named, weighted contribution (`explain()` returns the full breakdown
for inspection or tuning), with weights centralized and env-overridable in
`ptcg_bot/config.py`:

- **Prize lead dominates** (`W_PRIZE_LEAD = 100`) — net prize difference is worth far
  more than any other term, because taking and denying prizes is literally how the game
  is won.
- Board-shape terms carry the rest of the signal: own/opponent active HP, own bench HP,
  Pokémon in play, attached energy, hand size, and bench Basics (evolution potential).
- **Attack readiness** (`W_ATTACK_READY = 25`) and **type advantage** (weakness match,
  `W_TYPE_ADVANTAGE = 18`) are scored explicitly, not left implicit in HP totals.
- Terminal positions are clamped to `±1e6` so a genuine win/loss always dominates
  accumulated heuristic value.

Nothing here is a black box: the writeup and any judge can read `config.py` and see
exactly what the agent values and by how much.

**Engine adapter (`ptcg_bot/engine_adapter.py`).** Translates the live simulator's raw
observation dict into our `GameState`, with zero dependency on the engine's own `cg`
package — it's pure standard library, parsing `obs["current"]["players"][i]` fields
directly. This keeps the module importable (and testable) even when the engine isn't
on the machine, and satisfies the no-import-egress constraint in `SECURITY.md`.
Ambiguous energy types (Rainbow, Team Rocket) are conservatively mapped to Colorless —
a safe under-approximation that never over-claims a typed requirement.

**The agent (`ptcg_bot/main.py`).** `agent(obs_dict) -> list[int]` is the full engine
contract: return indices into `obs["select"]["option"]`. The current policy is
**attack-first**: if any attack option is available, take the highest-damage one
(damage looked up from the engine's own attack metadata via `ptcg_bot/metadata.py`);
otherwise develop the board in priority order (ability, attach energy, evolve, play a
card); otherwise end the turn. Retreat is intentionally never chosen. A try/except
wraps every decision so a bad or unexpected observation always degrades to a legal
fallback selection rather than raising — an uncaught exception would forfeit the game,
so `main.agent` is built to never crash, verified by driving full games against the
live engine to a decided result.

### 2.3 Why attack-first, not just "an" agent

This policy wasn't picked by intuition — it was the result of measuring an earlier,
worse one. Our first working policy developed the board (ability → attach → evolve →
play) and only attacked once nothing else was available. Measured with the self-play
harness (`tools/tournament.py`, which drives the real `libcg` engine end to end) over
games against a random legal-move baseline, that develop-first policy **lost decisively**
(representative run: 20.8% of decided games). Switching the priority so the agent attacks
whenever it legally can — before developing further — flipped the same board evaluation
logic to **consistently winning** (representative run: 24/24). The engine's internal RNG
is un-seeded, so exact percentages vary run to run (attack-first lands roughly 75–100% vs
random, develop-first roughly 5–25%); the qualitative gap is large and stable. The lesson, which is the honest
finding of this project so far, is that over-developing the board while sitting on a
usable attack was the dominant failure mode, not attack *selection*. That is exactly
the kind of result a heuristic-only, well-instrumented agent can surface cheaply before
any search is added — and it's why the harness, not intuition, drove the fix.

### 2.4 Robustness and safety

The whole agent stack sits behind an enforced quality and security gate: `ruff`,
`isort`, and `black` for style; `mypy` for static types; `bandit` and `pip-audit` for
static and dependency-CVE scanning; and 60 pytest tests (unit tests with synthetic
observations plus data/engine-gated integration tests) — all run by `make ci` and
mirrored in GitHub Actions on every push. `make audit` greps the bundled agent
package for forbidden `socket`/`urllib`/`requests`/`subprocess` imports, enforcing the
no-network-egress rule at the source level, not just by policy. `tools/bundle.py`
packages a self-contained submission folder (`main.py` + `deck.csv` + `ptcg_bot/` +
the provided `cg/`) and self-checks it by importing the folder in a subprocess with the
repository off `sys.path` — proving the shipped agent needs nothing but the standard
library and its own files.

## 3. Deck construction (20%)

`deckbuilder/` builds a legal, competitive 60-card deck offline from the same
1,267-card pool, entirely deterministically (no RNG — same pool in, same deck out, so
results are reproducible and testable):

- **`archetype.py`** picks a single, focused primary-attacker evolution line rather
  than a speculative multi-type toolbox, ranked by an offensive heuristic: highest
  fixed damage-per-energy, plus HP for durability, minus a prize-liability penalty
  (`ex` costs the opponent 2 prizes on KO, `Mega ex` costs 3 — so a fragile `ex`
  attacker is discounted relative to a resilient single-prize one). The line's primary
  energy type is inferred from its best attack's typed cost.
- **`constraints.py`** enforces legality straight from `ptcg_bot/rules.py`: exactly 60
  cards, at most 4 copies per card name (Basic Energy is unlimited, `ACE SPEC` is
  capped at 1 per deck), and at least one Basic Pokémon so the deck can actually open.
- **`roles.py`** classifies every Trainer and Special Energy card by a transparent
  keyword heuristic over its rules text (draw / search / switch / disruption / heal /
  energy acceleration, plus Stadium/Tool subtypes) — not a full effect parser, but
  enough to assemble a balanced, explainable Trainer package and to describe *why*
  each card is in the deck.
- **`optimize.py`** assembles the legal 60: the attacker line and its energy base,
  secondary single-prize Basic attackers on the same energy type (to reduce mulligan
  risk and raise consistency), and a role-balanced Trainer package filled in priority
  order (draw, then search, then switch, then energy acceleration, then other).
- **`score.py`** is an offline, explainable deck-quality heuristic (0–100) — consistency
  (draw+search density), energy ratio, opening reliability (Basic count), attacker
  power, and prize-safety (exposure to multi-prize KOs) — used as the fitness signal
  until the same tournament harness can score decks by real self-play win-rate instead
  (an explicit, documented placeholder — see `ASSUMPTIONS.md`).

The resulting deck is a single-attacker-line Water archetype backed by ~12 Basic
Energy and a handful of single-prize secondary Basics on the same energy type, filled
out with a role-balanced Trainer package. It scores roughly 97–98/100 on the offline
heuristic, and — more importantly — the **live engine accepts it outright**
(`errorType=0` from `game.battle_start`, checked by `tools/verify_env.py`), so deck
legality is not just asserted by our own validator but confirmed against the actual
simulator the agent will be judged in.

## 4. Report and evidence (10%)

Every claim above is backed by a runnable measurement, not a guess: `make ci` for the
quality/security gate, `make verify` for live-engine deck acceptance and observation
round-tripping, and `make tournament` for the self-play win-rate numbers. The figures
referenced in this writeup's Media Gallery (`writeup/figures.py`, run via `make
figures`) are generated directly from this codebase — deck composition, the deck-score
breakdown, and the develop-first-vs-attack-first win-rate comparison — and are original
charts, not card art, in compliance with the competition's media rules.

## 5. Honesty: what this is, and isn't, yet

The win-rate is measured **against a random legal-move baseline** — a useful, cheap
signal for catching gross policy errors (which it did), not a claim of strength against a
competent opponent. Because the engine's RNG is un-seeded, the figures are representative
single runs, not fixed constants (attack-first ≈75–100% vs random).

A first cut of the IS-MCTS layer **is now implemented** — a one-ply determinized lookahead
(`ptcg_bot/search.py` + `belief.py`) over the engine's `search_begin/step/end` hooks — and
was A/B-tested against the shipped heuristic. It **currently loses** (≈15–45% vs the
heuristic): with a single determinized world and a purely positional leaf evaluator, one-ply
search drifts back toward over-developing. So the heuristic remains the shipped agent, and
the honest next step is scaling to multi-world PIMC with a tempo-aware evaluator. Runtime
card metadata (building the pool from the engine's own card data rather than our offline
CSV, absent in the Kaggle runtime) is also a tracked follow-up in `ASSUMPTIONS.md`.

## 6. What's next

In priority order: (1) wire `legal.py` (legal-move generation over `GameState`) so
search has something to search over; (2) `belief.py` to sample hidden-state worlds
from public information and a deck prior; (3) `search.py`, bounded MCTS per world with
`evaluate.state_value` as the rollout leaf, aggregated by PIMC voting at the root; (4)
re-run `tools/tournament.py` against stronger baselines (not just random) to measure
the actual lift from search over the attack-first heuristic; (5) `tools/tune.py`
coordinate-descent tuning of the weights in `config.py` against that stronger signal.
Each stage remains a valid, submittable agent on its own, so risk to the Simulation
deadline is bounded even if search doesn't land in time.

---

*Word count target: ≤2000 words (this document, excluding this notice and headings
metadata).*
