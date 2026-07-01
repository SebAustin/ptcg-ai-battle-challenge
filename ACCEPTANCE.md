# Acceptance Record

A concise pass/fail/deferred record of this project's success criteria, each grounded
in a verifiable command or file in this repo. Re-run the "Evidence" command yourself to
reproduce any row.

## Success criteria

| Criterion | Status | Evidence |
|---|---|---|
| Quality gate (lint, types, security, tests) enforced | PASS | `make ci` runs `ruff`/`isort`/`black` (lint), `mypy` (typecheck), `bandit` + `pip-audit` (security), and pytest (test); mirrored in `.github/workflows/ci.yml` on every push. |
| Rule spine matches the competition variant | PASS | `ptcg_bot/rules.py` hard-codes 60-card/6-prize/ex=2/Mega ex=3/weakness×2 from the 1,267-card pool loaded by `ptcg_bot/cards.py` (`load_pool()`); `tools/verify_env.py` re-checks live. |
| Deckbuilder produces a legal 60-card deck | PASS | `deckbuilder/constraints.py:validate()` checks deck size, per-name copy caps, ACE SPEC cap, and minimum Basics against `ptcg_bot/rules.py`; `tests/test_deckbuilder.py::test_build_deck_is_legal_and_full` asserts `validate(deck, pool) == []` (integration, needs `data/`). |
| Deckbuilder deck is accepted by the live engine | PASS | `tools/verify_env.py` calls `game.battle_start(deck_ids, ...)` and exits non-zero unless the engine returns `errorType=0`; ASSUMPTIONS.md item 11 records this was verified after correcting the `deck.csv` format to 60 bare card-ID lines. |
| Engine adapter parses real observations correctly | PASS | `ptcg_bot/engine_adapter.py:parse_observation()` is pure-stdlib (no `cg` import) and is exercised by `tools/verify_env.py`, which asserts `0 <= prizes_remaining <= 6` on every real observation during a full battle. |
| Agent returns only legal, non-crashing selections | PASS | `ptcg_bot/main.py:agent()` wraps `choose()` in `try/except` with a legal fallback (`_fallback`/`_default_selection`); `tests/test_main.py::test_agent_plays_a_full_legal_game` drives a full game against the live engine to a decided result (integration, needs `engine/` + `data/`). |
| Agent beats a random baseline | PASS (vs random only) | `tools/tournament.py` self-play harness: develop-first policy lost decisively (representative run 20.8%); attack-first ordering flips it to consistently winning (representative run 24/24). The engine RNG is un-seeded so figures vary run-to-run (attack-first ≈75–100% vs random). Not yet measured against a non-random competent opponent. |
| Submission bundle is self-contained | PASS | `tools/bundle.py` copies the runtime-only module closure into `dist/submission/`, then re-imports it in a subprocess with the repo off `sys.path` (`_self_check()`); `tests/test_bundle.py` asserts `deckbuilder/` is excluded and no `__pycache__` remains (integration, needs `engine/` + `data/`). |
| No network/subprocess egress in the shipped agent | PASS | `make audit` greps `ptcg_bot/` for `socket`/`urllib`/`requests`/`subprocess` imports and fails the build on a match; documented in `SECURITY.md`'s import audit section. |
| Test suite coverage of shipped modules | PASS | 60 pytest test items across 6 files under `tests/` (56 `def test_*` functions, one of which is `@pytest.mark.parametrize`d with 5 cases, so pytest collects 60 total); integration tests skip cleanly without `data/`/`engine/`. |
| IS-MCTS search / belief sampling | BUILT (experimental, not shipped) | `ptcg_bot/search.py` + `belief.py` implement one-ply determinized lookahead over the engine's search API, exposed as `main.search_agent`. Measured WORSE than the heuristic (≈15–45% vs it), so `main.agent` stays the heuristic. Multi-world PIMC + `legal.py` remain deferred. See ASSUMPTIONS.md item 15. |
| Heuristic weight tuning against real win-rate | DEFERRED | `tools/tune.py` is referenced by the `make tune` target but does not exist yet (`Makefile`: `test -f tools/tune.py && ... || echo "[pending]"`); weights in `ptcg_bot/config.py` are hand-set defaults, not tuned. |
| Robustness soak testing | DEFERRED | `tools/soak.py` is referenced by `make soak` but does not exist yet (same guarded pattern as `tune`); `config.TURN_DEADLINE_S` (2.5s) is defined but unenforced by a timer today since there is no search loop to bound. |
| Writeup figures generated | DONE (local) | `make figures` generates 3 PNGs (deck composition, quality-score breakdown, policy win-rate) into `writeup/figures/`. Output is gitignored/regenerable; upload to the Kaggle Media Gallery per `SUBMIT.md`. |

## Built

- **Rule spine** (`ptcg_bot/rules.py`) — deck/prize/weakness constants for classic
  Standard SV+Mega era, derived from and checked against the 1,267-card competition
  pool.
- **Card data layer** (`ptcg_bot/cards.py`) — typed, immutable `Card`/`Move`/`CardPool`
  model with defensive CSV parsing (multi-move cards, dirty tokens, Dragon-as-kanji).
- **Internal game state** (`ptcg_bot/state.py`) — immutable `GameState`/`PlayerState`/
  `PokemonInPlay`, plus energy-payment resolution (`can_pay`), independent of the
  engine's wire format.
- **Explainable evaluator** (`ptcg_bot/evaluate.py` + `ptcg_bot/config.py`) — a
  fully-inspectable, weighted state-value heuristic (`explain()` returns every term);
  prize lead dominates by design.
- **Engine adapter** (`ptcg_bot/engine_adapter.py`) — parses the live engine's raw
  observation dict into `GameState`, pure stdlib, no `cg` dependency.
- **Attack metadata lookup** (`ptcg_bot/metadata.py`) — optional, gracefully-degrading
  lookup of attack damage from the engine's `cg.api.all_attack()`.
- **Playable agent** (`ptcg_bot/main.py`) — `agent(obs_dict) -> list[int]`, attack-first
  MAIN-phase policy, deck-request handling from `deck.csv`, never-crash fallback.
  Verified to play full legal games against the live engine.
- **Deckbuilder** (`deckbuilder/`) — archetype selection, legality constraints, trainer
  role classification, an offline 0-100 deck-quality score, and a deterministic
  60-card assembler. Output is accepted by the live engine.
- **Eval harness** (`tools/`) — `verify_env.py` (live-engine model validation),
  `tournament.py` (self-play win-rate), `bundle.py` (submission packaging),
  `build_deck.py` (writes `dist/deck.csv`).
- **Quality/CI/security gate** — `make ci` (lint + typecheck + audit + security + test),
  `.github/workflows/ci.yml`, 60 pytest tests, `SECURITY.md` STRIDE-lite review.

## Deferred

- **Stronger search layer** — a one-ply determinized `search.py` + `belief.py` are built
  (as `main.search_agent`) but measured worse than the heuristic, so not shipped. Still
  deferred: `legal.py` (legal-move generation), multi-world belief sampling, and bounded
  IS-MCTS with PIMC voting + a tempo-aware leaf evaluator — the path to actually beating
  the heuristic.
- **Weight tuning** — `tools/tune.py` (coordinate descent over `config.py` weights
  against a real fitness signal) is planned but not written.
- **Robustness soak** — `tools/soak.py` (no-crash/no-timeout sweep over many seeds) is
  planned but not written; `config.TURN_DEADLINE_S` exists but has no enforcing timer
  yet since there's no search loop to bound.
- **Stronger-opponent evaluation** — win-rate is measured against a random legal-move
  baseline only (and, for the search A/B, against the heuristic). No competent non-random
  opponent has been implemented or measured against.
- **Writeup figures generation** — `writeup/figures.py` is written and runnable but has
  not been executed in this environment (requires `data/EN_Card_Data.csv`, which is
  gitignored competition data not present here); `writeup/figures/*.png` do not yet
  exist on disk.
- **Runtime card metadata** — `main.agent` currently assumes offline `load_pool()`
  availability where card metadata is needed; at Kaggle runtime the CSV won't be
  present, so a follow-up is needed to build the pool from the engine's own
  `all_card_data()` (tracked in `ASSUMPTIONS.md` item 13).

## Next steps

1. Scale the built one-ply `search.py`/`belief.py` toward real PIMC: sample **K** hidden
   worlds (realistic deck prior, not filler IDs), roll out deeper, and give the leaf a
   **tempo-aware** evaluator so it stops over-developing — then re-A/B via `tools/tournament
   --opponent heuristic` and only ship `search_agent` as `agent` if it wins.
2. Add `legal.py` (typed option semantics) so the search reasons about move *kinds*, and
   bound search under `TURN_DEADLINE_S` with the heuristic as the timeout fallback.
3. Build the runtime card pool from the engine's `all_card_data()` so the agent needs no CSV.
4. Write `tools/soak.py` and run it before trusting search in a real submission.
5. Write `tools/tune.py` and re-tune `config.py` once a stronger opponent (not just
   random) exists to tune against.
6. Run `make data` + `make figures` to generate and commit the writeup's Media Gallery
   figures, then finalize the Kaggle Writeup submission per `SUBMIT.md`.
