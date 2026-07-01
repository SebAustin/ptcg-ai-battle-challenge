# Assumptions — Deckbuilder pass

Documented decisions made under `/agency` full autonomy for the offline deckbuilder
(`deckbuilder/`, the Strategy 20% deliverable). Revisit any of these when the official
simulator / competition page resolves the open items.

1. **Offline fitness proxy.** Decks are meant to be scored by the same tournament harness as the
   agent (`deckbuilder/__init__.py`), but that harness is engine-blocked (plan §W2/§W4). Until the
   simulator is wired, `optimize` maximizes an **offline deck-quality heuristic** (`score.py`:
   consistency, energy ratio, opening reliability, attacker power, prize risk, evolution
   completeness). The fitness function is isolated so it can be swapped for real self-play win-rate
   later without touching archetype/constraints.

2. **`deck.csv` schema.** SUBMIT.md says to confirm exact packaging on the competition page. We emit
   a simple, adaptable `Card ID,Card Name,Count` (one row per distinct card). `Deck.to_csv` is the
   only place this is defined, so re-targeting the engine's expected format is a one-function change.

3. **Card model extension.** The loader dropped Trainer / Special-Energy rules text (rows with an
   empty `Move Name` were skipped). We add `Card.text` (the head row's `Effect Explanation` when a
   card has no moves) so trainers can be classified by what they do. Pokémon text stays on `moves`;
   Basic Energy has no text. Behavior-preserving for all existing fields.

4. **Trainer roles are keyword-heuristic.** `roles.py` classifies trainer/energy text by keywords
   (draw / search / switch / disruption / heal / energy-accel) plus subtype (Stadium / Tool). This
   is a transparent approximation, not a full effect parser — good enough to assemble a balanced,
   explainable package, and easy to inspect for the writeup.

5. **Focused single-attacker archetype.** For consistency and explainability we build around one
   primary attacker line and its primary energy type (the shape the Strategy rubric rewards), rather
   than speculative multi-type toolboxes. `archetype.py` selects the line deterministically by an
   offensive heuristic (best fixed damage per energy, HP, minus prize liability).

6. **Deterministic.** No RNG in construction — same pool → same deck — so results are reproducible
   and unit-testable. (`Math.random`-style variety can be added later once fitness is real.)

# Assumptions — Decision-core heuristics + security scan pass

7. **Engine is published but rules-gated (not a hard block anymore).** As of 2026-07-01 the
   `pokemon-tcg-ai-battle` competition lists the engine (`sample_submission/cg/*.py` + native
   `libcg`, and `ptcg_engine/`). `kaggle competitions files` lists them, but
   `competitions download` returns **403** — the files require accepting that competition's
   rules. Accepting rules is an outward-facing legal action for the **user** to take; the agency
   guardrails forbid doing it automatically. **To unblock W2 wiring:** accept the
   `pokemon-tcg-ai-battle` rules (join it) and download `sample_submission/`; then
   `engine_adapter` is implemented against `cg/api.py` and `tools/verify_env.py` validates it.

8. **The heuristic core is engine-independent, so it was built now.** `ptcg_bot/state.py` (our
   own `GameState`) and `ptcg_bot/evaluate.py` (the explainable state-value heuristic) depend
   only on `rules.py` + `config.py`, not on the wire schema — the adapter maps INTO `state.py`
   later. `evaluate` computes exactly the features `config.py` already names. Still gated on the
   schema (deferred): `sim`, `legal`, `belief`, `search`, `main`, and `engine_adapter` — these
   need the running engine to build/verify against, so they were NOT built speculatively.

9. **Dependency-CVE + static scan wired into the gate.** `make security` runs `bandit` (static,
   on `ptcg_bot`+`deckbuilder`) and `pip-audit` (against `requirements.txt`), and is part of
   `make ci`. The two CVEs it initially surfaced were fixed by bumping the (dev-only) pins:
   `pytest` → `>=9.0.3` (CVE-2025-71176), `black` → `>=26.3.1` (CVE-2026-32274). `pip-audit`
   requires network to fetch its advisory DB.

# Assumptions — Engine wiring (§W2) pass

10. **Engine downloaded; adapter wired.** The `pokemon-tcg-ai-battle` rules were accepted and
    `make engine` fetched `sample_submission/` (the `cg/` Python package + native `libcg`) into
    the gitignored `engine/`. `libcg.dylib` loads and runs on this machine.

11. **deck.csv format corrected.** The real engine deck.csv is **60 lines, one integer card ID
    per line, copies repeated, no header** (engine `main.py` does `int(csv[i])` for `i` in
    `range(60)`). `Deck.to_csv()` now emits that; the earlier `Card ID,Card Name,Count` guess was
    wrong. Verified: the engine ACCEPTS our deckbuilder deck (`errorType=0`).

12. **Adapter parses the raw dict (no `cg` dependency).** The agent contract is
    `agent(obs_dict) -> list[int]` (indices into `obs["select"]["option"]`).
    `engine_adapter.parse_observation` walks the plain dict into our `GameState`;
    `encode_action` is a validated pass-through of option indices. This keeps `ptcg_bot`
    pure-stdlib (no import egress) and importable in CI without the engine. RAINBOW/TEAM_ROCKET
    energy are mapped to Colorless for affordability (safe under-approximation).

13. **Runtime card metadata (deferred).** `parse_observation` takes a `CardPool` for card lookups;
    offline that is `load_pool()` (our CSV). At agent runtime on Kaggle the CSV is absent, so
    `main` will build the pool from the engine's `all_card_data()` instead — a follow-up.

14. **`make verify` is the live gate (§W2-3).** `tools/verify_env.py` loads `libcg`, starts a
    battle with our deck, drives a random legal playthrough, and runs `parse_observation` on
    every real observation. It is engine-gated (needs `make engine`), not part of `make ci`.

15. **Determinized rollout search (PIMC) is built but NOT shipped — now roughly on par,
    not a clear win.** `ptcg_bot/search.py` + `ptcg_bot/belief.py` implement a K-world rollout
    search over the engine's `search_begin/step/end` API: for each MAIN option, run `K`
    determinized rollouts with an attack-first base policy to depth `D`, score the leaf **from
    the fixed root perspective**, and pick the mean-value argmax (PIMC). Measured evolution vs
    the attack-first heuristic (`make tournament --opponent heuristic`, sides alternated):
    - naïve one-ply, single world, leaf-`yourIndex` eval → **~10%** (much worse);
    - + perspective-bug fix (evaluate as the root player), MAIN-only gating, K-world depth-D
      rollouts → **~40–47%** (behind, closing);
    - + **realistic per-world determinization** (`belief.py`: our unseen cards = *decklist
      minus what's visible in the observation*, shuffled per world; opponent hidden cards a
      *varied legal sample*; one determinization per world for genuine PIMC diversity) →
      **~52%** (31W/29L over 60 games — **parity, within noise of 50%**).
    So realistic determinization moved search from behind to level with the heuristic, but
    it is still not a *decisive* win and it is slower — **therefore `agent` (shipped) stays the
    heuristic and `search_agent` remains experimental.** Engine RNG is un-seeded, so all
    figures are run-to-run ranges. The remaining levers to break past parity: deeper/terminal
    rollouts and more worlds (both currently capped for speed — `_WORLDS`/`_DEPTH` in
    `search.py`), a learned opponent deck prior, and a tempo-aware leaf evaluator.

16. **`main.agent` is a v1 heuristic option policy.** `agent(obs_dict) -> list[int]` is
    pure-stdlib and reads the raw observation dict (no `cg` import), so it drops into the
    submission bundle unchanged. Policy: deck request -> 60 IDs from `deck.csv`; MAIN phase ->
    ATTACK-FIRST over OptionType (attack when able, else develop ABILITY>ATTACH>EVOLVE>PLAY,
    else END; never voluntarily RETREAT); other selections -> take the allowed options; a catch-all guarantees a
    legal (never raised) selection. It needs **no card metadata**, so it runs without the CSV.
    Verified: it plays full legal games against the live engine to a decided result. Deferred:
    scoring options with `evaluate.state_value` + the engine's search hooks (`legal`/`belief`/
    `search`), which need a runtime pool built from the engine's `all_card_data`.
