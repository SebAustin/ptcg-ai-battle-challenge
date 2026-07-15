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
    it is still not a *decisive* win and it is slower. We then pushed for a decisive win:
    uncapped `_WORLDS`/`_DEPTH` (now env-tunable via `config`, bounded only by the wall-clock
    `TURN_DEADLINE_S`) and swept — **deeper rollouts (depth 30) and more worlds stayed at ~50%.**
    Diagnostic (instrumented one game): search picks a *different* MAIN option from the
    heuristic **65% of the time**, yet the net win-rate is ~50% — so its different choices are
    **neutral, not better**. That localizes the ceiling: it is **the value estimate**
    (leaf evaluator + determinization noise), NOT search depth/breadth. Flat rollout with an
    attack-first *base policy* provably converges toward that base policy's own value, so more
    rollouts can't exceed it by much. **Therefore `agent` (shipped) stays the heuristic and
    `search_agent` remains experimental at parity.** Engine RNG is un-seeded → figures are
    run-to-run ranges. The genuine path to a decisive win is a **stronger value function** — a
    learned leaf evaluator (train on self-play) rather than the hand-crafted positional one —
    which is a separate, larger effort with uncertain payoff given the heuristic already wins
    ~100% vs random. That is the honest stopping point for the hand-crafted search line.

16. **`main.agent` is a v1 heuristic option policy.** `agent(obs_dict) -> list[int]` is
    pure-stdlib and reads the raw observation dict (no `cg` import), so it drops into the
    submission bundle unchanged. Policy: deck request -> 60 IDs from `deck.csv`; MAIN phase ->
    ATTACK-FIRST over OptionType (attack when able, else develop ABILITY>ATTACH>EVOLVE>PLAY,
    else END; never voluntarily RETREAT); other selections -> take the allowed options; a catch-all guarantees a
    legal (never raised) selection. It needs **no card metadata**, so it runs without the CSV.
    Verified: it plays full legal games against the live engine to a decided result. Deferred:
    scoring options with `evaluate.state_value` + the engine's search hooks (`legal`/`belief`/
    `search`), which need a runtime pool built from the engine's `all_card_data`.

17. **Deck fitness tuning by self-play is built; no deck robustly beats the default.**
    `tools/tune.py` (`make tune`) generates candidate decks along two axes — attacker-line
    **rank** × **energy count** — and scores each by **real win-rate vs the default**, both
    sides piloted by the heuristic (fast, ~0.15 s/game), sides alternated. Findings:
    (a) the offline `score.py` is a **weak predictor** — an offline-100 deck won only ~53% while
    an offline-97.5 deck spiked to ~70% in one sample; (b) but that spike was **noise** — the
    un-seeded engine RNG makes 30-game evals unreliable, and the top candidate regressed to
    **48.8% over 80 games** (parity). So under the heuristic pilot the rank×energy variants are
    all ~equal, **no candidate robustly beats the default, and the default deck is kept.**
    The tuner is the tool for future deck search but needs large samples; its own verdict now
    warns about small-sample noise and asks to confirm any leader with a large `--games` run.

# Assumptions — Live-ladder diagnosis + v3 card-aware policy

18. **Live Kaggle diagnosis (2026-07-04): the agent runs clean but was too weak for the
    ladder.** Submission 54243849 (v2, 2026-07-01) scored **192.7** vs a 1224.9 leader. Episode
    evidence (replays via the Kaggle CLI): every episode COMPLETED, all agent statuses DONE —
    **no errors or timeouts** — and ~46% win-rate (6W/7L sampled) in its low-rating bracket.
    Replay inspection showed the v2 agent answered almost every non-MAIN prompt with option
    `[0]` (first option) while opponents made deliberate choices. Diagnosis: placement losses
    against mid-tier agents sank the rating; "first option everywhere" is the biggest gap.

19. **v3 card-aware policy (shipped).** MAIN: prefer the *cheapest lethal* attack (damage >=
    opponent Active's remaining HP) over raw max damage; energy/evolution target the Active.
    Card selections resolve each option's `area/index` against the observation and rank by the
    card behind it (engine metadata `card_power`): strongest for setup/switch/promote/fetch
    contexts, weakest for discard/to-deck/to-prize. Offline (no `cg`) everything degrades to
    the legacy first-option behavior; never-crash preserved. Measured: **53.3%** vs the v2
    `legacy_agent` (60-game mirror A/B, sides alternated) and 100% vs random; bundled runtime
    verified standalone (card_power loads, full legal games). Mirror A/Bs understate
    fundamentals vs *different* decks, but per our own discipline we do not promise a specific
    rating gain. Note: each Kaggle submission is rated separately, so resubmitting also gets a
    fresh placement rather than the 192-anchored rating.

20. **v4: charge-then-feed energy + gated retreat (shipped).** Two mirror-blind fundamentals:
    (a) once the Active has the energy its best attack needs (`metadata.attack_cost`), energy
    goes to the strongest un-charged bench Pokémon (builds the next attacker instead of
    overcharging); (b) retreat fires only when the Active cannot attack but a charged attacker
    waits on the bench (the ranked SWITCH pick then promotes it). Measured: 52.8% aggregate vs
    v2 over 180 mirror games (statistically level with v3's 54.4% — mirror A/Bs cannot see
    retreat, which fired ~0.1x/game there) and 100% vs random; offline both features disable
    (no metadata -> legacy behavior); never-crash preserved. Shipped on not-worse evidence +
    strategic rationale; no specific rating gain promised.

# Assumptions — Learned evaluator (gen0) pass

21. **Learned win-probability evaluator: the first DECISIVE win — search is now the shipped
    agent.** Pipeline: `tools/selfplay.py` generated 30k games (~1.78M MAIN-decision positions;
    pilot mix 50% mirror / 35% eps-heuristic / 15% random; 30% opponent-deck variants);
    `tools/train_eval.py` (numpy, dev-only) trained LR + a 32-unit tanh MLP — MLP won (val
    logloss 0.458 vs 0.693 base, AUC 0.867, near-diagonal calibration) — and exported
    `ptcg_bot/eval_weights.py` (pure-stdlib literals, ~1.3k weights). `search.py`'s leaf is now
    win-probability-scaled: terminal 1/0/0.5, else `eval_weights.predict(features.extract(obs,
    me))` (FEATURE_COUNT skew guard; offline falls back to the sigmoid-squashed heuristic).
    `ptcg_bot/features.py` (41 public-info features, dict+dataclass tolerant) is shared by
    training and inference. Screens: depth-0 45%, depth-8 70% — the rollout resolves tactics,
    the model values the outcome. **Promotion gate: pooled 78.5% over 200 mirror games (5x40:
    82.5/85/65/72.5/87.5, sides alternated) vs the v4 heuristic — every run >= 65%.** Worst
    decision 0.15s vs the 2.5s budget. `agent` = search+learned (heuristic fallback for
    non-MAIN/offline/error paths; never-crash preserved); the old policy ships as
    `heuristic_agent` for A/B. Ladder rating gain still not promised — but this is the first
    change that decisively beat its predecessor under the house gate. Next: gen1 self-play with
    the promoted pilot.

22. **Gen1 self-play iteration: honest plateau — gen0 weights kept.** Gen1 data = 12k games
    with the promoted search pilot at reduced budget (`PTCG_SEARCH_WORLDS=4`,
    `PTCG_TURN_DEADLINE_S=0.5`) mixed with ~20% gen0 heuristic-pilot shards (961k + ~450k
    rows; `--id-offset` added to selfplay to avoid game-id collisions across generations).
    Gen1 MLP: val logloss 0.507, AUC 0.831 on the harder search-pilot distribution
    (not comparable to gen0's raw 0.458/0.867). **Gate (mechanical, pre-registered): ship at
    pooled >= 80% vs heuristic_agent over 200 games. Result: 70/77.5/80/65/67.5 -> pooled
    72.0% — below the bar and below gen0's 77.3% band.** Reverted to gen0 weights
    (`git checkout`), confirmed GEN=gen0 + sanity A/B 72.5%. Interpretation: one generation of
    naive self-play iteration did not improve this evaluator (distribution shift without
    stronger supervision); the shipped v5 remains gen0. Future gens should change something
    structural (deeper features, more data from *mixed* strength pilots, or tree reuse)
    rather than re-running the same loop.

23. **Meta intelligence: the deck was the bottleneck — v6 ships a netdecked meta list.**
    With v5 at ~300, loss forensics over 31 live replays showed we took the FIRST prize in all
    22 losses and 12 were decided by <=1 prize: strong opening, losing close endgames — prize
    economics, i.e. a deck problem. The replays include both players' decklists (step-1
    actions), so we extracted the 22 real decks that beat us (`data/meta_decks.json`,
    gitignored) and measured: EVERY deck our builder makes loses ~80% to that meta set (best
    candidate 22.7%, shipped deck 20.5%) under our own strong pilot — the deck, not the agent,
    was the ceiling. Round-robin of the 22 meta decks under our pilot crowned a **Marnie's
    Grimmsnarl ex** list (71.4% RR), which scores **54.5% vs the meta set** (+34 points over
    ours) and **80% head-to-head vs our deck**. v6 = same v5 agent + this deck
    (`make bundle ARGS="--deck data/best_meta_deck.csv"`).
    **Disclosure:** the decklist is netdecked from public replays of our own ladder games —
    standard competitive practice; the Strategy writeup keeps OUR deckbuilder as the original
    deck-construction contribution and will disclose that the Simulation entry runs an
    observed meta list for win-rate. Deck data files stay gitignored (competition-derived).

24. **Gen2 (meta-vs-meta training) + Dwebble/Crustle deck = v7 — both pre-registered gates
    cleared.** Meta set refreshed from v6's bracket (22 -> 59 unique real decklists; v6
    forensics: 16W/30L, first blood OURS in all 30 losses, 17 close — same endgame signature).
    `selfplay --meta-decks` samples BOTH sides' decks from the real meta, so gen2 trained on
    the distribution the agent actually faces (12k games + 20% gen0 mix; val 0.570/0.773 on
    the much harder meta distribution). Gates: G1a vs 59-deck meta set — gen0 baseline 55.9%,
    bar 63.9%, **gen2 pooled 64.8% over 236 games — PASS**; G1b vs heuristic — **76.3%**,
    in gen0's band, no regression. G2 deck refresh under the gen2 pilot: screen found four 6-0
    challengers vs Grimmsnarl; confirm round crowned **deck22 (Dwebble/Crustle)** at **90% H2H
    (18-2)** and **78.0% vs the meta set (+13.2 over Grimmsnarl)** — both G2 conditions
    cleared. v7 = gen2 weights + Dwebble/Crustle list (netdecked; same disclosure as §23).
    make ci green (pipefail); bundle self-check; standalone 6/0. The flywheel (refresh meta ->
    retrain -> re-gate -> re-deck) is now the documented, repeatable improvement loop.

25. **Flywheel turn 3: three honest negatives — v7 stands.** Meta refreshed from v7's
    685-bracket episodes (59 -> 102 unique decks; v7 record 32W/28L, the first winning band;
    losses keep the first-blood-ours 28/0 + 15-close signature). All three levers failed their
    pre-registered gates and were mechanically rejected:
    (a) **worlds=32**: 85.7% vs 85.1% baseline on the 102-deck set (+0.6 < +5 bar; also ~2x
    slower per game) — worlds stays 12;
    (b) **gen3** (12k games on the 102-deck meta): 86.2% vs 85.1% (bar 93.1%) — reverted to
    gen2; second confirmation (after gen1) that re-running the loop without changing the
    training DISTRIBUTION plateaus;
    (c) **deck65 (Cinderace)**: 90% H2H vs our Dwebble/Crustle but only 77.8% vs the broad
    meta (ours: 85.1%) — the G2 AND-rule (H2H >= 60% AND meta >= +8) correctly rejected a
    rock-paper-scissors COUNTER that would lose the field to win the mirror.
    Notes for gen4's pre-registration (decided now, before any gen4 numbers): the +8-POINT
    bars are ceiling-compressed at an 85% baseline; future gates should use relative error
    reduction (e.g. ship if losses-vs-meta drop >= 25%: 85.1% -> >= 88.8%). The next
    structural levers: features v2 (evolution-line potential, energy tempo, per-archetype
    opponent conditioning), search tree reuse across decisions, and mixed-strength pilots.

26. **Flywheel turn 4: features v2 gated OUT at exactly zero gain — v7 stands.** The §25
    structural lever, built and measured end-to-end: 20 new public-board features appended
    to the 41 (evolution stage + headroom via `evolvesFrom`, KO prize liability ex=2/megaEx=3
    — aimed at the persistent "first blood ours, lose close" signature — energy-tempo
    deficits, retreat cost, discard composition), new lazy `cg` metadata maps, 12k
    meta-vs-meta self-play games (945k rows, 61 features) and a gen4 MLP (val logloss
    0.5742, AUC 0.785, well calibrated). Measurement moved to a COMMITTED harness,
    `tools/pilot_gate.py` (turn 3 used a scratch script since lost): our deck vs every meta
    deck, same pilot both sides, 2 games/deck, PTCG_TURN_DEADLINE_S=0.5. Same-day v7
    baseline on this harness: **165W/38L = 81.3%/203** (recorded turn-3 number was 85.1%,
    ~1.4 SE apart — harness/noise; the bar was re-anchored per the §25 relative rule BEFORE
    training: 25% loss reduction = **86.0%**). Gen4 G1a: batch1 83.7%/203 (inside the +/-3
    band -> one pre-authorized extra batch), batch2 78.8%/203 — **pooled 330W/76L =
    81.3%/406 — identical to baseline; FAIL, reverted**. Honest caveats logged: (i) the
    self-play pilot ran with the learned leaf skew-guarded off (41-dim gen2 weights vs
    61-dim features), i.e. a heuristic-leaf bootstrap — same class as gen0, which DID ship,
    so bootstrap alone doesn't explain zero effect; (ii) better calibration did not
    translate into better play — the search's decisions are evidently not bottlenecked on
    the leaf's knowledge of prize liability/tempo at 4-world/0.5s budgets. Work preserved on
    branch `wip/features-v2-gen4` (a gen5 two-stage retrain — pilot USING gen4 weights —
    stays a candidate for when the meta distribution next changes). Remaining §25 levers:
    search tree reuse across decisions; mixed-strength pilots. Fourth consecutive
    confirmation that only training-DISTRIBUTION changes have shipped.

27. **Turn 5 pre-registrations + the daily-publish policy (user-directed).** Decided BEFORE
    any turn-5 gate numbers exist. (a) **gen5 two-stage bootstrap**: 12k meta-vs-meta games
    generated in a worktree of `wip/features-v2-gen4` with the pilot now USING the gen4
    learned leaf (61==61; addresses §26 caveat (i)); gen5 trains on those rows; ships iff
    pooled `tools/pilot_gate.py` >= 86.0% (the §25 relative rule vs the same-harness 81.3%
    baseline) over >= 400 games (204-game batches; one extra batch only if batch 1 lands
    within +-3 of the bar) AND >= 65% vs heuristic. (b) **rollout-depth sweep** (search
    lever; §26 pointed at the search, not the leaf): PTCG_SEARCH_ROLLOUT_DEPTH 16 and 24 at
    worlds=12 on v7 code — deeper rollouts resolve close endgames to EXACT terminal values
    (our loss signature is close endgames); same bar, plus worst-decision timing < 2.0s at
    ship budget. One lever ships at a time; if both gate, the combination must re-confirm
    with its own 400-game run. (c) **Daily publish** (user: "publish every day a better
    version on kaggle"): each daily run submits EXACTLY ONE bundle — the gated winner when a
    gate passes; otherwise the current best bundle re-submitted unchanged and labeled
    "duplicate for episode collection" (parallel copies climb independently and roughly
    double replay/meta data per day; "better every day" is the goal, but ONLY gates decide
    what ships as new). Honest framing preserved: no gate, no version bump.

28. **Turn 5 results: both levers gated OUT — v7 stands; five same-distribution negatives.**
    (a) **gen5 two-stage bootstrap** (12k games, pilot USING the gen4 leaf; 870k rows; val
    calibration clean): G1a batch1 84.3%/204 (within +-3 -> extra batch), batch2 81.3%/203 —
    **pooled 337W/70L = 82.8%/407 vs bar 86.0% — FAIL** (+1.5 over baseline ~= 0.8 SE).
    §26 caveat (i) is now answered: the bootstrap pilot was NOT the reason gen4 flatlined.
    (b) **Rollout-depth sweep** on v7 code: depth16 82.6%/201, depth24 79.2%/202 — both
    outside the +-3 band low, **rejected at screen** (depth trades worlds for plies and
    LOSES at the 0.5s gate budget; monotone worse with depth). Honest confound logged: the
    harness runs at PTCG_TURN_DEADLINE_S=0.5 while the shipped agent gets 2.5s — a
    ship-budget depth test would need a fresh 2.5s baseline + screen (~3.5h) and stays in
    the lever backlog. Score of the day: five consecutive same-distribution negatives
    (gen1, gen3, worlds=32, features-v2/gen4, gen5 + depth) vs zero — the flywheel's only
    proven fuel remains NEW META DATA (episodes from the live ladder). Daily-publish policy
    (§27c) therefore submitted a v7 duplicate today to double the episode collection rate.
    Lever backlog for future turns: depth-at-ship-budget (needs 2.5s baseline), mixed
    opponent-strength pilots in self-play, meta refresh whenever >=15 new episodes land.

29. **Turn 6: scouted the summit — harvester shipped, both levers gated OUT, and the real
    bottleneck identified.** Built `tools/harvest.py` (committed; replaces the lost §23
    scratch pipeline): public `ListEpisodes(submissionId)` reveals both agents' submission
    ids + ratings per episode, so a greedy ladder-crawl reached the 1100+ bracket from our
    v7 seed in TWO hops (317 submissions known, best 1272). Pulled 320 top replays; deck
    extraction validated on our own episode (exact deck.csv recovery). **47 of 49 unique
    1000+-rated decks were absent from our 102-deck set** — we had never seen the top meta.
    Results, all mechanical: (a) same-day baseline 175W/28L = **86.2%/203** (vs 81.3%
    yesterday, identical code — ~5-point day-to-day harness variance is real; same-day
    anchors are mandatory); (b) **deck lever**: screen found ONE challenger (deck4, 5/6),
    20-game H2H confirm **7/20 = 35% — G2 FAIL** (screen was noise; 48/49 top decks LOSE to
    Dwebble/Crustle under OUR pilot — deck strength is pilot-relative, third confirmation);
    (c) **gen6** (12k games on the 149-deck merged meta): G1a batch1 165W/37L =
    **81.7%/202** vs bar 89.7% — FAIL outside the band, reverted to gen2. Interpretation:
    training on decks our bracket never fields DILUTED fit to the deployment distribution;
    "distribution change" only pays when it matches what we actually face (gen2 did).
    **Forensics (399 top agent-games, >=1150):** MAIN mix PLAY 42.7 / ATTACH 15.9 / END
    15.7 / ATTACK 10.4 / ABILITY 6.6 / EVOLVE 6.3 / RETREAT 2.5, avg 141 steps — vs our
    attack-heavy short turns (ATTACK ~30%, no ability/evolve/retreat in sample; 1-game
    caveat). The gap to 1200+ is BEHAVIORAL: trainer-driven card advantage and long
    developed turns, which no evaluator retrain can add while the pilot's action policy
    and rollouts stay attack-first. Backlog, in order: (1) develop-before-attack MAIN
    policy + rollout policy (code lever, gateable same-day); (2) within-turn action-
    sequence search; (3) sub-selection (non-MAIN) quality; (4) depth-at-ship-budget.
    Daily publish (§27c): v7 duplicate submitted (no gate passed).

30. **Turn 7 (cron): own-bracket refresh + gen7 — FAIL; the evaluator lever is formally
    exhausted.** Refresh via `tools/harvest.py`: 260 own replays (v7 134 + two duplicates)
    -> 150 unique decks seen -> **meta_decks.json 102 -> 202** (100 new; the 102 set backed
    up to meta_decks_102_backup.json). The bracket has churned hard past v7 (rating slide
    691 -> 640 corroborates). Same-day baseline on the refreshed set: 324W/78L =
    **80.6%/402** (vs 86.2% on the stale set — the new decks are harder). Gen7 (12k games,
    deployment-matched distribution, the exact gen2 recipe): G1a 321W/79L = **80.2%/400**
    vs bar 85.5% — FAIL below the band at batch 1, reverted to gen2. Verdict after SEVEN
    consecutive negatives (gen1, gen3, worlds, features-v2/gen4, gen5, gen6, gen7 —
    same-loop, two-stage, richer features, top-bracket data, own-bracket refresh): at
    4-world/0.5s-2.5s budgets the leaf evaluator is SATURATED; retraining cannot move game
    outcomes regardless of distribution. gen2 shipped when the baseline was ~64%; at ~80%
    vs own bracket there is nothing left for a leaf to add. The one lever with mechanistic
    evidence behind it is BEHAVIORAL (§29 forensics: top bots PLAY 42.7%/ATTACK 10.4% over
    141-step turns vs our attack-first 30%): develop-before-attack MAIN + rollout policy,
    then within-turn sequence search. Next turn should run that code lever FIRST (gate:
    same-day baseline on the 202 set, relative rule, one 404-game batch +-3 extra).
    Daily publish (§27c): v7 duplicate submitted.

31. **Turn 8 (cron): develop-before-attack — gated OUT at -8.2 vs baseline; frequency is
    not quality.** The §29/§30 behavioral lever, implemented minimally (lethal attacks kept
    on top; otherwise ability/attach/evolve/play before the closing attack, in BOTH
    `_choose_main` and the search rollout policy; 84 tests green incl. two updated priority
    specs). Same-day baseline (v7 HEAD, 202-deck set): 335W/67L = **83.3%/402**; bar 87.5%.
    Candidate: 302W/100L = **75.1%/402 — FAIL far below the band**, reverted. Autopsy
    (hypotheses for follow-ups, not conclusions): (a) our ~640 bracket is tempo-driven —
    delaying attacks loses races; (b) imitating top bots' MAIN-mix FREQUENCIES (PLAY 42.7%)
    without their card-choice QUALITY plays more bad trainers, not better turns; (c) the
    rollout change is confounding: at depth 8, ~5 dev actions/turn means rollouts no longer
    reach attack exchanges, blinding the search to aggression — the leaf sees cosmetic
    development instead of resolved trades. Refined backlog (one per future turn):
    (i) dev-first heuristic with rollouts kept attack-first (isolates (c));
    (ii) dev-first everywhere + rollout depth x3 (dev-first needs deeper sims);
    (iii) trainer QUALITY: rank which trainer/ability to play via metadata instead of
    first-of-type (the likely real gap vs 1200+). Daily publish: v7 duplicate submitted.

32. **Turn 9 (cron): meta refresh (202 -> 238) + lever (i) dev-first-heuristic-only —
    gated OUT at -1.6.** Refresh: 86 replays from the two newest duplicates -> 36 new
    decks. Same-day baseline on the 238 set: 385W/90L = **81.1%/475**; bar 85.8%.
    Candidate (§31 lever (i): dev-first in `_choose_main` only, rollouts kept
    attack-first, lethal still on top): 376W/97L = **79.5%/473 — FAIL below the band**,
    reverted. Read together with §31 (-8.2 for the full variant): the rollout confound
    was indeed the bulk of turn 8's damage, but heuristic-side dev-first alone is still
    a small negative — at this bracket, tempo beats sequencing-without-card-quality.
    Backlog remaining: (iii) trainer-QUALITY ranking (metadata-driven choice of which
    trainer/ability to play — the best-motivated lever left); (ii) dev-first + depth x3
    (now weakly motivated). Daily publish: v7 duplicate submitted.

33. **Turn 10 (cron): trainer-quality PLAY ranking (§32 lever iii) — gated OUT at +0.8.**
    Minimal build: `metadata.card_kind()` + `_best_play()` in `_choose_main`'s PLAY step
    (strongest Pokémon -> Supporter -> Item/Tool/Stadium, resolved from our visible hand;
    ordering semantics untouched, first-play fallback offline; 27 unit tests green).
    Same-day baseline on the 238 set: 383W/88L = **81.3%/471**; bar 86.0%. Candidate:
    389W/85L = **82.1%/474 — FAIL below the +-3 band**, reverted. First behavioral lever
    with a POSITIVE sign (+0.8 ~= 0.6 SE, noise-compatible) — kind-ranking the play choice
    neither helps enough nor hurts; the PLAY-choice surface (heuristic fires only when
    search declines: >12 options or sub-selects) is evidently too narrow to move pooled
    outcomes. Backlog now nearly dry: (ii) dev-first + rollout depth x3 remains (weakly
    motivated, ~2h of gate time). The honest strategic read after 10 turns: v7 is at the
    local optimum reachable by single-variable changes under this architecture; the loop's
    standing value is now episode collection (duplicates), meta refreshes, and gate-ready
    infrastructure for when a structural idea (within-turn sequence search over ALL MAIN
    chains; skill-semantics-aware trainer play) is worth a build day.
    Daily publish: v7 duplicate submitted.
