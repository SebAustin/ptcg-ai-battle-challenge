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
