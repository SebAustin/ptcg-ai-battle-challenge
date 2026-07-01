# Media Gallery captions

**deck_composition.png**

Deck composition — the built 60-card deck: a focused attacker line, single-prize support Basics, ~12 Basic Energy, and a role-balanced Trainer package.

**deck_score.png**

Offline deck-quality heuristic (0-100) — the transparent proxy used during construction: consistency, energy balance, opening reliability, attacker power, and prize safety.

**winrate.png**

Why the policy fix mattered: switching the agent from develop-first to attack-first lifted its win-rate vs a random baseline from ~21% to ~100% (representative run; engine RNG un-seeded).

**search_progress.png**

Honest search progress: determinized rollout-PIMC climbed from ~10% to ~52% vs the heuristic (perspective-bug fix + K worlds + realistic determinization) — parity, not a decisive win, so the heuristic still ships.
