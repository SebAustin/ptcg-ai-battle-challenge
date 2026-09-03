# Security & Compliance Review — PTCG Battle Agent

Scope: the submitted agent — `ptcg_bot/` as bundled into `dist/submission/` (`main.py` +
`deck.csv` + `ptcg_bot/` + the provided `cg/`) by `tools/bundle.py`. The `tools/` harness, the
`deckbuilder/`, and the `writeup/` are **not** submitted with the agent and are out of scope for
runtime compliance.

> Status: enforced. The constraints below are checked automatically by `make ci`
> (`make audit` import grep + `bandit` static scan + `pip-audit`) and by `make verify` against
> the live engine. `make bundle` self-checks that the shipped folder is stdlib-only self-contained.

## Competition rule compliance (binding constraints)

| Rule | Status | Evidence |
|------|--------|----------|
| **Pure, open-source-compatible deps** | ✅ | Bundled `ptcg_bot/` imports **standard library only** (`bandit` clean; `make bundle` self-check imports the folder with the repo off `sys.path`). `config.py`'s `import os` is stdlib (env-var weight overrides) and permitted — it is not network/process egress. |
| **No ingress/egress during evaluation** | ✅ | No `socket`/`urllib`/`requests`/`subprocess` in `ptcg_bot/`; `make audit` enforces this on every `make ci` run and in CI. `engine_adapter` parses the observation dict directly (no `cg` import). |
| **No reading hidden state** (opp hand, deck order, prizes, RNG seed) | ✅ (v1) | `main.agent` and `engine_adapter.parse_observation` consume only the official observation dict. The opponent's hand is `None` in the observation; we never read it. (Belief sampling for search is future work.) |
| **Deterministic, original work** (§3.14) | ✅ | Heuristics authored for this project; `main.agent` is deterministic. Any RNG lives in the dev harness (`verify_env`), explicitly seeded — never in the shipped agent. |
| **Competition Data not redistributed** (§2.4) | ✅ | `data/*.csv` / `*.pdf` and the engine `engine/` are gitignored; downloaded locally via `make data` / `make engine`. |
| **Pokémon Elements not republished** (§2.5, media) | ✅ | No card images committed; writeup figures are our own charts/diagrams, not card art. |

## Import audit (the agent must stay stdlib-only)

`make audit` runs this on every `make ci`:

```bash
grep -rEn '^[[:space:]]*(import|from)[[:space:]]+(socket|urllib|requests|subprocess)' ptcg_bot --include='*.py'
# expected: no matches. (config.py's stdlib `import os` for env overrides is allowed — not egress.)
```

## STRIDE-lite (sandboxed game agent)

- **Tampering / Information disclosure:** the agent reads only the provided observation; it cannot
  reach the opponent's hidden zones or the RNG seed. Belief sampling reconstructs *plausible* hidden
  state from public info — it never inspects the real one.
- **Denial of service (self):** `main.agent` wraps every decision in a catch-all that always
  returns a legal selection, so a bug or malformed observation can never raise or forfeit (the
  live full-game test exercises this). The v1 heuristic is O(options) with no lookahead, so it is
  well under the turn timeout; when search lands, `config.TURN_DEADLINE_S` bounds it and
  `tools/soak.py` becomes the gate.
