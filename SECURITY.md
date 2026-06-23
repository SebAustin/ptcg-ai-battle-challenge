# Security & Compliance Review — PTCG Battle Agent

Scope: the submitted agent (`ptcg_bot/` → bundled `dist/main.py`). The `tools/` harness, the
`deckbuilder/`, and the `writeup/` are **not** submitted with the agent and are out of scope for
runtime compliance.

> Status: scaffold. This document states the constraints the build is held to; the evidence rows
> are filled in as the corresponding tools (`bundle.py` import audit, `verify_env.py`) land.

## Competition rule compliance (binding constraints)

| Rule | Status | Evidence / plan |
|------|--------|-----------------|
| **Pure, open-source-compatible deps** | Target | Bundled agent imports **standard library only**; `tools/bundle.py` will freeze tuned weights to literals and drop `os`. |
| **No ingress/egress during evaluation** | Target | No `socket`/`urllib`/`requests`/`http`/`subprocess` in the agent. Audit command below runs in CI/`make check`. |
| **No reading hidden state** (opp hand, deck order, prizes, RNG seed) | Target | The agent consumes only the official observation via `engine_adapter.parse_observation`; hidden info is *estimated* by `belief.py`, never read. |
| **Deterministic, original work** (§3.14) | Target | Heuristics authored for this project; any RNG in search is explicitly seeded and confined to the harness, not to hidden-state access. |
| **Competition Data not redistributed** (§2.4) | ✅ | `data/*.csv` / `*.pdf` are gitignored; `data/README.md` documents local download only. |
| **Pokémon Elements not republished** (§2.5, media) | ✅ | No card images committed; writeup figures are our own charts/diagrams, not card art. |

## Import audit (the agent must stay stdlib-only)

```bash
grep -rnE "socket|urllib|requests|http|subprocess|eval\(|exec\(|__import__|pickle" ptcg_bot/ dist/main.py
# expected: no matches (the dev-only config.py `import os` is dropped by bundle.py)
```

## STRIDE-lite (sandboxed game agent)

- **Tampering / Information disclosure:** the agent reads only the provided observation; it cannot
  reach the opponent's hidden zones or the RNG seed. Belief sampling reconstructs *plausible* hidden
  state from public info — it never inspects the real one.
- **Denial of service (self):** a hard per-decision deadline (`config.TURN_DEADLINE_S`) with the
  heuristic 1-ply policy as a never-crash fallback keeps the agent under the engine's turn timeout;
  `tools/soak.py` is the gate that proves it.
