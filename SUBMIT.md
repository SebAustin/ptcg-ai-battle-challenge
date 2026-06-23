# Submitting

Two linked submissions are required (Simulation agent + Strategy writeup), under the **same team**
across both divisions (Rules §2.1.c). Uploading/publishing is **outward-facing — you run it**, not
the tooling.

## Auth (already configured on this machine)

This machine authenticates with the **new Kaggle OAuth flow**:
`~/.kaggle/credentials.json` (a `refresh_token`, not the legacy `kaggle.json`). It requires the
**2.x** CLI, which `make setup` installs into `.venv`. Verify:

```bash
.venv/bin/kaggle config view          # should show auth_method: OAUTH, username: <you>
.venv/bin/kaggle competitions files -c pokemon-tcg-ai-battle-challenge-strategy
```

If that lists the four data files, auth + rules acceptance are good.

## 1. Download the data (one-time)

```bash
make data
```

## 2. Build the agent artifact (once the agent exists)

```bash
make bundle      # writes dist/main.py (frozen, stdlib-only) + tarballs
make check       # tests + engine verify + bundle must all be green first
```

## 3. Submit the Simulation agent

The Simulation division is a **separate competition**: **`pokemon-tcg-ai-battle`**
(deadline **2026-08-16**, ~4 weeks before the Strategy deadline). Its submission is a folder —
`main.py` (the agent) + `deck.csv` (the submitted deck) + the provided `cg/` engine package —
not a single file. Confirm the exact packaging on its page before submitting:

```bash
# .venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle -f <submission.zip> -m "phase-H heuristic"
```

> You must **accept that competition's rules** (join it) before you can download its
> `sample_submission/` engine or submit — see the note at the top of this file.

## 4. Submit the Strategy writeup

The Strategy submission is a **Kaggle Writeup** (≤2000 words) with a Media Gallery, created via the
competition's *New Writeup* button — not a CLI upload. Attach:

- `writeup/WRITEUP.md` content (title, subtitle, analysis).
- Figures from `writeup/figures/` as the Media Gallery (our own charts only — **no card art**).
- Optionally the public code repo / notebook.

Select a **Track** before submitting, then click **Submit** (top-right). Draft/un-submitted
writeups are not judged.

> Reminder: keep the team identical across the Simulation and Strategy divisions, or the entry is
> ineligible (Rules §2.1.c).
