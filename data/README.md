# Competition data

The card files are **gitignored** — they are Competition Data and must not be
redistributed (Rules §2.4). Download them locally:

```bash
make data            # all four files (CSVs + the two large PDFs, ~320 MB)
# or just the CSVs the code needs:
.venv/bin/kaggle competitions download -c pokemon-tcg-ai-battle-challenge-strategy -f EN_Card_Data.csv -p data/
.venv/bin/kaggle competitions download -c pokemon-tcg-ai-battle-challenge-strategy -f JP_Card_Data.csv -p data/
```

## Files

| File | Size | Used for |
|------|------|----------|
| `EN_Card_Data.csv` | ~350 KB | **the code** — `ptcg_bot/cards.py` loads this |
| `JP_Card_Data.csv` | ~430 KB | Japanese names (same schema) |
| `Card_ID List_EN.pdf` | ~138 MB | visual card reference |
| `Card_ID List_JP.pdf` | ~182 MB | visual card reference (JP) |

## Confirmed facts (from `EN_Card_Data.csv`, 2026-06 export)

- **1,267 distinct cards** (2,022 CSV rows — multi-attack cards span rows).
- **Variant: classic Pokémon TCG, Standard format, Scarlet & Violet + Mega era**
  (in-deck Basic/Special Energy, Item/Tool/Supporter/Stadium, `Pokémon ex` /
  `Mega Pokémon ex` / `ACE SPEC` rules, HP 30–380). → 60-card deck, 6 prizes.
- Energy/Pokémon types: `{G}{R}{W}{L}{P}{F}{D}{M}{C}` + Dragon (leaks as `竜`).
- Cost notation: `●` = one Colorless requirement; `{X}` = typed energy.
