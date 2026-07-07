"""Package the Simulation submission folder (plan §W4).

Produces ``dist/submission/`` — the folder Kaggle wants for the
``pokemon-tcg-ai-battle`` division:

    dist/submission/
      main.py        # entry the harness imports; re-exports ptcg_bot.main.agent
      deck.csv       # our deckbuilder's 60-card deck (one card ID per line)
      ptcg_bot/      # our agent — RUNTIME modules only, pure stdlib
      cg/            # the engine package (provided; required by the harness)

…and zips it to ``dist/submission.zip``. deckbuilder / tools / tests are NOT
shipped. A self-check imports the bundled entry in a subprocess with the repo
OFF sys.path, proving the folder stands alone (stdlib + its own files only).

Run: ``make bundle`` (needs ``make engine`` and ``make data``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DIST = _ROOT / "dist" / "submission"
_ZIP_BASE = _ROOT / "dist" / "submission"
_CG_SRC = _ROOT / "engine" / "sample_submission" / "sample_submission" / "cg"

# The import closure of ptcg_bot.main — the only modules the agent needs at
# runtime. (deckbuilder is a build-time tool and is deliberately excluded.)
_RUNTIME_MODULES = (
    "__init__.py",
    "main.py",
    "search.py",
    "belief.py",
    "features.py",
    "eval_weights.py",
    "metadata.py",
    "engine_adapter.py",
    "state.py",
    "evaluate.py",
    "cards.py",
    "rules.py",
    "config.py",
)

_ENTRY = '''"""Kaggle submission entry — the harness imports this and calls agent()."""

from ptcg_bot.main import agent  # noqa: F401  (re-exported for the harness)
'''


def _build_deck_csv() -> str:
    from deckbuilder import build_deck, validate
    from ptcg_bot.cards import load_pool

    pool = load_pool()
    deck = build_deck(pool)
    problems = validate(deck, pool)
    if problems:
        raise SystemExit("refusing to bundle an illegal deck: " + "; ".join(problems))
    return deck.to_csv()


def _self_check() -> None:
    """Import the bundled entry with the repo OFF the path — proves standalone."""
    env = {**os.environ, "PYTHONPATH": ""}
    proc = subprocess.run(  # noqa: S603 - fixed args, no shell, dev-only tool
        [sys.executable, "-c", "import main; assert callable(main.agent)"],
        cwd=_DIST,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "bundle self-check FAILED — the folder is not self-contained:\n"
            + proc.stderr.strip()
        )


def _read_deck_file(path: str) -> str:
    """A pre-made deck.csv (60 integer lines) to ship instead of the builder's."""
    ids = [
        int(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(ids) != 60:
        raise SystemExit(f"--deck {path}: expected 60 card ids, got {len(ids)}")
    return "\n".join(str(i) for i in ids) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="package the submission bundle")
    parser.add_argument(
        "--deck",
        default=None,
        help="optional path to a 60-line deck.csv to ship instead of the built deck",
    )
    args = parser.parse_args(argv)

    if not _CG_SRC.exists():
        raise SystemExit("engine cg/ not found at engine/ — run `make engine` first")

    deck_csv = _read_deck_file(args.deck) if args.deck else _build_deck_csv()

    if _DIST.exists():
        shutil.rmtree(_DIST)
    (_DIST / "ptcg_bot").mkdir(parents=True)

    for name in _RUNTIME_MODULES:
        shutil.copy(_ROOT / "ptcg_bot" / name, _DIST / "ptcg_bot" / name)
    shutil.copytree(_CG_SRC, _DIST / "cg")
    (_DIST / "main.py").write_text(_ENTRY, encoding="utf-8")
    (_DIST / "deck.csv").write_text(deck_csv, encoding="utf-8")

    _self_check()

    # Strip caches (the self-check + any copied cg/ caches) so the zip is clean.
    for cache in _DIST.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

    archive = shutil.make_archive(str(_ZIP_BASE), "zip", _DIST)
    deck_lines = deck_csv.strip().splitlines()
    print(f"[bundle] wrote {_DIST}/ (self-check passed)")
    print(f"[bundle]   main.py + deck.csv ({len(deck_lines)} cards) + ptcg_bot/ + cg/")
    print(f"[bundle] zipped -> {archive}")
    print("[bundle] submit per SUBMIT.md (outward-facing — you run it).")


if __name__ == "__main__":
    main()
