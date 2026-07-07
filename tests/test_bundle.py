"""Test the submission packager (tools.bundle).

Integration only: bundling needs the deckbuilder (competition data) and the
engine's cg/ to copy. Skips when either is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CG = _ROOT / "engine" / "sample_submission" / "sample_submission" / "cg"


@pytest.mark.integration
def test_bundle_produces_self_contained_submission():
    if not (_CG / "api.py").exists():
        pytest.skip("engine not present (run `make engine`)")

    from ptcg_bot.cards import DEFAULT_CSV

    if not DEFAULT_CSV.exists():
        pytest.skip("competition dataset not present (run `make data`)")

    from tools import bundle

    # Explicit empty argv (bare main() would parse pytest's own sys.argv).
    bundle.main(
        []
    )  # runs the subprocess self-check; raises SystemExit if not standalone

    sub = _ROOT / "dist" / "submission"
    assert (sub / "main.py").exists()
    assert (sub / "ptcg_bot" / "main.py").exists()
    assert (sub / "ptcg_bot" / "engine_adapter.py").exists()
    assert (sub / "cg" / "api.py").exists()
    assert (_ROOT / "dist" / "submission.zip").exists()

    deck_lines = [
        ln for ln in (sub / "deck.csv").read_text().splitlines() if ln.strip()
    ]
    assert len(deck_lines) == 60
    assert all(ln.lstrip("-").isdigit() for ln in deck_lines)

    # deckbuilder must NOT ship; caches must be stripped from the zip'd folder.
    assert not (sub / "deckbuilder").exists()
    assert not list(sub.rglob("__pycache__"))
