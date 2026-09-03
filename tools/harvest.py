"""Harvest top-bracket Kaggle episodes: crawl -> pull -> extract (dev-only).

Rebuilds the meta-deck pipeline (ASSUMPTIONS §23) as a committed tool, and
extends it beyond our own bracket: simulation episodes are public, and each
episode's metadata names both agents' ``submissionId`` and rating, so we can
LADDER-CRAWL from our own games up to the leaders and netdeck the summit.

Subcommands (typical turn-6 chain):

  crawl    BFS over ``ListEpisodes(submissionId)`` starting from our own
           submission, always expanding the highest-rated unvisited opponent;
           stops when --want submissions rated >= --target-score are found (or
           frontier/hop budget runs out). Writes ``data/top_submissions.json``.
  pull     Download replays (authenticated ``kaggle competitions replay`` CLI,
           polite delay, cached) + per-episode metadata JSON for the chosen
           submissions into ``data/replays/``.
  extract  Decklists live in ``steps[1][agent].action`` (60 card ids; validated
           against our own deck as ground truth). Keep decks whose agent rating
           >= --min-score, dedupe canonically, write ``data/top_decks.json``
           and a merged ``data/meta_decks_v2.json``. The frozen gate set
           ``data/meta_decks.json`` is never modified.

All outputs are competition-derived and stay gitignored (data/). The public
``ListEpisodes`` endpoint is the same service the leaderboard page calls; no
credentials are sent to it (replay downloads use the authenticated CLI).
"""

from __future__ import annotations

import argparse
import json
import subprocess  # noqa: S404 - drives the authenticated kaggle CLI (dev-only)
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"
_KAGGLE = _ROOT / ".venv" / "bin" / "kaggle"
_LIST_EPISODES_URL = (
    "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
)
_POLITE_DELAY_S = 0.8
_DECK_SIZE = 60


# --- pure helpers (unit-tested) ------------------------------------------------


def decks_from_replay(replay: dict[str, Any]) -> list[list[int]]:
    """Both players' 60-card decklists from a replay dict, or [] if malformed.

    The deck registration is each agent's ``steps[1]`` action (validated: it
    reproduces our own deck.csv exactly on our own episodes).
    """
    try:
        step1 = replay["steps"][1]
        decks = []
        for agent in step1:
            action = agent.get("action") or []
            if len(action) == _DECK_SIZE and all(isinstance(c, int) for c in action):
                decks.append([int(c) for c in action])
        return decks
    except Exception:
        return []


def canonical(deck: list[int]) -> tuple[int, ...]:
    """Order-independent identity of a decklist."""
    return tuple(sorted(deck))


def merge_unique(
    base: list[list[int]], new: list[list[int]]
) -> tuple[list[list[int]], int]:
    """``base`` plus decks from ``new`` not already present; returns (merged, added)."""
    seen = {canonical(d) for d in base}
    merged = [list(d) for d in base]
    added = 0
    for deck in new:
        key = canonical(deck)
        if key not in seen:
            seen.add(key)
            merged.append(list(deck))
            added += 1
    return merged, added


def episode_agent_scores(episode: dict[str, Any]) -> list[tuple[int, float]]:
    """(submissionId, updatedScore) per agent of a ListEpisodes episode entry."""
    out = []
    for agent in episode.get("agents") or []:
        sub = agent.get("submissionId")
        if sub is not None:
            out.append((int(sub), float(agent.get("updatedScore") or 0.0)))
    return out


# --- network helpers ------------------------------------------------------------


def list_episodes(submission_id: int) -> list[dict[str, Any]]:
    """Public episode metadata for one submission (empty list on any failure)."""
    body = json.dumps({"submissionId": submission_id}).encode()
    request = urllib.request.Request(  # noqa: S310 - fixed https URL
        _LIST_EPISODES_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode())
        episodes = payload.get("episodes") or []
        return episodes if isinstance(episodes, list) else []
    except Exception:
        return []


def download_replay(episode_id: int, dest: Path) -> Path | None:
    """Fetch one replay via the authenticated CLI; cached; None on failure."""
    out = dest / f"episode-{episode_id}-replay.json"
    if out.exists():
        return out
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, dev-only tool
        [str(_KAGGLE), "competitions", "replay", str(episode_id), "-p", str(dest)],
        capture_output=True,
        text=True,
    )
    time.sleep(_POLITE_DELAY_S)
    return out if (proc.returncode == 0 and out.exists()) else None


# --- subcommands ------------------------------------------------------------------


def cmd_crawl(args: argparse.Namespace) -> None:
    best_score: dict[int, float] = {}
    visited: set[int] = set()
    frontier: dict[int, float] = {int(s): 0.0 for s in args.seed}

    for hop in range(args.max_hops):
        if not frontier:
            break
        # Expand the highest-rated unvisited submission (greedy ladder climb).
        sub = max(frontier, key=lambda s: frontier[s])
        frontier.pop(sub)
        visited.add(sub)
        episodes = list_episodes(sub)
        time.sleep(_POLITE_DELAY_S)
        for episode in episodes:
            for other, score in episode_agent_scores(episode):
                best_score[other] = max(best_score.get(other, 0.0), score)
                if other not in visited:
                    frontier[other] = max(frontier.get(other, 0.0), score)
        found = sorted(
            (s for s, sc in best_score.items() if sc >= args.target_score),
            key=lambda s: -best_score[s],
        )
        top = max(best_score.values(), default=0.0)
        print(
            f"[crawl] hop {hop + 1}: expanded {sub}, known subs "
            f"{len(best_score)}, best score {top:.0f}, targets {len(found)}"
        )
        if len(found) >= args.want:
            break

    found = sorted(
        (s for s, sc in best_score.items() if sc >= args.target_score),
        key=lambda s: -best_score[s],
    )[: args.want]
    result = [{"submissionId": s, "score": best_score[s]} for s in found]
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(
        f"[crawl] wrote {len(result)} submissions >= {args.target_score} -> {args.out}"
    )


def cmd_pull(args: argparse.Namespace) -> None:
    subs = json.loads(Path(args.subs).read_text(encoding="utf-8"))
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    pulled = 0
    for entry in subs:
        sub = int(entry["submissionId"])
        episodes = list_episodes(sub)
        time.sleep(_POLITE_DELAY_S)
        for episode in episodes[: args.per_sub]:
            eid = int(episode["id"])
            meta_path = dest / f"episode-{eid}-meta.json"
            if not meta_path.exists():
                meta_path.write_text(json.dumps(episode), encoding="utf-8")
            if download_replay(eid, dest):
                pulled += 1
        print(f"[pull] submission {sub}: cumulative replays {pulled}")
    print(f"[pull] done — {pulled} replays under {dest}")


def cmd_extract(args: argparse.Namespace) -> None:
    replays_dir = Path(args.replays)
    top: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    scanned = 0
    for replay_path in sorted(replays_dir.glob("episode-*-replay.json")):
        eid = replay_path.stem.split("-")[1]
        meta_path = replays_dir / f"episode-{eid}-meta.json"
        scores: list[float] = []
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            scores = [
                float(a.get("updatedScore") or 0.0) for a in meta.get("agents") or []
            ]
        try:
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        scanned += 1
        for i, deck in enumerate(decks_from_replay(replay)):
            score = scores[i] if i < len(scores) else 0.0
            if score < args.min_score:
                continue
            key = canonical(deck)
            if key not in seen:
                seen.add(key)
                top.append(deck)
    Path(args.top_out).write_text(json.dumps(top), encoding="utf-8")

    base: list[list[int]] = []
    if Path(args.base).exists():
        base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    merged, added = merge_unique(base, top)
    Path(args.merged_out).write_text(json.dumps(merged), encoding="utf-8")
    print(
        f"[extract] {scanned} replays -> {len(top)} unique decks with agent "
        f"rating >= {args.min_score} -> {args.top_out}"
    )
    print(
        f"[extract] merged {len(base)} base + {added} new = {len(merged)} "
        f"-> {args.merged_out} (base file untouched)"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="top-bracket episode harvester")
    sub = parser.add_subparsers(dest="cmd", required=True)

    crawl = sub.add_parser("crawl", help="ladder-crawl submission ids upward")
    crawl.add_argument("--seed", type=int, nargs="+", required=True)
    crawl.add_argument("--target-score", type=float, default=1100.0)
    crawl.add_argument("--want", type=int, default=6)
    crawl.add_argument("--max-hops", type=int, default=25)
    crawl.add_argument("--out", default=str(_DATA / "top_submissions.json"))
    crawl.set_defaults(func=cmd_crawl)

    pull = sub.add_parser("pull", help="download replays + metadata")
    pull.add_argument("--subs", default=str(_DATA / "top_submissions.json"))
    pull.add_argument("--per-sub", type=int, default=40)
    pull.add_argument("--dest", default=str(_DATA / "replays"))
    pull.set_defaults(func=cmd_pull)

    extract = sub.add_parser("extract", help="decklists from downloaded replays")
    extract.add_argument("--replays", default=str(_DATA / "replays"))
    extract.add_argument("--min-score", type=float, default=1000.0)
    extract.add_argument("--top-out", default=str(_DATA / "top_decks.json"))
    extract.add_argument("--base", default=str(_DATA / "meta_decks.json"))
    extract.add_argument("--merged-out", default=str(_DATA / "meta_decks_v2.json"))
    extract.set_defaults(func=cmd_extract)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
