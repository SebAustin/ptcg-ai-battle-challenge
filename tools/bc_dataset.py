"""Behavior-cloning dataset from harvested top-bracket replays (dev-only).

For every decision a highly rated agent made (rating from the episode meta
written by ``tools/harvest.py pull``), record the STATE features
(``features.extract`` + ``option_features.extract_decision``), one feature row
per legal OPTION (``option_features.extract_options``) and which option(s) it
chose. Label alignment is NOT a fixed offset: empirically (verified against
~1,600 decisions across 15 replays) the agent's response to the select shown
at ``steps[t][i]`` lands at ``steps[t][i]["action"]`` itself 72% of the time
and at ``steps[t+1][i]["action"]`` the other 28% (no clean rule by select
type — a per-turn engine-timing artifact, not our concern to explain). We
therefore scan a small forward window (``_LABEL_WINDOW`` steps) for the FIRST
in-range, non-empty, distinct action for the SAME agent index and take that —
this covered 100% of the sampled decisions at window <= 1 (kept wider here
for margin). ``yourIndex == i`` is required at the decision step.

Options with identical feature vectors (e.g. four copies of the same Energy)
collapse into one CLASS — at runtime they score identically anyway — which
removes an artificial accuracy ceiling. Also recorded per decision: the v7
heuristic's pick (agreement baseline), the first option's class, the acting
agent's canonical deck key (leave-one-deck-out splits), rating and game id.

Output: ``data/bc/<gen>/shard_<w>.npz`` (gitignored). Run:
``.venv/bin/python -m tools.bc_dataset --replays data/replays --out data/bc/gen0``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_ENGINE = _ROOT / "engine" / "sample_submission" / "sample_submission"
_LABEL_WINDOW = 4  # empirically only 0-1 needed; wider for safety margin

Decision = tuple[int, dict[str, Any], dict[str, Any], list[int], int]


def agent_ratings(meta: dict[str, Any]) -> dict[int, float]:
    """``agent index -> rating`` from a ListEpisodes meta entry."""
    out: dict[int, float] = {}
    for k, agent in enumerate(meta.get("agents") or []):
        idx = agent.get("index", k)
        out[int(idx)] = float(agent.get("updatedScore") or 0.0)
    return out


def _valid_action(action: Any, n_options: int) -> bool:
    if not isinstance(action, list) or not action:
        return False
    if any((not isinstance(a, int)) or a < 0 or a >= n_options for a in action):
        return False
    return len(set(action)) == len(action)


def _find_action(
    steps: list[list[dict[str, Any]]], t: int, i: int, n: int
) -> int | None:
    """Forward-window scan: the offset (0..``_LABEL_WINDOW``) of the first
    in-range, non-empty, distinct action for agent ``i``, or ``None``."""
    for dt in range(_LABEL_WINDOW + 1):
        if t + dt >= len(steps) or i >= len(steps[t + dt]):
            break
        action = steps[t + dt][i].get("action")
        if _valid_action(action, n):
            return dt
    return None


def iter_decisions(
    replay: dict[str, Any],
    ratings: dict[int, float],
    min_rating: float,
    select_types: frozenset[int],
) -> Iterator[Decision]:
    """Yield ``(agent_idx, observation, select, chosen_indices, offset)`` for rated agents."""
    steps = replay.get("steps") or []
    for t in range(len(steps)):
        for i, agent in enumerate(steps[t]):
            if ratings.get(i, 0.0) < min_rating:
                continue
            status = agent.get("status")
            if status not in (None, "ACTIVE"):
                continue
            obs = agent.get("observation") or {}
            select = obs.get("select")
            if not isinstance(select, dict) or select.get("type") not in select_types:
                continue
            options = select.get("option") or []
            if len(options) < 2:
                continue
            if int((obs.get("current") or {}).get("yourIndex", -1)) != i:
                continue
            offset = _find_action(steps, t, i, len(options))
            if offset is None:
                continue
            action = steps[t + offset][i]["action"]
            yield i, obs, select, list(action), offset


def deck_key(replay: dict[str, Any], agent_idx: int) -> int:
    """Canonical, order-independent id of the acting agent's 60-card deck."""
    try:
        deck = replay["steps"][1][agent_idx].get("action") or []
        return hash(tuple(sorted(int(c) for c in deck))) & 0x7FFFFFFFFFFFFFFF
    except Exception:
        return 0


def encode(
    obs: dict[str, Any], select: dict[str, Any], chosen: list[int]
) -> tuple[list[float], list[list[float]], list[float], int, int]:
    """(state, unique option rows, per-class target weights, first_class, v7_class)."""
    from ptcg_bot import features, main, option_features  # noqa: PLC0415

    me = int((obs.get("current") or {}).get("yourIndex") or 0)
    state = features.extract(obs, me) + option_features.extract_decision(obs, select)
    rows = option_features.extract_options(obs, select)
    mask = option_features.DEDUP_MASK

    # Dedupe by CONTENT (excluding position-only features): two options for the
    # same card in different hand slots are the same choice. The class row kept
    # is whichever raw row was seen first — the model still trains on real
    # per-option vectors (incl. r_index), matching what runtime inference sees
    # (which never dedupes and scores every option individually).
    classes: dict[tuple[float, ...], int] = {}
    class_of: list[int] = []
    unique: list[list[float]] = []
    for row in rows:
        key = tuple(v for v, keep in zip(row, mask, strict=True) if keep)
        if key not in classes:
            classes[key] = len(unique)
            unique.append(row)
        class_of.append(classes[key])

    chosen_classes = sorted({class_of[c] for c in chosen})
    weights = [0.0] * len(unique)
    for c in chosen_classes:
        weights[c] = 1.0 / len(chosen_classes)

    try:
        v7 = main._heuristic_selection(obs)
        v7_class = class_of[v7[0]] if v7 else -1
    except Exception:
        v7_class = -1
    return state, unique, weights, class_of[0], v7_class


def _worker(args: tuple[int, list[str], str, float, tuple[int, ...]]) -> str:
    worker_id, files, out_dir, min_rating, select_types = args
    sys.path.insert(0, str(_ENGINE))
    sys.path.insert(0, str(_ROOT))
    import numpy as np  # noqa: PLC0415

    types = frozenset(select_types)
    states: list[list[float]] = []
    opt_rows: list[list[float]] = []
    opt_dec: list[int] = []
    targets: list[float] = []
    game_ids: list[int] = []
    sel_type: list[int] = []
    n_classes: list[int] = []
    ratings_out: list[float] = []
    deck_keys: list[int] = []
    v7_classes: list[int] = []
    first_classes: list[int] = []
    offsets = Counter()

    for path in files:
        replay_path = Path(path)
        meta_path = replay_path.with_name(replay_path.name.replace("-replay", "-meta"))
        if not meta_path.exists():
            continue
        try:
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ratings = agent_ratings(meta)
        # replay["id"] is a UUID; the numeric episode id lives in the filename.
        try:
            game_id = int(replay_path.stem.split("-")[1])
        except (IndexError, ValueError):
            game_id = abs(hash(replay_path.stem)) % (2**62)
        for agent_idx, obs, select, chosen, offset in iter_decisions(
            replay, ratings, min_rating, types
        ):
            offsets[offset] += 1
            state, unique, weights, first_class, v7_class = encode(obs, select, chosen)
            d = len(states)
            states.append(state)
            for row, w in zip(unique, weights, strict=True):
                opt_rows.append(row)
                opt_dec.append(d)
                targets.append(w)
            game_ids.append(game_id)
            sel_type.append(int(select.get("type") or 0))
            n_classes.append(len(unique))
            ratings_out.append(ratings.get(agent_idx, 0.0))
            deck_keys.append(deck_key(replay, agent_idx))
            v7_classes.append(v7_class)
            first_classes.append(first_class)

    out = Path(out_dir) / f"shard_{worker_id}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        state=np.asarray(states, np.float32),
        opt=np.asarray(opt_rows, np.float32),
        opt_dec=np.asarray(opt_dec, np.int32),
        target=np.asarray(targets, np.float32),
        game_id=np.asarray(game_ids, np.int64),
        sel_type=np.asarray(sel_type, np.int8),
        n_classes=np.asarray(n_classes, np.int16),
        rating=np.asarray(ratings_out, np.float32),
        deck_key=np.asarray(deck_keys, np.int64),
        v7_class=np.asarray(v7_classes, np.int16),
        first_class=np.asarray(first_classes, np.int16),
    )
    n = max(1, len(states))
    off_summary = {k: round(v / n, 3) for k, v in sorted(offsets.items())}
    print(
        f"[bc] worker {worker_id}: {len(states)} decisions, {len(opt_rows)} option "
        f"rows from {len(files)} replays | label offset distribution {off_summary} "
        f"-> {out}"
    )
    return str(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="behavior-cloning dataset builder")
    parser.add_argument("--replays", default="data/replays")
    parser.add_argument("--out", default="data/bc/gen0")
    parser.add_argument("--min-rating", type=float, default=1150.0)
    parser.add_argument("--select-types", default="0,1", help="MAIN=0, CARD=1")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    if not (_ENGINE / "cg" / "api.py").exists():
        raise SystemExit("engine not found — run `make engine` first")
    files = sorted(str(p) for p in Path(args.replays).glob("episode-*-replay.json"))
    if not files:
        raise SystemExit(f"no replays under {args.replays} — run tools.harvest pull")
    types = tuple(int(x) for x in args.select_types.split(",") if x.strip())
    workers = max(1, min(args.workers, len(files)))
    jobs = [
        (w, files[w::workers], args.out, args.min_rating, types) for w in range(workers)
    ]
    if workers == 1:
        shards = [_worker(jobs[0])]
    else:
        import multiprocessing as mp  # noqa: PLC0415

        with mp.get_context("spawn").Pool(workers) as pool:
            shards = pool.map(_worker, jobs)
    print(f"[bc] wrote {len(shards)} shards under {args.out}")


if __name__ == "__main__":
    main()
