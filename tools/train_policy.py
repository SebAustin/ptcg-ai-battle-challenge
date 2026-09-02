"""Train the behavior-cloned option policy (dev-only, numpy).

Late-fusion model, chosen for rollout-time cost: a state tower runs once per
decision, ``h = tanh(Ws·s + bs)``; the option weights are ``m = a + V·h``; each
option then costs one dot product, ``score(o) = m·o + b``. Training: per
decision softmax over its (deduplicated) option classes, cross-entropy against
the recorded choice(s), Adam. Splits BY GAME (no leakage) plus a
leave-one-deck-out (LODO) split on the most frequent top deck — the transfer
proxy for our own deck, which no top agent plays.

Reports top-1 accuracy against three baselines: chance (mean 1/classes),
first-listed option, and the v7 heuristic's agreement with the top agents
(G0 fail-fast gate in ASSUMPTIONS §38). Exports ``ptcg_bot/policy_weights.py``
— pure stdlib ``prepare``/``score``/``score_all``.

Run: ``.venv/bin/python -m tools.train_policy --data data/bc/gen0 --gen bc0``
Self-test (no data): ``... --self-test``
"""

from __future__ import annotations

import argparse
import subprocess  # noqa: S404 - formats a generated file with the repo's own black (dev-only)
import sys
from pathlib import Path

import numpy as np

from tools.train_eval import _Adam

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "ptcg_bot" / "policy_weights.py"
_HIDDEN = 32


class Data:
    """Decision-major arrays; option rows of one decision are contiguous."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.state = arrays["state"].astype(np.float64)
        self.opt = arrays["opt"].astype(np.float64)
        self.opt_dec = arrays["opt_dec"].astype(np.int64)
        self.target = arrays["target"].astype(np.float64)
        self.game_id = arrays["game_id"]
        self.sel_type = arrays["sel_type"]
        self.n_classes = arrays["n_classes"].astype(np.int64)
        self.deck_key = arrays["deck_key"]
        self.v7_class = arrays["v7_class"].astype(np.int64)
        self.first_class = arrays["first_class"].astype(np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.n_classes)[:-1]])

    def __len__(self) -> int:
        return len(self.state)

    def subset(self, mask: np.ndarray) -> Data:
        idx = np.nonzero(mask)[0]
        row_mask = mask[self.opt_dec]
        new_dec = np.cumsum(mask) - 1  # old decision index -> new (valid where mask)
        return Data(
            {
                "state": self.state[idx],
                "opt": self.opt[row_mask],
                "opt_dec": new_dec[self.opt_dec[row_mask]],
                "target": self.target[row_mask],
                "game_id": self.game_id[idx],
                "sel_type": self.sel_type[idx],
                "n_classes": self.n_classes[idx],
                "deck_key": self.deck_key[idx],
                "v7_class": self.v7_class[idx],
                "first_class": self.first_class[idx],
            }
        )


def load_data(data_dir: str) -> Data:
    shards = sorted(Path(data_dir).glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards under {data_dir} — run tools.bc_dataset first")
    parts: dict[str, list[np.ndarray]] = {}
    offset = 0
    for shard in shards:
        with np.load(shard) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["opt_dec"] = arrays["opt_dec"].astype(np.int64) + offset
        offset += len(arrays["state"])
        for k, v in arrays.items():
            parts.setdefault(k, []).append(v)
    return Data({k: np.concatenate(v) for k, v in parts.items()})


class Model:
    def __init__(self, n_state: int, n_opt: int, hidden: int = _HIDDEN, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.ws = rng.normal(0, 1 / np.sqrt(n_state), (hidden, n_state))
        self.bs = np.zeros(hidden)
        self.v = rng.normal(0, 1 / np.sqrt(hidden), (n_opt, hidden))
        self.a = np.zeros(n_opt)
        self.b = np.zeros(())

    def params(self) -> list[np.ndarray]:
        return [self.ws, self.bs, self.v, self.a, self.b]

    def option_weights(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(state @ self.ws.T + self.bs)
        return h, h @ self.v.T + self.a  # [D,H], [D,n_opt]


def _forward(model: Model, data: Data, d0: int, d1: int):  # noqa: ANN202
    r0 = int(data.starts[d0])
    r1 = int(data.starts[d1]) if d1 < len(data) else len(data.opt)
    rows = data.opt[r0:r1]
    dec_local = data.opt_dec[r0:r1] - d0
    h, m = model.option_weights(data.state[d0:d1])
    s = np.einsum("ij,ij->i", rows, m[dec_local]) + float(model.b)
    starts = data.starts[d0:d1] - r0
    s_max = np.maximum.reduceat(s, starts)
    e = np.exp(s - s_max[dec_local])
    z = np.add.reduceat(e, starts)
    p = e / z[dec_local]
    return rows, dec_local, h, m, s, p, starts


def train(
    model: Model, data: Data, epochs: int = 15, batch: int = 512, lr: float = 1e-3
) -> None:
    opt = _Adam([p.shape for p in model.params()], lr=lr)
    n = len(data)
    for epoch in range(epochs):
        total = 0.0
        for d0 in range(0, n, batch):
            d1 = min(n, d0 + batch)
            rows, dec_local, h, _m, _s, p, starts = _forward(model, data, d0, d1)
            r0 = int(data.starts[d0])
            y = data.target[r0 : r0 + len(rows)]
            nb = d1 - d0
            total += float(-(y * np.log(np.clip(p, 1e-9, 1.0))).sum())
            ds = (p - y) / nb
            g = np.add.reduceat(ds[:, None] * rows, starts, axis=0)  # [D,n_opt]
            ga = g.sum(axis=0)
            gb = np.asarray(ds.sum())
            gv = g.T @ h
            dz = (g @ model.v) * (1 - h * h)
            gws = dz.T @ data.state[d0:d1]
            gbs = dz.sum(axis=0)
            opt.step(model.params(), [gws, gbs, gv, ga, gb])
        if epoch % 5 == 4 or epoch == epochs - 1:
            print(f"[policy] epoch {epoch + 1}: train CE {total / max(1, n):.4f}")


def evaluate(model: Model, data: Data, label: str) -> dict[str, float]:
    """Top-1 accuracy vs chance / first-option / v7-agreement; MAIN-only too."""
    hits, hits_main, n_main = 0, 0, 0
    chance = float(np.mean(1.0 / data.n_classes)) if len(data) else 0.0
    first_hit = v7_hit = v7_n = 0
    for d0 in range(0, len(data), 2048):
        d1 = min(len(data), d0 + 2048)
        rows, dec_local, _h, _m, s, _p, starts = _forward(model, data, d0, d1)
        r0 = int(data.starts[d0])
        y = data.target[r0 : r0 + len(rows)]
        ends = np.append(starts[1:], len(rows))
        for k, (a, b) in enumerate(zip(starts, ends, strict=True)):
            d = d0 + k
            pick = a + int(np.argmax(s[a:b]))
            hit = y[pick] > 0
            hits += hit
            if data.sel_type[d] == 0:
                hits_main += hit
                n_main += 1
            first_hit += y[a + data.first_class[d]] > 0
            if data.v7_class[d] >= 0:
                v7_hit += y[a + data.v7_class[d]] > 0
                v7_n += 1
    n = max(1, len(data))
    out = {
        "top1": hits / n,
        "top1_main": hits_main / max(1, n_main),
        "chance": chance,
        "first": first_hit / n,
        "v7": v7_hit / max(1, v7_n),
        "n": float(n),
        "n_main": float(n_main),
    }
    print(
        f"[policy] {label}: top1 {out['top1']:.3f} (MAIN {out['top1_main']:.3f}, "
        f"n={n}, main={n_main}) | chance {chance:.3f} | first {out['first']:.3f} | "
        f"v7-agreement {out['v7']:.3f} (n={v7_n})"
    )
    return out


def export_module(model: Model, gen: str, metrics: str) -> None:
    def _fmt(arr: np.ndarray) -> str:
        return repr(tuple(round(float(v), 6) for v in arr))

    ws_lit = "(" + ",\n    ".join(_fmt(row) for row in model.ws) + ",)"
    v_lit = "(" + ",\n    ".join(_fmt(row) for row in model.v) + ",)"
    _OUT.write_text(
        f'''"""AUTO-GENERATED by tools/train_policy.py — do not edit.

Behavior-cloned option policy ({gen}), late fusion. Pure stdlib inference.
{metrics}
"""

from __future__ import annotations

import math
from collections.abc import Sequence

STATE_COUNT = {model.ws.shape[1]}
OPTION_COUNT = {model.v.shape[0]}
GEN = "{gen}"

WS: tuple[tuple[float, ...], ...] = {ws_lit}
BS: tuple[float, ...] = {_fmt(model.bs)}
V: tuple[tuple[float, ...], ...] = {v_lit}
A: tuple[float, ...] = {_fmt(model.a)}
B: float = {round(float(model.b), 6)}


def prepare(state: Sequence[float]) -> tuple[float, ...]:
    """Per-decision option weights m = A + V·tanh(WS·state + BS)."""
    hidden = [
        math.tanh(sum(w * f for w, f in zip(row, state, strict=False)) + b)
        for row, b in zip(WS, BS, strict=False)
    ]
    return tuple(
        a + sum(v * h for v, h in zip(row, hidden, strict=False))
        for row, a in zip(V, A, strict=False)
    )


def score(ctx: Sequence[float], option: Sequence[float]) -> float:
    """Imitation score of one option given ``prepare(state)``."""
    return sum(m * o for m, o in zip(ctx, option, strict=False)) + B


def score_all(state: Sequence[float], options: Sequence[Sequence[float]]) -> list[float]:
    ctx = prepare(state)
    return [score(ctx, o) for o in options]
''',
        encoding="utf-8",
    )
    subprocess.run(  # noqa: S603 - fixed argv, no shell; keeps `make lint` green
        [sys.executable, "-m", "black", "-q", str(_OUT)], check=False
    )
    print(f"[policy] exported {gen} -> {_OUT}")


def self_test() -> None:
    """Synthetic bilinear ground truth must be recoverable (top-1 > 0.85)."""
    rng = np.random.default_rng(3)
    n_dec, n_state, n_opt, hidden = 3000, 12, 9, 8
    true = Model(n_state, n_opt, hidden, seed=7)
    true.ws *= 2.0
    true.v *= 2.0
    states, opts, opt_dec, targets, n_classes = [], [], [], [], []
    for d in range(n_dec):
        k = int(rng.integers(2, 7))
        s = rng.normal(0, 1, n_state)
        o = rng.normal(0, 1, (k, n_opt))
        _h, m = true.option_weights(s[None, :])
        best = int(np.argmax(o @ m[0]))
        states.append(s)
        opts.extend(o)
        opt_dec.extend([d] * k)
        targets.extend([1.0 if j == best else 0.0 for j in range(k)])
        n_classes.append(k)
    data = Data(
        {
            "state": np.asarray(states),
            "opt": np.asarray(opts),
            "opt_dec": np.asarray(opt_dec),
            "target": np.asarray(targets),
            "game_id": np.arange(n_dec),
            "sel_type": np.zeros(n_dec, np.int8),
            "n_classes": np.asarray(n_classes),
            "deck_key": np.zeros(n_dec, np.int64),
            "v7_class": -np.ones(n_dec, np.int64),
            "first_class": np.zeros(n_dec, np.int64),
        }
    )
    model = Model(n_state, n_opt, hidden)
    train_mask = np.arange(n_dec) < 2400
    train(model, data.subset(train_mask), epochs=40, batch=128, lr=3e-3)
    out = evaluate(model, data.subset(~train_mask), "self-test val")
    assert out["top1"] > 0.85, f"self-test failed: top1={out['top1']:.3f}"
    print("[policy] self-test OK")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="train the behavior-cloned policy")
    parser.add_argument("--data", default="data/bc/gen0")
    parser.add_argument("--gen", default="bc0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return

    data = load_data(args.data)
    print(
        f"[policy] {len(data)} decisions, {len(data.opt)} option rows, "
        f"state {data.state.shape[1]} / option {data.opt.shape[1]} features"
    )
    val_mask = (data.game_id % 997) < 100
    train_set, val_set = data.subset(~val_mask), data.subset(val_mask)
    model = Model(data.state.shape[1], data.opt.shape[1])
    train(model, train_set, epochs=args.epochs)
    val = evaluate(model, val_set, "game-split val")

    # Leave-one-deck-out on the most frequent top deck: transfer proxy.
    keys, counts = np.unique(data.deck_key, return_counts=True)
    held = keys[int(np.argmax(counts))]
    lodo_mask = data.deck_key == held
    lodo_model = Model(data.state.shape[1], data.opt.shape[1])
    train(lodo_model, data.subset(~lodo_mask), epochs=args.epochs)
    lodo = evaluate(lodo_model, data.subset(lodo_mask), "LODO (top deck held out)")

    metrics = (
        f"game-split val: top1 {val['top1']:.3f} (MAIN {val['top1_main']:.3f}), "
        f"chance {val['chance']:.3f}, first {val['first']:.3f}, v7 {val['v7']:.3f}; "
        f"LODO MAIN top1 {lodo['top1_main']:.3f}"
    )
    if not args.no_export:
        export_module(model, args.gen, metrics)


if __name__ == "__main__":
    main()
