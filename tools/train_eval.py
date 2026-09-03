"""Train the learned win-probability evaluator (dev-only, numpy).

Loads self-play CSV shards (``tools/selfplay.py``), splits train/val BY GAME
(no position leakage), trains a logistic-regression baseline and a tiny MLP
(1 hidden layer, tanh) with Adam + binary cross-entropy, reports val
logloss/AUC/calibration, and exports the better model (MLP only if it clearly
beats LR) as **``ptcg_bot/eval_weights.py``** — a generated, committed,
pure-stdlib module whose ``predict(features) -> float`` runs in ~µs.

Run: ``.venv/bin/python -m tools.train_eval --data data/selfplay/gen0``
Self-test (no data needed): ``... -m tools.train_eval --self-test``
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "ptcg_bot" / "eval_weights.py"
_HIDDEN = 32
_MLP_MARGIN = 0.003  # ship MLP only if val logloss beats LR by this much


def load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, game_ids) from every shard under ``data_dir``."""
    xs, ys, gids = [], [], []
    shards = sorted(Path(data_dir).glob("shard_*.csv"))
    if not shards:
        raise SystemExit(f"no shards under {data_dir} — run tools/selfplay.py first")
    for shard in shards:
        with shard.open(encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)  # header
            for row in reader:
                gids.append(int(row[0]))
                ys.append(float(row[1]))
                xs.append([float(v) for v in row[2:]])
    return np.asarray(xs, np.float64), np.asarray(ys, np.float64), np.asarray(gids)


def split_by_game(
    x: np.ndarray, y: np.ndarray, gids: np.ndarray, val_frac: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_mask = (gids % 997) < int(997 * val_frac)  # deterministic per-game hash
    return x[~val_mask], y[~val_mask], x[val_mask], y[val_mask]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank AUC over win(1)/loss(0) rows (draw 0.5 rows excluded)."""
    mask = (y == 0.0) | (y == 1.0)
    y, p = y[mask], p[mask]
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1.0
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class _Adam:
    def __init__(self, shapes: list[tuple[int, ...]], lr: float = 1e-3) -> None:
        self.lr, self.t = lr, 0
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads, strict=True)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            m_hat = self.m[i] / (1 - 0.9**self.t)
            v_hat = self.v[i] / (1 - 0.999**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def train_lr(
    x: np.ndarray, y: np.ndarray, epochs: int = 12, batch: int = 4096
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(0)
    w = np.zeros(x.shape[1])
    b = 0.0
    opt = _Adam([w.shape, ()], lr=3e-3)
    for _ in range(epochs):
        idx = rng.permutation(len(x))
        for start in range(0, len(x), batch):
            sl = idx[start : start + batch]
            p = _sigmoid(x[sl] @ w + b)
            err = p - y[sl]
            gw = x[sl].T @ err / len(sl)
            gb = float(err.mean())
            b_arr = np.asarray(b)
            opt.step([w, b_arr], [gw, np.asarray(gb)])
            b = float(b_arr)
    return w, b


def train_mlp(
    x: np.ndarray, y: np.ndarray, epochs: int = 12, batch: int = 4096
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(0)
    n_in = x.shape[1]
    w1 = rng.normal(0, 1 / np.sqrt(n_in), ((_HIDDEN), n_in))
    b1 = np.zeros(_HIDDEN)
    w2 = rng.normal(0, 1 / np.sqrt(_HIDDEN), _HIDDEN)
    b2 = 0.0
    opt = _Adam([w1.shape, b1.shape, w2.shape, ()], lr=1e-3)
    for _ in range(epochs):
        idx = rng.permutation(len(x))
        for start in range(0, len(x), batch):
            sl = idx[start : start + batch]
            xb, yb = x[sl], y[sl]
            h = np.tanh(xb @ w1.T + b1)
            p = _sigmoid(h @ w2 + b2)
            err = (p - yb) / len(sl)
            gw2 = h.T @ err
            gb2 = float(err.sum())
            dh = np.outer(err, w2) * (1 - h * h)
            gw1 = dh.T @ xb
            gb1 = dh.sum(axis=0)
            b2_arr = np.asarray(b2)
            opt.step([w1, b1, w2, b2_arr], [gw1, gb1, gw2, np.asarray(gb2)])
            b2 = float(b2_arr)
    return w1, b1, w2, b2


def _mlp_predict(x, w1, b1, w2, b2):  # noqa: ANN001, ANN202 - internal numpy helper
    return _sigmoid(np.tanh(x @ w1.T + b1) @ w2 + b2)


def _calibration(y: np.ndarray, p: np.ndarray) -> str:
    lines = ["  bin    n      pred    actual"]
    for lo in np.arange(0.0, 1.0, 0.1):
        mask = (p >= lo) & (p < lo + 0.1)
        if mask.sum():
            lines.append(
                f"  {lo:.1f}  {int(mask.sum()):>7}   {p[mask].mean():.3f}   {y[mask].mean():.3f}"
            )
    return "\n".join(lines)


def export_module(
    kind: str,
    w1: np.ndarray | None,
    b1: np.ndarray | None,
    w2: np.ndarray,
    b2: float,
    feature_count: int,
    gen: str,
    metrics: str,
) -> None:
    def _fmt(arr: np.ndarray) -> str:
        return repr(tuple(round(float(v), 6) for v in arr))

    if kind == "mlp":
        assert w1 is not None and b1 is not None
        w1_lit = "(" + ",\n    ".join(_fmt(row) for row in w1) + ",)"  # tuple of tuples
        b1_lit, w2_lit = _fmt(b1), _fmt(w2)
    else:
        w1_lit, b1_lit, w2_lit = "()", "()", _fmt(w2)

    _OUT.write_text(
        f'''"""AUTO-GENERATED by tools/train_eval.py — do not edit.

Learned win-probability evaluator ({kind}, {gen}). Pure stdlib inference.
{metrics}
"""

from __future__ import annotations

import math
from collections.abc import Sequence

FEATURE_COUNT = {feature_count}
GEN = "{gen}"

W1: tuple[tuple[float, ...], ...] = {w1_lit}
B1: tuple[float, ...] = {b1_lit}
W2: tuple[float, ...] = {w2_lit}
B2: float = {round(float(b2), 6)}


def predict(features: Sequence[float]) -> float:
    """Win probability in (0, 1) for a FEATURE_COUNT-long feature vector."""
    if W1:
        hidden = [
            math.tanh(sum(w * f for w, f in zip(row, features, strict=False)) + b)
            for row, b in zip(W1, B1, strict=False)
        ]
        logit = sum(w * h for w, h in zip(W2, hidden, strict=False)) + B2
    else:
        logit = sum(w * f for w, f in zip(W2, features, strict=False)) + B2
    logit = max(-30.0, min(30.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))
''',
        encoding="utf-8",
    )
    print(f"[train] exported {kind} model -> {_OUT}")


def self_test() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (2000, 10))
    y = (x[:, 0] + 0.5 * x[:, 1] + 0.1 * rng.normal(0, 1, 2000) > 0).astype(float)
    # Small data needs small batches (batch > n would mean ~1 step/epoch).
    w, b = train_lr(x[:1600], y[:1600], epochs=60, batch=128)
    auc = _auc(y[1600:], _sigmoid(x[1600:] @ w + b))
    assert auc > 0.9, f"self-test failed: AUC={auc:.3f}"
    print(f"[train] self-test OK (AUC={auc:.3f})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="train the learned evaluator")
    parser.add_argument("--data", default="data/selfplay/gen0")
    parser.add_argument("--gen", default="gen0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return

    x, y, gids = load_data(args.data)
    x_tr, y_tr, x_val, y_val = split_by_game(x, y, gids)
    base = np.full_like(y_val, y_tr.mean())
    print(
        f"[train] {len(x_tr)} train / {len(x_val)} val rows "
        f"({len(np.unique(gids))} games) | base logloss {_logloss(y_val, base):.4f}"
    )

    w, b = train_lr(x_tr, y_tr)
    p_lr = _sigmoid(x_val @ w + b)
    ll_lr, auc_lr = _logloss(y_val, p_lr), _auc(y_val, p_lr)
    print(f"[train] LR : val logloss {ll_lr:.4f}  AUC {auc_lr:.4f}")

    w1, b1, w2, b2 = train_mlp(x_tr, y_tr)
    p_mlp = _mlp_predict(x_val, w1, b1, w2, b2)
    ll_mlp, auc_mlp = _logloss(y_val, p_mlp), _auc(y_val, p_mlp)
    print(f"[train] MLP: val logloss {ll_mlp:.4f}  AUC {auc_mlp:.4f}")

    if ll_mlp < ll_lr - _MLP_MARGIN:
        metrics = (
            f"val_logloss={ll_mlp:.4f} auc={auc_mlp:.4f} (LR {ll_lr:.4f}/{auc_lr:.4f})"
        )
        print("[train] calibration (MLP):\n" + _calibration(y_val, p_mlp))
        export_module("mlp", w1, b1, w2, b2, x.shape[1], args.gen, metrics)
    else:
        metrics = (
            f"val_logloss={ll_lr:.4f} auc={auc_lr:.4f} (MLP {ll_mlp:.4f}/{auc_mlp:.4f})"
        )
        print("[train] calibration (LR):\n" + _calibration(y_val, p_lr))
        export_module("lr", None, None, w, b, x.shape[1], args.gen, metrics)


if __name__ == "__main__":
    main()
