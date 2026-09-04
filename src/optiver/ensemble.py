"""Phase 3 blend: a convex combination of two out-of-fold predictions, weighted forward in time.

The trap a blend sets for a harness. Fitting one weight on the pooled
out-of-fold predictions and then scoring the blend on those same predictions
is a leak the fold loop cannot see: the weight has been chosen with knowledge
of every validation label it is about to be scored on. It is a small leak —
one parameter against three million rows — and that is exactly why it is
worth refusing: a harness whose guarantees hold except for the small leaks
does not have guarantees.

So the weight moves forward in time. For fold k it is fitted on the OOF
predictions of folds 0..k-1 only. Those rows are validation dates of EARLIER
folds, which means they are all strictly before fold k's validation block and
inside fold k's own training window — information a live system would have
had. Fold 0 has no prior folds and takes the a priori weight of 0.5. The
fixed 50/50 blend is reported beside it as the arm that fits nothing at all.

Two things the forward weight is not. It is not the weight a live system would
use on fold k's dates — that system would have refitted both models on all
history up to k, and their relative merit could differ. And it is not tuned:
one scalar, found on a grid, against the metric, on rows the fold never
scores. The grid rather than a solver for the reason `baselines._mae_optimal_scale`
gives — the objective is convex and piecewise linear in the weight, so a
coarse-to-fine grid is exact to three decimals with no dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import splits
from .evaluate import mae


def mae_optimal_weight(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """The w in [0, 1] minimising MAE(y, w*a + (1-w)*b), found coarse-to-fine on a grid."""
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[ok], a[ok], b[ok]
    if y.size == 0:
        raise ValueError("no rows to fit a blend weight on")

    def loss(w: float) -> float:
        return float(np.abs(y - (w * a + (1.0 - w) * b)).mean())

    coarse = np.linspace(0.0, 1.0, 21)
    best = float(coarse[int(np.argmin([loss(w) for w in coarse]))])
    fine = np.linspace(max(0.0, best - 0.05), min(1.0, best + 0.05), 21)
    return float(fine[int(np.argmin([loss(w) for w in fine]))])


def blend_fixed(a: np.ndarray, b: np.ndarray, weight_a: float = 0.5) -> np.ndarray:
    """The arm that fits nothing: a fixed convex combination."""
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError(f"weight_a must lie in [0, 1], got {weight_a}")
    return weight_a * np.asarray(a, np.float64) + (1.0 - weight_a) * np.asarray(b, np.float64)


def blend_forward(
    a: np.ndarray,
    b: np.ndarray,
    df: pd.DataFrame,
    folds: list[splits.Fold],
    *,
    prior: float = 0.5,
) -> dict:
    """Per-fold weight on `a`, fitted only on the OOF rows of earlier folds.

    Returns the blended OOF vector (NaN wherever either input is NaN or the row
    is in no validation block) and the per-fold weight table for the log.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    y = df["target"].to_numpy(np.float64)
    if not (len(a) == len(b) == len(y)):
        raise ValueError(f"length mismatch: a {len(a)}, b {len(b)}, df {len(y)}")

    pred = np.full(len(y), np.nan)
    table: list[dict] = []
    for fold in sorted(folds, key=lambda f: f.index):
        _, va = splits.fold_masks(df, fold)
        earlier = [f for f in folds if f.index < fold.index]
        fit_rows = np.zeros(len(y), dtype=bool)
        for f in earlier:
            # Forward only: an earlier fold's validation block must end before
            # this one begins. make_folds guarantees it; asserting is cheap.
            if f.val_date_max >= fold.val_date_min:
                raise AssertionError(f"{f} is not strictly before {fold}")
            fit_rows |= splits.fold_masks(df, f)[1]
        fit_rows &= np.isfinite(a) & np.isfinite(b) & np.isfinite(y)

        fitted = bool(fit_rows.any())
        w = mae_optimal_weight(y[fit_rows], a[fit_rows], b[fit_rows]) if fitted else float(prior)
        pred[va] = w * a[va] + (1.0 - w) * b[va]
        table.append({
            "fold": fold.index,
            "weight_a": w,
            "fitted": fitted,
            "n_fit_rows": int(fit_rows.sum()),
            "fit_dates": [int(min(f.val_date_min for f in earlier)), int(max(f.val_date_max for f in earlier))]
            if earlier else None,
            "val_mae_bps": mae(y[va], pred[va]),
        })
    return {"pred": pred, "weights": table}
