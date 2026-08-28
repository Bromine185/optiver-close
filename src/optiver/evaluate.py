"""MAE, and the breakdowns that stop a single number from hiding things.

The competition metric is mean absolute error in basis points, unweighted, over
every scored row. That is the headline. It is also, on its own, close to useless
for deciding whether a change helped: the target's cross-sectional dispersion
varies by a factor of several between stocks and rises sharply through the
auction, so a model can improve the overall MAE by getting slightly better at the
loud stocks while getting worse everywhere else, and one number will not say so.

Hence the breakdowns: per fold (is the gain stable through time, or one lucky
block?), per stock (is it a handful of names?), per bucket (is it only the easy
early buckets, where the target is smallest?).

A note on what MAE rewards here. The target is index-relative and therefore very
nearly zero-mean and symmetric, and MAE's optimal constant prediction is the
MEDIAN, not the mean. The median target is -0.060 bps. So predict-zero is not
merely a naive floor, it is within 0.06 bps of the best constant predictor that
exists, which is most of why it is so hard to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    """Mean absolute error in bps over every LABELLED row. Null targets excluded.

    Nulls are excluded rather than treated as zero. 88 rows in the full fixture
    have no label; scoring them as if the truth were 0 would credit a
    predict-zero model with 88 perfect predictions it never earned.

    A non-finite *prediction* on a labelled row raises instead. Dropping it would
    be far worse than the NaN it hides: it silently scores that model on fewer
    rows than every other model in the same fold, and since the hard rows are
    exactly the ones a model is most likely to fail on, the reward for failing is
    a lower MAE. Raising is also what makes `fold_table` comparable — with this
    guard, "the labelled rows of the fold" is the row set for every model in it,
    by construction rather than by hope. A model that cannot produce a number for
    a row it was asked about is a bug, not a row to skip.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    ok = np.isfinite(y_true)
    if not ok.any():
        raise ValueError("no labelled rows to score")
    bad = int((~np.isfinite(y_pred[ok])).sum())
    if bad:
        raise ValueError(
            f"{bad} of {int(ok.sum())} labelled rows have a non-finite prediction; "
            f"scoring would silently drop them and flatter the model"
        )
    return float(np.abs(y_true[ok] - y_pred[ok]).mean())


def breakdown(df: pd.DataFrame, y_pred, by: str) -> pd.DataFrame:
    """MAE grouped by a column of `df`, with the group's own zero-baseline beside it.

    The `mae_zero` column is what makes this readable. A per-stock MAE of 11 bps
    means nothing until you know that stock's predict-zero MAE is 11.2; the
    difference is the only quantity with any information in it.
    """
    y_true = df["target"].to_numpy(np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    g = pd.DataFrame(
        {
            by: df[by].to_numpy()[ok],
            "abs_err": np.abs(y_true[ok] - y_pred[ok]),
            "abs_target": np.abs(y_true[ok]),
        }
    ).groupby(by, observed=True)
    out = g.agg(n=("abs_err", "size"), mae=("abs_err", "mean"), mae_zero=("abs_target", "mean"))
    out["improvement_bps"] = out["mae_zero"] - out["mae"]
    out["improvement_pct"] = 100.0 * out["improvement_bps"] / out["mae_zero"]
    return out.reset_index()


def scorecard(results: dict[str, np.ndarray], df: pd.DataFrame) -> pd.DataFrame:
    """One row per named prediction vector, sorted best first.

    `improvement_bps` is signed against predict-zero deliberately: a negative
    number is a model that is worse than doing nothing, and that has to be as
    easy to read as a positive one.
    """
    y = df["target"].to_numpy(np.float64)
    zero = mae(y, np.zeros_like(y))
    rows = []
    for name, pred in results.items():
        m = mae(y, pred)
        rows.append(
            {
                "model": name,
                "mae_bps": m,
                "vs_zero_bps": zero - m,
                "vs_zero_pct": 100.0 * (zero - m) / zero,
                "coverage": float(np.isfinite(np.asarray(pred, dtype=np.float64)).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_bps").reset_index(drop=True)


def fold_table(per_fold: list[dict]) -> pd.DataFrame:
    """Fold-by-fold MAE for each model, plus the mean and the spread across folds.

    The spread matters more than it looks. Fold MAEs here differ by ~1 bps
    between blocks of dates — far more than any model's ~0.05 bps improvement —
    so a model comparison that is not paired within fold is measuring the
    calendar, not the model.
    """
    long = pd.DataFrame(per_fold)
    wide = long.pivot(index="model", columns="fold", values="mae_bps")
    wide.columns = [f"fold{c}" for c in wide.columns]
    wide["mean"] = wide.mean(axis=1)
    wide["std"] = wide.std(axis=1, ddof=0)
    # Paired against predict-zero within each fold, then averaged: the correct
    # comparison when fold-to-fold variance dwarfs the effect being measured.
    if "zero" in wide.index:
        fold_cols = [c for c in wide.columns if c.startswith("fold")]
        wide["mean_vs_zero"] = (wide.loc["zero", fold_cols] - wide[fold_cols]).mean(axis=1)
    return wide.sort_values("mean")


def describe_target(y) -> dict:
    """The distribution every claim in RESEARCH.md is measured against."""
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    qs = [1e-4, 1e-3, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1 - 1e-4]
    s = pd.Series(y)
    return {
        "n": int(y.size),
        "mean": float(y.mean()),
        "median": float(np.median(y)),
        "std": float(y.std(ddof=0)),
        "mean_abs": float(np.abs(y).mean()),
        "min": float(y.min()),
        "max": float(y.max()),
        "skew": float(s.skew()),
        "excess_kurtosis": float(s.kurt()),
        "quantiles": {q: float(np.quantile(y, q)) for q in qs},
    }
