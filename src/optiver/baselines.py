"""Phase 1 baselines: the floor, the carry, and one linear model.

No gradient boosting. The point of Phase 1 is a harness whose numbers can be
believed and a floor that later work must clear; a strong model on an untrusted
split teaches nothing, and a strong model on a trusted split teaches nothing
either until you know what "strong" means relative to doing nothing.

Four models, each answering one question.

`zero`      Predict 0.0. The floor. Because the target is index-relative it is
            nearly zero-mean by construction, so this is not a straw man — it is
            a genuinely strong predictor and the number everything is measured
            against.

`constant`  Predict the training median. MAE's optimal constant is the median,
            so this is the best any constant predictor can do. Its gap to `zero`
            is the entire value of knowing the target's location, and it bounds
            how much of any model's improvement is just re-centring.

`carry`     Predict the previous auction's target for the same stock and bucket,
            optionally shrunk by a factor fitted on training rows. Tests whether
            the target autocorrelates across auctions. This is legal: the live
            timeseries API hands you all of date d-1's targets before you predict
            date d (see `data.add_revealed_target`). It is the one baseline that
            uses information from an embargoed date, and it does so at PREDICTION
            time, not training time — the embargo governs what the model may be
            fitted on, not what the exchange tells you.

`ridge`     Ridge regression on the obvious microstructure features. Tests
            whether there is any linear signal at all.

A mismatch worth naming: ridge minimises squared error and the metric is MAE. On
a target with excess kurtosis of 22 those disagree — least squares chases the
tails that MAE barely notices. Two guards, both fitted on training rows only:
the target is winsorised before fitting (`cfg.fit_clip_bps`), and the fitted
predictions are rescaled by a single MAE-optimal scalar. Both are reported, so
the effect of each is visible rather than baked in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import features as F
from . import splits
from .config import Config
from .evaluate import mae


def _mae_optimal_scale(y: np.ndarray, pred: np.ndarray, grid: np.ndarray | None = None) -> float:
    """The scalar a minimising MAE(y, a*pred), found on a grid.

    A grid rather than a solver because the objective is piecewise linear in `a`
    with a kink at every observation: it is convex, so a coarse-to-fine grid is
    exact enough at 3 decimal places and needs no dependency. Returns 0.0 when
    the prediction is degenerate, which correctly collapses the model to `zero`.
    """
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    if y.size == 0 or not np.any(pred != 0):
        return 0.0
    grid = np.linspace(0.0, 2.0, 41) if grid is None else grid
    losses = [np.abs(y - a * pred).mean() for a in grid]
    best = float(grid[int(np.argmin(losses))])
    fine = np.linspace(max(0.0, best - 0.05), best + 0.05, 21)
    losses = [np.abs(y - a * pred).mean() for a in fine]
    return float(fine[int(np.argmin(losses))])


# --------------------------------------------------------------------------
# Models. Each is fit on a training slice and predicts on any slice.
# --------------------------------------------------------------------------

@dataclass
class Zero:
    name: str = "zero"

    def fit(self, df: pd.DataFrame, X: pd.DataFrame, cfg: Config) -> "Zero":
        return self

    def predict(self, df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df))


@dataclass
class ConstantMedian:
    name: str = "constant_median"
    value: float = 0.0

    def fit(self, df: pd.DataFrame, X: pd.DataFrame, cfg: Config) -> "ConstantMedian":
        y = df["target"].to_numpy(np.float64)
        self.value = float(np.nanmedian(y))
        return self

    def predict(self, df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.value)


@dataclass
class Carry:
    """Previous auction's target, same stock, same bucket, optionally shrunk.

    `shrink=None` fits the shrinkage on training rows by minimising MAE; that is
    the honest version, because an unshrunk carry is a bet that the
    autocorrelation is 1.0 and nobody believes that. Rows with no previous
    auction fall back to 0.0 — the baseline's own floor — and the fraction of
    such rows is reported so the result cannot be quietly driven by them. On the
    full fixture that is 0.21% of rows: date 0, the 11 stock-days that re-enter
    after an absence, and the 88 rows whose d-1 target is null. See
    `data.add_revealed_target`.
    """

    name: str = "carry"
    shrink: float | None = None
    fitted_shrink: float = 1.0
    column: str = "revealed_target"

    def fit(self, df: pd.DataFrame, X: pd.DataFrame, cfg: Config) -> "Carry":
        if self.shrink is not None:
            self.fitted_shrink = float(self.shrink)
            return self
        y = df["target"].to_numpy(np.float64)
        r = np.nan_to_num(df[self.column].to_numpy(np.float64), nan=0.0)
        self.fitted_shrink = _mae_optimal_scale(y, r)
        return self

    def predict(self, df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        r = np.nan_to_num(df[self.column].to_numpy(np.float64), nan=0.0)
        return self.fitted_shrink * r


@dataclass
class RidgeMicro:
    """Ridge on `features.FEATURE_NAMES`, standardised on training rows only.

    Standardisation matters more than usual here because the columns span six
    orders of magnitude (`size_imbalance` in [-1, 1] against
    `far_minus_wap_bps` in the thousands), and a single ridge penalty applied to
    unscaled columns penalises them by wildly different amounts — the
    regularisation would land almost entirely on the small-scale features.

    `clip_in_fit` / `clip_in_predict` exist so `scripts/run_ablations.py` can take
    the feature winsorisation apart without forking this class. They are not
    tuning knobs and nothing in `default_models` changes them. They are here
    because the published ablation row for "no feature clipping" turned out to
    have been produced by a hand-edited copy of this file that dropped the clip
    from `predict` only, and a variant reachable solely by editing source is a
    variant nobody can check.
    """

    name: str = "ridge"
    alpha: float = 1.0
    rescale: bool = True
    clip_in_fit: bool = True
    clip_in_predict: bool = True
    #: None = features.FEATURE_NAMES exactly (the Phase 1 model, unchanged).
    #: Phase 2 passes a wider list plus the matching `extra_skip` so the same
    #: class can serve as the "linear model, memory features" arm of its 2x2.
    columns: tuple[str, ...] | None = None
    extra_skip: tuple[str, ...] = ()
    mu: np.ndarray = field(default=None, repr=False)
    sd: np.ndarray = field(default=None, repr=False)
    bounds: dict = field(default_factory=dict, repr=False)
    model: Ridge = field(default=None, repr=False)
    scale: float = 1.0

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[list(self.columns)] if self.columns is not None else X

    def fit(self, df: pd.DataFrame, X: pd.DataFrame, cfg: Config) -> "RidgeMicro":
        X = self._select(X)
        self.fitted_columns = tuple(X.columns)
        self.bounds = F.quantile_bounds(X, extra_skip=self.extra_skip)
        Xc = (F.clip_outliers(X, self.bounds) if self.clip_in_fit else X).to_numpy(np.float64)
        self.mu = Xc.mean(axis=0)
        # Zero-variance columns (an indicator that is constant within a fold)
        # would divide by zero; mapping their scale to 1 leaves them as a
        # constant column, which the intercept absorbs.
        self.sd = np.where(Xc.std(axis=0) > 0, Xc.std(axis=0), 1.0)
        Z = (Xc - self.mu) / self.sd

        y = df["target"].to_numpy(np.float64)
        ok = np.isfinite(y)
        y_fit = y[ok]
        if cfg.fit_clip_bps is not None:
            y_fit = np.clip(y_fit, -cfg.fit_clip_bps, cfg.fit_clip_bps)

        self.model = Ridge(alpha=self.alpha, fit_intercept=True, solver="cholesky")
        self.model.fit(Z[ok], y_fit)

        # One MAE-optimal scalar, fitted on the SAME training rows against the
        # UNCLIPPED target: the clip protects the fit, but the metric is scored
        # on real targets, so the rescaling must be too.
        self.scale = _mae_optimal_scale(y[ok], self.model.predict(Z[ok])) if self.rescale else 1.0
        return self

    def predict(self, df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        X = self._select(X)
        Xc = (F.clip_outliers(X, self.bounds) if self.clip_in_predict else X).to_numpy(np.float64)
        Z = (Xc - self.mu) / self.sd
        return self.scale * self.model.predict(Z)

    def coefficients(self) -> pd.Series:
        """Coefficients on the STANDARDISED features, i.e. bps of target per 1 sd of feature.

        Indexed by the columns the model was actually FITTED on, recorded at fit
        time — not by `FEATURE_NAMES` assumed. The two diverged once already:
        with `columns=None` this model fits whatever frame the harness hands it,
        and under Phase 2's wider builder that is 31 columns, not 14.
        """
        return pd.Series(self.scale * self.model.coef_, index=list(self.fitted_columns)).sort_values(
            key=np.abs, ascending=False
        )


def default_models(cfg: Config) -> list:
    return [
        Zero(),
        ConstantMedian(),
        Carry(name="carry_raw", shrink=1.0),
        Carry(name="carry_shrunk", shrink=None),
        RidgeMicro(name="ridge", alpha=cfg.ridge_alpha, rescale=True),
        RidgeMicro(name="ridge_noscale", alpha=cfg.ridge_alpha, rescale=False),
    ]


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def run_cv(
    df: pd.DataFrame,
    cfg: Config,
    models: list | None = None,
    *,
    feature_builder=None,
    verbose: bool = True,
) -> dict:
    """Fit every model on every fold and score on the fold's validation dates.

    Features are built ONCE for the whole frame and then sliced. Phase 1's
    argument for that was strict row-wiseness; Phase 2's features have memory,
    so the argument is now the weaker condition that actually matters:
    every column `feature_builder` produces must be CAUSAL — a row's value
    depends only on rows that precede it in auction/date order, never on which
    fold it lands in. `tests/test_features2.py` asserts this by truncation for
    every Phase 2 family. A non-causal builder here is a leak the fold loop
    would not catch either, since validation rows exist in the frame when
    training features are computed.

    `feature_builder` defaults to the Phase 1 matrix, so every existing caller
    and every Phase 1 number is untouched.
    """
    from . import data as D

    models = models or default_models(cfg)
    if "revealed_target" not in df.columns:
        df = D.add_revealed_target(df)
    missing_rate = float(df["revealed_target"].isna().mean())
    if missing_rate > 0.25 and verbose:
        # A date-strided frame has no date d-1 for any d, so every carry model
        # silently collapses onto predict-zero. Say so rather than reporting a
        # tie as if it were a measurement.
        print(
            f"WARNING: {missing_rate:.1%} of rows have no previous auction, so the carry "
            f"baselines are mostly predicting 0.0 and their MAE is not a measurement of carry."
        )
    X = (feature_builder or F.build)(df)
    folds = splits.make_folds(D.date_ids(df), cfg)

    per_fold: list[dict] = []
    oof = {m.name: np.full(len(df), np.nan) for m in models}
    coefs: dict[int, pd.Series] = {}
    importances: dict[str, dict[int, pd.Series]] = {}

    for fold in folds:
        tr, va = splits.fold_masks(df, fold)
        df_tr, df_va = df.loc[tr], df.loc[va]
        X_tr, X_va = X.loc[tr], X.loc[va]
        y_va = df_va["target"].to_numpy(np.float64)
        if verbose:
            print(f"{fold}  train {tr.sum():,} rows / val {va.sum():,} rows", flush=True)

        for m in models:
            m.fit(df_tr, X_tr, cfg)
            p = m.predict(df_va, X_va)
            oof[m.name][np.flatnonzero(va)] = p
            per_fold.append({"fold": fold.index, "model": m.name, "mae_bps": mae(y_va, p),
                             "n": int(np.isfinite(y_va).sum())})
            if isinstance(m, RidgeMicro) and m.name == "ridge":
                coefs[fold.index] = m.coefficients()
            if hasattr(m, "feature_importance"):
                importances.setdefault(m.name, {})[fold.index] = m.feature_importance()

    scored = np.isfinite(np.stack([oof[m.name] for m in models]).sum(axis=0))
    return {
        "folds": folds,
        "fold_table": splits.describe(folds),
        "per_fold": per_fold,
        "oof": oof,
        "scored_mask": scored,
        "df": df,
        "coefficients": coefs,
        "importances": importances,
        "carry_missing_rate": float(df["revealed_target"].isna().mean()),
    }
