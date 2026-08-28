"""Purged, embargoed, forward-chaining cross-validation on date_id.

THE LOAD-BEARING FILE. Every number this project reports is only as trustworthy
as this module. A model evaluated on a leaky split is not a weak result, it is
not a result at all, and the failure mode is silent: leakage makes the score
better, so nothing ever prompts you to look.

There are three distinct ways this dataset leaks, and each guard here answers
exactly one of them. They are worth separating, because the usual reflex ("use
TimeSeriesSplit") only closes the first.

1. ROW-LEVEL leakage: one auction is 55 correlated rows
   ------------------------------------------------------------------
   A random K-fold puts some of a stock-day's 55 buckets in train and the rest
   in validation. Those rows are near-duplicates of each other — the book at
   s=300 and s=310 differ by ten seconds — and, worse, their labels physically
   overlap: the target at s is the move over [s, s+60], so the labels at
   s=300 and s=310 share 50 of their 60 seconds. Memorising the neighbour is
   most of the job.
   GUARD: never split on rows. The unit of assignment is a date_id, and a date
   takes all 200 stocks and all 55 buckets with it.

2. CROSS-SECTIONAL leakage: the target is index-relative
   ------------------------------------------------------------------
   target_i(t) = (WAP_i(t+60)/WAP_i(t) - Index(t+60)/Index(t)) * 10^4.
   Every stock on the same (date, second) is quoted against the SAME index leg.
   Verified directly on the fixture: (stock 60-second WAP return in bps) minus
   (target) is constant across stocks within a (date, second) to a cross-
   sectional standard deviation of 0.005 bps — the source data's own price
   rounding, i.e. exactly constant. See RESEARCH.md.
   So a split that assigns stock A's date-d rows to train and stock B's date-d
   rows to validation hands the model the index move it is being asked to
   subtract. This is *not* fixed by grouping on stock-day; it is only fixed by
   keeping whole dates together.
   GUARD: same one. Dates are indivisible.

3. TEMPORAL leakage: neighbouring dates are not independent
   ------------------------------------------------------------------
   Note carefully what is NOT the issue: the label horizon is 60 seconds and the
   auction is 540 seconds long, so no label ever crosses a date boundary. There
   is no label overlap between date d and date d+1, and a purge in the strict
   Lopez de Prado sense (drop training labels whose horizon reaches into the
   validation window) removes nothing here.
   What does cross the boundary is STATE. Per-stock volatility, imbalance
   regimes, index composition, and any rolling feature computed over recent
   dates all persist day to day, so date d and date d+1 are far more alike than
   two dates a month apart. Training right up to the validation boundary
   therefore flatters any model whose features have memory.
   GUARD: an embargo of `embargo_dates` trading days between the end of training
   and the start of validation, plus strict forward chaining (training data is
   always in the past).

The embargo is stated as a deviation, not smuggled in: we apply it for feature
autocorrelation, not for label overlap, and Phase 1's features have no memory at
all, so the embargo costs 5 dates per fold and buys nothing *yet*. It is here so
that when Phase 2 adds rolling per-stock statistics, the harness does not have to
change and the Phase 1 numbers stay comparable to the Phase 2 ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config


@dataclass(frozen=True)
class Fold:
    """One forward-chaining fold, expressed as date_ids rather than row indices.

    Dates, not rows, because that is the unit that makes the guarantee checkable:
    `check_fold` can assert a property of two small integer arrays, and it stays
    true no matter which stocks or buckets happen to be present.
    """

    index: int
    train_dates: np.ndarray
    val_dates: np.ndarray
    #: The dates actually removed by the embargo, kept so `check_fold` can assert
    #: the gap was applied rather than inferring it from date_id arithmetic. Those
    #: two are the same thing only when the date axis is contiguous, and SMOKE's
    #: is not: it is 8 blocks of 8 consecutive dates, so positions inside a block
    #: are 1 date_id apart but the step from 7 to 67 spans 60. Embargoing two
    #: POSITIONS across that seam removes two dates and a 60-date gap, which
    #: date_id arithmetic would report as an embargo of 59. Only this array
    #: records what was really dropped.
    embargoed_dates: np.ndarray
    embargo_dates: int

    @property
    def train_date_max(self) -> int:
        return int(self.train_dates.max())

    @property
    def val_date_min(self) -> int:
        return int(self.val_dates.min())

    @property
    def val_date_max(self) -> int:
        return int(self.val_dates.max())

    @property
    def gap(self) -> int:
        """Dates strictly between the last training date and the first validation date."""
        return self.val_date_min - self.train_date_max - 1

    def __repr__(self) -> str:
        return (
            f"Fold({self.index}: train {int(self.train_dates.min())}..{self.train_date_max} "
            f"[{len(self.train_dates)}d], gap {self.gap}, "
            f"val {self.val_date_min}..{self.val_date_max} [{len(self.val_dates)}d])"
        )


def make_folds(dates: np.ndarray, cfg: Config) -> list[Fold]:
    """Forward-chaining folds over the sorted unique date_ids.

    Layout, for n_folds=3, val_dates_per_fold=4, embargo=2 on dates 0..19:

        fold 0  train 0..5   embargo 6,7    val  8..11
        fold 1  train 0..9   embargo 10,11  val 12..15
        fold 2  train 0..13  embargo 14,15  val 16..19

    Validation blocks are contiguous, non-overlapping, and take the TAIL of the
    timeline; the head is reserved as fold 0's training history. Training is
    expanding by default (fold k sees everything before its embargo, including
    earlier folds' validation dates, which is correct — by the time fold k is
    fit, those dates are in the past). `cfg.max_train_dates` switches to a
    rolling window instead.

    Raises rather than returning a degenerate fold. A silently empty training set
    scores like a bug in the model, and the point of this module is that its
    failures are loud.
    """
    dates = np.asarray(np.sort(np.unique(dates)))
    n = dates.size
    v, k, e = cfg.val_dates_per_fold, cfg.n_folds, cfg.embargo_dates

    # k*v dates go to validation blocks, e more are eaten by fold 0's embargo, and
    # at least one must survive as fold 0's training history. That single check is
    # the only gate: satisfy it and no fold can come out empty, which is why there
    # is no second "empty training set" branch further down pretending to catch
    # something this one already made impossible.
    need = k * v + e + 1
    if n < need:
        raise ValueError(
            f"{n} dates is not enough for {k} folds x {v} validation dates with an "
            f"embargo of {e} (need at least {need}). Lower n_folds, val_dates_per_fold, "
            f"or embargo_dates."
        )

    first_val = n - k * v
    folds: list[Fold] = []
    for i in range(k):
        vs = first_val + i * v
        val = dates[vs : vs + v]
        # Purge + embargo: everything within `e` positions of the validation
        # block's start is dropped from training.
        train_end = vs - e
        train = dates[:train_end]
        embargoed = dates[train_end:vs]
        if cfg.max_train_dates is not None:
            train = train[-cfg.max_train_dates :]
        folds.append(Fold(i, train, val, embargoed, e))

    for f in folds:
        check_fold(f)
    return folds


def check_fold(fold: Fold) -> None:
    """The invariant, asserted. Called on construction and again by the tests.

    `train_date_max <= val_date_min - embargo - 1` is the whole contract: no
    training date may sit inside the embargo window, and none may sit at or after
    the validation block. Written as an explicit comparison of two integers so
    that it means something on a fixture with gaps in its date_ids, where "the
    previous 5 dates" and "date_id - 5" are different things.
    """
    if fold.train_dates.size == 0:
        raise AssertionError(f"fold {fold.index}: empty training set")
    if fold.val_dates.size == 0:
        raise AssertionError(f"fold {fold.index}: empty validation set")
    if not (np.diff(fold.train_dates) > 0).all():
        raise AssertionError(f"fold {fold.index}: train_dates not strictly increasing")
    if not (np.diff(fold.val_dates) > 0).all():
        raise AssertionError(f"fold {fold.index}: val_dates not strictly increasing")
    if np.intersect1d(fold.train_dates, fold.val_dates).size:
        raise AssertionError(f"fold {fold.index}: train and validation dates overlap")

    # (a) The date_id-arithmetic property: no training date within `embargo` units
    #     of the validation block. This is the guarantee stated in CLAUDE.md.
    limit = fold.val_date_min - fold.embargo_dates
    if fold.train_date_max >= limit:
        raise AssertionError(
            f"fold {fold.index}: train_date_max={fold.train_date_max} violates the embargo "
            f"(must be < val_date_min - embargo = {limit})"
        )

    # (b) The position property: `embargo` *available* dates were actually
    #     dropped, and they sit strictly in the gap. Stronger than (a) whenever
    #     the date axis has holes, where (a) alone would pass on an unpurged fold.
    emb = fold.embargoed_dates
    if emb.size != fold.embargo_dates:
        raise AssertionError(
            f"fold {fold.index}: embargo removed {emb.size} dates, expected {fold.embargo_dates}"
        )
    if emb.size and not (emb.min() > fold.train_date_max and emb.max() < fold.val_date_min):
        raise AssertionError(
            f"fold {fold.index}: embargoed dates {emb.tolist()} are not strictly between "
            f"train_date_max={fold.train_date_max} and val_date_min={fold.val_date_min}"
        )


def fold_masks(df: pd.DataFrame, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Boolean row masks for a fold. The only place rows meet dates."""
    d = df["date_id"].to_numpy()
    return np.isin(d, fold.train_dates), np.isin(d, fold.val_dates)


def describe(folds: list[Fold]) -> pd.DataFrame:
    """One row per fold: the table that goes in the research log."""
    return pd.DataFrame(
        [
            {
                "fold": f.index,
                "train_dates": len(f.train_dates),
                "train_from": int(f.train_dates.min()),
                "train_to": f.train_date_max,
                "embargo": f.embargo_dates,
                "gap": f.gap,
                "purged_dates": len(f.embargoed_dates),
                "val_dates": len(f.val_dates),
                "val_from": f.val_date_min,
                "val_to": f.val_date_max,
            }
            for f in folds
        ]
    )
