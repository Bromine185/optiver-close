"""The split guarantees, asserted.

These are the tests that matter most in the repo. Every reported MAE is a claim
about generalisation, and a claim about generalisation is only as good as the
promise that no validation date was visible during fitting. That promise is four
properties, and each has a test:

  1. no training date at or after any validation date          (forward chaining)
  2. no training date within `embargo` of the validation start (the embargo)
  3. every fold has non-empty train and validation sets        (no silent no-ops)
  4. train and validation never share a date                   (no overlap)

Plus the one that is easy to forget: the guarantee must survive a date axis with
holes, because the committed smoke fixture has them and because stock-days go
missing in the real data.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiver import config as C
from optiver import data as D
from optiver import splits

CFG = C.get_config("SMOKE", n_folds=5, val_dates_per_fold=10, embargo_dates=3)


@pytest.fixture
def dates():
    return np.arange(100)


@pytest.fixture
def folds(dates):
    return splits.make_folds(dates, CFG)


def test_no_training_date_reaches_the_validation_block(folds):
    """The headline invariant, stated exactly as CLAUDE.md states it."""
    for f in folds:
        assert f.train_dates.max() < f.val_dates.min() - f.embargo_dates
        # and, more bluntly: no individual training date may be >= any validation
        # date minus the embargo.
        assert (f.train_dates[:, None] < f.val_dates[None, :] - f.embargo_dates).all()


def test_every_fold_is_non_empty(folds):
    for f in folds:
        assert f.train_dates.size > 0
        assert f.val_dates.size == CFG.val_dates_per_fold


def test_train_and_validation_never_share_a_date(folds):
    for f in folds:
        assert np.intersect1d(f.train_dates, f.val_dates).size == 0


def test_the_embargo_actually_removes_dates(folds, dates):
    """Dates in the gap belong to NEITHER set. Without this, an implementation
    that computed the right boundaries but forgot to drop anything would pass
    every other test in this file."""
    for f in folds:
        assert f.embargoed_dates.size == CFG.embargo_dates
        assigned = np.concatenate([f.train_dates, f.val_dates])
        assert np.intersect1d(f.embargoed_dates, assigned).size == 0
        assert (f.embargoed_dates > f.train_dates.max()).all()
        assert (f.embargoed_dates < f.val_dates.min()).all()


def test_validation_blocks_march_forward_and_do_not_overlap(folds):
    starts = [f.val_dates.min() for f in folds]
    assert starts == sorted(starts)
    for a, b in zip(folds, folds[1:]):
        assert a.val_dates.max() < b.val_dates.min()
    assert np.intersect1d(*[f.val_dates for f in folds[:2]]).size == 0


def test_training_sets_expand(folds):
    for a, b in zip(folds, folds[1:]):
        assert b.train_dates.size > a.train_dates.size
        assert set(a.train_dates).issubset(set(b.train_dates))


def test_validation_covers_the_tail_of_the_timeline(dates, folds):
    covered = np.concatenate([f.val_dates for f in folds])
    assert covered.max() == dates.max()
    assert covered.size == CFG.n_folds * CFG.val_dates_per_fold


def test_rolling_window_caps_the_training_set(dates):
    cfg = C.get_config("SMOKE", n_folds=3, val_dates_per_fold=10, embargo_dates=2, max_train_dates=20)
    for f in splits.make_folds(dates, cfg):
        assert f.train_dates.size == 20
        assert f.train_dates.max() < f.val_dates.min() - f.embargo_dates


# --- the gappy axis -------------------------------------------------------

def test_guarantee_holds_when_date_ids_are_not_contiguous():
    """A strided axis (dates 0, 7, 14, ...) is the case where "drop 3 positions"
    and "drop 3 date_ids" mean different things. Both properties must still hold,
    and the position property is the one that would silently be violated by an
    implementation that filtered on `date_id < val_start - embargo`."""
    dates = np.arange(0, 350, 7)
    cfg = C.get_config("SMOKE", n_folds=4, val_dates_per_fold=8, embargo_dates=3)
    for f in splits.make_folds(dates, cfg):
        assert f.train_dates.max() < f.val_dates.min() - f.embargo_dates
        assert f.embargoed_dates.size == 3
        # 3 embargoed POSITIONS is 21 date_id units on this axis; the date_id
        # check alone would have accepted a gap of 3.
        assert f.val_dates.min() - f.train_dates.max() == 4 * 7


def test_check_fold_rejects_a_hand_built_leaky_fold():
    leaky = splits.Fold(
        index=0,
        train_dates=np.arange(0, 20),
        val_dates=np.arange(18, 28),          # overlaps train
        embargoed_dates=np.array([], dtype=int),
        embargo_dates=0,
    )
    with pytest.raises(AssertionError, match="overlap"):
        splits.check_fold(leaky)

    touching = splits.Fold(
        index=0,
        train_dates=np.arange(0, 20),
        val_dates=np.arange(20, 30),          # no gap, but embargo claims 3
        embargoed_dates=np.array([], dtype=int),
        embargo_dates=3,
    )
    with pytest.raises(AssertionError, match="embargo"):
        splits.check_fold(touching)

    unpurged = splits.Fold(
        index=0,
        train_dates=np.arange(0, 20),
        val_dates=np.arange(25, 35),
        embargoed_dates=np.array([], dtype=int),   # gap exists but nothing was dropped
        embargo_dates=3,
    )
    with pytest.raises(AssertionError, match="removed 0 dates"):
        splits.check_fold(unpurged)


def test_too_few_dates_raises_rather_than_returning_a_degenerate_fold():
    with pytest.raises(ValueError, match="not enough"):
        splits.make_folds(np.arange(20), C.get_config("SMOKE", n_folds=5, val_dates_per_fold=10))


def test_the_minimum_date_count_accounts_for_the_embargo():
    """Exactly at the boundary: k*v + e dates is one short, k*v + e + 1 works and
    leaves fold 0 with a single training date. An implementation that forgot to
    charge the embargo against the budget would accept the first case and hand
    back a fold with an empty training set."""
    cfg = C.get_config("SMOKE", n_folds=3, val_dates_per_fold=10, embargo_dates=2)
    with pytest.raises(ValueError, match="not enough"):
        splits.make_folds(np.arange(3 * 10 + 2), cfg)
    folds = splits.make_folds(np.arange(3 * 10 + 2 + 1), cfg)
    assert folds[0].train_dates.tolist() == [0]


# --- against the real fixture --------------------------------------------

def test_fold_masks_partition_the_rows_they_claim(smoke_df, smoke_cfg):
    for f in splits.make_folds(D.date_ids(smoke_df), smoke_cfg):
        tr, va = splits.fold_masks(smoke_df, f)
        assert not (tr & va).any()
        assert tr.sum() > 0 and va.sum() > 0
        assert set(smoke_df.loc[tr, "date_id"]) == set(f.train_dates.tolist())
        assert set(smoke_df.loc[va, "date_id"]) == set(f.val_dates.tolist())
        # the property restated at ROW level, which is where leakage would bite
        assert smoke_df.loc[tr, "date_id"].max() < smoke_df.loc[va, "date_id"].min() - f.embargo_dates


def test_smoke_preset_produces_usable_folds(smoke_df, smoke_cfg):
    folds = splits.make_folds(D.date_ids(smoke_df), smoke_cfg)
    assert len(folds) == smoke_cfg.n_folds
    tbl = splits.describe(folds)
    assert (tbl["train_dates"] > 0).all() and (tbl["val_dates"] > 0).all()
    assert (tbl["purged_dates"] == smoke_cfg.embargo_dates).all()
