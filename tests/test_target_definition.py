"""What the target actually is, pinned as a test rather than quoted from a forum.

    target_i(t) = ( WAP_i(t+60)/WAP_i(t) - Index(t+60)/Index(t) ) * 10^4

The index weights are not in the staged data, so the index leg cannot be
reconstructed directly. It does not need to be: the leg is COMMON to every stock
in a given (date_id, seconds_in_bucket), so subtracting it out is as simple as
checking that

    (stock's own 60-second WAP return in bps) - target

is constant across stocks within that (date, second). It is, to a cross-sectional
standard deviation of ~0.005 bps — the source data's own six-decimal price
quantisation.

Two things follow, and both are load-bearing elsewhere:

  * the horizon is exactly 6 buckets, which is why `config.HORIZON_BUCKETS` is 6
    and why the target for s >= 490 cannot be reconstructed at all (t+60 lands
    past the end of the auction, in prices the fixture does not contain);
  * every stock on a date shares an index term, which is the second and less
    obvious reason splits.py refuses to divide a date across folds.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiver import config as C
from optiver.data import index_leg as _index_leg


def test_target_is_a_60_second_return_minus_a_common_index_leg(smoke_df):
    d = _index_leg(smoke_df, C.HORIZON_BUCKETS)
    spread = d.groupby(["date_id", "seconds_in_bucket"])["leg"].std()
    assert spread.max() < 0.02, "the residual index leg is not constant across stocks"
    assert spread.median() < 0.01


def test_the_horizon_is_six_buckets_and_not_five_or_seven(smoke_df):
    """The same check at the wrong lag must fail loudly, or the test above proves
    nothing about the horizon."""
    right = _index_leg(smoke_df, 6).groupby(["date_id", "seconds_in_bucket"])["leg"].std().median()
    for wrong in (5, 7):
        got = _index_leg(smoke_df, wrong).groupby(["date_id", "seconds_in_bucket"])["leg"].std().median()
        assert got > 50 * right, f"lag {wrong} is indistinguishable from lag 6"


def test_the_index_is_not_equally_weighted(smoke_df):
    """Recorded because it is the reason the equal-weighted cross-sectional mean
    target is not exactly zero — a fact that would otherwise look like a bug in
    the fixture."""
    d = _index_leg(smoke_df, C.HORIZON_BUCKETS)
    g = d.groupby(["date_id", "seconds_in_bucket"])
    leg = g["leg"].mean()
    equal_weighted = g["ret_bps"].mean()
    assert np.corrcoef(leg, equal_weighted)[0, 1] > 0.9      # close, as it must be
    assert (leg - equal_weighted).std() > 0.1                # but not the same thing


def test_the_target_is_near_zero_mean_by_construction(smoke_df):
    """The whole reason predict-zero is a hard baseline rather than a straw man."""
    t = smoke_df["target"].dropna()
    assert abs(t.mean()) < 0.1
    assert abs(t.median()) < 0.1
    assert t.std() == pytest.approx(9.4, abs=0.5)
