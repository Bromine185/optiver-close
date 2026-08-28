"""Feature construction, and specifically the ways it could quietly corrupt data.

Most of these tests exist because the obvious wrong implementation would still
run, still produce finite numbers, and still train — it would just be wrong by a
factor of ten thousand on half the rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optiver import config as C
from optiver import features as F


def frame(**over) -> pd.DataFrame:
    base = {
        "stock_id": 1, "date_id": 0, "seconds_in_bucket": 0,
        "imbalance_size": 1_000.0, "imbalance_buy_sell_flag": 1,
        "reference_price": 1.0, "matched_size": 10_000.0,
        "far_price": np.nan, "near_price": np.nan,
        "bid_price": 0.9999, "bid_size": 100.0,
        "ask_price": 1.0001, "ask_size": 300.0, "wap": 1.0,
    }
    base.update(over)
    return pd.DataFrame({k: [v] for k, v in base.items()})


def test_price_differences_are_in_basis_points():
    """wap 1.0005 against reference 1.0000 is 5 bps, the same unit as the target,
    so a coefficient of 1.0 means 'one bps in moves one bps out'."""
    X = F.build(frame(wap=1.0005))
    assert X["wap_minus_reference_bps"].iloc[0] == pytest.approx(5.0, abs=1e-6)


def test_spread_is_in_bps_of_wap():
    X = F.build(frame(bid_price=0.9990, ask_price=1.0010, wap=1.0))
    assert X["spread_bps"].iloc[0] == pytest.approx(20.0, abs=1e-6)


def test_imbalance_ratio_is_signed_by_the_flag():
    buy = F.build(frame(imbalance_buy_sell_flag=1))["imbalance_ratio"].iloc[0]
    sell = F.build(frame(imbalance_buy_sell_flag=-1))["imbalance_ratio"].iloc[0]
    none = F.build(frame(imbalance_buy_sell_flag=0))["imbalance_ratio"].iloc[0]
    assert buy == pytest.approx(0.1)
    assert sell == pytest.approx(-0.1)
    assert none == 0.0


def test_size_imbalance_is_bounded():
    X = F.build(frame(bid_size=100.0, ask_size=300.0))
    assert X["size_imbalance"].iloc[0] == pytest.approx(-0.5)
    assert abs(X["size_imbalance"].iloc[0]) <= 1.0


def test_zero_denominators_do_not_produce_infinities():
    """bid_size, ask_size and imbalance_size all reach exactly 0 in the real data."""
    X = F.build(frame(bid_size=0.0, ask_size=0.0, imbalance_size=0.0, matched_size=0.0))
    assert np.isfinite(X.to_numpy()).all()
    assert X["size_imbalance"].iloc[0] == 0.0
    assert X["imbalance_ratio"].iloc[0] == 0.0


def test_missing_cross_prices_get_a_neutral_value_and_an_indicator():
    """The bug this prevents: fillna(0) on a price that lives at 1.0 produces a
    -10,000 bps deviation on 55% of the rows in the dataset."""
    X = F.build(frame(near_price=np.nan, far_price=np.nan))
    assert X["near_is_missing"].iloc[0] == 1.0
    assert X["far_is_missing"].iloc[0] == 1.0
    assert X["near_minus_wap_bps"].iloc[0] == 0.0        # neutral, NOT -10000
    assert X["far_minus_wap_bps"].iloc[0] == 0.0


def test_present_cross_prices_are_used():
    X = F.build(frame(near_price=1.0002, far_price=0.9997, wap=1.0, seconds_in_bucket=300))
    assert X["near_is_missing"].iloc[0] == 0.0
    assert X["near_minus_wap_bps"].iloc[0] == pytest.approx(2.0, abs=1e-6)
    assert X["far_minus_wap_bps"].iloc[0] == pytest.approx(-3.0, abs=1e-6)


def test_a_completely_absent_book_is_named_rather_than_filled():
    """Four stock-days in the full fixture have no book at all. They still carry
    targets, so they must remain predictable without inventing a price."""
    X = F.build(frame(wap=np.nan, reference_price=np.nan, bid_price=np.nan,
                      ask_price=np.nan, imbalance_size=np.nan, matched_size=np.nan))
    assert X["book_is_missing"].iloc[0] == 1.0
    assert np.isfinite(X.to_numpy()).all()
    assert X["wap_minus_reference_bps"].iloc[0] == 0.0


def test_a_nan_on_an_observable_book_raises_instead_of_being_swallowed():
    """The zero-fill is deliberately narrow: it applies only where the absence has
    already been recorded. A NaN anywhere else is a bug in this module and has to
    be loud."""
    df = frame()
    df["reference_price"] = np.nan          # wap present, so book_is_missing is 0
    with pytest.raises(ValueError, match="guard is missing"):
        F.build(df)


def test_seconds_fraction_spans_zero_to_one(smoke_df):
    X = F.build(smoke_df)
    assert X["seconds_frac"].min() == 0.0
    assert X["seconds_frac"].max() == pytest.approx(1.0)
    assert X["seconds_frac_sq"].equals(X["seconds_frac"] ** 2)


def test_features_are_finite_and_ordered_on_real_data(smoke_df):
    X = F.build(smoke_df)
    assert list(X.columns) == list(F.FEATURE_NAMES)
    assert len(X) == len(smoke_df)
    assert np.isfinite(X.to_numpy()).all()


def test_clip_bounds_come_from_training_rows_only(smoke_df):
    """far_price reaches 437 against a wap of 1.0, i.e. ~4.4 million bps. Clipping
    is necessary for least squares and must not be allowed to see validation
    extremes."""
    X = F.build(smoke_df)
    train = X.iloc[: len(X) // 2]
    bounds = F.quantile_bounds(train)
    clipped = F.clip_outliers(X, bounds)
    lo, hi = bounds["far_minus_wap_bps"]
    assert clipped["far_minus_wap_bps"].max() <= hi
    assert clipped["far_minus_wap_bps"].min() >= lo


def test_no_indicator_column_is_ever_clipped(smoke_df):
    """Assert the whole skip list, not one member of it — the spot-check this
    replaces named `far_is_missing`, which is why `book_is_missing` being absent
    from it went unnoticed."""
    X = F.build(smoke_df)
    bounds = F.quantile_bounds(X)
    for col in F.INDICATOR_NAMES + F.BOUNDED_NAMES:
        assert col in X.columns, f"{col} is in the skip list but not in the design matrix"
        assert col not in bounds, f"{col} must not be winsorised"
    clipped = F.clip_outliers(X, bounds)
    for col in F.INDICATOR_NAMES + F.BOUNDED_NAMES:
        assert clipped[col].equals(X[col]), f"{col} was altered by the clip"


def test_a_rare_indicator_is_not_flattened_by_the_clip():
    """The regression this is here for. `book_is_missing` fires on 220 of 5.2 M
    rows, so BOTH of its 0.1% quantiles are 0.0: bounding it clips the column to
    (0, 0), the model never sees the feature it was added for, and the ridge
    coefficient reads as a measured 0.000 rather than as a clip. Built here
    rather than taken from the smoke fixture, which contains none of the four
    no-book stock-days."""
    rows = pd.concat([frame(stock_id=i, seconds_in_bucket=10 * (i % 55)) for i in range(2_000)])
    prices = ["reference_price", "bid_price", "ask_price", "wap", "imbalance_size", "matched_size"]
    rows.iloc[:2, [rows.columns.get_loc(c) for c in prices]] = np.nan
    X = F.build(rows)
    assert X["book_is_missing"].sum() == 2.0

    clipped = F.clip_outliers(X, F.quantile_bounds(X))
    assert clipped["book_is_missing"].sum() == 2.0, "the clip deleted the indicator"
