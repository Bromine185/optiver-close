"""Phase 2 features: causality, determinism, and the null policy.

The claim `features2.py` makes is stronger than Phase 1's and easier to break:
not "row-wise" but CAUSAL — a row's features may use the past, so the tests must
prove they use ONLY the past. Both causality tests work by truncation: delete
the future, recompute, and demand the surviving rows are bit-identical. A
rolling window that peeks one bucket ahead, a cross-sectional statistic that
pools tomorrow, or a state feature built without the one-date shift all fail
this immediately, which is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optiver import config as C
from optiver import data as D
from optiver import features as F
from optiver import features2 as F2


@pytest.fixture(scope="module")
def df():
    return D.add_revealed_target(D.load(C.get_config("SMOKE")))


@pytest.fixture(scope="module")
def X2(df):
    return F2.build2(df)


def test_columns_and_order_are_pinned(X2):
    assert tuple(X2.columns) == F2.FEATURE2_NAMES


def test_build_is_deterministic(df, X2):
    again = F2.build2(df)
    pd.testing.assert_frame_equal(X2, again)


def test_no_nans_and_no_infs(X2):
    assert np.isfinite(X2.to_numpy()).all()


# --- causality, the load-bearing pair --------------------------------------

def test_within_auction_features_survive_truncating_the_future(df, X2):
    """Truncate the LAST date at bucket s; every surviving row must not change.

    The counterfactual is the live API's actual information set mid-auction:
    all previous dates complete (their targets were revealed overnight), today
    cut off at bucket s. An earlier draft truncated every date at s and failed
    on the state family — correctly, since the state uses ALL of yesterday's
    buckets, and by prediction time it legitimately has them. Equality here IS
    the no-lookahead property for every family at once: rolling may not read
    today's future buckets, the cross-section may not pool them, and the state
    may not read today at all.
    """
    last = df["date_id"].max()
    for s in (0, 250, 480):
        cut = df[(df["date_id"] < last) | (df["seconds_in_bucket"] <= s)]
        Xcut = F2.build2(cut)
        pd.testing.assert_frame_equal(Xcut, X2.loc[cut.index], check_exact=True)


def test_state_features_survive_truncating_future_dates(df, X2):
    """Delete every date after d; rows at or before d must not change.

    The state family's trailing window and the revealed-target join both walk
    the date axis, so this is the test that would catch a shift in the wrong
    direction or a window centred instead of trailing.
    """
    dates = np.sort(df["date_id"].unique())
    for d in (dates[8], dates[len(dates) // 2], dates[-1]):
        cut = df[df["date_id"] <= d]
        Xcut = F2.build2(cut)
        pd.testing.assert_frame_equal(Xcut, X2.loc[cut.index], check_exact=True)


def test_state_features_do_not_use_the_current_date(df, X2):
    """Scaling every target on date d must leave date d's OWN state features fixed.

    Truncation cannot catch a feature that reads the current date's targets —
    they are in the frame either way. Perturbation can: multiply date d's
    targets by 10 and rebuild. Date d's rows may not move (their state comes
    from dates < d); dates AFTER d must move (d is now in their trailing
    window), which also proves the test has teeth.
    """
    dates = np.sort(df["date_id"].unique())
    d = dates[len(dates) // 2]
    mutated = df.copy()
    on_d = mutated["date_id"] == d
    mutated.loc[on_d, "target"] *= 10.0
    # the revealed-target column is derived from targets; rebuild it too
    mutated = D.add_revealed_target(mutated.drop(columns="revealed_target"))
    Xmut = F2.build2(mutated)

    state = ["stock_vol_20d_bps", "stock_vol_is_missing"]
    pd.testing.assert_frame_equal(Xmut.loc[on_d, state], X2.loc[on_d, state])
    after = mutated["date_id"] > d
    assert not Xmut.loc[after, "stock_vol_20d_bps"].equals(X2.loc[after, "stock_vol_20d_bps"])


# --- the families behave like their definitions -----------------------------

def test_first_bucket_rolling_features_are_neutral_zero(df, X2):
    """At s=0 there is no lookback; every rolling column must be exactly 0.

    No indicator column accompanies this — the absence is a deterministic
    function of `seconds_in_bucket`, already a feature. The module docstring
    argues it; this test pins the neutral value side of the bargain.
    """
    first = df["seconds_in_bucket"] == 0
    for col in ("wap_ret_1b_bps", "wap_ret_6b_bps", "wap_vol_6b_bps", "imb_ratio_chg_1b",
                "size_imb_chg_1b", "matched_chg_1b", "spread_chg_1b_bps", "near_wap_chg_1b_bps"):
        assert (X2.loc[first, col] == 0.0).all(), col


def test_momentum_matches_a_hand_computed_auction(df, X2):
    """One auction, wap momentum recomputed with a plain loop. Bit-agreement."""
    s0, d0 = df.iloc[0][["stock_id", "date_id"]]
    a = df[(df["stock_id"] == s0) & (df["date_id"] == d0)].sort_values("seconds_in_bucket")
    wap = a["wap"].to_numpy(np.float64)
    want = np.zeros(len(a))
    want[1:] = (wap[1:] / wap[:-1] - 1.0) * 1e4
    np.testing.assert_array_equal(X2.loc[a.index, "wap_ret_1b_bps"].to_numpy(), want)


def test_near_change_is_gated_at_the_publication_boundary(df, X2):
    """At s=300 near_price appears; the change must read 0, not (value - 0).

    An ungated diff would hand the model a spike whose size is the LEVEL of the
    just-published cross — a duplicate of `near_minus_wap_bps` wearing a change
    feature's name — on exactly the bucket where the level first matters.
    """
    at_boundary = df["seconds_in_bucket"] == C.CROSS_PRICE_FIRST_SECOND
    has_near = df["near_price"].notna()
    assert (X2.loc[at_boundary & has_near, "near_wap_chg_1b_bps"] == 0.0).all()


def test_cross_sectional_columns_demean_within_the_instant(df, X2):
    """De-meaned columns must average ~0 within every (date, second) group."""
    g = X2.groupby([df["date_id"], df["seconds_in_bucket"]], observed=True)
    for col in ("wap_ret_1b_cs", "wap_ret_6b_cs", "wap_ref_cs"):
        assert g[col].mean().abs().max() < 1e-9, col


def test_ranks_are_bounded_and_centred(df, X2):
    for col in F2.BOUNDED2_NAMES:
        v = X2[col]
        assert v.min() >= -0.5 and v.max() <= 0.5, col


def test_state_indicator_fires_exactly_where_there_is_no_history(df, X2):
    """stock_vol_is_missing = 1 iff the stock has no earlier date in the frame.

    On the smoke fixture that is every stock's first date in each... no — the
    trailing window walks the stock's own date SEQUENCE, so only the very first
    observed date per stock qualifies. Gaps between the smoke fixture's date
    blocks do not reset it, exactly as the live revealed-target stream would
    not: the previous revealed date is simply further back.
    """
    first_date = df.groupby("stock_id", observed=True)["date_id"].transform("min")
    expected = (df["date_id"] == first_date).astype(np.float64)
    np.testing.assert_array_equal(X2["stock_vol_is_missing"].to_numpy(), expected.to_numpy())
    assert (X2.loc[expected == 1.0, "stock_vol_20d_bps"] == 0.0).all()


def test_revealed_abs_is_the_abs_of_the_revealed_target(df, X2):
    rev = df["revealed_target"].astype(np.float64)
    np.testing.assert_array_equal(
        X2["revealed_is_missing"].to_numpy(), rev.isna().astype(np.float64).to_numpy()
    )
    ok = rev.notna()
    np.testing.assert_array_equal(
        X2.loc[ok, "revealed_abs_bps"].to_numpy(), rev[ok].abs().to_numpy()
    )


def test_build_all_is_phase1_then_phase2(df, X2):
    Xall = F2.build_all(df)
    assert tuple(Xall.columns) == F2.ALL_NAMES
    pd.testing.assert_frame_equal(Xall[list(F.FEATURE_NAMES)], F.build(df))
    pd.testing.assert_frame_equal(Xall[list(F2.FEATURE2_NAMES)], X2)


def test_quantile_bounds_skip_extends_to_phase2_exemptions(X2):
    """The Phase 1 lesson, re-applied: a quantile bound on `revealed_is_missing`
    (fires on ~1% of smoke rows) or on a [-0.5, 0.5] rank is deletion, not
    winsorisation. The skip must actually reach them through `extra_skip`."""
    skip = tuple(F2.INDICATOR2_NAMES) + tuple(F2.BOUNDED2_NAMES)
    bounds = F.quantile_bounds(X2, extra_skip=skip)
    for name in skip:
        assert name not in bounds, name
    assert "wap_ret_1b_bps" in bounds
