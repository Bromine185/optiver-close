"""Phase 2 features: the ones with memory.

Phase 1's features were deliberately row-wise — no rolling windows, no group
statistics — because a feature with memory needs the embargo to be doing real
work before its number can be believed. The harness paid for that embargo up
front (5 dates per fold, buying nothing in Phase 1); this module is what it was
bought for.

Three families, each using a different, individually-argued source of past
information. Everything is CAUSAL: a row's features depend only on information
the live timeseries API would have delivered before that row's prediction is
due. `tests/test_features2.py` asserts this by truncation — recompute on data
cut off at bucket s (or date d) and the surviving rows must be bit-identical.

**Within-auction rolling** (past buckets of the same auction). The API delivers
buckets in order, so at bucket s every earlier bucket of today's auction is on
the table. Momentum, realised vol so far, and one-bucket changes in the book
state. The first buckets of each auction have no lookback; that absence gets a
NEUTRAL value (0 = "no move observed") and — deliberately — NO indicator column,
breaking with the Phase 1 convention for a reason worth stating: truncation here
is a deterministic function of `seconds_in_bucket`, which is already a feature,
so an indicator would be an exact copy of `seconds_frac < k/54` and carry no
information the model does not have. Phase 1's indicators earn their keep where
absence varies row to row (a stock with no book); these do not.

**Cross-sectional** (other stocks, same instant). All 200 stocks arrive in one
API response per bucket, so the cross-section at (date, second) is available at
prediction time by construction. The target is index-relative, so the natural
coordinate for every predictor is also relative: a stock's momentum only means
something against what the market did in the same 10 seconds. De-meaning by the
(date, second) group is a first-order stand-in for the index leg itself (the
true weights are not in the staged data; the leg correlates 0.957 with
equal-weighted).

**Cross-auction state** (previous dates' revealed targets). Phase 1 killed
"yesterday's move persists" — carrying the revealed target as a LEVEL is 42.6%
worse than zero at ρ = 0.027 — and the verdict named the survivor: revealed
targets are worth carrying as STATE. So the level does not appear here; its
absolute value does, twice. A trailing per-stock scale (how loud is this name
lately) and yesterday's same-bucket |move| (how loud was this exact moment
yesterday). Both are volatility estimates, not direction bets. These are the two
families whose feature autocorrelation the embargo exists to contain.

Null policy for the state family follows Phase 1 exactly: absence (a stock's
first observed date, a missing d−1) is an indicator plus a neutral 0 — here
absence is NOT a function of time-of-day, it varies by stock history, so the
indicator carries real information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import features as F

BPS = 1e4

#: Phase 2 columns, fixed order, same reason as `features.FEATURE_NAMES`.
FEATURE2_NAMES = (
    # within-auction rolling
    "wap_ret_1b_bps",
    "wap_ret_6b_bps",
    "wap_vol_6b_bps",
    "imb_ratio_chg_1b",
    "size_imb_chg_1b",
    "matched_chg_1b",
    "spread_chg_1b_bps",
    "near_wap_chg_1b_bps",
    # cross-sectional
    "wap_ret_1b_cs",
    "wap_ret_6b_cs",
    "imb_ratio_cs_rank",
    "wap_ref_cs",
    "spread_cs_rank",
    # cross-auction state
    "stock_vol_20d_bps",
    "stock_vol_is_missing",
    "revealed_abs_bps",
    "revealed_is_missing",
)

#: Extra names for `features.quantile_bounds`'s skip list, same taxonomy.
INDICATOR2_NAMES = ("stock_vol_is_missing", "revealed_is_missing")
BOUNDED2_NAMES = ("imb_ratio_cs_rank", "spread_cs_rank")

ALL_NAMES = tuple(F.FEATURE_NAMES) + FEATURE2_NAMES


def _by_auction(df: pd.DataFrame):
    """Group by one auction. Rows within a group are already in bucket order
    because `data.load` sorts by (date_id, seconds_in_bucket, stock_id) and
    groupby preserves the frame's row order inside each group."""
    return df.groupby(["stock_id", "date_id"], observed=True, sort=False)


def _by_instant(df: pd.DataFrame):
    """Group by (date, second): the cross-section one API response delivers."""
    return df.groupby(["date_id", "seconds_in_bucket"], observed=True, sort=False)


def build2(df: pd.DataFrame) -> pd.DataFrame:
    """The Phase 2 columns alone. `build_all` is the entry point models use."""
    out = {}
    wap = df["wap"].astype(np.float64)
    auct = _by_auction(df)

    # --- within-auction rolling ------------------------------------------
    # One- and six-bucket own-WAP momentum. shift(k) inside the auction never
    # crosses a date or a stock, so the only NaNs are the auction's first k
    # buckets — the deterministic-in-time case argued in the module docstring.
    for k, name in ((1, "wap_ret_1b_bps"), (C.HORIZON_BUCKETS, "wap_ret_6b_bps")):
        prev = auct["wap"].shift(k).astype(np.float64)
        out[name] = ((wap / prev - 1.0) * BPS).fillna(0.0)

    # Realised vol of the auction so far: rolling std of the 1-bucket return.
    # min_periods=2 because a std of one observation is not a dispersion.
    r1 = pd.Series(out["wap_ret_1b_bps"], index=df.index)
    out["wap_vol_6b_bps"] = (
        r1.groupby([df["stock_id"], df["date_id"]], observed=True)
        .rolling(C.HORIZON_BUCKETS, min_periods=2).std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
    )

    # One-bucket changes in the book state. Built from the same definitions as
    # the Phase 1 levels so the pairs (level, change) share units exactly.
    flag = df["imbalance_buy_sell_flag"].astype(np.float64)
    imb_ratio = flag * pd.Series(
        F._safe_ratio(df["imbalance_size"], df["matched_size"]), index=df.index
    )
    size_imb = pd.Series(
        F._safe_ratio(
            df["bid_size"].astype(np.float64) - df["ask_size"].astype(np.float64),
            df["bid_size"].astype(np.float64) + df["ask_size"].astype(np.float64),
        ),
        index=df.index,
    )
    spread_bps = pd.Series(
        F._safe_ratio(df["ask_price"].astype(np.float64) - df["bid_price"].astype(np.float64), wap)
        * BPS,
        index=df.index,
    )
    grp = [df["stock_id"], df["date_id"]]
    for name, s in (
        ("imb_ratio_chg_1b", imb_ratio),
        ("size_imb_chg_1b", size_imb),
        ("spread_chg_1b_bps", spread_bps),
    ):
        out[name] = (s - s.groupby(grp, observed=True).shift(1)).fillna(0.0)

    # Matched size growth as a ratio, not a difference: matched_size spans four
    # orders of magnitude across stocks and a raw difference is a stock-identity
    # proxy, which Phase 1's size convention exists to keep out.
    matched = df["matched_size"].astype(np.float64)
    prev_m = auct["matched_size"].shift(1).astype(np.float64)
    out["matched_chg_1b"] = pd.Series(
        F._safe_ratio(matched - prev_m, prev_m), index=df.index
    ).fillna(0.0)

    # Movement of the indicative cross. Gated on BOTH buckets having a published
    # near_price: at s = 300 the previous bucket has none by construction, and
    # "appeared" must not read as a (value − 0) jump of the level's whole size.
    near_dev = (df["near_price"].astype(np.float64) - wap) * BPS  # NaN pre-300s
    chg = near_dev - near_dev.groupby(grp, observed=True).shift(1)
    out["near_wap_chg_1b_bps"] = chg.fillna(0.0)

    # --- cross-sectional --------------------------------------------------
    # De-meaned momentum: own move minus the cross-section's mean move over the
    # same window — the index-relative coordinate the target itself lives in.
    inst = [df["date_id"], df["seconds_in_bucket"]]
    for src, name in (("wap_ret_1b_bps", "wap_ret_1b_cs"), ("wap_ret_6b_bps", "wap_ret_6b_cs")):
        s = pd.Series(out[src], index=df.index)
        out[name] = s - s.groupby(inst, observed=True).transform("mean")

    wref = (wap - df["reference_price"].astype(np.float64)) * BPS
    out["wap_ref_cs"] = (wref - wref.groupby(inst, observed=True).transform("mean")).fillna(0.0)

    # Ranks in [-0.5, 0.5]: scale-free position in today's cross-section, robust
    # to the fat tails that make the raw columns need winsorising.
    for src, name in ((imb_ratio, "imb_ratio_cs_rank"), (spread_bps, "spread_cs_rank")):
        out[name] = (src.groupby(inst, observed=True).rank(pct=True) - 0.5).fillna(0.0)

    # --- cross-auction state ---------------------------------------------
    # Trailing per-stock scale from revealed targets: mean |target| per stock-day,
    # shifted one date (d sees only dates < d), trailing 20-day mean. The shift is
    # on the per-stock DATE SEQUENCE, so a stock absent for a gap sees its last
    # traded date, exactly as the API's revealed-targets stream would deliver.
    day_scale = (
        df.assign(abs_t=df["target"].abs())
        .groupby(["stock_id", "date_id"], observed=True)["abs_t"]
        .mean()
    )
    state = (
        day_scale.groupby(level=0, observed=True)
        .apply(lambda s: s.shift(1).rolling(20, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    joined = pd.MultiIndex.from_frame(df[["stock_id", "date_id"]]).map(state)
    vol = pd.Series(np.asarray(joined, dtype=np.float64), index=df.index)
    out["stock_vol_is_missing"] = vol.isna().astype(np.float64)
    out["stock_vol_20d_bps"] = vol.fillna(0.0)

    # Yesterday's |move| at this exact bucket — the state form of the killed
    # carry level. Requires data.add_revealed_target to have run.
    if "revealed_target" not in df.columns:
        raise ValueError("build2 needs the revealed_target column; call data.add_revealed_target first")
    rev = df["revealed_target"].astype(np.float64)
    out["revealed_is_missing"] = rev.isna().astype(np.float64)
    out["revealed_abs_bps"] = rev.abs().fillna(0.0)

    res = pd.DataFrame(out, index=df.index)[list(FEATURE2_NAMES)]

    # Same guard, same argument as features.build: a non-finite value that is
    # not explained by a whole-book hole is a missing guard in THIS module.
    book_missing = ~np.isfinite(wap.to_numpy())
    vals = res.to_numpy()
    unexplained = ~np.isfinite(vals) & ~book_missing[:, None]
    if unexplained.any():
        bad = res.columns[unexplained.any(axis=0)].tolist()
        raise ValueError(f"non-finite Phase 2 features in {bad} on rows with an observable book")
    return res.fillna(0.0).replace([np.inf, -np.inf], 0.0)


def build_all(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 1 + Phase 2 design matrix, columns in `ALL_NAMES` order.

    Safe to build ONCE for the whole frame and slice per fold — the condition is
    not "row-wise" (Phase 1's argument) but the strictly weaker one that matters:
    every column is CAUSAL, so a row's value cannot depend on which fold it lands
    in, only on rows that precede it in auction/date order. The truncation tests
    make that claim executable.
    """
    return pd.concat([F.build(df), build2(df)], axis=1)[list(ALL_NAMES)]
