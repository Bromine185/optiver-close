"""The obvious microstructure features, and nothing clever.

Phase 1's model exists to answer one question — is there any linear signal at all
above predict-zero — so the feature set is deliberately the short list any
auction trader would name first, computed row-wise with no memory of other rows.
No rolling windows, no cross-sectional ranks, no per-stock statistics. That
restraint is the point: a rich feature set on an unvalidated split teaches you
nothing, and every feature with memory would need the embargo to be doing real
work before its number could be believed.

Two conventions run through the whole file.

**Everything is in basis points or is already dimensionless.** The staged prices
are normalised per stock to sit near 1.0, so a price *difference* is directly
comparable across stocks once multiplied by 10^4 — and 10^4 is the target's unit
too, which makes a coefficient of 1.0 mean "one bps of feature moves the
prediction one bps". Sizes are only ever used as ratios, never as levels: raw
size is a proxy for market cap and would let the model learn stock identity
rather than auction state.

**Missing stays missing.** `far_price` and `near_price` do not exist before the
five-minute mark. They are filled with a NEUTRAL value (the deviation a fully
uninformative cross would imply, i.e. zero bps) and paired with an explicit
indicator column, so the model can price the absence separately from the value.
The tempting `fillna(0)` on the raw price is a catastrophe hiding in plain
sight: these prices live at 1.0, so zero is a -10,000 bps deviation, three
orders of magnitude outside the target's entire range, applied to 55% of rows.

The same treatment, for the same reason, covers a second and much rarer hole:
four stock-days in the full fixture — (19, 438), (101, 328), (131, 35),
(158, 388) — have no PRICE book at all. For all 55 buckets `reference_price`,
`bid_price`, `ask_price`, `wap`, `imbalance_size` and `matched_size` are null,
yet 132 of those 220 rows still carry a target. The hole is partial, not total:
`bid_size`/`ask_size` survive (non-zero on 133 of the 220 rows) and
`imbalance_buy_sell_flag` is non-zero on 69, so `size_imbalance`,
`imbalance_flag` and the time columns still carry real values on these rows while
every price-derived column is zeroed. `book_is_missing = 1` names that state —
a row you must predict and can only half observe — rather than leaving a NaN to
be swept up by whatever `fillna` runs last.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

BPS = 1e4

#: The design matrix, in a fixed order. Fixed because the ridge coefficients are
#: reported by name in RESEARCH.md and a reordering would silently relabel them.
FEATURE_NAMES = (
    "imbalance_ratio",
    "imbalance_flag",
    "wap_minus_reference_bps",
    "spread_bps",
    "size_imbalance",
    "matched_share",
    "reference_minus_mid_bps",
    "near_minus_wap_bps",
    "far_minus_wap_bps",
    "near_is_missing",
    "far_is_missing",
    "book_is_missing",
    "seconds_frac",
    "seconds_frac_sq",
)

#: Columns `quantile_bounds` must never bound. Two kinds, one reason each.
#:
#: The indicators are 0/1 (or -1/0/+1) flags, and a quantile bound on a rare flag
#: is not winsorisation, it is deletion — see `quantile_bounds`.
INDICATOR_NAMES = ("imbalance_flag", "near_is_missing", "far_is_missing", "book_is_missing")
#: The time coordinates are a deterministic function of `seconds_in_bucket`,
#: bounded in [0, 1] with no tail to trim.
BOUNDED_NAMES = ("seconds_frac", "seconds_frac_sq")


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num/den with a zero denominator mapped to 0 rather than inf.

    `bid_size`, `ask_size` and `imbalance_size` all reach exactly 0 in the raw
    data (a one-sided or perfectly balanced book), so this is a real branch, not
    defensive padding.
    """
    den = np.asarray(den, dtype=np.float64)
    out = np.zeros_like(den)
    ok = den > 0
    np.divide(np.asarray(num, dtype=np.float64), den, out=out, where=ok)
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features. Returns a float64 frame with `FEATURE_NAMES` columns.

    float64 despite the float32 fixture: the design matrix is ~13 columns and the
    ridge normal equations are solved on X^T X, where accumulating 4 M rows in
    float32 loses more precision than the storage saves. The fixture is float32;
    the arithmetic is not.
    """
    f = {}
    ref = df["reference_price"].to_numpy(np.float64)
    wap = df["wap"].to_numpy(np.float64)
    bid, ask = df["bid_price"].to_numpy(np.float64), df["ask_price"].to_numpy(np.float64)
    bsz, asz = df["bid_size"].to_numpy(np.float64), df["ask_size"].to_numpy(np.float64)
    imb = df["imbalance_size"].to_numpy(np.float64)
    matched = df["matched_size"].to_numpy(np.float64)
    flag = df["imbalance_buy_sell_flag"].to_numpy(np.float64)
    secs = df["seconds_in_bucket"].to_numpy(np.float64)

    # How lopsided the auction is, signed by which side is short. Scaled by
    # matched_size rather than by shares outstanding because matched_size is the
    # auction's own notion of "how big is this cross" and is already per-stock.
    f["imbalance_ratio"] = flag * _safe_ratio(imb, matched)
    # The sign on its own: a stock with a tiny buy imbalance is not the same
    # animal as one with a tiny sell imbalance, and the ratio alone cannot say
    # which when the magnitude is near zero. 0 = balanced, a genuine third state.
    f["imbalance_flag"] = flag

    # Where the continuous book sits relative to the auction's reference price.
    f["wap_minus_reference_bps"] = (wap - ref) * BPS
    # Cost of crossing, in bps of wap. Wider = less confident quote = noisier target.
    f["spread_bps"] = _safe_ratio(ask - bid, wap) * BPS
    # Depth asymmetry, bounded in [-1, 1] by construction, so no winsorising needed.
    f["size_imbalance"] = _safe_ratio(bsz - asz, bsz + asz)
    # What fraction of the crossing interest is the imbalance rather than the
    # matched book. Same numerator as imbalance_ratio, different normaliser;
    # bounded in [0, 1], which the ratio is not.
    f["matched_share"] = _safe_ratio(imb, imb + matched)
    # Reference price against the quote midpoint. Non-zero means the auction's
    # indicative clear is off the continuous market.
    f["reference_minus_mid_bps"] = (ref - 0.5 * (bid + ask)) * BPS

    # Indicative crossing prices. Missing before 300 s BY CONSTRUCTION (see the
    # module docstring); the indicator carries that, the value carries 0 bps.
    for name, col in (("near", "near_price"), ("far", "far_price")):
        v = df[col].to_numpy(np.float64)
        miss = ~np.isfinite(v)
        dev = np.where(miss, 0.0, (v - wap) * BPS)
        f[f"{name}_minus_wap_bps"] = dev
        f[f"{name}_is_missing"] = miss.astype(np.float64)

    # Time through the auction, in [0, 1]. The squared term is the cheapest way
    # to let the fit bend: uncertainty collapses non-linearly as the cross nears.
    frac = secs / C.AUCTION_SECONDS
    f["seconds_frac"] = frac
    f["seconds_frac_sq"] = frac * frac

    # A whole-book hole, distinct from the two crossing prices above.
    book_missing = ~np.isfinite(wap)
    f["book_is_missing"] = book_missing.astype(np.float64)

    out = pd.DataFrame(f, index=df.index)[list(FEATURE_NAMES)]

    # Zero-fill ONLY where we have already accounted for the absence. Anything
    # non-finite on a row with an observable book is a bug in this module, and it
    # must raise rather than be quietly absorbed — the whole argument of this file
    # is that blanket filling is how silent corruption gets in.
    vals = out.to_numpy()
    unexplained = ~np.isfinite(vals) & ~book_missing[:, None]
    if unexplained.any():
        bad = out.columns[unexplained.any(axis=0)].tolist()
        raise ValueError(
            f"non-finite feature values in {bad} on rows with an observable book; a guard is missing"
        )
    return out.fillna(0.0).replace([np.inf, -np.inf], 0.0)


def clip_outliers(X: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Clip to bounds learned on TRAINING rows only.

    `far_price` runs from 7.7e-05 to 437.95 in the raw data against a wap of ~1.0
    — an eight-order-of-magnitude range on a column whose sane values span a few
    percent. Those are real published far prices from thin one-sided crosses, not
    parse errors, and they turn `far_minus_wap_bps` into a column with a few
    values near 4.4 million bps.

    Where that hurts is not where it looks. The obvious story is that the extremes
    dominate the normal equations and clipping the fitted rows rescues the fit;
    `scripts/run_ablations.py` measures it and the story does not hold. What the
    clip changes first is the STANDARDISATION: unclipped, `far_minus_wap_bps`'s
    training standard deviation is 47-73x larger across the five folds, because a
    few rows near 4.4 million bps set it. Everything downstream divides by that
    number, so the ridge sees the other 99.9% of the column squashed into a
    sliver, and at predict time every row is scaled by it too.

    That is why the two halves of the clip are not interchangeable. Over the five
    FULL folds: clipping in both (the shipped 6.32288), in neither (6.32878), only
    at fit with predictions left unclipped (6.33042 — the WORST of the four, since
    the extreme rows then meet a mu/sd learned without them), and only at predict
    (6.32248 — the best). Both variants that clip at predict time beat both that
    do not. Removing the clip entirely costs 0.0059 bps, 9.4% of the model's edge,
    and not evenly: on fold 0 the unclipped model is 0.0009 bps BETTER, and the
    cost grows across folds 2 to 4.

    Bounds are computed per fold on training rows only — computing them on the
    full column would let validation-set extremes set the clipping level, which
    is a small leak but a leak. Applying those training bounds to validation rows
    is a different act and is not a leak: nothing about the validation rows is
    being learned, they are being pushed through a rule fixed before they were
    seen, exactly as a live model would push the next auction's rows through it.
    """
    out = X.copy()
    for col, (lo, hi) in bounds.items():
        if col in out.columns:
            out[col] = out[col].clip(lo, hi)
    return out


def quantile_bounds(
    X: pd.DataFrame, q: float = 0.001, extra_skip: tuple[str, ...] = ()
) -> dict[str, tuple[float, float]]:
    """Symmetric quantile bounds for every column that is not an indicator.

    The skip list is load-bearing, not tidiness. A quantile bound on a rare 0/1
    column is not winsorisation, it is deletion: `book_is_missing` is 1 on 220 of
    5.2 M rows, so both of its 0.1% quantiles are 0.0 and `clip_outliers` would
    flatten the column to a constant zero in every fold. The feature would never
    reach the model, and its coefficient would read as an honest measured 0.000
    rather than as a clip. That is exactly what happened before `INDICATOR_NAMES`
    existed, and `tests/test_features.py` now asserts the whole list rather than
    spot-checking one member of it.

    `extra_skip` lets a caller extend the list for columns this module does not
    own — Phase 2's indicators and rank features have the same two exemption
    reasons and are declared next to their definitions in `features2.py`, not
    here, so neither module has to know the other's names.
    """
    skip = set(INDICATOR_NAMES) | set(BOUNDED_NAMES) | set(extra_skip)
    return {
        col: (float(X[col].quantile(q)), float(X[col].quantile(1 - q)))
        for col in X.columns
        if col not in skip
    }
