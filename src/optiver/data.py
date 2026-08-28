"""Fixture loading.

Reads `data/fixtures/*.parquet` (produced once by `scripts/build_fixture.py`).
Never opens the raw CSV — see CLAUDE.md non-negotiable #2. The loader is
deliberately strict: it checks dtypes and auction geometry on every load rather
than trusting that whoever built the fixture built it correctly, because a
silently float64 fixture or a fixture missing a bucket would show up much later
as a mysterious memory blowup or an off-by-one in the splits.

One row = one (stock, date, 10-second bucket) snapshot of the closing auction
book. 55 rows make one auction; 200 auctions make one date.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .config import Config

#: What the fixture must look like. Pinned here rather than inferred so a change
#: in build_fixture.py that silently widens a column fails loudly at load.
EXPECTED_DTYPES = {
    "stock_id": "int16",
    "date_id": "int16",
    "seconds_in_bucket": "int16",
    "imbalance_size": "float32",
    "imbalance_buy_sell_flag": "int8",
    "reference_price": "float32",
    "matched_size": "float32",
    "far_price": "float32",
    "near_price": "float32",
    "bid_price": "float32",
    "bid_size": "float32",
    "ask_price": "float32",
    "ask_size": "float32",
    "wap": "float32",
    "time_id": "int32",
}
KEY_COLUMNS = ("stock_id", "date_id", "seconds_in_bucket")


def load_manifest(path: Path | None = None) -> dict:
    """The build manifest. Committed, so it is available even without the full fixture."""
    p = path or C.MANIFEST
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `python scripts/build_fixture.py`")
    return json.loads(p.read_text())


def load(
    cfg: Config | None = None,
    *,
    path: Path | None = None,
    columns: list[str] | None = None,
    check: bool = True,
) -> pd.DataFrame:
    """Load a fixture, sorted by (date_id, seconds_in_bucket, stock_id).

    `cfg` selects which fixture (SMOKE's committed subsample or FULL's) and
    applies its `date_stride`. `path` overrides both, for tests.

    The sort order is not cosmetic. Splits, the revealed-target join, and every
    groupby-shift feature assume rows are in time order within a stock; sorting
    once here means none of them have to re-sort, and none of them can be quietly
    wrong if the parquet was written in a different order.
    """
    cfg = cfg or C.get_config("SMOKE")
    p = path or cfg.fixture
    if not p.exists():
        alt = "" if p == C.SMOKE_PARQUET else (
            f"\n(The SMOKE fixture at {C.SMOKE_PARQUET.name} is committed and needs no rebuild.)"
        )
        raise FileNotFoundError(
            f"{p} not found. Fixtures are built once from the raw Kaggle CSV:\n"
            f"    python scripts/build_fixture.py{alt}"
        )

    df = pd.read_parquet(p, columns=columns)
    if check:
        _check(df, columns)

    if cfg.date_stride > 1:
        keep = np.sort(df["date_id"].unique())[:: cfg.date_stride]
        df = df[df["date_id"].isin(keep)]
    if cfg.stocks is not None:
        df = df[df["stock_id"].isin(cfg.stocks)]

    sort_by = [c for c in ("date_id", "seconds_in_bucket", "stock_id") if c in df.columns]
    return df.sort_values(sort_by, kind="stable").reset_index(drop=True)


def _check(df: pd.DataFrame, columns: list[str] | None) -> None:
    for col, want in EXPECTED_DTYPES.items():
        if col not in df.columns:
            if columns is None:
                raise ValueError(f"fixture is missing column {col!r}")
            continue
        got = str(df[col].dtype)
        if got != want:
            raise ValueError(f"fixture column {col!r} has dtype {got}, expected {want}")

    if "target" in df.columns and str(df["target"].dtype) not in ("float32", "float64"):
        raise ValueError(f"target dtype {df['target'].dtype} is not floating point")

    if "seconds_in_bucket" in df.columns:
        secs = np.sort(df["seconds_in_bucket"].unique())
        expected = np.arange(0, C.AUCTION_SECONDS + 1, C.BUCKET_SECONDS)
        if not np.array_equal(secs.astype(int), expected):
            raise ValueError(
                f"bucket grid is {secs.tolist()}, expected 0..{C.AUCTION_SECONDS} step {C.BUCKET_SECONDS}"
            )


def row_id(df: pd.DataFrame) -> pd.Series:
    """Rebuild Kaggle's `row_id`, dropped from the fixture as pure restatement.

    `row_id` is `f"{date_id}_{seconds_in_bucket}_{stock_id}"` and nothing else.
    Storing 5.2 M copies of a string derived from three integers already present
    costs ~90 MB and buys nothing; submission time can pay the string-formatting
    cost on the 33 k rows that actually need it.
    """
    return (
        df["date_id"].astype(str) + "_"
        + df["seconds_in_bucket"].astype(str) + "_"
        + df["stock_id"].astype(str)
    )


def date_ids(df: pd.DataFrame) -> np.ndarray:
    """Sorted unique date_ids. The axis every split is defined on."""
    return np.sort(df["date_id"].unique())


def drop_null_targets(df: pd.DataFrame, *, verbose: bool = False) -> pd.DataFrame:
    """Remove rows Kaggle could not label.

    88 rows in the full fixture have a null target — 55 of them are one entire
    auction (stock 158, date 388) and 31 more are most of another (stock 131,
    date 35). They are kept in the fixture so its row count reconciles with
    Kaggle's, and dropped here, at the point of use, because a null label is not
    trainable and MAE against it is undefined. Never fill it.
    """
    if "target" not in df.columns:
        return df
    mask = df["target"].notna()
    if verbose and not mask.all():
        print(f"drop_null_targets: dropped {int((~mask).sum())} of {len(df)} rows with a null target")
    return df.loc[mask].reset_index(drop=True)


def add_revealed_target(df: pd.DataFrame, *, column: str = "revealed_target") -> pd.DataFrame:
    """Join the previous date's target for the same stock and bucket.

    This is *not* leakage, and the distinction is worth being precise about
    because the whole harness depends on getting it right. In the live timeseries
    API, the first call of test date d delivers `revealed_targets.csv`: every one
    of date d-1's 11,000 targets, all at once, before any prediction for date d is
    made. Verified against the staged example: for date 478 the revealed values
    match `train.csv`'s date-477 targets for the same (stock, seconds_in_bucket)
    to 3.5e-6 bps — i.e. exactly, up to the fixture's float32 storage.

    So `target(stock, d-1, s)` is legitimately available when predicting
    `(stock, d, s)`. What is NOT available is anything from date d itself:
    `target(stock, d, s-10)` is 60 seconds in the future at bucket s-10 and will
    not be revealed until date d+1. Any feature built from within-date targets is
    a leak, and this function deliberately offers no way to build one.

    A row gets NaN when its stock did not trade on d-1, or when d-1's target for
    that bucket was itself null. That is rarer than the 964 missing stock-days
    suggest, because those gaps sit in 11 stocks and are almost all contiguous
    leading runs — a stock that is absent for its first 300 dates loses its
    previous auction exactly once, on the date it appears. So the full fixture
    has 11,198 NaN rows, 0.21%: 10,505 for the 191 stocks present on date 0,
    605 for the 11 stock-days that re-enter after a gap, and 88 that inherit a
    null d-1 target. Asserted in `tests/test_data.py` so the arithmetic stays
    honest.
    """
    prev = df[["stock_id", "date_id", "seconds_in_bucket", "target"]].copy()
    prev["date_id"] = (prev["date_id"].astype("int32") + 1).astype("int16")
    prev = prev.rename(columns={"target": column})
    out = df.merge(prev, on=list(KEY_COLUMNS), how="left")
    return out


def index_leg(df: pd.DataFrame, lag_buckets: int = C.HORIZON_BUCKETS) -> pd.DataFrame:
    """(own `lag`-bucket WAP return in bps) − target, per row: the index leg.

    The target is a 60-second WAP return minus a weighted index's return over the
    same 60 seconds, and the index weights are not in the staged data. They do not
    need to be: the index leg is COMMON to every stock in a
    `(date_id, seconds_in_bucket)`, so this difference recovers it up to a
    per-row constant, and its cross-sectional spread within a (date, second) is a
    direct measure of whether the target definition is what we think it is.

    Lives here rather than in the test that first needed it because
    `scripts/run_ablations.py` re-runs the same check on the full fixture, and two
    copies of an eight-line derivation are two chances to check different things
    and believe they agree. Returns only rows where the horizon stays inside the
    auction (`seconds_in_bucket <= 480`); past that the forward price does not
    exist in the fixture and the leg cannot be recovered at all.
    """
    d = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"]).copy()
    fwd = d.groupby(["stock_id", "date_id"], observed=True)["wap"].shift(-lag_buckets)
    d["ret_bps"] = (fwd / d["wap"] - 1.0) * 1e4
    d["leg"] = d["ret_bps"] - d["target"]
    keep = d["seconds_in_bucket"] <= C.AUCTION_SECONDS - C.TARGET_HORIZON_SECONDS
    return d[keep].dropna(subset=["leg"])


def coverage(df: pd.DataFrame) -> dict:
    """Row counts by stock and by date, and the stock-days that are simply absent.

    Reported rather than repaired. The gaps are structural — a stock that did not
    trade the closing auction that day has no rows, and inventing them would
    invent an auction.
    """
    stocks = np.sort(df["stock_id"].unique())
    date_ids_ = date_ids(df)
    per_stock = df.groupby("stock_id").size()
    per_date = df.groupby("date_id").size()
    stock_days = df.groupby(["stock_id", "date_id"], observed=True).size()
    return {
        "rows": int(len(df)),
        "n_stocks": int(stocks.size),
        "n_dates": int(date_ids_.size),
        "date_id_min": int(date_ids_.min()),
        "date_id_max": int(date_ids_.max()),
        "stock_days_present": int(stock_days.size),
        "stock_days_possible": int(stocks.size * date_ids_.size),
        "stock_days_missing": int(stocks.size * date_ids_.size - stock_days.size),
        "partial_auctions": int((stock_days != C.BUCKETS_PER_AUCTION).sum()),
        "rows_per_stock_min": int(per_stock.min()),
        "rows_per_stock_max": int(per_stock.max()),
        "rows_per_date_min": int(per_date.min()),
        "rows_per_date_max": int(per_date.max()),
        "thinnest_stocks": {int(k): int(v) for k, v in per_stock.nsmallest(5).items()},
    }
