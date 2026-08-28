#!/usr/bin/env python
"""Read data/raw/train.csv ONCE, write data/fixtures/*.parquet, never look back.

This is the only file in the project permitted to open the raw CSV. Everything
downstream — loader, splits, baselines, tests — reads the parquet. That is what
makes a run on one machine reproduce a run on another, and what keeps a 640 MB
CSV parse out of every test session.

Usage
-----
    python scripts/build_fixture.py                # full + smoke fixtures
    python scripts/build_fixture.py --smoke-only   # rebuild just the committed one
    python scripts/build_fixture.py --chunk-rows 500000

What this script is careful about
---------------------------------
**Downcasting is verified, not assumed.** The target is in basis points, has a
standard deviation near 9.5, and the whole project is a fight over ~0.05 bps of
MAE. Casting it to float32 to save 21 MB would be a foolish way to lose the
argument, so the round-trip error is measured on every float column and the
target's cast is *gated* on it: if float32 moved the zero-predictor MAE by more
than `MAE_SHIFT_TOL_BPS`, the target stays float64 and the manifest says so.

**Nulls are structure, not damage.** `far_price` and `near_price` are null for
every row before seconds_in_bucket == 300 because the exchange publishes no
indicative crossing prices until the five-minute mark. That is the auction, not
corruption. This script records the null rate per bucket so the distinction is
visible in the manifest, and refuses to fill anything. A downstream
`fillna(0)` on a price series that lives at 1.0 would invent a -10,000 bps
deviation and quietly poison every model that used it.

**Nothing is dropped.** 88 rows have a null target. They stay in the fixture and
are counted in the manifest; the consumers decide. A build step that silently
deletes rows is a build step whose row counts cannot be reconciled with Kaggle's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from optiver import config as C  # noqa: E402
from optiver import data as D  # noqa: E402
from optiver.seeding import ROOT_SEED, fork  # noqa: E402

# --- downcast targets -----------------------------------------------------
# stock_id 0..199, date_id 0..480, seconds_in_bucket 0..540 all fit int16 with
# room to spare; the flag is -1/0/+1. time_id 0..26454 needs int32.
ID_DTYPES = {
    "stock_id": "int16",
    "date_id": "int16",
    "seconds_in_bucket": "int16",
    "imbalance_buy_sell_flag": "int8",
    "time_id": "int32",
}
FLOAT_COLUMNS = C.PRICE_COLUMNS + C.SIZE_COLUMNS   # everything but target

#: The target must survive the cast. Both gates must pass or it stays float64.
TARGET_ABS_TOL_BPS = 1e-3      # worst single-row error
MAE_SHIFT_TOL_BPS = 1e-6       # movement in the number every baseline is judged by

#: Committed fixture: enough stocks that the cross-sectional structure is still
#: visible, and dates taken as BLOCKS of consecutive days spread across the
#: timeline rather than as a stride. The block structure is deliberate. A strided
#: sample never contains date d-1 for any date d, so the revealed-target carry
#: baseline would have 100% missing history and score identically to predict-zero
#: — a SMOKE run that silently fails to exercise the code path it exists to test.
#: The gaps BETWEEN blocks are useful too: they make the date axis genuinely
#: non-contiguous, which is the case where a position-based embargo and a
#: date_id-arithmetic embargo diverge (see splits.check_fold).
SMOKE_N_STOCKS = 40
SMOKE_DATE_BLOCKS = 8
SMOKE_DATES_PER_BLOCK = 8


# --------------------------------------------------------------------------
# Streaming accumulators
# --------------------------------------------------------------------------

class PrecisionProbe:
    """Worst-case float32 round-trip error, accumulated across chunks.

    Absolute *and* relative, because the two columns families fail differently:
    prices sit at 1.0 where float32 resolves 6e-8 (finer than the source's own
    6-decimal quantum, so the cast is free), while sizes reach 7.7e9 where
    float32 cannot represent the cents at all. The relative error is 6e-8 in both
    cases; only the absolute number is alarming, and only for a column nobody
    reads in absolute units.
    """

    def __init__(self) -> None:
        self.max_abs: dict[str, float] = {}
        self.max_rel: dict[str, float] = {}

    def update(self, name: str, values: np.ndarray) -> None:
        v = values[np.isfinite(values)]
        if v.size == 0:
            return
        err = np.abs(v - v.astype(np.float32).astype(np.float64))
        rel = err / np.maximum(np.abs(v), np.finfo(np.float64).tiny)
        self.max_abs[name] = max(self.max_abs.get(name, 0.0), float(err.max()))
        self.max_rel[name] = max(self.max_rel.get(name, 0.0), float(rel.max()))

    def report(self) -> dict:
        return {
            name: {"max_abs_err": self.max_abs[name], "max_rel_err": self.max_rel[name]}
            for name in sorted(self.max_abs)
        }


def _accumulate(acc, new):
    """Sum a per-chunk pandas object into a running total.

    `Series.add(..., fill_value=0)` cannot align an empty unnamed index against a
    named or MultiIndex one, so the first chunk seeds the accumulator instead of
    being added to a zero-length placeholder.
    """
    return new if acc is None else acc.add(new, fill_value=0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------

def read_and_downcast(csv: Path, chunk_rows: int) -> tuple[pd.DataFrame, dict]:
    """One streaming pass: downcast, and collect everything the manifest needs.

    `row_id` is dropped. It is exactly `f"{date_id}_{seconds_in_bucket}_{stock_id}"`
    — a 12-byte string per row restating three columns we already have — and
    carrying 5.2 M of them through every load costs more than it explains.
    `data.row_id()` reconstructs it for submission time.
    """
    probe = PrecisionProbe()
    parts: list[pd.DataFrame] = []
    nulls = None            # column -> null rows
    group_sizes = None      # (stock_id, date_id) -> rows
    cross_nulls = None      # seconds_in_bucket -> null far/near rows
    bucket_rows = None      # seconds_in_bucket -> rows
    n_rows = 0
    tgt_abs_sum_64 = 0.0
    tgt_abs_sum_32 = 0.0
    tgt_sum = 0.0
    tgt_sq_sum = 0.0
    tgt_n = 0
    tgt_max_abs_err = 0.0
    target_samples: list[np.ndarray] = []

    usecols = [c for c in C.RAW_COLUMNS if c != "row_id"]
    reader = pd.read_csv(csv, usecols=usecols, dtype=ID_DTYPES, chunksize=chunk_rows)

    for i, chunk in enumerate(reader):
        n_rows += len(chunk)
        nulls = _accumulate(nulls, chunk.isna().sum())
        group_sizes = _accumulate(
            group_sizes, chunk.groupby(["stock_id", "date_id"], observed=True).size()
        )
        bucket_rows = _accumulate(bucket_rows, chunk.groupby("seconds_in_bucket").size())
        cross_nulls = _accumulate(
            cross_nulls,
            chunk.groupby("seconds_in_bucket")[list(C.CROSS_PRICE_COLUMNS)].apply(
                lambda g: g.isna().sum()
            ),
        )

        for col in FLOAT_COLUMNS:
            probe.update(col, chunk[col].to_numpy(dtype=np.float64))

        t = chunk["target"].to_numpy(dtype=np.float64)
        t = t[np.isfinite(t)]
        t32 = t.astype(np.float32).astype(np.float64)
        tgt_max_abs_err = max(tgt_max_abs_err, float(np.abs(t - t32).max()) if t.size else 0.0)
        tgt_abs_sum_64 += float(np.abs(t).sum())
        tgt_abs_sum_32 += float(np.abs(t32).sum())
        tgt_sum += float(t.sum())
        tgt_sq_sum += float((t * t).sum())
        tgt_n += int(t.size)
        target_samples.append(t.astype(np.float64))

        for col in FLOAT_COLUMNS:
            chunk[col] = chunk[col].astype("float32")
        parts.append(chunk)
        print(f"  chunk {i}: {n_rows:>9,} rows", flush=True)

    df = pd.concat(parts, ignore_index=True)
    del parts

    # --- the gate ---------------------------------------------------------
    mae64 = tgt_abs_sum_64 / tgt_n
    mae32 = tgt_abs_sum_32 / tgt_n
    mae_shift = abs(mae64 - mae32)
    target_f32_ok = (tgt_max_abs_err <= TARGET_ABS_TOL_BPS) and (mae_shift <= MAE_SHIFT_TOL_BPS)
    df["target"] = df["target"].astype("float32" if target_f32_ok else "float64")

    targets = np.concatenate(target_samples)
    qs = [1e-4, 1e-3, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1 - 1e-4]
    stats = {
        "rows": n_rows,
        "nulls": {k: int(v) for k, v in nulls.items()},
        "group_sizes": group_sizes.astype("int64"),
        "bucket_rows": bucket_rows.astype("int64"),
        "cross_nulls": cross_nulls.astype("int64"),
        "precision": probe.report(),
        "target": {
            "n_non_null": tgt_n,
            "n_null": n_rows - tgt_n,
            "mean": tgt_sum / tgt_n,
            "std": float(np.sqrt(tgt_sq_sum / tgt_n - (tgt_sum / tgt_n) ** 2)),
            "min": float(targets.min()),
            "max": float(targets.max()),
            "mean_abs_float64": mae64,
            "mean_abs_float32": mae32,
            "quantiles": {str(q): float(np.quantile(targets, q)) for q in qs},
            "skew": float(pd.Series(targets).skew()),
            "excess_kurtosis": float(pd.Series(targets).kurt()),
        },
        "target_float32": {
            "max_abs_round_trip_err_bps": tgt_max_abs_err,
            "zero_predictor_mae_shift_bps": mae_shift,
            "abs_tol_bps": TARGET_ABS_TOL_BPS,
            "mae_shift_tol_bps": MAE_SHIFT_TOL_BPS,
            "accepted": bool(target_f32_ok),
        },
    }
    return df, stats


def check_invariants(df: pd.DataFrame, stats: dict) -> dict:
    """Assert the geometry we claim in config.py, and report what we found.

    These are cheap and they earn their keep: every one of them is a statement
    made elsewhere in the repo (in a docstring, in CLAUDE.md, in a feature) that
    would otherwise be folklore.

    The coverage half of the answer comes from `data.coverage`, the same function
    the runtime loader's callers use, rather than from a second implementation of
    the same groupbys. That is not only less code: the manifest is the document a
    rebuild is checked against, so a manifest whose coverage numbers were computed
    by a private copy of the logic could agree with the CSV and disagree with what
    every later run sees. The build-only extras — the medians, the ten thinnest
    stocks, the partial-auction histogram, contiguity of the date axis — are added
    on top, and `stats["group_sizes"]` (accumulated chunk by chunk during the read)
    is cross-checked against the whole-frame count as a third opinion.
    """
    secs = np.sort(df["seconds_in_bucket"].unique())
    expected = np.arange(0, C.AUCTION_SECONDS + 1, C.BUCKET_SECONDS, dtype=secs.dtype)
    if not np.array_equal(secs, expected):
        raise AssertionError(f"unexpected bucket grid: {secs.tolist()}")

    # time_id is fully determined by (date_id, seconds_in_bucket) — it carries no
    # information the other two do not. We keep it because the timeseries API
    # groups on it, but nothing may treat it as an independent feature.
    derived = df["date_id"].astype("int32") * C.BUCKETS_PER_AUCTION + df["seconds_in_bucket"] // C.BUCKET_SECONDS
    time_id_is_derived = bool((derived == df["time_id"]).all())
    if not time_id_is_derived:
        raise AssertionError("time_id is not date_id*55 + seconds/10; splits assumptions need revisiting")

    if not df["time_id"].is_monotonic_increasing:
        raise AssertionError("rows are not in time order; downstream assumes they are")

    cov = D.coverage(df)
    gs = stats["group_sizes"]
    if int(gs.size) != cov["stock_days_present"]:
        raise AssertionError(
            f"streaming pass counted {gs.size} stock-days, the whole frame has "
            f"{cov['stock_days_present']}; the chunk accumulator is wrong"
        )
    partial = gs[gs != C.BUCKETS_PER_AUCTION]

    dates = np.sort(df["date_id"].unique())
    per_stock = df.groupby("stock_id").size()
    per_date = df.groupby("date_id").size()

    return {
        "bucket_grid_ok": True,
        "time_id_is_derived_from_date_and_second": time_id_is_derived,
        "rows_time_ordered": True,
        "n_stocks": cov["n_stocks"],
        "n_dates": cov["n_dates"],
        "date_id_min": cov["date_id_min"],
        "date_id_max": cov["date_id_max"],
        "date_ids_contiguous": bool(np.array_equal(dates, np.arange(dates.min(), dates.max() + 1))),
        "stock_days_present": cov["stock_days_present"],
        "stock_days_possible": cov["stock_days_possible"],
        "stock_days_missing": cov["stock_days_missing"],
        "stock_days_with_partial_auction": int(partial.size),
        "partial_auction_sizes": {str(k): int(v) for k, v in partial.value_counts().items()},
        "rows_per_stock": {
            "min": cov["rows_per_stock_min"], "max": cov["rows_per_stock_max"],
            "median": int(per_stock.median()),
            "stocks_below_max": int((per_stock < per_stock.max()).sum()),
            "ten_thinnest": {str(k): int(v) for k, v in per_stock.nsmallest(10).items()},
        },
        "rows_per_date": {
            "min": cov["rows_per_date_min"], "max": cov["rows_per_date_max"],
            "median": int(per_date.median()),
            "dates_below_max": int((per_date < per_date.max()).sum()),
        },
    }


def null_structure(stats: dict) -> dict:
    """far/near null rate per bucket, split at the five-minute mark.

    Reported as two blocks rather than one 55-row table so the manifest states
    the *claim* — 100% null before 300 s, sparse after — rather than leaving a
    reader to eyeball 55 numbers and infer it.
    """
    cn = stats["cross_nulls"]
    rows = stats["bucket_rows"]
    rate = cn.div(rows, axis=0)
    before = rate.loc[rate.index < C.CROSS_PRICE_FIRST_SECOND]
    after = rate.loc[rate.index >= C.CROSS_PRICE_FIRST_SECOND]
    return {
        "cross_price_first_second": C.CROSS_PRICE_FIRST_SECOND,
        "before_first_second": {
            col: {"min_null_rate": float(before[col].min()), "max_null_rate": float(before[col].max())}
            for col in before.columns
        },
        "at_or_after_first_second": {
            col: {
                "min_null_rate": float(after[col].min()),
                "max_null_rate": float(after[col].max()),
                "null_rate_by_second": {str(int(s)): float(v) for s, v in after[col].items()},
            }
            for col in after.columns
        },
        "note": (
            "Null before 300s is BY CONSTRUCTION: the exchange publishes no indicative "
            "crossing price until the five-minute mark. Residual nulls after 300s are "
            "genuine absences (no cross could be computed). Neither may be filled with 0 "
            "— these are prices normalised to ~1.0, and 0 is 10,000 bps away from every "
            "real value."
        ),
    }


def write_smoke(df: pd.DataFrame, path: Path) -> dict:
    """A committed subsample: `SMOKE_N_STOCKS` seeded stocks, 8 blocks of 8 dates.

    Stocks are sampled rather than head-sliced so the sample is not biased toward
    low stock_ids, and the sample is *not* filtered to fully-covered stocks — a
    smoke fixture that silently excludes the ragged stocks would let the
    coverage-gap code path go untested exactly where it matters.
    """
    rng = fork("smoke-stocks")
    all_dates = np.sort(df["date_id"].unique())
    stocks = np.sort(rng.choice(np.sort(df["stock_id"].unique()), size=SMOKE_N_STOCKS, replace=False))
    starts = np.linspace(0, all_dates.size - SMOKE_DATES_PER_BLOCK, SMOKE_DATE_BLOCKS).astype(int)
    dates = np.unique(np.concatenate([all_dates[s : s + SMOKE_DATES_PER_BLOCK] for s in starts]))
    sub = df[df["stock_id"].isin(stocks) & df["date_id"].isin(dates)].reset_index(drop=True)
    sub.to_parquet(path, index=False, compression="zstd")
    return {
        "path": str(path.relative_to(REPO)),
        "root_seed": ROOT_SEED,
        "fork_label": "smoke-stocks",
        "n_stocks": int(len(stocks)),
        "stock_ids": [int(s) for s in stocks],
        "date_blocks": int(SMOKE_DATE_BLOCKS),
        "dates_per_block": int(SMOKE_DATES_PER_BLOCK),
        "date_ids": [int(d) for d in dates],
        "n_dates": int(len(dates)),
        "rows": int(len(sub)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=C.TRAIN_CSV)
    ap.add_argument("--chunk-rows", type=int, default=1_000_000)
    ap.add_argument("--smoke-only", action="store_true",
                    help="rebuild train_smoke.parquet from an existing train.parquet")
    args = ap.parse_args()

    C.FIXTURES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.smoke_only:
        if not C.TRAIN_PARQUET.exists():
            print(f"FATAL: --smoke-only needs {C.TRAIN_PARQUET}", file=sys.stderr)
            return 1
        info = write_smoke(pd.read_parquet(C.TRAIN_PARQUET), C.SMOKE_PARQUET)
        # Update the manifest in place. A manifest whose smoke sha256 no longer
        # matches the file it names is worse than no manifest at all.
        if C.MANIFEST.exists():
            man = json.loads(C.MANIFEST.read_text())
            man.setdefault("fixtures", {})["smoke"] = {**info, "committed": True}
            man["smoke_rebuilt_at_utc"] = datetime.now(UTC).isoformat()
            C.MANIFEST.write_text(json.dumps(man, indent=2) + "\n")
        print(json.dumps(info, indent=2))
        return 0

    if not args.csv.exists():
        print(
            f"FATAL: {args.csv} not found. The raw Kaggle download is gitignored; "
            f"re-download it from the competition page into data/raw/.",
            file=sys.stderr,
        )
        return 1

    print(f"reading {args.csv} ({args.csv.stat().st_size / 1e6:.0f} MB) in chunks of {args.chunk_rows:,}")
    df, stats = read_and_downcast(args.csv, args.chunk_rows)
    print(f"  parsed in {time.time() - t0:.0f}s")

    geom = check_invariants(df, stats)

    df.to_parquet(C.TRAIN_PARQUET, index=False, compression="zstd")
    smoke = write_smoke(df, C.SMOKE_PARQUET)

    tf = stats["target_float32"]
    manifest = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "build_seconds": round(time.time() - t0, 1),
        "source": {
            "path": str(args.csv.relative_to(REPO)),
            "bytes": args.csv.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(args.csv.stat().st_mtime, UTC).isoformat(),
        },
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
        "rows": stats["rows"],
        "columns": list(df.columns),
        "dropped_columns": {"row_id": "exactly f'{date_id}_{seconds_in_bucket}_{stock_id}'; see data.row_id()"},
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "geometry": geom,
        "nulls": stats["nulls"],
        "null_structure": null_structure(stats),
        "target": stats["target"],
        "float32_round_trip": stats["precision"],
        "target_float32_gate": tf,
        "fixtures": {
            "full": {
                "path": str(C.TRAIN_PARQUET.relative_to(REPO)),
                "bytes": C.TRAIN_PARQUET.stat().st_size,
                "sha256": _sha256(C.TRAIN_PARQUET),
                "committed": False,
            },
            "smoke": {**smoke, "committed": True},
        },
    }
    C.MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")

    mb = manifest["fixtures"]["full"]["bytes"] / 1e6
    print(f"\nwrote {C.TRAIN_PARQUET.relative_to(REPO)}  ({mb:.1f} MB, {stats['rows']:,} rows)")
    print(f"wrote {C.SMOKE_PARQUET.relative_to(REPO)}  "
          f"({smoke['bytes'] / 1e6:.1f} MB, {smoke['rows']:,} rows, "
          f"{smoke['n_stocks']} stocks x {smoke['n_dates']} dates)")
    print(f"wrote {C.MANIFEST.relative_to(REPO)}")
    print(f"\ntarget float32 gate: max |err| = {tf['max_abs_round_trip_err_bps']:.3g} bps "
          f"(tol {TARGET_ABS_TOL_BPS:g}), zero-MAE shift = {tf['zero_predictor_mae_shift_bps']:.3g} bps "
          f"(tol {MAE_SHIFT_TOL_BPS:g}) -> {'ACCEPTED' if tf['accepted'] else 'REJECTED, kept float64'}")
    print(f"zero-predictor MAE over the whole fixture: {stats['target']['mean_abs_float64']:.6f} bps")
    print(f"stock-days missing: {geom['stock_days_missing']:,} of {geom['stock_days_possible']:,}; "
          f"partial auctions: {geom['stock_days_with_partial_auction']}")
    print(f"null targets: {stats['target']['n_null']} (kept in the fixture, not dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
