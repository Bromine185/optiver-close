"""Loader contract: dtypes, geometry, and the joins that could silently leak."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optiver import config as C
from optiver import data as D
from optiver.seeding import fork

requires_full = pytest.mark.skipif(
    not C.TRAIN_PARQUET.exists(),
    reason="full fixture absent; run `python scripts/build_fixture.py`",
)


def test_dtypes_are_the_downcast_ones(smoke_df):
    """Pinned, not inferred. A fixture that quietly came back float64 would still
    work and would still be right — it would just use 2x the memory and mean the
    build script's precision gate never ran."""
    for col, want in D.EXPECTED_DTYPES.items():
        assert str(smoke_df[col].dtype) == want, f"{col}: {smoke_df[col].dtype} != {want}"
    assert str(smoke_df["target"].dtype) in ("float32", "float64")


def test_row_id_was_dropped_and_can_be_rebuilt(smoke_df):
    assert "row_id" not in smoke_df.columns
    rid = D.row_id(smoke_df)
    for i in (0, 7, len(smoke_df) - 1):
        d, s, k = (int(smoke_df[c].iloc[i]) for c in ("date_id", "seconds_in_bucket", "stock_id"))
        assert rid.iloc[i] == f"{d}_{s}_{k}"


def test_auction_geometry(smoke_df):
    secs = np.sort(smoke_df["seconds_in_bucket"].unique())
    assert secs.tolist() == list(range(0, C.AUCTION_SECONDS + 1, C.BUCKET_SECONDS))
    assert len(secs) == C.BUCKETS_PER_AUCTION
    sizes = smoke_df.groupby(["stock_id", "date_id"], observed=True).size()
    # every stock-day is a WHOLE auction or absent entirely; there are no
    # half-auctions in this dataset, and several downstream shortcuts rely on it
    assert set(sizes.unique().tolist()) == {C.BUCKETS_PER_AUCTION}


def test_rows_are_sorted_for_the_shift_based_joins(smoke_df):
    key = smoke_df[["date_id", "seconds_in_bucket", "stock_id"]]
    assert key.equals(key.sort_values(list(key.columns), kind="stable"))


def test_loader_rejects_a_fixture_with_the_wrong_dtype(tmp_path, smoke_df):
    bad = smoke_df.head(1000).copy()
    bad["wap"] = bad["wap"].astype("float64")
    p = tmp_path / "bad.parquet"
    bad.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="wap"):
        D.load(path=p)


def test_loader_gives_a_useful_error_for_a_missing_fixture(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_fixture.py"):
        D.load(path=tmp_path / "nope.parquet")


def test_date_stride_spans_the_timeline_rather_than_truncating_it(smoke_df):
    """A stride, not a head. Shrinking a run by taking the first N dates would
    confine it to one slab of an anonymised, unmappable calendar; striding keeps
    whatever regime variation the 481 dates contain."""
    all_dates = D.date_ids(smoke_df)
    strided = D.date_ids(D.load(C.get_config("SMOKE", date_stride=4)))
    assert strided.size == (all_dates.size + 3) // 4
    assert strided.min() == all_dates.min()
    head = all_dates[: strided.size]
    assert strided.max() > head.max()
    assert strided.max() >= all_dates[-4]


# --- the revealed-target join --------------------------------------------

def test_revealed_target_is_the_previous_dates_same_bucket_target(smoke_df):
    out = D.add_revealed_target(smoke_df)
    assert len(out) == len(smoke_df)
    lookup = smoke_df.set_index(["stock_id", "date_id", "seconds_in_bucket"])["target"]
    have = out[out["revealed_target"].notna()]
    matched = have.sample(200, random_state=fork("revealed-target-spotcheck"))
    for r in matched.itertuples():
        want = lookup.loc[(r.stock_id, r.date_id - 1, r.seconds_in_bucket)]
        assert r.revealed_target == pytest.approx(want, nan_ok=True)


def test_revealed_target_never_uses_the_same_date(smoke_df):
    """The leak this join is one typo away from. A shift within the auction would
    hand the model a target from 60 seconds ago on the SAME day, which the live
    API never reveals."""
    out = D.add_revealed_target(smoke_df)
    prev = smoke_df.set_index(["stock_id", "date_id", "seconds_in_bucket"])["target"]
    have = out[out["revealed_target"].notna()]
    idx = pd.MultiIndex.from_arrays(
        [have["stock_id"], have["date_id"], have["seconds_in_bucket"]]
    )
    same_day = prev.reindex(idx).to_numpy()
    # revealed values must differ from the same-row target essentially always;
    # equality on more than a handful of rows means the wrong date was joined
    both = np.isfinite(same_day) & np.isfinite(have["revealed_target"].to_numpy())
    assert (same_day[both] == have["revealed_target"].to_numpy()[both]).mean() < 0.01


def test_revealed_target_is_missing_where_there_is_no_previous_auction(smoke_df):
    out = D.add_revealed_target(smoke_df)
    first = out["date_id"].min()
    assert out.loc[out["date_id"] == first, "revealed_target"].isna().all()


# --- nulls ----------------------------------------------------------------

def test_null_targets_are_dropped_not_filled(smoke_df):
    kept = D.drop_null_targets(smoke_df)
    assert kept["target"].notna().all()
    assert len(kept) == int(smoke_df["target"].notna().sum())


def test_cross_prices_are_null_before_the_five_minute_mark_and_are_not_filled(smoke_df):
    early = smoke_df[smoke_df["seconds_in_bucket"] < C.CROSS_PRICE_FIRST_SECOND]
    late = smoke_df[smoke_df["seconds_in_bucket"] >= C.CROSS_PRICE_FIRST_SECOND]
    for col in C.CROSS_PRICE_COLUMNS:
        assert early[col].isna().all(), f"{col} should be entirely absent before 300s"
        assert late[col].notna().mean() > 0.9
        # and specifically: nothing was zero-filled on the way in
        assert (smoke_df[col] == 0).sum() == 0


def test_coverage_reports_gaps_rather_than_hiding_them(smoke_df):
    cov = D.coverage(smoke_df)
    assert cov["stock_days_missing"] == cov["stock_days_possible"] - cov["stock_days_present"]
    assert cov["partial_auctions"] == 0
    assert cov["rows"] == len(smoke_df)


# --- manifest -------------------------------------------------------------

def test_manifest_matches_the_committed_smoke_fixture(manifest, smoke_df):
    import hashlib

    smoke = manifest["fixtures"]["smoke"]
    assert smoke["rows"] == len(smoke_df)
    assert smoke["n_stocks"] == smoke_df["stock_id"].nunique()
    h = hashlib.sha256(C.SMOKE_PARQUET.read_bytes()).hexdigest()
    assert h == smoke["sha256"], "committed smoke fixture does not match the manifest"


def test_manifest_records_that_the_float32_target_cast_was_verified(manifest):
    gate = manifest["target_float32_gate"]
    assert gate["accepted"] is True
    assert gate["max_abs_round_trip_err_bps"] < gate["abs_tol_bps"]
    assert gate["zero_predictor_mae_shift_bps"] < gate["mae_shift_tol_bps"]


@requires_full
def test_full_fixture_has_the_shape_the_manifest_claims(manifest):
    df = D.load(C.get_config("FULL"), columns=["stock_id", "date_id", "seconds_in_bucket", "target"])
    assert len(df) == manifest["rows"] == 5_237_980
    assert df["stock_id"].nunique() == C.N_STOCKS
    assert df["date_id"].nunique() == C.N_DATES
    assert int(df["target"].isna().sum()) == manifest["nulls"]["target"] == 88


@requires_full
def test_carry_history_is_missing_on_exactly_the_rows_we_say_it_is():
    """The docstring on `add_revealed_target` used to reason from the 964 missing
    stock-days and land 64x too high. The gaps are 11 stocks' contiguous leading
    runs, so a stock loses its previous auction once, on the date it appears —
    not once per missing day."""
    df = D.load(C.get_config("FULL"), columns=["stock_id", "date_id", "seconds_in_bucket", "target"])
    na = D.add_revealed_target(df)["revealed_target"].isna()
    assert int(na.sum()) == 11_198
    assert na.mean() == pytest.approx(0.00214, abs=1e-5)

    on_date_zero = int((na & (df["date_id"] == 0)).to_numpy().sum())
    assert on_date_zero == 191 * C.BUCKETS_PER_AUCTION == 10_505   # 191 of 200 stocks exist on date 0
    later = df.loc[(na & (df["date_id"] > 0)).to_numpy()]
    assert len(later) == 693
    sizes = later.groupby(["stock_id", "date_id"]).size()
    assert len(sizes) == 15

    # Two mechanisms, and only the first has anything to do with the 964 gaps:
    # 11 stock-days re-enter after an absence (55 rows each), and the remaining
    # 88 rows simply inherit a null d-1 target.
    present = set(map(tuple, df[["stock_id", "date_id"]].drop_duplicates().to_numpy()))
    reentry = [k for k in sizes.index if (k[0], k[1] - 1) not in present]
    assert len(reentry) == 11
    assert int(sizes[reentry].sum()) == 605
    assert 693 - 605 == int(df["target"].isna().sum()) == 88
