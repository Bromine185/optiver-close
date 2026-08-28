"""Paths, auction geometry, and the SMOKE / FULL presets.

One dataclass, two presets. SMOKE is not a separate toy code path — it is the
same code with smaller numbers, so whatever runs locally is exactly what a full
run executes. Its job is to catch shape errors, fold-boundary off-by-ones, and
silently-empty validation sets. It runs against a committed 40-stock fixture and
finishes in seconds; the numbers it produces are not comparable to FULL's and
must never be quoted as results.

The geometry constants below are properties of the Nasdaq closing auction as
Optiver staged it, not tunables. They are asserted against the data in
`scripts/build_fixture.py` rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
FIXTURES = REPO / "data" / "fixtures"

#: The only file in the project that reads this is scripts/build_fixture.py.
TRAIN_CSV = RAW / "train.csv"
EXAMPLE_TEST_DIR = RAW / "example_test_files"

#: Everything downstream reads these and only these.
TRAIN_PARQUET = FIXTURES / "train.parquet"
SMOKE_PARQUET = FIXTURES / "train_smoke.parquet"
MANIFEST = FIXTURES / "manifest.json"

# --- Auction geometry -----------------------------------------------------
# The last ten minutes of the closing auction, sampled every 10 seconds:
#     seconds_in_bucket = 0, 10, ..., 540   -> 55 observations per stock-day
# The target looks 60 seconds ahead, i.e. 6 buckets. For the last 6 buckets
# (s >= 490) that horizon runs past 540 into prices the fixture does not
# contain, which is why any attempt to reconstruct the target from `wap`
# succeeds only for s <= 480. See RESEARCH.md, "verifying the target".
BUCKET_SECONDS = 10
AUCTION_SECONDS = 540
BUCKETS_PER_AUCTION = 55           # 0..540 inclusive, step 10
TARGET_HORIZON_SECONDS = 60
HORIZON_BUCKETS = TARGET_HORIZON_SECONDS // BUCKET_SECONDS   # 6

N_STOCKS = 200
N_DATES = 481                      # date_id 0..480, anonymised (see CLAUDE.md)

#: Columns as they appear in train.csv, in order.
RAW_COLUMNS = (
    "stock_id", "date_id", "seconds_in_bucket",
    "imbalance_size", "imbalance_buy_sell_flag",
    "reference_price", "matched_size", "far_price", "near_price",
    "bid_price", "bid_size", "ask_price", "ask_size", "wap",
    "target", "time_id", "row_id",
)

#: Prices are already normalised per stock to sit near 1.0 in the source data.
PRICE_COLUMNS = ("reference_price", "far_price", "near_price", "bid_price", "ask_price", "wap")
SIZE_COLUMNS = ("imbalance_size", "matched_size", "bid_size", "ask_size")

#: far_price and near_price do not exist before the 5-minute mark, by
#: construction of the auction: the exchange publishes no indicative crossing
#: prices until then. Null there is INFORMATION, not corruption. Anything that
#: fills these with 0 has invented a price 10,000 bps away from every real one.
CROSS_PRICE_COLUMNS = ("far_price", "near_price")
CROSS_PRICE_FIRST_SECOND = 300


@dataclass(frozen=True)
class Config:
    name: str

    # --- data ---
    fixture: Path
    #: Keep only every k-th date_id after loading. 1 = everything. This is a
    #: *stride*, not a head, so a reduced run still spans the whole timeline
    #: instead of one contiguous slab of it.
    date_stride: int = 1
    #: Keep only these stock_ids (None = all). Used by SMOKE's fixture, not by
    #: the loader, so FULL and SMOKE share the loader code path exactly.
    stocks: tuple[int, ...] | None = None

    # --- cross-validation (see splits.py; this is the load-bearing part) ---
    n_folds: int = 5
    #: Contiguous date_ids per validation block.
    val_dates_per_fold: int = 60
    #: Training dates within `embargo` days of a validation block's start are
    #: dropped. Not because labels overlap across dates — they do not — but
    #: because auction state is autocorrelated day to day. splits.py argues this
    #: out in full.
    embargo_dates: int = 5
    #: None = expanding window (each fold trains on all prior history).
    max_train_dates: int | None = None

    # --- baselines ---
    ridge_alpha: float = 1.0
    #: Winsorise the target before FITTING (never before scoring). MAE is robust
    #: but the ridge normal equations are least-squares, so a -385 bps outlier
    #: moves the fit far more than it moves the metric.
    fit_clip_bps: float | None = 60.0

    def __post_init__(self) -> None:
        if self.n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if self.embargo_dates < 0:
            raise ValueError("embargo_dates must be >= 0")


SMOKE = Config(
    name="SMOKE",
    # The committed 40-stock fixture: 8 blocks of 8 CONSECUTIVE dates, not a
    # date stride. Small enough to live in git, wide enough that a fold still
    # contains several thousand auctions, and blocked rather than strided so that
    # date d-1 exists for most d and the carry baselines are actually exercised
    # (see scripts/build_fixture.py and RESEARCH.md, "the first smoke fixture
    # silently disabled a baseline").
    fixture=SMOKE_PARQUET,
    n_folds=3,
    val_dates_per_fold=8,
    embargo_dates=2,
)

FULL = Config(
    name="FULL",
    fixture=TRAIN_PARQUET,
    # 5 folds x 60 validation dates = 300 of 481 dates scored (62% of the
    # timeline), leaving 181 dates of history before the first validation block.
    # Fewer, larger folds would score more history but start fold 0 on too
    # little; more, smaller folds would make each fold's MAE noisier than the
    # differences we are trying to resolve (which are ~0.05 bps on a ~6.4 bps
    # baseline).
    n_folds=5,
    val_dates_per_fold=60,
    embargo_dates=5,
)

PRESETS = {"SMOKE": SMOKE, "FULL": FULL}


def get_config(name: str = "SMOKE", **overrides) -> Config:
    """Fetch a preset, optionally tweaked. `get_config("FULL", n_folds=3)`."""
    key = name.upper()
    if key not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; expected one of {sorted(PRESETS)}")
    cfg = PRESETS[key]
    return replace(cfg, **overrides) if overrides else cfg


def auto_config(**overrides) -> Config:
    """FULL if the full fixture exists, SMOKE otherwise.

    The full fixture is gitignored (see CLAUDE.md -> Data policy), so a fresh
    clone gets SMOKE without editing anything, and a machine that has run
    `scripts/build_fixture.py` gets FULL.
    """
    return get_config("FULL" if TRAIN_PARQUET.exists() else "SMOKE", **overrides)
