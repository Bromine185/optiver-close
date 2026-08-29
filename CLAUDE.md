# optiver-close — build spec

Kaggle, *Optiver — Trading at the Close*. Predict the 60-second-ahead,
index-relative move of a stock's weighted average price during the last ten
minutes of the Nasdaq closing auction.

Phase 1 is a **harness and a floor**, not a model. What is delivered here is a
fixture whose provenance is recorded, a cross-validation scheme whose guarantees
are asserted rather than assumed, and an honest number that everything later has
to beat.

## The problem, precisely

For each of 200 stocks, at each of 55 ten-second buckets covering
`seconds_in_bucket` 0 … 540 of the closing auction:

```
target_i(t) = ( WAP_i(t+60) / WAP_i(t)  -  Index(t+60) / Index(t) ) * 10^4
```

in **basis points**, where `Index` is a weighted average of all 200 stocks' WAP.
The metric is **mean absolute error** in bps, unweighted over rows.

Three consequences run through everything else in the repo:

* **It is index-relative, so it is nearly zero-mean by construction.** The
  cross-sectional average of the target is pinned near zero by the subtraction.
  Predict-zero is therefore not a straw man; it scores **6.4078 bps** over the
  whole fixture and is within 0.0002 bps of the best constant predictor that
  exists. Every result in this project is quoted as a difference against it.
* **The horizon is exactly 6 buckets.** Verified, not assumed — see
  `tests/test_target_definition.py`. For `seconds_in_bucket >= 490` the horizon
  runs past 540 into prices the dataset does not contain, so the target for the
  last six buckets cannot be reconstructed from the fixture at all.
* **All 200 stocks on a date share one index leg.** `tests/test_target_definition.py`
  shows that (own 60-second WAP return in bps) minus (target) is constant across
  stocks within a `(date_id, seconds_in_bucket)` to a cross-sectional standard
  deviation of 0.005 bps. That shared term is the second, less obvious reason a
  split may never divide a date across folds.

## Non-negotiables

1. **The split is the product.** Nothing is ever split on rows. The unit of
   assignment is a `date_id`, and a date takes all 200 stocks and all 55 buckets
   with it. `src/optiver/splits.py` names the three leakage channels this closes
   and `tests/test_splits.py` asserts the guarantee, including on a date axis with
   holes. A model number produced by any other split does not go in the log.
2. **No live reads at runtime.** `data/raw/train.csv` is opened exactly once, by
   `scripts/build_fixture.py`. Loader, splits, baselines and tests read
   `data/fixtures/*.parquet` and nothing else. (Convention carried over from the
   sibling `diff_model` project, and from `3d-night` before it.)
3. **Nulls are never blanket-filled.** `far_price` and `near_price` are absent
   for every row before the five-minute mark *by construction of the auction*, and
   four stock-days have no *price* book at all (their `bid_size`/`ask_size` and
   imbalance flag survive; see RESEARCH.md). Absence is encoded as an explicit
   indicator plus a neutral value; `fillna(0)` on a price series that lives at 1.0
   invents a −10,000 bps deviation and is a silent catastrophe on 55% of rows. The
   indicator columns are then exempt from feature winsorisation — a quantile bound
   on a flag that fires on 220 of 5.2 M rows deletes it rather than trimming it.
4. **Downcasting is verified, not assumed.** The build script measures the
   float32 round-trip error on every float column and *gates* the target's cast on
   it: max single-row error 1.47e-05 bps and a zero-predictor MAE shift of
   2.5e-10 bps, against tolerances of 1e-03 and 1e-06. If the gate fails the
   target stays float64 and the manifest records it.
5. **Deterministic.** All randomness flows through `seeding.fork("<label>")`.
   Forking by label, not by counter, so re-running one component cannot perturb
   another's draws. Bare `np.random.*` is a defect.
6. **SMOKE is the same code path with smaller numbers**, never a toy branch.
   SMOKE numbers are for proving the code runs and are never quoted as results.
7. **Every number in `RESEARCH.md` is reproducible by a named command**, and the
   commands are in the log next to the numbers.
8. **Null results get written down.** The log records what failed and what was
   killed. A log that only contains wins is a marketing document.

## Known limitations, stated up front

**`date_id` is anonymised and unmapped, and this is the big one.** The 481 dates
are integers 0…480 with no calendar attached. There is no way to know which
period they cover, whether they include an earnings season, a rate decision, a
volatility event, or a market structure change. It follows that:

* no macro-regime analysis is possible;
* no event study is possible;
* no day-of-week, month-end, index-rebalance or triple-witching feature is
  possible;
* **any claim in this repo that depends on knowing *when* these days were is
  unsupportable and must not be made.** Fold-to-fold MAE varies by 1.3 bps
  (7.13 down to 5.82 for predict-zero), which is more than twenty times any model
  effect measured so far, and we cannot say what drives it.

Intraday time *is* known: `seconds_in_bucket` is genuine seconds into the
auction, so within-auction seasonality is fair game and is used.

Further limitations, in descending order of how much they constrain conclusions:

* **The index weights are not in the staged data.** The index leg can be shown to
  be common across stocks but cannot be reconstructed, so no feature can use the
  index level directly. The weights were published on the competition forum; they
  are deliberately not pulled in, because that would be a network read.
* **481 dates is one sample of one market.** Roughly two years of trading days on
  one exchange's closing auction. Nothing here says anything about other venues,
  other auction mechanisms, or other periods.
* **`optiver2023` is a Linux-only `.so`** and will not import on this Mac, so the
  live timeseries API cannot be replayed locally. `public_timeseries_testing_util.py`
  (Kaggle's `MockApi`) is available if a Phase 4 submission harness needs it. The
  example test set is three dates; it is used to *verify semantics* (the
  revealed-target join) and never to measure anything.
* **The last six buckets of each auction are unverifiable.** The target's 60-second
  horizon leaves the data after `seconds_in_bucket = 480`. Those rows are trained
  and scored like any other; we simply cannot independently check their labels.
* **Ridge minimises squared error; the metric is MAE.** On a target with excess
  kurtosis 22.6 those objectives disagree. Two train-only corrections are applied
  and both are reported separately, but the mismatch is real and a Phase 2 model
  should optimise MAE directly.

## Deviations from the competition setup, and why

| Kaggle | Here | Reason |
|---|---|---|
| Score by submitting through the `optiver2023` timeseries API | offline purged forward-chaining CV over `train.csv` | the API module is a Linux-only `.so`; and replaying the 3-date example set measures nothing |
| Public/private LB over unseen future months | 5 folds × 60 held-out dates, covering dates 181–480 (62% of the timeline) | 300 scored dates with a stated embargo beats one opaque leaderboard number |
| Ensembles of LightGBM / NN | ridge on 14 row-wise features | Phase 1 is the harness; a strong model on an untrusted split teaches nothing |
| `row_id` carried through the data | dropped, rebuilt on demand by `data.row_id()` | it is exactly `f"{date_id}_{seconds_in_bucket}_{stock_id}"` — 5.2 M copies of three columns we already have |

## Cross-validation, and what each guard prevents

`FULL`: 5 folds, 60 validation dates each, embargo 5 dates, expanding training window.

```
fold 0  train   0..175   embargo 176..180   val 181..240
fold 1  train   0..235   embargo 236..240   val 241..300
fold 2  train   0..295   embargo 296..300   val 301..360
fold 3  train   0..355   embargo 356..360   val 361..420
fold 4  train   0..415   embargo 416..420   val 421..480
```

| Leak | What it looks like | Guard |
|---|---|---|
| Row-level | a random K-fold puts an auction's 55 buckets in both sets; the labels at `s` and `s+10` share 50 of their 60 seconds | dates are indivisible |
| Cross-sectional | stock A's date-*d* rows in train, stock B's in validation, both quoted against the same index leg | dates are indivisible |
| Temporal | training right up to the validation boundary, where auction state is autocorrelated day to day | 5-date embargo + strict forward chaining |

The embargo is stated as a **deviation, not a purge**: the label horizon is 60
seconds and the auction is 540 seconds long, so no label crosses a date boundary
and a strict Lopez de Prado purge would remove nothing. The embargo exists for
*feature* autocorrelation. Phase 1's features have no memory, so it currently
costs 5 dates per fold and buys nothing — it is in place so that Phase 2's
rolling features do not require the harness to change and do not invalidate
comparison with these numbers.

## Layout

```
scripts/build_fixture.py    ONE pass over train.csv -> fixtures + manifest
scripts/run_baselines.py    coverage, target stats, purged CV, MAE tables
scripts/run_ablations.py    ablations, carry autocorrelation, per-stock spread
scripts/run_phase2.py       the 2x2 (ridge/lgbm x row/memory) + family ablations
src/optiver/
  config.py                 SMOKE / FULL presets, auction geometry, paths
  seeding.py                label-forked seeded RNG
  data.py                   fixture loader, revealed-target join, coverage
  splits.py                 purged, embargoed, forward-chaining folds  <- load-bearing
  features.py               row-wise microstructure features (Phase 1)
  features2.py              CAUSAL features with memory: rolling / cross-sectional / state
  evaluate.py               MAE + per-fold / per-stock / per-bucket breakdowns
  baselines.py              zero, constant-median, carry, ridge; the CV runner
  boosted.py                LightGBM optimising MAE directly, params fixed a priori
tests/                      98 tests, seconds; green on a fresh clone via the smoke fixture
reports/phase1_baselines.json   machine-readable copy of the log's numbers
reports/phase1_ablations.json   the ablation table and the three analyses beside it
reports/phase2_lgbm.json        the 2x2 scorecard, importances, consistency slices
```

## Phase 2, and what its guards inherit

Phase 2 keeps the Phase 1 harness byte-for-byte — same folds, same embargo,
same floor — and re-runs Phase 1's ridge inside every Phase 2 report as a
replica check. Its own rules:

* **Features must be causal, and causality is tested by truncation.** A row may
  use past buckets of its own auction, the current cross-section, and previous
  dates' revealed targets — exactly the live API's information set. Cutting the
  frame at bucket s of the last date (or at date d) and rebuilding must leave
  every surviving row bit-identical; `tests/test_features2.py` runs both cuts,
  plus a perturbation test truncation cannot catch (scaling date d's targets
  must not move date d's own state features).
* **The carry verdict binds.** Revealed targets enter as STATE (|target|,
  trailing scale) and never as a signed level — Phase 1 measured the level at
  42.6% worse than zero, ρ = 0.027, and killed the family.
* **LightGBM hyperparameters were fixed a priori and are not tuned on any
  validation fold.** The first boosted number in the log is a measurement, not
  the argmax of a search. Tuning, if ever, gets a nested split inside training
  dates and its own log entry.
* **Rolling-feature truncation at the auction open carries no indicator** — a
  deliberate, argued exception to non-negotiable #3: the absence is a
  deterministic function of `seconds_in_bucket`, which is already a feature, so
  an indicator would duplicate it. State-family absence varies by stock history
  and keeps the indicator+neutral treatment.

## Data policy

`data/raw/` is gitignored: 641 MB of CSV plus a Linux-only `.so`, redownloadable
from the competition page, read exactly once.

`data/fixtures/train.parquet` (130 MB, 5,237,980 rows) is **also gitignored** —
a deviation from the sibling project, which commits its fixtures. 130 MB is too
much for git, and the reproducibility guarantee is provided instead by
`data/fixtures/manifest.json`, which is committed and records row counts, dtypes,
per-column null counts, coverage, the precision gate, and the parquet's sha256. A
rebuild that does not reproduce that hash is a rebuild that changed something.

`data/fixtures/train_smoke.parquet` (4.1 MB, 140,360 rows: 40 seeded stocks × 8
blocks of 8 consecutive dates) **is committed**. Every test runs against it, so
`python -m pytest -q` is green on a fresh clone with no raw data present. The
date *blocks* are deliberate: a strided sample would contain no date *d−1* for
any *d*, and the revealed-target carry baseline would silently collapse onto
predict-zero without ever exercising its code path.

## Running it

```bash
source /Users/raghavsharma/.venvs/optiver/bin/activate
python scripts/build_fixture.py            # needs data/raw/train.csv
python scripts/run_baselines.py            # FULL preset, 4.5 GB peak RSS
python scripts/run_ablations.py            # FULL preset, the slowest of the three
python scripts/run_baselines.py --preset SMOKE   # committed fixture only
python -m pytest -q                        # 78 tests
```

Timings are deliberately not quoted above. Each script records its own in the
artifact it writes (`build_seconds` in the manifest, `runtime_seconds` in the two
reports). They swing by 4x on one machine between a cold and a warm page cache —
49 s and 175 s on the first run of the day, 12 s and 43 s ten minutes later — so
a figure typed into this document is the one number in the repo that nothing
regenerates and everything contradicts.
