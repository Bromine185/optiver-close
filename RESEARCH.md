# Research log

Phase-by-phase findings. Null results and killed hypotheses are recorded here
too — a log that only contains wins is a marketing document.

Every number below is reproducible with the command named beside it, on the
committed `data/fixtures/manifest.json`, `reports/phase1_baselines.json` and
`reports/phase1_ablations.json`.

---

## Phase 1 — fixture, harness, and the floor

Three commands produce everything below, and every section names the one it came
from:

```bash
python scripts/build_fixture.py     # -> data/fixtures/ + manifest.json
python scripts/run_baselines.py     # -> reports/phase1_baselines.json
python scripts/run_ablations.py     # -> reports/phase1_ablations.json
```

Runtimes are deliberately not quoted here as prose. Each run records its own in
the artifact it writes: `build_seconds` in the manifest, `runtime_seconds` in the
two reports. They are the one class of number in this log that a rerun cannot
confirm — the same two scripts took 49 s and 175 s on one machine and then 12 s
and 43 s on the same machine ten minutes later, purely on page cache — and the
line this replaces claimed 7 s and 24 s, which had drifted past every one of
those. Read the field, not the sentence.

### Fixture built

**5,237,980 rows**, 200 stocks × 481 dates (`date_id` 0…480) × 55 buckets
(`seconds_in_bucket` 0…540 step 10). 130 MB as zstd parquet, gitignored; a 4.1 MB
40-stock smoke fixture is committed beside it and every test runs off that one.

`row_id` was dropped — it is exactly `f"{date_id}_{seconds_in_bucket}_{stock_id}"`.
`time_id` was kept but is equally redundant: `time_id == date_id*55 + seconds/10`
holds for all 5.2 M rows, asserted at build time. It carries no information and
nothing may use it as a feature; it is retained only because the live timeseries
API groups on it.

### Coverage: the gaps are whole stock-days, never partial auctions

95,236 stock-days present of a possible 96,200 → **964 missing stock-days**, and
**zero partial auctions**. Every stock-day that exists has all 55 buckets. That
is a stronger regularity than expected and several downstream shortcuts lean on
it, so it is asserted in `tests/test_data.py` rather than assumed.

The gaps are concentrated: 11 of 200 stocks are below full coverage, and the
thinnest is **stock 102 with 10,230 rows — 186 of 481 dates**, so it is absent for
61% of the sample. Stocks 135 (290 dates), 79 (300) and 199 (393) follow.
Per-date row counts run 10,505 to 11,000; 296 of 481 dates are below full.

Nothing is imputed. A stock that did not run a closing auction that day has no
rows, and inventing them would invent an auction.

### The null structure, and the fill that would have been a silent bug

| column | nulls | share | why |
|---|---|---|---|
| `far_price` | 2,894,342 | 55.3% | absent before `seconds_in_bucket = 300` **by construction** |
| `near_price` | 2,857,180 | 54.5% | same |
| `target` | 88 | 0.002% | Kaggle could not label these |
| `imbalance_size`, `matched_size`, `reference_price`, `bid_price`, `ask_price`, `wap` | 220 each | 0.004% | four stock-days with no *price* book (sizes survive; see below) |

Before 300 s the two indicative crossing prices are **100.0% null** — the exchange
publishes no cross until the five-minute mark. After 300 s, `far_price` nulls fall
from 4.50% at s=300 to 0.025% at s=540, and `near_price` sits at a flat 0.0042%
throughout. So the null pattern is two different things stacked: a structural
absence, and a thin residual of genuinely uncomputable crosses.

These prices live at ~1.0. **`fillna(0)` would encode a −10,000 bps deviation on
55% of the dataset** — three orders of magnitude outside the target's entire
range. `features.py` fills with a neutral 0 bps *deviation* and adds an explicit
`near_is_missing` / `far_is_missing` indicator instead.

**Found while building this: four stock-days have no PRICE book at all** —
(19, 438), (101, 328), (131, 35), (158, 388). For all 55 buckets
`reference_price`, `bid_price`, `ask_price`, `wap`, `imbalance_size` and
`matched_size` are null, yet **132 of those 220 rows still carry a target**. They
get a third indicator, `book_is_missing`, rather than a NaN for whatever `fillna`
runs last to sweep up.

The hole is partial, and the earlier version of this paragraph said otherwise —
it claimed every price *and size* column was null and the imbalance flag was 0,
which contradicts the null table three lines above it. What is actually true
(`run_ablations.py`, `no_book_stock_days`): `bid_size` and `ask_size` have **no
nulls at all** on these rows and are non-zero on 133 of the 220, and
`imbalance_buy_sell_flag` is non-zero on 69 of them (151 zeros, 55 sells, 14
buys). Only (158, 388) is dark on every bucket. So on these rows
`size_imbalance`, `imbalance_flag` and the time columns still carry real values
while every price-derived feature is zeroed — a half-observable row, not a blank
one. `book_is_missing` flags absent prices, not an absent row.

### float32 is safe for the target — measured, not assumed

| | max abs round-trip error | max relative |
|---|---|---|
| prices (`wap`, `reference_price`, `bid`/`ask`, `near`) | 5.96e-08 | 5.96e-08 |
| `far_price` (range 7.7e-05 … 437.95) | 5.52e-06 | 5.96e-08 |
| sizes (up to 7.71e+09) | 255 | 5.96e-08 |
| **`target`** | **1.47e-05 bps** | — |

The build gates the target's cast on two numbers: worst single-row error
**1.47e-05 bps** (tolerance 1e-03) and shift in the zero-predictor MAE
**2.5e-10 bps** (tolerance 1e-06). Both pass by orders of magnitude, and the
manifest records them; if either failed the target would stay float64.

Worth stating for the prices, since "float32 loses precision" is the reflex
objection: near 1.0 float32 resolves 6e-08, which is **17× finer than the source
data's own six-decimal quantum**. The cast is free there. It is *not* free for
sizes, where 255 units of a 7.7e+09 value are discarded — which is why sizes are
only ever used as ratios, never as levels.

### Finding: the target definition can be verified without the index weights

The index weights are not in the staged data, so the index leg looks
unreconstructable. It does not need to be reconstructed. The leg is **common to
every stock** in a `(date_id, seconds_in_bucket)`, so:

```
(own 60-second WAP return, in bps)  -  target   ==   the index leg,  same for all i
```

Measured on the full fixture over its first 40 dates (`run_ablations.py`,
`index_leg`; 1,960 auction-seconds, `s <= 480`): the cross-sectional standard
deviation of that residual is **0.00501 bps**, max 0.00582, over 200 stocks. That
is the source data's own six-decimal price quantisation. It is constant.

Three things fall out, and all three are load-bearing:

* **The horizon is exactly 6 buckets.** At lag 5 or 7 the same residual's spread
  is **691× larger**. `config.HORIZON_BUCKETS = 6` is now a tested fact (the test
  asserts a conservative >50×; the measured ratio is 691).
* **The index is not equally weighted.** Over the same 40 dates the recovered leg
  correlates 0.9568 with the equal-weighted mean stock return but differs from it
  with a standard deviation of 0.8043 bps. This is why the equal-weighted
  cross-sectional mean target is not exactly zero — over **all** 481 dates it has
  mean −0.048 and std 1.21 — a fact that would otherwise read as a bug in the
  fixture. (That last pair is a whole-fixture number, not a first-40-dates one;
  on the 40-date window it is −0.088 / 0.80.)
* **Every stock on a date shares an index term**, which is the second and less
  obvious reason `splits.py` refuses to divide a date across folds. A split that
  put stock A's date-*d* rows in train and stock B's in validation would hand the
  model the exact quantity it is being asked to subtract.

This is `tests/test_target_definition.py`, four tests. It only covers
`seconds_in_bucket <= 480`; beyond that `t+60` leaves the auction and the target
cannot be checked at all.

### Target distribution

```
n            5,237,892 (88 null)
mean            -0.0476 bps
median          -0.0602 bps
std              9.4529 bps
mean |target|    6.4078 bps      <- the floor
min           -385.29    max  446.07
skew             0.205   excess kurtosis  22.56
```

| q | 0.01% | 0.1% | 1% | 5% | 25% | 50% | 75% | 95% | 99% | 99.9% | 99.99% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bps | −92.76 | −49.38 | −25.65 | −14.05 | −4.56 | −0.06 | +4.41 | +13.96 | +25.99 | +51.82 | +100.37 |

Near-symmetric, very fat-tailed. The tails are not artefacts: the 0.01% quantile
is 10× the median absolute target, and the extremes reach 47 standard deviations
of the *metric*. MAE barely notices them; least squares does, which is the whole
reason `features.clip_outliers` exists.

Dispersion is far from uniform across stocks (`run_ablations.py`,
`per_stock_zero_mae`). Per-stock predict-zero MAE runs **3.4763 (stock 112) to
13.3054 (stock 82) bps, a factor of 3.83**, median 5.8956. Any aggregate number is
therefore dominated by the loud names, which is why `evaluate.breakdown` reports
each group's own zero baseline beside its MAE.

### The floor

`zero` scores **6.40777 bps** over the whole fixture and **6.38518 bps** over the
out-of-fold window (dates 181–480, 3,293,068 scored rows). All model comparisons
below are on the out-of-fold window.

Per-fold predict-zero MAE: **7.130, 6.040, 6.474, 6.469, 5.815** — a spread of
**1.31 bps**. Every model effect measured in Phase 1 is at most 0.063 bps, i.e.
**twenty times smaller than the fold-to-fold variation**. Model comparisons must
therefore be *paired within fold*, and `evaluate.fold_table` reports
`mean_vs_zero` for exactly that reason. What drives the fold-to-fold variation is
unknowable — see the `date_id` limitation in CLAUDE.md.

### Out-of-fold scorecard

5 folds × 60 validation dates, embargo 5, expanding window. Dates 181–480.

| model | MAE (bps) | vs zero (bps) | vs zero (%) |
|---|---|---|---|
| `ridge` | **6.32235** | **+0.06283** | **+0.98%** |
| `ridge_noscale` | 6.32314 | +0.06204 | +0.97% |
| `carry_shrunk` | 6.38300 | +0.00218 | +0.03% |
| `constant_median` | 6.38499 | +0.00019 | +0.003% |
| `zero` | 6.38518 | 0 | 0 |
| `carry_raw` | 9.10227 | **−2.71709** | **−42.55%** |

### Null result: the target does not carry across auctions

Predicting the previous auction's target for the same stock and bucket — which is
legal, since the live API reveals all of date *d−1*'s targets before date *d* is
predicted — is **42.6% WORSE than predicting zero**: 9.102 against 6.385 bps.

The reason is straightforward once measured (`run_ablations.py`,
`carry_autocorrelation`). Cross-auction autocorrelation of the target, same stock
and same bucket, over 5,226,694 pairs, is **ρ = 0.02702** (per-bucket: min
−0.0025, median 0.0222, max 0.0530). Carrying a value forward at
full weight when ρ ≈ 0 adds roughly √2 times the target's own dispersion.

Shrinking it does not rescue the idea. The MAE-optimal shrinkage fitted on
training rows comes out at **0.0225–0.025** across all five folds — the model
learns to throw away 97.75% of the signal it is given — and the resulting
`carry_shrunk` beats zero by **0.0022 bps**, which is 3.5% of what the ridge
achieves and is not a usable edge.

**Killed:** any Phase 2 feature whose thesis is "yesterday's move persists". Also
killed by implication: revealed targets are worth carrying as *state* (a
volatility estimate, a stock-level scale) but not as a *level*.

### Null result: the median is not better than zero

`constant_median` beats `zero` by **0.00019 bps**. The training median is
−0.060 bps and MAE's optimal constant is the median, so this is the *entire*
value of knowing the target's location. Predict-zero is within 0.0002 bps of the
best constant predictor that exists. It is not a straw man, and it never was.

### The ridge does beat zero — by 0.063 bps, consistently

`ridge`: **6.32235 vs 6.38518, +0.0628 bps (+0.98%)**. It is small but it is not
noise:

* better in **5 of 5 folds** (7.027/7.130, 5.984/6.040, 6.419/6.474, 6.414/6.469, 5.770/5.815);
* better on **97.7% of the 300 validation dates** (worst date −0.0134 bps, best +0.2231);
* better on **80.0% of the 200 stocks** (worst stock −1.1724 bps, best +0.8175).

So there *is* linear signal in the obvious microstructure features, and a
harness this project can trust says it is worth about one percent of the floor.
That is the entire Phase 1 result, and it is deliberately quoted as a fraction of
the floor rather than compared to anything external: nothing here was scored
through the competition API, so there is no leaderboard number to sit it beside,
and inventing one would be exactly the kind of claim CLAUDE.md forbids. Whether
0.98% is a lot or a little is a Phase 2 question.

### Where the improvement comes from: the open of the auction

Improvement over zero by `seconds_in_bucket`:

| s | 0 | 10 | 100 | 220 | 300 | 400 | 490 | 540 |
|---|---|---|---|---|---|---|---|---|
| improvement (bps) | **0.434** | 0.151 | 0.064 | 0.067 | 0.127 | 0.028 | 0.009 | 0.065 |
| improvement (%) | **5.44** | 2.24 | 1.08 | 0.92 | 1.69 | 0.51 | 0.19 | 0.98 |

The first bucket is 5.4% better than zero — seven times the average — and the
edge decays monotonically to near nothing by s ≈ 490. There is a clean secondary
bump at **s = 300**, exactly where the indicative crossing prices first appear,
which is a satisfying independent confirmation that `near_price`/`far_price` are
carrying real information rather than just an indicator flag.

But this is not a "the model only works at s=0" story: bucket 0 contributes
0.434/55 ≈ 0.0079 bps, **12.6% of the total 0.0628**. The other 87% is spread
across the auction.

Note also the shape of the *difficulty*: predict-zero MAE rises from 5.95 bps at
s=100 to **9.69 bps at s=270**, then falls to 4.87 by s=490 before climbing again
to 6.66 at s=540. Uncertainty peaks in the middle of the auction, not at either
end.

### Coefficients: three features do the work, three are noise

Ridge coefficients on standardised features, bps of target per 1 sd, mean over
five folds. "Stable" = the same sign in all five.

| feature | mean | stable |
|---|---|---|
| `wap_minus_reference_bps` | **−2.558** | yes |
| `reference_minus_mid_bps` | **−1.683** | yes |
| `seconds_frac` | −0.340 | yes |
| `near_minus_wap_bps` | +0.239 | yes |
| `seconds_frac_sq` | +0.222 | yes |
| `far_is_missing` | −0.133 | yes |
| `far_minus_wap_bps` | −0.127 | yes |
| `imbalance_ratio` | +0.091 | yes |
| `imbalance_flag` | −0.080 | yes |
| `matched_share` | +0.072 | yes |
| `size_imbalance` | −0.038 | **no** |
| `near_is_missing` | −0.023 | **no** |
| `spread_bps` | +0.022 | **no** |
| `book_is_missing` | +0.009 | yes |

**Bug, found in review, fixed:** `book_is_missing` used to appear in this table as
exactly **0.000 in all five folds**, and the earlier version of this log wrote
that off as a small-sample result — "220 rows". It was not. The column was being
destroyed before the model saw it: `features.quantile_bounds` skipped the other
three indicators but not this one, and an indicator that is 1 on 220 of 5.2 M rows
has both of its 0.1% quantiles at 0.0, so `clip_outliers` flattened it to a
constant zero in every fold. A feature that is identically zero has an identically
zero coefficient, which looks exactly like an honest measurement of nothing. With
the clip skipped it is +0.009 and sign-stable across all five folds — still the
smallest coefficient in the table, and still a feature firing on 132 labelled
rows, but now that number means what it says. `features.INDICATOR_NAMES` now names
the whole skip list and `tests/test_features.py` asserts every member of it plus a
synthetic rare indicator. Every model number in this log was regenerated after the
fix; the effect on the headline is +0.00005 bps.

The signal is **mean reversion of the continuous book toward the auction's
reference price**: `wap` above `reference_price` predicts a negative
index-relative move, and it is 1.5× the next feature and 10× everything below the
top three. `reference_minus_mid_bps` says the same thing from the other side.

**Null result: the imbalance features are nearly worthless here.**
`imbalance_ratio` (+0.091) and `matched_share` (+0.072) are stable but tiny —
1/28th of the leading feature — despite being the first thing anyone reaches for
in an auction dataset. Signed imbalance simply is not, on its own and without any
per-stock normalisation, a linear predictor of the index-relative 60-second move.

**Null result: the bid-ask spread carries nothing linear.** `spread_bps` flips
sign across folds and averages +0.022. Same for `size_imbalance` (flips at fold
3, which is the only place it becomes non-trivial) and `near_is_missing`. Three
of fourteen features are not measurably doing anything, and saying so is cheaper
than leaving them in and implying they are.

### Ablation: which guards actually matter

`python scripts/run_ablations.py` → `reports/phase1_ablations.json`. Mean fold
MAE, five folds, everything else held fixed (zero baseline = 6.38574 by the same
averaging). Note this is the mean of five fold MAEs, not the pooled out-of-fold
MAE in the scorecard above — the folds are unequal in size, so 6.32288 here and
6.32235 there are the same model, differently averaged.

| variant | MAE | vs zero |
|---|---|---|
| shipped: feature clip + target clip 60 + MAE rescale | 6.32288 | +0.0629 |
| **no feature clipping, in fit or predict** | **6.32878** | **+0.0570** |
| — clip the fitted rows only, predict unclipped | 6.33042 | +0.0553 |
| — clip the predicted rows only, fit unclipped | 6.32248 | +0.0633 |
| no target winsorisation (`fit_clip_bps=None`) | 6.32307 | +0.0627 |
| target clip at 20 instead of 60 | 6.32191 | +0.0638 |
| `alpha = 1e4` instead of 1.0 | 6.32269 | +0.0631 |
| `alpha = 1e6` | 6.32936 | +0.0564 |
| no MAE-optimal rescale | 6.32366 | +0.0621 |

* **Feature winsorisation is the guard that earns its keep, but not for the
  reason it was given.** Removing it costs 0.0059 bps, **9.4% of the entire
  signal** — still much the largest of the three guards the model actually ships
  with (the target clip is worth 0.0002 and the rescale 0.0008). The old story was that
  `far_price` reaches 437.95 against a wap of 1.0 (about 4.4 million bps) and
  those rows dominate the normal equations. The two extra rows above say
  otherwise: clipping the *fitted* rows while predicting on unclipped ones is
  **worse than not clipping anywhere** (6.33042 vs 6.32878), and clipping only the
  rows being predicted is the best of the four (6.32248). Both variants that clip
  at predict time beat both that do not. What the clip actually moves first is the
  **standardisation** — unclipped, `far_minus_wap_bps` has a training standard
  deviation **47–73× larger** (`sd_inflation_without_clip` in the report), set by
  a handful of rows, and `RidgeMicro` divides every row by it at fit *and* predict
  time. Clip one side only and the two sides disagree about what one standard
  deviation means, which is the 6.33042 row. The effect is also not uniform: on
  fold 0 the unclipped model is 0.0009 bps *better*, and the cost grows across
  folds 2 to 4 (per-fold MAEs are in the JSON).
* **The target winsorisation is a no-op** (0.00019 bps). Clipping tighter at 20
  is marginally better (+0.001) but tuning a clip against validation folds is
  itself a leak, so it stays at 60 and is not tuned.
* **The ridge penalty is a no-op** until it is absurd. At n = 4 M and 14
  standardised features, `alpha=1` and `alpha=1e4` are indistinguishable; the fit
  is effectively OLS. This is worth knowing before anyone spends a Phase 2
  afternoon tuning it.
* The MAE-optimal rescale is worth 0.0008 bps — nearly nothing, because the
  feature clip has already dealt with the tails. It stays because a Phase 2 model
  with more capacity will face the MSE/MAE mismatch harder, but it is not what is
  producing the result.

**Correction, and the reason this section now has a command.** The row published
here as "no feature clipping" was **6.33048**, and the bullet under it claimed the
clip was worth 0.0076 bps / 12% and that the extremes "dominate the normal
equations". Both were wrong, in the same way. 6.33048 is not the no-clip number
at all: it is what you get by leaving `clip_outliers` in `RidgeMicro.fit` and
removing it from `RidgeMicro.predict` — this table's `clip_fit_only` row, which
lands on 6.33042 with the `book_is_missing` fix above and on 6.33048 without it.
No committed code produced that variant, and no committed code produced this
table at all; the other five rows happen to reproduce to five decimals, which is
exactly why the sixth went unnoticed. A number that only one uncommitted working
copy can regenerate is a number nobody can check, and this one had a conclusion
about least squares resting on it. The variants are now
`RidgeMicro(clip_in_fit=..., clip_in_predict=...)` keywords and the table is
written by `scripts/run_ablations.py` into a committed JSON.

### Build finding: the first smoke fixture silently disabled a baseline

The committed smoke fixture was first built as **every 8th date**, which spans the
timeline and looked correct. It is not: a strided sample contains no date *d−1*
for any *d*, so `add_revealed_target` returned 100% nulls, both carry baselines
collapsed onto predict-zero, and the SMOKE run reported a **three-way tie at
exactly 6.20065 bps** as though it were a measurement.

Rebuilt as **8 blocks of 8 consecutive dates**, which restores 87.5% carry
coverage and, as a bonus, gives a genuinely non-contiguous date axis — the case
where a position-based embargo and a `date_id`-arithmetic embargo diverge, and
which `tests/test_splits.py` now covers explicitly. `run_cv` also warns when
carry coverage drops below 75%, so a tie can never again be mistaken for a
result. Recorded because the failure mode was silent, plausible, and produced a
number.

### Not resolved

* **One stock is 1.17 bps worse under ridge than under predict-zero**, and 20.0% of
  stocks are worse at all. Whether that is a handful of low-liquidity names where
  the mean-reversion sign genuinely inverts, or just noise on a small per-stock
  sample, is not established. A per-stock intercept or a per-stock scale is the
  obvious next probe.
* **Fold 0's MAE (7.13 for zero) is 23% higher than fold 4's (5.82).** With
  `date_id` anonymised, the cause is unknowable. It might be a volatile period,
  it might be a change in the stock universe, it might be drift in how the data
  was assembled. Nothing in this repo can distinguish those, and no claim should
  be built on any of them.
* **The `near_price` residual null rate is a flat 0.0042% at every bucket from
  300 to 540**, whereas `far_price` decays smoothly from 4.5% to 0.025%. A
  perfectly flat rate across 25 buckets suggests whole stock-days missing
  `near_price` rather than individual buckets, but this was not chased down.
* **Whether the 0.98% edge survives a different model class.** Reproducing it with
  LightGBM under this same harness is the first job of Phase 2, and if the
  boosted model's *linear-feature* edge is not at least this large, the harness is
  wrong somewhere. **Resolved in Phase 2**: `lgbm_row` (same 14 features) scores
  +0.1009 bps against the ridge's +0.0628 — the edge survives and grows.

### What Phase 1 does not license anyone to say

* Anything about *when* these auctions happened, or about market regimes.
* Anything about the leaderboard. Nothing here was scored through the competition
  API; the `optiver2023` module is a Linux-only `.so` and did not run.
* That 0.98% is a good result. It is a *measured* result on a split whose
  guarantees are asserted. That is all it is, and it is the point of the phase.

**Tests:** 78, a few seconds (5.2 s on the machine that produced the numbers
above), all green on the committed smoke fixture with no raw data present. Two of
them are gated on the full fixture and skip without it.

---

## Phase 2 — Gradient boosting on features with memory

The two changes Phase 1 deferred, made separately measurable: a model class that
optimises MAE directly, and features that use the past. Same folds, same
embargo, same floor, same scorecard code — `run_cv` gained a `feature_builder`
parameter and nothing else about the harness moved. Every number below:

    python scripts/run_phase2.py --preset FULL             # the 2x2 (391 s)
    python scripts/run_phase2.py --preset FULL --ablate    # + family ablations (1,006 s)

**Replica check first.** The Phase 2 report re-runs Phase 1's zero and ridge
inside itself: `zero` 6.38518 and `ridge` 6.32235 (+0.06283), identical to
`phase1_baselines.json` to every printed digit. The comparison below is between
runs of one harness, not between two harnesses.

### The 2×2, plus the floor

Pooled OOF MAE over the same 3,293,068 scored rows, dates 181–480:

| model | features | MAE (bps) | vs zero (bps) | vs zero (%) |
|---|---|---|---|---|
| `lgbm_mem` | 31 (+memory) | **6.25592** | **+0.12926** | **+2.02%** |
| `lgbm_row` | 14 row-wise | 6.28432 | +0.10087 | +1.58% |
| `ridge_mem` | 31 (+memory) | 6.31036 | +0.07482 | +1.17% |
| `ridge` (Phase 1) | 14 row-wise | 6.32235 | +0.06283 | +0.98% |
| `zero` (floor) | — | 6.38518 | 0 | 0 |

The decomposition, all against the Phase 1 ridge:

| gain | bps |
|---|---|
| model class alone (`lgbm_row` − `ridge`) | +0.0380 |
| memory features under a linear model (`ridge_mem` − `ridge`) | +0.0120 |
| both together (`lgbm_mem` − `ridge`) | +0.0664 |
| interaction (both − sum of parts) | **+0.0164** |

The interaction term is a quarter of the whole move: the trees extract more from
the memory features than the sum of "better model" and "more features" predicts.
That is what `objective="l1"` on nonlinear structure buys — and it is also why a
2×2 was worth running instead of one confounded jump.

LightGBM hyperparameters were fixed a priori (`boosted.LGBM_PARAMS`, recorded
verbatim in the report) and were not moved after seeing any validation number.
+2.02% is a *measurement* of the default configuration, not the ceiling of a
search. Both Phase 1 ridge corrections (target winsorisation, MAE rescale) fall
away here — the objective is the metric.

### Consistency

`lgbm_mem` vs zero: better in **5 of 5 folds** (paired mean +0.1293); better on
**100.0% of the 300 validation dates** — the worst single date is still
+0.0369 bps; better at **all 55 buckets**; better on **96.0% of the 200 stocks**
(worst −0.2249, best +0.9532). The date-level sweep is the strongest consistency
statement this project has produced: Phase 1's ridge lost on 2.3% of dates,
Phase 2's model loses on none.

Per-bucket structure, against the ridge's monotone decay: the edge is 6.8% at
s = 0, holds between 1.0% and 2.5% through the middle, bumps to **3.9% at
exactly s = 300** (the indicative-cross publication, the same landmark Phase 1
found) — and adds a new landmark, **6.7% at s = 540**, the auction's final
bucket, where predict-zero's MAE jumps from 4.87 to 6.66 and the model recovers
most of the jump. The last bucket's labels are among the unverifiable ones (the
horizon leaves the data), so this is noted, not interpreted.

### Importance is not value: the family ablation

Share of total gain (mean over folds, `lgbm_mem`): Phase 1 row features 56.6%,
cross-sectional 24.4%, rolling 14.6%, state 4.3%. Two findings and a lesson.

**`size_imbalance` alone carries 24.2% of total gain** — the single largest
feature — after five folds of ridge gave it a stable but tiny −0.09
coefficient. Bounded in [−1, 1], no winsorisation, no scale issues: the signal
was always there, it just is not linear. This is the cleanest single
demonstration of what the model-class change bought.

**The drop-one-family ablation, pooled OOF (cost of removal, bps):**

| family removed | pooled | per-fold |
|---|---|---|
| rolling | **+0.0184** | positive in 5/5 (0.0148–0.0202) |
| cross-sectional | +0.0063 | positive in 5/5, decaying (0.0134 → 0.0013) |
| state | **−0.0002** | straddles zero (−0.0018 to +0.0015) |
| all three (= `lgbm_row`) | +0.0284 | — |

The marginal costs do not sum to the total (0.0245 vs 0.0284): the families are
partially redundant, and the redundancy is itself information — cross-sectional
context can be partially reconstructed from row features plus rolling state.

**Null result: the state family adds nothing.** Removing
`stock_vol_20d_bps` + `revealed_abs_bps` (and their indicators) leaves the model
*equal* — −0.0002 bps, per-fold noise on both sides of zero — despite the pair
showing 4.3% of gain importance. Importance measures where the trees spent
splits, not what the splits were worth; the vol state the revealed targets
carry is evidently already spanned by the book itself (spread, depth, and the
rolling realised vol of the auction under way). The carry epitaph now has a
second line: yesterday's target was worth nothing as a level (Phase 1, 42.6%
worse than zero), and its scale is worth nothing *on top of the live book*
either. Killed for Phase 3: any feature whose only input is the revealed-target
stream. The embargo, note, is now genuinely load-bearing either way — the
rolling and cross-sectional families are what it was pre-paid for.

**And the cross-sectional caveat.** 24.4% of importance, +0.0063 of marginal
value, and a decaying per-fold profile (fold 0 +0.0134 → fold 4 +0.0013). With
`date_id` anonymised the decay cannot be attributed — more training data
teaching the trees to reconstruct the cross-section from row features is the
boring candidate, a regime change is the untestable one. Recorded, not resolved.

### What Phase 2 does not license anyone to say

* Still nothing about the leaderboard — nothing was scored through the
  competition API.
* Not that +2.02% is what LightGBM "can do" here: the parameters are
  deliberately untuned defaults. A tuned number would need a nested split and
  its own log entry.
* Not that the state family would fail in a live setting with richer history —
  481 dates cap the trailing window at one regime of one market.

**Tests:** 98 (Phase 1's 78 + 20), all green on the committed smoke fixture; the
new causality tests prove every Phase 2 feature is truncation-invariant at the
live API's actual information set (past dates complete, today cut at bucket s),
and a perturbation test covers the case truncation cannot: scaling date d's
targets moves later dates' state features and does not move date d's own.

## Phase 3 — A neural model beside the boosted one, and a blend that fits forward

Two commands produce everything this section will quote:

    python scripts/run_phase3.py --preset FULL --device cuda   # -> reports/phase3_ensemble.json
    python scripts/compare_benchmarks.py                       # -> reports/benchmark_comparison.md

**The FULL run has not been executed yet.** It belongs to
`notebooks/colab_phase3.ipynb` on an A100 — the two arms run as parallel
processes and the network is the GPU's job — and this log quotes a Phase 3
number only once `reports/phase3_ensemble.json` is committed beside it. SMOKE
has run, on both CPU and MPS, and proves the code path in fourteen seconds;
its numbers are not results (non-negotiable #6). What follows is the design,
what it measures, and the null results named in advance so that none of them
can be quietly dropped after the fact.

### The arms, and what each isolates

Same folds, same embargo, same floor, same scorecard code. Every arm sees the
31-column Phase 2 matrix.

| arm | what it is | the question it answers |
|---|---|---|
| `zero` | the floor, recomputed | must agree with Phase 1 to the digit |
| `lgbm_mem` | Phase 2's headline, rerun unchanged | must agree with `phase2_lgbm.json` to the digit — the replica check the script prints |
| `mlp_mem` | an MLP with an 8-d stock embedding, L1 loss, early-stopped on an inner holdout | does a smooth function of the same columns, plus a per-stock vector the trees never see, extract anything on its own |
| `blend_fixed` | 0.5 `lgbm_mem` + 0.5 `mlp_mem` | what two function classes are worth together, with nothing fitted |
| `blend_forward` | per-fold weight on `lgbm_mem`, fitted on the OOF predictions of *earlier* folds only; fold 0 takes 0.5 | the headline, named before the run |

The comparison that matters is paired within fold: `blend_forward − lgbm_mem`
on each of the five validation blocks. The bar is positive in 5 of 5. A blend
that wins pooled and loses folds has measured the calendar, not the model —
the fold-to-fold swing of the floor is 1.3 bps and every effect in this
project is a hundredth of that.

### What the guards are, in one paragraph each

**Early stopping inside the training dates.** The MLP's one data-driven choice
is when to stop, and it is made on the last 10% of the *training* dates by
position, embargoed from the inner training dates by the same five dates the
outer folds use. The fold's validation dates are seen once, at prediction
time. `tests/test_neural.py` asserts the inner block is inside the training
window and the gap is exactly `embargo_dates`; a frame too small to hold an
embargoed inner block raises rather than silently skipping the holdout.

**The blend weight moves forward.** A weight fitted on the pooled
out-of-fold vector and scored on the same vector is a one-parameter leak.
Here fold *k*'s weight sees folds 0..*k*−1 only — dates that all precede its
validation block — and `tests/test_ensemble.py` constructs the case where a
peeking weight would choose differently and asserts the shipped one does not.
Fold 0 has nothing earlier and takes 0.5; the fixed blend is reported beside
the forward one as the arm that fits nothing, so the weight's value is itself
a measured quantity.

**Two processes.** torch and LightGBM each load their own OpenMP runtime, and
on this Mac one interpreter holding both segfaults (torch first: OMP Error
#179, `pthread_mutex_init`) or deadlocks (a small LightGBM fit first, then
torch) — measured across five orderings, not assumed, and not fixed by
`KMP_DUPLICATE_LIB_OK`. So each arm runs `run_cv` in its own interpreter and
the parent, which imports neither library, refuses to blend until both
children report the same frame fingerprint, the same fold table and the same
predict-zero vector. The test suite is split the same way: 108 tests in the
main process, 8 neural tests in a child (`tests/conftest.py`).

**Hyperparameters fixed a priori**, for the MLP as for the LightGBM: the
values in `neural.MLP_PARAMS` were written down before any fold was scored
and are recorded verbatim in the report. A GPU number is reproducible to
float noise, not to the bit; the report records its device.

### Null results named in advance

Three outcomes that would be worth writing down, listed now so that the
section after the run has to address each:

1. **`mlp_mem` alone does not beat `lgbm_mem`.** Likely, and not the
   question. Phase 2's trees have the interaction with the memory features on
   their side; the MLP's case is the blend.
2. **The forward weight lands near 0.5 and `blend_fixed` scores as well or
   better.** Then the weight bought nothing and the honest headline is the
   fixed blend, with the forward arm recorded as the null.
3. **The stock embedding adds nothing over the cross-sectional family.** Not
   separable from this run — it needs a drop-the-embedding arm, which is
   named here as the first Phase 4 ablation and is not run in Phase 3.

### Beside the published results

This part quotes numbers that already exist. `python scripts/compare_benchmarks.py`
renders them from `reports/benchmarks.json` (every entry with URL, leaderboard
and primary/secondary provenance) and the Phase 2 report; `BENCHMARKS.md`
argues the comparison out.

Raw MAE does not cross periods. Predict-zero scores 6.385 bps on this
harness's 300 dates and about 5.40 on the public leaderboard's period — a
gap eight times the largest model effect measured here — and the same
trivial predictor swings 7.13 → 5.82 across this harness's own folds. The one
statistic whose scale is the model's and not the calendar's is improvement
over predict-zero on the same rows:

| entry | period | zero MAE | model MAE | improvement |
|---|---|---:|---:|---:|
| leaderboard leader, Dec 2023 | public | 5.40 | 5.3070 | 1.72 % |
| open-source feature-engineering LightGBM | public | 5.40 | 5.33 | 1.30 % |
| single LightGBM, purged 5-fold ±2 dates (rank 186) | public | 5.40 | 5.3341 | 1.22 % |
| this harness, `lgbm_mem` | offline CV, dates 181..480 | 6.3852 | 6.2559 | 2.02 % |
| this harness, `lgbm_row` | offline CV, dates 181..480 | 6.3852 | 6.2843 | 1.58 % |
| this harness, `ridge` | offline CV, dates 181..480 | 6.3852 | 6.3224 | 0.98 % |

The public zero score is printed to two decimals in its source, so every
public percentage carries ±0.09 points; the ordering survives that, the third
digit does not. The 1st-place private score (5.4030, CatBoost/GRU/Transformer,
refit five times through the forecasting period) is recorded with the other
private numbers as MAE only: no private all-zeros score was found anywhere,
and a private improvement computed against the public floor would be exactly
the period confusion the comparison exists to refuse.

What that licenses: the improvement-over-zero of this harness's models,
measured on 300 held-out dates behind an embargo, is of the same order as what
the published solutions achieved on the public board — low single-digit
percent — and the rank ordering of model classes here matches what the
write-ups report about their own progressions. What it does not: a rank, a
"beats", or anything at all about the private board beyond the levels listed.
`BENCHMARKS.md` has the full list, in the same form as the sections above.

### What Phase 3 does not license anyone to say

* Nothing yet about `mlp_mem` or either blend — the FULL report is not
  committed. When it is, this section gets its numbers and the same three
  slices Phase 2 got (fold, date, bucket, stock), plus the per-fold weights
  and where early stopping landed.
* Still nothing about the leaderboard — nothing was scored through the
  competition API, and the benchmark table above is explicit about which
  column crosses periods.
* Not that the MLP's number is what an MLP "can do" here: untuned by
  construction, and 481 dates is one regime of one market.

**Tests:** 117 in the main process (Phase 2's 98 + 4 blend + 5 benchmark
schema + 9 manifest-check + 1 wrapper) and 8 neural tests in a child
interpreter, all green on the committed smoke fixture: bit-identical refits on CPU, the inner holdout
inside the training window with the exact embargo, an excluded column that
cannot move a prediction, a forward weight that cannot peek, and a benchmark
file that cannot carry a number without a URL. The manifest check moved out of
the notebooks and into `optiver.manifest` after the first Colab rebuild was
refused for a 2e-15 disagreement in kurtosis — a text diff of the JSON, in a
cell that a Colab auto-save had quietly reverted from the structural version
committed in 323ac45. Nine tests now pin what it forgives and what it refuses.
