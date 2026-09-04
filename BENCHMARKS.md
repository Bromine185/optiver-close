# Benchmarks — this repository beside the published competition results

`reports/benchmarks.json` holds every published number this file cites, with
its URL, which leaderboard it belongs to, and whether it is the author's own
report (primary) or a third party's transcription of a Kaggle page (secondary).
`python scripts/compare_benchmarks.py` renders `reports/benchmark_comparison.md`
from that file and this repository's reports. Nothing in this file was scored by
this repository; nothing from this repository was scored by Kaggle.

## What is being compared

The competition scored submissions on **mean absolute error in basis points**
over two hidden periods: a *public* leaderboard, on dates after the training
data, frozen at the 2023-12-20 deadline; and a *private* leaderboard, on a
forecasting period in early 2024 during which teams could resubmit and, most
of the top teams did, refit on the days as they arrived.

This repository scores **out-of-fold** on 300 of the 481 training dates
(181..480), five forward-chaining folds with a five-date embargo — see
`CLAUDE.md`, "Cross-validation". Same metric, different rows.

## The numbers

| entry | period | MAE (bps) | source |
|---|---|---:|---|
| all-zeros submission | public | 5.40 | [zhihu 678286556](https://www.zhihu.com/en/article/678286556) — secondary; printed as "540" |
| leaderboard leader, mid-competition (Dec 2023) | public | 5.3070 | [Medium recap](https://medium.com/@joehbridges/gauging-the-market-optivers-trading-at-the-close-kaggle-competition-27b73f7789c0) — secondary |
| open-source feature-engineering LightGBM | public | 5.33 | zhihu 678286556 — secondary; "533", gold zone then "532" |
| 14th place, LightGBM + CatBoost, 193 features | public / private | 5.3327 / 5.4458 | [zhihu 689096751](https://www.zhihu.com/en/article/689096751) — secondary (notes on the write-up) |
| single LightGBM, purged 5-fold by date, ±2-date embargo, public rank 186 | public | 5.3341 | [fan2goa1](https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close/) — primary |
| ConvNet, imbalance + raw features | public | 5.3439 | [nimashahbazi](https://github.com/nimashahbazi/optiver-trading-close) — primary |
| 1st place (HYD): CatBoost 0.5 / GRU 0.3 / Transformer 0.2, refit ×5 | private | 5.4030 | [modb.pro summary](https://www.modb.pro/db/1774620580892971008) — secondary; weights and refits corroborated by [docswell](https://www.docswell.com/s/8980249862/K6YQ3E-2024-05-23-200638) |
| 6th place: Transformer ×3 + GRU, daily refit | private | 5.4285 | modb.pro summary — secondary |
| forecasting leaderboard, first refresh (Jan 2024): gold zone / open-source zone | private, partial | 5.461 / 5.464 | zhihu 678286556 — secondary; levels of a running board, not final scores |
| study-session presenter, LightGBM + Transformer 8:2 | private | 5.482 | docswell — primary; components: LightGBM CV 5.858 → private 5.478 |
| **this repository, `lgbm_mem`** | **offline CV, dates 181..480** | **6.2559** | `reports/phase2_lgbm.json` |
| this repository, predict-zero | offline CV, dates 181..480 | 6.3852 | `reports/phase1_baselines.json`, `phase2_lgbm.json` |

Kaggle's own pages — the leaderboard, the write-ups, the discussion threads —
are client-rendered and returned no content to any fetch attempted while
compiling this. Every leaderboard figure above is therefore somebody's
transcription. The two primary sources are participants reporting their own
scores. Two of the secondary sources print scores to two decimals ("540",
"533"); the uncertainty that adds is stated below where it matters.

## Why the MAE column does not compare

Read down the MAE column and this repository looks a full basis point behind
everything on the board. That is not a model gap; it is a period gap, and it
shows up in three places that have nothing to do with the model:

* **The floor moves.** Predict-zero — the same trivial predictor — scores
  6.385 bps on this repository's 300 dates and about 5.40 bps on the public
  leaderboard's period. That 0.98 bps difference is eight times the largest
  model effect measured here (+0.129 bps). The target is index-relative and
  near zero-mean, so its MAE is essentially its mean absolute size, and the
  mean absolute size of a 60-second relative move is a property of the days
  scored, not of anything a model does.
* **The floor moves between the folds of this one harness.** Predict-zero
  scores 7.130, 6.040, 6.474, 6.469 and 5.815 bps on folds 0..4. A 1.3 bps
  swing across 60-date blocks of the *training* data is the same phenomenon,
  measured where it can be measured.
* **The floor moved for the competitors too.** The 14th-place team's unchanged
  submissions scored 5.3327 public and 5.4458 private — +0.11 bps for the
  same model on a different quarter. The study-session presenter's single
  LightGBM scored 5.858 on their own CV and 5.478 on the private board: 0.38
  bps apart, same model. Nobody's numbers cross periods, including theirs.

So a raw MAE from one period placed beside a raw MAE from another says which
period was quieter. It cannot say which model was better.

## The statistic that does compare

Divide out the floor. **Improvement over predict-zero on the same rows** is
the only quantity in either table whose scale is the model's and not the
calendar's:

| entry | period | zero MAE | model MAE | improvement |
|---|---|---:|---:|---:|
| leaderboard leader, Dec 2023 | public | 5.40 | 5.3070 | **1.72 %** |
| open-source LightGBM | public | 5.40 | 5.33 | 1.30 % |
| 14th place, public submission | public | 5.40 | 5.3327 | 1.25 % |
| purged-CV LightGBM (rank 186) | public | 5.40 | 5.3341 | 1.22 % |
| ConvNet | public | 5.40 | 5.3439 | 1.04 % |
| this repository, `lgbm_mem` | offline CV | 6.3852 | 6.2559 | **2.02 %** |
| this repository, `lgbm_row` | offline CV | 6.3852 | 6.2843 | 1.58 % |
| this repository, `ridge` | offline CV | 6.3852 | 6.3224 | 0.98 % |

The public zero score is printed to two decimals, so every public-leaderboard
percentage carries ±0.09 points of rounding; the ordering within that column
survives it, the third digit does not. **No private-leaderboard improvement
appears because no private zero score was found.** The 5.40..5.48 private
numbers are listed as MAE and nothing more.

Phase 3's arms (`mlp_mem`, the fixed blend, the forward-weighted blend) join
the repository rows automatically once `reports/phase3_ensemble.json` exists;
the comparison script reads it if it is there and says so if it is not.

## Why an offline improvement can exceed a leaderboard improvement

2.02 % offline against 1.72 % for the December leader is not evidence that
this repository's LightGBM would have led the board. Four reasons, none of
them exotic:

1. **Three hundred dates against one block.** The offline number is pooled
   over five validation blocks spanning 62 % of the timeline; a leaderboard
   number is one hidden period. Both are the same estimator; one has more
   variance, and the fold-to-fold spread above shows how much. A single fold
   of this harness reads anywhere from 1.7 % to 2.5 %.
2. **The leaderboard period is out of the training distribution; the
   validation blocks are inside it.** Every validation date here is preceded
   by an embargo and followed by nothing; the competition's periods came
   after a gap of days to months from the last training date, in a market
   the training data had not seen. The private board's uniform +0.1 bps
   shift is what that looks like. Forward chaining is the honest offline
   analogue, and it is still an analogue.
3. **The top teams refit during the forecasting period; this repository
   never does.** The 1st, 6th and 14th place write-ups all describe
   retraining on new days as they arrived, and the zhihu article credits
   online updates with about a thousandth of a bps on the public board. Their
   private scores include that; nothing here has an equivalent.
4. **Two of the four public numbers are rounded to the hundredth.** "540" is
   anything from 5.395 to 5.405; the leader's percentage is 1.63..1.81 before
   its own model differences are considered.

There is also the plain fact that the improvement column measures a *ratio*
whose denominator is quieter on the leaderboard: the same absolute edge in
bps is a larger percentage of 5.40 than of 6.39, which cuts the other way and
is why the bps column is shown beside the percentage.

## What this comparison does not license anyone to say

Following the convention of `RESEARCH.md`:

* Not a leaderboard rank, for any model in this repository. Nothing here was
  submitted, and the competition API does not run on this machine
  (`CLAUDE.md`, "Known limitations").
* Not that 2.02 % "beats" 1.72 %. Different rows, different periods, one
  number rounded, and the leader's later private score is unknown relative to
  its floor.
* Not anything about the private leaderboard at all beyond the MAE levels
  listed. Without a private zero score there is no comparable statistic, and
  inventing one from the public floor would be exactly the period confusion
  this file exists to refuse.
* Not that a purged, embargoed split is what separates this repository from
  the field: at least one published solution used one (fan2goa1, ±2 dates)
  and scored 1.22 % public. The split is the product because it makes the
  number believable, not because it makes the number large.

What it does license: the improvement-over-zero of this repository's models,
measured on 300 held-out dates behind an embargo, is of the same order as the
improvement-over-zero that published solutions achieved on the public board —
low single-digit percent — and the rank ordering of model classes here
(ridge < row-feature LightGBM < memory-feature LightGBM) matches what the
board's write-ups report about their own progressions.

## How to refresh

1. Add or correct an entry in `reports/benchmarks.json`. Every entry needs a
   URL, a leaderboard type, a source type, and a `notes` field that states
   the precision of the number as printed. If a private all-zeros score is
   ever found, put it in `zero_baseline.private` and the script will start
   computing private improvements on its own.
2. `python -m pytest tests/test_benchmarks.py -q` — the schema is asserted.
3. `python scripts/compare_benchmarks.py` — rewrites
   `reports/benchmark_comparison.md`. Commit both.
4. Do not edit the numbers in this file's tables by hand without editing the
   JSON; the JSON is what the script reads, and the two drifting apart is the
   failure mode of every benchmark table ever typed into a README.
