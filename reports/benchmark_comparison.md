# Benchmark comparison

Rendered by `python scripts/compare_benchmarks.py` from `reports/benchmarks.json` (retrieved 2026-09-04), `reports/phase2_lgbm.json` and `reports/phase3_ensemble.json`.

Improvement is over predict-zero **on the same rows**, and is the only column that
compares across periods. Rows with no published zero baseline for their period get
no improvement rather than a guessed one. Why MAE itself does not compare, and what
this table does not license anyone to say, is in `BENCHMARKS.md`.

| entry | period | MAE (bps) | zero MAE, same rows | improvement (bps) | improvement (%) | source |
|---|---|---:|---:|---:|---:|---|
| LightGBM + MLP, weight fitted forward | offline purged CV, dates 181..480 (this repo) | 6.2531 | 6.385 | 0.1320 | 2.07 | reports/phase3_ensemble.json (Phase 3, `blend_forward`) |
| 0.5 LightGBM + 0.5 MLP (fixed) | offline purged CV, dates 181..480 (this repo) | 6.2544 | 6.385 | 0.1308 | 2.05 | reports/phase3_ensemble.json (Phase 3, `blend_fixed`) |
| LightGBM, 31 features (+memory) | offline purged CV, dates 181..480 (this repo) | 6.2559 | 6.385 | 0.1293 | 2.02 | reports/phase2_lgbm.json (Phase 2, `lgbm_mem`) |
| MLP + stock embedding, 31 features | offline purged CV, dates 181..480 (this repo) | 6.2715 | 6.385 | 0.1137 | 1.78 | reports/phase3_ensemble.json (Phase 3, `mlp_mem`) |
| 1st place, mid-competition (2023-12): leaderboard leader | public leaderboard (hidden dates after train, scored to 2023-12-20) | 5.3070 | 5.400 | 0.0930 | 1.72 | https://medium.com/@joehbridges/gauging-the-market-optivers-trading-at-the-close-kaggle-competition-27b73f7789c0 (secondary) |
| LightGBM, 14 row-wise features | offline purged CV, dates 181..480 (this repo) | 6.2843 | 6.385 | 0.1009 | 1.58 | reports/phase2_lgbm.json (Phase 2, `lgbm_row`) |
| Open-source feature-engineering LightGBM (public notebooks, 2023-12) | public leaderboard (hidden dates after train, scored to 2023-12-20) | 5.3300 | 5.400 | 0.0700 | 1.30 | https://www.zhihu.com/en/article/678286556 (secondary) |
| 14th place: LightGBM + CatBoost, 193 features, zero-sum post-processing (public) | public leaderboard (hidden dates after train, scored to 2023-12-20) | 5.3327 | 5.400 | 0.0673 | 1.25 | https://www.zhihu.com/en/article/689096751 (secondary) |
| Single LightGBM, 5-fold purged K-fold by date_id, +/-2-date embargo (fan2goa1, public rank 186) | public leaderboard (hidden dates after train, scored to 2023-12-20) | 5.3341 | 5.400 | 0.0659 | 1.22 | https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close/ (primary) |
| ridge, 31 features (+memory) | offline purged CV, dates 181..480 (this repo) | 6.3104 | 6.385 | 0.0748 | 1.17 | reports/phase2_lgbm.json (Phase 2, `ridge_mem`) |
| ConvNet on imbalance + raw features (nimashahbazi) | public leaderboard (hidden dates after train, scored to 2023-12-20) | 5.3439 | 5.400 | 0.0561 | 1.04 | https://github.com/nimashahbazi/optiver-trading-close (primary) |
| ridge, 14 row-wise features | offline purged CV, dates 181..480 (this repo) | 6.3224 | 6.385 | 0.0628 | 0.98 | reports/phase2_lgbm.json (Phase 2, `ridge`) |
| predict-zero (the floor) | offline purged CV, dates 181..480 (this repo) | 6.3852 | 6.385 | 0.0000 | 0.00 | reports/phase2_lgbm.json (Phase 2, `zero`) |
| 1st place (HYD): CatBoost 0.5 / GRU 0.3 / Transformer 0.2, online refit x5 | private leaderboard (forecasting period, 2024) | 5.4030 | — | — | — | https://www.modb.pro/db/1774620580892971008 (secondary) |
| 6th place: Transformer x3 + GRU, seq2seq, daily incremental refit | private leaderboard (forecasting period, 2024) | 5.4285 | — | — | — | https://www.modb.pro/db/1774620580892971008 (secondary) |
| 14th place: LightGBM + CatBoost, 193 features, zero-sum post-processing (private) | private leaderboard (forecasting period, 2024) | 5.4458 | — | — | — | https://www.zhihu.com/en/article/689096751 (secondary) |
| Forecasting-period leaderboard, first refresh (2024-01): gold zone | private leaderboard (forecasting period, 2024) | 5.4610 | — | — | — | https://www.zhihu.com/en/article/678286556 (secondary) |
| Forecasting-period leaderboard, first refresh (2024-01): open-source zone | private leaderboard (forecasting period, 2024) | 5.4640 | — | — | — | https://www.zhihu.com/en/article/678286556 (secondary) |
| LightGBM + Transformer 8:2 (study-session presenter's own solution, rank not stated) | private leaderboard (forecasting period, 2024) | 5.4820 | — | — | — | https://www.docswell.com/s/8980249862/K6YQ3E-2024-05-23-200638 (primary) |

## Not sourced

* The all-zeros score on the private (forecasting-period) leaderboard. Without it no private-leaderboard improvement can be computed, and none is.
* The final medal cutoffs and the number of teams.
* Any top solution's CV MAE together with a description of its CV split; the two CV numbers found (5.8117 for 1st place, 5.858 for the study-session LightGBM) come without one and are recorded in notes only.
