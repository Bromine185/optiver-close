#!/usr/bin/env python
"""The numbers in RESEARCH.md that `run_baselines.py` does not produce.

    python scripts/run_ablations.py                  # FULL if the fixture exists
    python scripts/run_ablations.py --preset SMOKE   # committed fixture, seconds
    python scripts/run_ablations.py --out reports/phase1_ablations.json

Five blocks, one for each section of the log that was otherwise a number with no
command behind it — CLAUDE.md non-negotiable #7, which the repo was quietly
violating in four places at once:

    ablation     which guards actually matter: the feature clip (taken apart into
                 its fit half and its predict half), the target winsorisation,
                 the ridge penalty, the MAE-optimal rescale
    carry        cross-auction autocorrelation of the target, overall and per
                 bucket — the measurement behind "the target does not carry"
    dispersion   per-stock predict-zero MAE, the range every aggregate hides
    no_book      what is and is not missing on the four price-less stock-days
    index_leg    the target-definition check, run on the FULL fixture's first 40
                 dates rather than on the smoke fixture the tests use

The ablation block is why this file exists at all. The row published as "no
feature clipping" reproduces only if the clip is left in `RidgeMicro.fit` and
removed from `RidgeMicro.predict` — a third variant, not the one the row was
labelled with, and it carried a conclusion about the normal equations that the
real no-clip number does not support. No committed code produced it and none
could. So every variant here is a `RidgeMicro` keyword and every row lands in
JSON: the point of #7 is not tidiness, it is that a number nothing regenerates is
a number nobody re-checks.

This is a read-only analysis script: it fits models on training folds and scores
them, and it writes exactly one file, the report named by --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from optiver import baselines as B, config as C, data as D, features as F, splits  # noqa: E402
from optiver.evaluate import mae  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

#: (key, what it changes, RidgeMicro kwargs, Config overrides).
#:
#: The four clip variants are the whole point: the feature winsorisation is two
#: separate acts — bounds applied to the rows the model is fitted on, and bounds
#: applied to the rows it predicts — and they do not have to be the same act. The
#: log's original single "no feature clipping" row could not distinguish them.
VARIANTS = (
    ("shipped", "feature clip + target clip 60 + MAE rescale", {}, {}),
    ("no_feature_clip", "no feature clip in fit OR predict",
     {"clip_in_fit": False, "clip_in_predict": False}, {}),
    ("clip_fit_only", "clip the fitted rows, predict unclipped",
     {"clip_in_predict": False}, {}),
    ("clip_predict_only", "fit unclipped, clip the predicted rows",
     {"clip_in_fit": False}, {}),
    ("no_target_clip", "fit_clip_bps = None", {}, {"fit_clip_bps": None}),
    ("target_clip_20", "fit_clip_bps = 20 instead of 60", {}, {"fit_clip_bps": 20.0}),
    ("alpha_1e4", "ridge alpha 1e4", {"alpha": 1e4}, {}),
    ("alpha_1e6", "ridge alpha 1e6", {"alpha": 1e6}, {}),
    ("no_rescale", "no MAE-optimal rescale", {"rescale": False}, {}),
)


def ablation(df: pd.DataFrame, X: pd.DataFrame, folds: list, cfg) -> dict:
    """Every variant against the same folds, paired against predict-zero.

    Paired within fold, and reported as the mean of the five fold MAEs rather
    than as a pooled out-of-fold MAE, because fold-to-fold variation (1.3 bps) is
    twenty times any effect here and an unpaired comparison would measure the
    calendar. `run_baselines.py`'s scorecard pools; this table does not, so the
    two are NOT directly comparable and the log says so where it quotes them.
    """
    masks = [splits.fold_masks(df, f) for f in folds]
    y_va = [df.loc[va, "target"].to_numpy(np.float64) for _, va in masks]
    zero = [mae(y, np.zeros_like(y)) for y in y_va]
    rows = [{"variant": "zero", "note": "predict 0.0, the floor",
             "fold_mae": zero, "mean_mae": float(np.mean(zero)), "vs_zero": 0.0}]

    for key, note, kw, over in VARIANTS:
        c = replace(cfg, **over) if over else cfg
        kwargs = {"alpha": cfg.ridge_alpha, **kw}
        maes = []
        for (tr, va), y in zip(masks, y_va):
            m = B.RidgeMicro(**kwargs)
            m.fit(df.loc[tr], X.loc[tr], c)
            maes.append(mae(y, m.predict(df.loc[va], X.loc[va])))
        mean = float(np.mean(maes))
        rows.append({"variant": key, "note": note, "fold_mae": maes,
                     "mean_mae": mean, "vs_zero": float(np.mean(zero) - mean)})
        print(f"  {key:20s} {mean:9.5f}  vs zero {np.mean(zero) - mean:+8.5f}   {note}", flush=True)

    # What the clip is actually doing, measured rather than assumed: the ratio of
    # each column's training standard deviation without the clip to its standard
    # deviation with it. This is the quantity the ablation MAEs are downstream of
    # — `RidgeMicro` standardises, so a column whose sd is set by three rows near
    # 4.4 M bps arrives at the ridge with the other 99.9% of its range squashed
    # into a sliver, and the same sd is what divides the extreme rows at predict
    # time. A ratio near 1 means the clip changed nothing for that column.
    inflation = {}
    for (tr, _) in masks:
        Xt = X.loc[tr]
        raw, clipped = Xt.std(axis=0), F.clip_outliers(Xt, F.quantile_bounds(Xt)).std(axis=0)
        for col in X.columns:
            inflation.setdefault(col, []).append(
                float(raw[col] / clipped[col]) if clipped[col] > 0 else 1.0)
    return {"zero_mean_mae": float(np.mean(zero)), "rows": rows,
            "sd_inflation_without_clip": inflation}


def carry_autocorrelation(df: pd.DataFrame) -> dict:
    """Correlation of the target with the SAME stock and bucket one auction ago.

    The measurement behind the carry null result. Reported per bucket as well as
    overall because a single ρ over 5.2 M pairs could in principle be an average
    of a strong early-auction effect and a negative late one; it is not.
    """
    out = D.add_revealed_target(df)
    ok = out["target"].notna() & out["revealed_target"].notna()
    pairs = out.loc[ok, ["seconds_in_bucket", "target", "revealed_target"]]
    per_bucket = pairs.groupby("seconds_in_bucket").apply(
        lambda g: g["target"].corr(g["revealed_target"]), include_groups=False
    )
    return {
        "n_pairs": int(len(pairs)),
        "rho": float(pairs["target"].corr(pairs["revealed_target"])),
        "per_bucket_min": float(per_bucket.min()),
        "per_bucket_median": float(per_bucket.median()),
        "per_bucket_max": float(per_bucket.max()),
        "per_bucket": {int(k): float(v) for k, v in per_bucket.items()},
    }


def per_stock_dispersion(df: pd.DataFrame) -> dict:
    """Per-stock predict-zero MAE, i.e. mean |target| by stock.

    The number that says why `evaluate.breakdown` reports each group's own zero
    baseline: an aggregate MAE is dominated by whichever stocks are loudest, and
    they are 3.8x louder than the quietest.
    """
    s = df.dropna(subset=["target"]).groupby("stock_id")["target"].apply(lambda t: t.abs().mean())
    return {
        "n_stocks": int(s.size),
        "min": float(s.min()), "min_stock": int(s.idxmin()),
        "max": float(s.max()), "max_stock": int(s.idxmax()),
        "median": float(s.median()),
        "ratio_max_over_min": float(s.max() / s.min()),
    }


def no_book_stock_days(df: pd.DataFrame) -> dict:
    """What is actually missing on the four stock-days with no price book.

    Here because the log described these rows wrongly — "every price and size
    column null and the imbalance flag 0" — in a sentence that contradicted the
    manifest's own null table two paragraphs above it, and that was the stated
    justification for the `book_is_missing` feature. Takes the frame BEFORE null
    targets are dropped: 88 of the 220 rows are exactly the unlabelled ones.
    """
    nb = df[df["wap"].isna()]
    g = nb.groupby(["stock_id", "date_id"])
    return {
        "rows": int(len(nb)),
        "stock_days": [[int(s), int(d)] for s, d in g.size().index],
        "rows_with_a_target": int(nb["target"].notna().sum()),
        "null_counts": {c: int(nb[c].isna().sum()) for c in nb.columns},
        "bid_size_non_zero": int((nb["bid_size"] > 0).sum()),
        "ask_size_non_zero": int((nb["ask_size"] > 0).sum()),
        "imbalance_flag_counts": {int(k): int(v)
                                  for k, v in nb["imbalance_buy_sell_flag"].value_counts().items()},
        "per_stock_day": {
            f"{int(s)}_{int(d)}": {
                "n": int(len(part)),
                "bid_size_non_zero": int((part["bid_size"] > 0).sum()),
                "flag_non_zero": int((part["imbalance_buy_sell_flag"] != 0).sum()),
                "with_target": int(part["target"].notna().sum()),
            }
            for (s, d), part in g
        },
    }


def index_leg_check(df: pd.DataFrame, n_dates: int = 40) -> dict:
    """`tests/test_target_definition.py`'s check, on the full fixture.

    The tests run on the committed smoke fixture, as every test must; the log
    quotes the same quantities over the full fixture's first 40 dates, and those
    are the numbers this produces. Same helper (`data.index_leg`), so the two
    cannot drift apart.
    """
    dates = D.date_ids(df)[:n_dates]
    d = D.index_leg(df[df["date_id"].isin(dates)])
    g = d.groupby(["date_id", "seconds_in_bucket"])
    spread = g["leg"].std()
    leg, equal_weighted = g["leg"].mean(), g["ret_bps"].mean()

    wrong = {}
    for lag in (5, 7):
        w = D.index_leg(df[df["date_id"].isin(dates)], lag)
        wrong[lag] = float(w.groupby(["date_id", "seconds_in_bucket"])["leg"].std().median())

    # Over EVERY date, not just the 40: this is the consequence of an unequally
    # weighted index rather than a property of the window, and quoting it on a
    # 40-date slice would make it look like one.
    xsec_mean = df.groupby(["date_id", "seconds_in_bucket"])["target"].mean()
    return {
        "n_dates": int(len(dates)),
        "n_date_seconds": int(spread.size),
        "cross_sectional_std_median_bps": float(spread.median()),
        "cross_sectional_std_max_bps": float(spread.max()),
        "corr_leg_with_equal_weighted": float(np.corrcoef(leg, equal_weighted)[0, 1]),
        "std_leg_minus_equal_weighted_bps": float((leg - equal_weighted).std()),
        "equal_weighted_target_mean_bps_all_dates": float(xsec_mean.mean()),
        "equal_weighted_target_std_bps_all_dates": float(xsec_mean.std()),
        "wrong_lag_spread_median_bps": wrong,
        "wrong_lag_ratio": {k: v / float(spread.median()) for k, v in wrong.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=None, choices=["SMOKE", "FULL"])
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "phase1_ablations.json")
    args = ap.parse_args()

    cfg = C.get_config(args.preset) if args.preset else C.auto_config()
    t0 = time.time()
    print(f"=== preset {cfg.name}  fixture {cfg.fixture.name} ===")
    if cfg.name == "SMOKE":
        print("(SMOKE numbers prove the code runs; they are never quoted as results)")

    raw = D.load(cfg)
    df = D.drop_null_targets(raw, verbose=True)
    X = F.build(df)
    folds = splits.make_folds(D.date_ids(df), cfg)

    print(f"\n--- ablation: mean of {cfg.n_folds} fold MAEs, bps ---")
    abl = ablation(df, X, folds, cfg)
    infl = abl["sd_inflation_without_clip"]
    worst = sorted(infl, key=lambda c: -max(infl[c]))[:3]
    print("  training sd without the clip, as a multiple of the sd with it: " + ", ".join(
        f"{c} {min(infl[c]):.1f}-{max(infl[c]):.1f}x" for c in worst))

    print("\n--- carry autocorrelation ---")
    carry = carry_autocorrelation(df)
    print(f"  rho {carry['rho']:.5f} over {carry['n_pairs']:,} pairs; per bucket "
          f"min {carry['per_bucket_min']:.4f} median {carry['per_bucket_median']:.4f} "
          f"max {carry['per_bucket_max']:.4f}")

    print("\n--- per-stock predict-zero MAE (bps) ---")
    disp = per_stock_dispersion(df)
    print(f"  {disp['min']:.4f} (stock {disp['min_stock']}) to {disp['max']:.4f} "
          f"(stock {disp['max_stock']}), median {disp['median']:.4f}, "
          f"ratio {disp['ratio_max_over_min']:.2f}x")

    print("\n--- stock-days with no price book ---")
    nb = no_book_stock_days(raw)
    print(f"  {nb['rows']} rows over {len(nb['stock_days'])} stock-days "
          f"{nb['stock_days']}, {nb['rows_with_a_target']} of them labelled")
    print(f"  bid_size/ask_size nulls {nb['null_counts']['bid_size']}/"
          f"{nb['null_counts']['ask_size']}, non-zero on "
          f"{nb['bid_size_non_zero']}/{nb['ask_size_non_zero']} rows; "
          f"imbalance flag {nb['imbalance_flag_counts']}")

    print("\n--- index leg, first 40 dates ---")
    leg = index_leg_check(df)
    print(f"  cross-sectional std of (own return - target): median "
          f"{leg['cross_sectional_std_median_bps']:.5f} bps, max "
          f"{leg['cross_sectional_std_max_bps']:.5f}, over {leg['n_date_seconds']:,} date-seconds")
    print(f"  wrong lags: " + ", ".join(
        f"lag {k} is {v:.0f}x wider" for k, v in leg["wrong_lag_ratio"].items()))
    print(f"  corr with equal-weighted mean return {leg['corr_leg_with_equal_weighted']:.4f}, "
          f"std of the difference {leg['std_leg_minus_equal_weighted_bps']:.4f} bps")
    print(f"  equal-weighted cross-sectional mean target, ALL dates: mean "
          f"{leg['equal_weighted_target_mean_bps_all_dates']:+.4f}, "
          f"std {leg['equal_weighted_target_std_bps_all_dates']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "preset": cfg.name,
        "fixture": cfg.fixture.name,
        "runtime_seconds": round(time.time() - t0, 1),
        "folds": splits.describe(folds).to_dict("records"),
        "ablation": abl,
        "carry_autocorrelation": carry,
        "per_stock_zero_mae": disp,
        "no_book_stock_days": nb,
        "index_leg": leg,
    }
    args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    shown = args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out
    print(f"\nwrote {shown}   ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
