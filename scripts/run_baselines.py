#!/usr/bin/env python
"""Phase 1 baselines end to end: coverage, target distribution, purged CV, MAE.

    python scripts/run_baselines.py                  # FULL if the fixture exists
    python scripts/run_baselines.py --preset SMOKE   # committed 40-stock fixture
    python scripts/run_baselines.py --out reports/phase1_baselines.json

Everything printed here is what goes in RESEARCH.md. The script writes a JSON
alongside so the log's numbers can be re-checked against a machine-readable copy
rather than retyped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from optiver import baselines, config as C, data as D, evaluate as E, splits  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=None, choices=["SMOKE", "FULL"])
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "phase1_baselines.json")
    ap.add_argument("--date-stride", type=int, default=None,
                    help="keep every k-th date; a way to shrink a run without "
                         "restricting it to one slab of the timeline")
    args = ap.parse_args()

    overrides = {} if args.date_stride is None else {"date_stride": args.date_stride}
    cfg = C.get_config(args.preset, **overrides) if args.preset else C.auto_config(**overrides)
    t0 = time.time()
    print(f"=== preset {cfg.name}  fixture {cfg.fixture.name} ===")

    df = D.load(cfg)
    cov = D.coverage(df)
    print("\n--- coverage ---")
    for k, v in cov.items():
        print(f"  {k:24s} {v}")

    tstats = E.describe_target(df["target"])
    print("\n--- target distribution (bps) ---")
    for k in ("n", "mean", "median", "std", "mean_abs", "min", "max", "skew", "excess_kurtosis"):
        print(f"  {k:16s} {tstats[k]:,.6f}" if isinstance(tstats[k], float) else f"  {k:16s} {tstats[k]:,}")
    print("  quantiles " + "  ".join(f"{q:g}:{v:.2f}" for q, v in tstats["quantiles"].items()))

    df = D.drop_null_targets(df, verbose=True)

    print("\n--- folds ---")
    res = baselines.run_cv(df, cfg)
    print(splits.describe(res["folds"]).to_string(index=False))
    print(f"\ncarry: {res['carry_missing_rate']:.4%} of rows have no previous auction "
          f"(date 0 and missing stock-days); those predict 0.0")

    print("\n--- MAE by fold (bps) ---")
    ft = E.fold_table(res["per_fold"])
    print(ft.to_string(float_format=lambda x: f"{x:8.4f}"))

    scored = res["df"].loc[res["scored_mask"]]
    preds = {k: v[res["scored_mask"]] for k, v in res["oof"].items()}
    print(f"\n--- out-of-fold scorecard ({len(scored):,} scored rows, "
          f"dates {scored['date_id'].min()}..{scored['date_id'].max()}) ---")
    sc = E.scorecard(preds, scored)
    print(sc.to_string(index=False, float_format=lambda x: f"{x:10.5f}"))

    best = sc.iloc[0]["model"]
    print(f"\n--- best non-trivial model: {best} ---")
    for by in ("date_id", "seconds_in_bucket", "stock_id"):
        bd = E.breakdown(scored, preds[best], by)
        if by == "seconds_in_bucket":
            print("\nby seconds_in_bucket:")
            print(bd.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
        else:
            print(f"\nby {by}: improvement over zero, bps — "
                  f"mean {bd['improvement_bps'].mean():+.4f}, "
                  f"better on {(bd['improvement_bps'] > 0).mean():.1%} of groups, "
                  f"worst {bd['improvement_bps'].min():+.4f}, best {bd['improvement_bps'].max():+.4f}")

    if res["coefficients"]:
        print("\n--- ridge coefficients (bps of target per 1 sd of feature) ---")
        cdf = pd.DataFrame(res["coefficients"])
        cdf["mean"] = cdf.mean(axis=1)
        cdf["sign_stable"] = (np.sign(cdf.drop(columns="mean")).nunique(axis=1) == 1)
        print(cdf.reindex(cdf["mean"].abs().sort_values(ascending=False).index)
              .to_string(float_format=lambda x: f"{x:8.4f}"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "preset": cfg.name,
        "fixture": cfg.fixture.name,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items()},
        "runtime_seconds": round(time.time() - t0, 1),
        "coverage": cov,
        "target": tstats,
        "folds": splits.describe(res["folds"]).to_dict("records"),
        "per_fold_mae": res["per_fold"],
        "fold_table": ft.reset_index().to_dict("records"),
        "scorecard": sc.to_dict("records"),
        "carry_missing_rate": res["carry_missing_rate"],
        "ridge_coefficients_mean": (
            pd.DataFrame(res["coefficients"]).mean(axis=1).to_dict() if res["coefficients"] else {}
        ),
        "by_seconds": E.breakdown(scored, preds[best], "seconds_in_bucket").to_dict("records"),
        "best_model": best,
    }
    args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    shown = args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out
    print(f"\nwrote {shown}   ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
