#!/usr/bin/env python
"""Phase 2: gradient boosting on features with memory, scored by the Phase 1 harness.

    python scripts/run_phase2.py                     # FULL if the fixture exists
    python scripts/run_phase2.py --preset SMOKE      # committed 40-stock fixture
    python scripts/run_phase2.py --ablate            # + drop-one-family ablations

Same folds, same embargo, same floor, same scorecard code as Phase 1 — that is
the entire point. A Phase 2 number produced by a different harness would not be
comparable to the 6.3852/6.3224 pair it exists to beat.

The experiment is a 2x2 plus the floor, so the two changes Phase 2 makes are
measured separately instead of as one confounded jump:

    zero          the floor, recomputed (must agree with Phase 1 to the digit)
    ridge         Phase 1's shipped model, rerun unchanged (same check)
    lgbm_row      NEW MODEL, old features  -> the model-class gain alone
    ridge_mem     old model, NEW FEATURES  -> the feature gain a linear model can see
    lgbm_mem      new model, new features  -> the headline

`--ablate` reruns lgbm_mem three more times, each with one Phase 2 feature
family removed (rolling / cross-sectional / state), so every family's marginal
value is measured on exactly the folds that produced the headline.
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

from optiver import baselines, boosted, config as C, data as D, evaluate as E  # noqa: E402
from optiver import features as F, features2 as F2, splits  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

#: The three Phase 2 families, by column, for the ablation. Kept here rather
#: than inferred from name prefixes so a renamed feature breaks this loudly.
FAMILIES = {
    "rolling": (
        "wap_ret_1b_bps", "wap_ret_6b_bps", "wap_vol_6b_bps", "imb_ratio_chg_1b",
        "size_imb_chg_1b", "matched_chg_1b", "spread_chg_1b_bps", "near_wap_chg_1b_bps",
    ),
    "cross_sectional": (
        "wap_ret_1b_cs", "wap_ret_6b_cs", "imb_ratio_cs_rank", "wap_ref_cs", "spread_cs_rank",
    ),
    "state": (
        "stock_vol_20d_bps", "stock_vol_is_missing", "revealed_abs_bps", "revealed_is_missing",
    ),
}


def phase2_models(cfg: C.Config, ablate: bool) -> list:
    row_cols = tuple(F.FEATURE_NAMES)
    all_cols = tuple(F2.ALL_NAMES)
    skip2 = tuple(F2.INDICATOR2_NAMES) + tuple(F2.BOUNDED2_NAMES)
    models = [
        baselines.Zero(),
        # Phase 1's shipped arm, PINNED to the row columns: with columns=None it
        # would silently fit whatever the wider builder hands it and stop being
        # the replica whose agreement with phase1_baselines.json is the check.
        baselines.RidgeMicro(name="ridge", alpha=cfg.ridge_alpha, rescale=True,
                             columns=row_cols),
        boosted.LightGBMMae(name="lgbm_row", columns=row_cols),
        baselines.RidgeMicro(name="ridge_mem", alpha=cfg.ridge_alpha, rescale=True,
                             columns=all_cols, extra_skip=skip2),
        boosted.LightGBMMae(name="lgbm_mem", columns=all_cols),
    ]
    if ablate:
        for fam, cols in FAMILIES.items():
            kept = tuple(c for c in all_cols if c not in cols)
            models.append(boosted.LightGBMMae(name=f"lgbm_mem_minus_{fam}", columns=kept))
    return models


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=None, choices=["SMOKE", "FULL"])
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "phase2_lgbm.json")
    ap.add_argument("--ablate", action="store_true",
                    help="also run the three drop-one-family lgbm ablations (3x the lgbm cost)")
    args = ap.parse_args()

    cfg = C.get_config(args.preset) if args.preset else C.auto_config()
    t0 = time.time()
    print(f"=== Phase 2  preset {cfg.name}  fixture {cfg.fixture.name} ===")

    # A family list that drifts from the feature list is an ablation that
    # silently measures something else. Checked before any compute is spent.
    fam_cols = {c for cols in FAMILIES.values() for c in cols}
    assert fam_cols == set(F2.FEATURE2_NAMES), (
        sorted(fam_cols ^ set(F2.FEATURE2_NAMES)))

    df = D.drop_null_targets(D.load(cfg), verbose=True)

    print("\n--- folds ---")
    res = baselines.run_cv(df, cfg, phase2_models(cfg, args.ablate),
                           feature_builder=F2.build_all)
    print(splits.describe(res["folds"]).to_string(index=False))

    print("\n--- MAE by fold (bps) ---")
    ft = E.fold_table(res["per_fold"])
    print(ft.to_string(float_format=lambda x: f"{x:8.4f}"))

    scored = res["df"].loc[res["scored_mask"]]
    preds = {k: v[res["scored_mask"]] for k, v in res["oof"].items()}
    print(f"\n--- out-of-fold scorecard ({len(scored):,} scored rows, "
          f"dates {scored['date_id'].min()}..{scored['date_id'].max()}) ---")
    sc = E.scorecard(preds, scored)
    print(sc.to_string(index=False, float_format=lambda x: f"{x:10.5f}"))

    best = "lgbm_mem"  # the headline arm, fixed a priori — not min-picked
    consistency = {}
    for by in ("date_id", "seconds_in_bucket", "stock_id"):
        bd = E.breakdown(scored, preds[best], by)
        consistency[by] = {
            "mean_improvement_bps": float(bd["improvement_bps"].mean()),
            "share_of_groups_better": float((bd["improvement_bps"] > 0).mean()),
            "worst_bps": float(bd["improvement_bps"].min()),
            "best_bps": float(bd["improvement_bps"].max()),
        }
        if by == "seconds_in_bucket":
            print("\nby seconds_in_bucket (lgbm_mem):")
            print(bd.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
        else:
            c = consistency[by]
            print(f"\nby {by}: improvement over zero, bps — mean {c['mean_improvement_bps']:+.4f}, "
                  f"better on {c['share_of_groups_better']:.1%} of groups, "
                  f"worst {c['worst_bps']:+.4f}, best {c['best_bps']:+.4f}")

    imp_mean = {}
    for name, per_fold_imp in res["importances"].items():
        imp_mean[name] = pd.DataFrame(per_fold_imp).mean(axis=1).sort_values(ascending=False)
    if "lgbm_mem" in imp_mean:
        print("\n--- lgbm_mem feature importance (share of total gain, mean over folds) ---")
        print(imp_mean["lgbm_mem"].round(4).to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "preset": cfg.name,
        "fixture": cfg.fixture.name,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items()},
        "lgbm_params": boosted.LGBM_PARAMS,
        "feature_columns": {"row": list(F.FEATURE_NAMES), "all": list(F2.ALL_NAMES)},
        "families": {k: list(v) for k, v in FAMILIES.items()},
        "runtime_seconds": round(time.time() - t0, 1),
        "folds": splits.describe(res["folds"]).to_dict("records"),
        "per_fold_mae": res["per_fold"],
        "fold_table": ft.reset_index().to_dict("records"),
        "scorecard": sc.to_dict("records"),
        "consistency_lgbm_mem": consistency,
        "by_seconds": E.breakdown(scored, preds[best], "seconds_in_bucket").to_dict("records"),
        "importance_mean": {k: v.to_dict() for k, v in imp_mean.items()},
        "ablate": args.ablate,
    }
    args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    shown = args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out
    print(f"\nwrote {shown}   ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
