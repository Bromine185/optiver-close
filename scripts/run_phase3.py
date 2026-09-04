#!/usr/bin/env python
"""Phase 3: a neural model beside the boosted one, and a blend weighted forward in time.

    python scripts/run_phase3.py                     # FULL if the fixture exists
    python scripts/run_phase3.py --preset SMOKE      # committed 40-stock fixture
    python scripts/run_phase3.py --device cuda       # pin the torch device (else cuda > mps > cpu)
    python scripts/run_phase3.py --serial            # one arm at a time (memory-constrained machines)

Same folds, same embargo, same floor, same scorecard code as Phases 1 and 2.
Every arm sees the 31-column Phase 2 matrix; the experiment is the function
class, and what a blend of two of them is worth:

    zero            the floor, recomputed (must agree with Phase 1 to the digit)
    lgbm_mem        Phase 2's headline, rerun unchanged (same check, against phase2_lgbm.json)
    mlp_mem         NEW: an MLP with a stock embedding, L1 loss, early-stopped on an
                    inner holdout inside the training dates (neural.py)
    blend_fixed     0.5 * lgbm_mem + 0.5 * mlp_mem — fits nothing
    blend_forward   the headline arm, fixed a priori: per-fold weight fitted on the
                    out-of-fold predictions of EARLIER folds only (ensemble.py)

The headline was named before the run, not min-picked after it.

Two processes, not one. torch and LightGBM each carry their own OpenMP
runtime, and on macOS a process holding both either segfaults or deadlocks
depending on which initialised first (OMP Error #179, pthread_mutex_init) —
measured, not assumed, and not fixed by KMP_DUPLICATE_LIB_OK. So each arm
runs `run_cv` in its own interpreter (`--arm lgbm` / `--arm mlp`, the same
code path this script calls) and hands back its out-of-fold vectors; the
parent never imports either library. Both children rebuild the same fixture,
the same folds and the same predict-zero vector, and the parent checks all
three agree before blending anything. The arms run concurrently by default —
trees on the CPU and the network on the GPU at the same time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from optiver import baselines, boosted, config as C, data as D, ensemble, evaluate as E  # noqa: E402
from optiver import features as F, features2 as F2, neural, splits  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

HEADLINE = "blend_forward"
ARMS = ("lgbm", "mlp")
PHASE2_REPORT = REPO / "reports" / "phase2_lgbm.json"


@dataclass
class RecordingMlp(neural.MlpMae):
    """`run_cv` refits one object per fold; keep each fold's training summary for the log."""

    summaries: list = field(default_factory=list, repr=False)

    def fit(self, df, X, cfg):
        super().fit(df, X, cfg)
        self.summaries.append(self.training_summary())
        return self


def arm_models(arm: str, device: str | None) -> list:
    all_cols = tuple(F2.ALL_NAMES)
    skip2 = tuple(F2.INDICATOR2_NAMES) + tuple(F2.BOUNDED2_NAMES)
    if arm == "lgbm":
        return [baselines.Zero(), boosted.LightGBMMae(name="lgbm_mem", columns=all_cols)]
    if arm == "mlp":
        return [baselines.Zero(), RecordingMlp(name="mlp_mem", columns=all_cols, extra_skip=skip2, device=device)]
    raise ValueError(arm)


def load_frame(cfg: C.Config) -> pd.DataFrame:
    return D.drop_null_targets(D.load(cfg), verbose=True)


def frame_fingerprint(df: pd.DataFrame) -> dict:
    """What two processes must agree on before their OOF vectors can be combined."""
    y = df["target"].to_numpy(np.float64)
    return {
        "n_rows": int(len(df)),
        "target_sum": float(np.nansum(y)),
        "first": [int(df["date_id"].iloc[0]), int(df["seconds_in_bucket"].iloc[0]), int(df["stock_id"].iloc[0])],
        "last": [int(df["date_id"].iloc[-1]), int(df["seconds_in_bucket"].iloc[-1]), int(df["stock_id"].iloc[-1])],
    }


# --------------------------------------------------------------------------
# child: one arm
# --------------------------------------------------------------------------

def run_arm(arm: str, cfg: C.Config, device: str | None, out: Path) -> int:
    t0 = time.time()
    df = load_frame(cfg)
    models = arm_models(arm, device)
    res = baselines.run_cv(df, cfg, models, feature_builder=F2.build_all)
    meta = {
        "arm": arm,
        "fingerprint": frame_fingerprint(df),
        "folds": splits.describe(res["folds"]).to_dict("records"),
        "per_fold": res["per_fold"],
        "importance_mean": {k: pd.DataFrame(v).mean(axis=1).sort_values(ascending=False).to_dict()
                            for k, v in res["importances"].items()},
        "seconds": round(time.time() - t0, 1),
    }
    if arm == "mlp":
        import torch

        mlp = models[1]
        meta["mlp_training"] = mlp.summaries
        meta["device"] = mlp.fitted_device
        meta["torch_version"] = torch.__version__
    np.savez(out.with_suffix(".npz"), scored_mask=res["scored_mask"], **res["oof"])
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"[{arm}] done in {meta['seconds']}s", flush=True)
    return 0


def _relay(arm: str, pipe, t0: float) -> None:
    """Forward a child's lines as they arrive, prefixed with the arm and elapsed seconds."""
    for line in iter(pipe.readline, ""):
        print(f"[{arm} {time.time() - t0:5.0f}s] {line.rstrip()}", flush=True)
    pipe.close()


def spawn_arms(cfg: C.Config, device: str | None, workdir: Path, serial: bool) -> dict[str, dict]:
    """Run both arms as child interpreters, streaming their output live; return meta + OOF arrays."""
    t0 = time.time()
    procs: dict[str, tuple[subprocess.Popen, threading.Thread]] = {}
    for arm in ARMS:
        # -u: unbuffered, so a child's progress lines reach the parent as they are printed.
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--preset", cfg.name,
               "--arm", arm, "--arm-out", str(workdir / arm)]
        if arm == "mlp" and device:
            cmd += ["--device", device]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=REPO,
                                text=True, bufsize=1)
        relay = threading.Thread(target=_relay, args=(arm, proc.stdout, t0), daemon=True)
        relay.start()
        procs[arm] = (proc, relay)
        print(f"[{arm}] started pid {proc.pid}", flush=True)
        if serial:
            proc.wait()
            relay.join()

    out: dict[str, dict] = {}
    for arm, (proc, relay) in procs.items():
        rc = proc.wait()
        relay.join()
        if rc != 0:
            raise RuntimeError(f"{arm} arm exited with {rc}; its output is above")
        meta = json.loads((workdir / f"{arm}.json").read_text())
        with np.load(workdir / f"{arm}.npz") as z:
            meta["oof"] = {k: z[k] for k in z.files if k != "scored_mask"}
            meta["scored_mask"] = z["scored_mask"]
        out[arm] = meta
    return out


def reconcile(arms: dict[str, dict], df: pd.DataFrame, folds: list) -> None:
    """The two children and the parent must have built the same frame and the same folds."""
    fp = frame_fingerprint(df)
    fold_desc = splits.describe(folds).to_dict("records")
    for arm, m in arms.items():
        if m["fingerprint"] != fp:
            raise AssertionError(f"{arm} arm built a different frame: {m['fingerprint']} vs parent {fp}")
        if m["folds"] != fold_desc:
            raise AssertionError(f"{arm} arm built different folds")
    a, b = arms["lgbm"], arms["mlp"]
    if not np.array_equal(a["scored_mask"], b["scored_mask"]):
        raise AssertionError("the two arms scored different rows")
    if not np.array_equal(a["oof"]["zero"], b["oof"]["zero"], equal_nan=True):
        raise AssertionError("the two arms disagree on the predict-zero vector")


def replica_check(sc: pd.DataFrame, preset: str) -> dict:
    """zero and lgbm_mem must reproduce phase2_lgbm.json to the printed digit."""
    if not PHASE2_REPORT.exists():
        return {"available": False, "reason": "no phase2 report"}
    prev = json.loads(PHASE2_REPORT.read_text())
    if prev.get("preset") != preset:
        return {"available": False, "reason": f"phase2 report is {prev.get('preset')}, this run is {preset}"}
    ref = {r["model"]: r["mae_bps"] for r in prev["scorecard"]}
    now = dict(zip(sc["model"], sc["mae_bps"]))
    return {"available": True, "arms": {
        arm: {"phase2": ref[arm], "phase3": now[arm], "abs_diff": abs(ref[arm] - now[arm])}
        for arm in ("zero", "lgbm_mem")
    }}


# --------------------------------------------------------------------------
# parent: the blend, the tables, the report
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=None, choices=["SMOKE", "FULL"])
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "phase3_ensemble.json")
    ap.add_argument("--device", default=None, help="torch device for the mlp arm; default cuda > mps > cpu")
    ap.add_argument("--serial", action="store_true", help="run the two arms one after the other")
    ap.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)        # child mode
    ap.add_argument("--arm-out", type=Path, help=argparse.SUPPRESS)       # child mode
    args = ap.parse_args()

    cfg = C.get_config(args.preset) if args.preset else C.auto_config()
    if args.arm:
        return run_arm(args.arm, cfg, args.device, args.arm_out)

    t0 = time.time()
    print(f"=== Phase 3  preset {cfg.name}  fixture {cfg.fixture.name}  "
          f"arms {'serial' if args.serial else 'parallel'} ===")
    with tempfile.TemporaryDirectory(prefix="phase3-") as tmp:
        arms = spawn_arms(cfg, args.device, Path(tmp), args.serial)

    print("\n--- parent frame ---")
    df = load_frame(cfg)
    folds = splits.make_folds(D.date_ids(df), cfg)
    reconcile(arms, df, folds)
    print("\n--- folds ---")
    print(splits.describe(folds).to_string(index=False))

    oof = {"zero": arms["lgbm"]["oof"]["zero"], "lgbm_mem": arms["lgbm"]["oof"]["lgbm_mem"],
           "mlp_mem": arms["mlp"]["oof"]["mlp_mem"]}
    scored_mask = arms["lgbm"]["scored_mask"]
    # zero is in both children's tables; keep one copy.
    per_fold = arms["lgbm"]["per_fold"] + [r for r in arms["mlp"]["per_fold"] if r["model"] != "zero"]

    # The blends are post-hoc arithmetic on OOF vectors; they enter the same
    # tables as the fitted arms so every comparison is paired within fold.
    fwd = ensemble.blend_forward(oof["lgbm_mem"], oof["mlp_mem"], df, folds, prior=0.5)
    oof["blend_fixed"] = ensemble.blend_fixed(oof["lgbm_mem"], oof["mlp_mem"], 0.5)
    oof["blend_forward"] = fwd["pred"]
    y = df["target"].to_numpy(np.float64)
    for fold in folds:
        _, va = splits.fold_masks(df, fold)
        for name in ("blend_fixed", "blend_forward"):
            per_fold.append({"fold": fold.index, "model": name, "mae_bps": E.mae(y[va], oof[name][va]),
                             "n": int(np.isfinite(y[va]).sum())})

    print("\n--- MAE by fold (bps) ---")
    ft = E.fold_table(per_fold)
    print(ft.to_string(float_format=lambda x: f"{x:8.4f}"))

    print("\n--- forward blend weights (on lgbm_mem; 1 - w on mlp_mem) ---")
    print(pd.DataFrame(fwd["weights"]).to_string(index=False))

    scored = df.loc[scored_mask]
    preds = {k: v[scored_mask] for k, v in oof.items()}
    print(f"\n--- out-of-fold scorecard ({len(scored):,} scored rows, "
          f"dates {scored['date_id'].min()}..{scored['date_id'].max()}) ---")
    sc = E.scorecard(preds, scored)
    print(sc.to_string(index=False, float_format=lambda x: f"{x:10.5f}"))

    rep = replica_check(sc, cfg.name)
    if rep["available"]:
        print("\n--- replica check against reports/phase2_lgbm.json ---")
        for arm, r in rep["arms"].items():
            print(f"{arm:<10s} phase2 {r['phase2']:.5f}  phase3 {r['phase3']:.5f}  |diff| {r['abs_diff']:.2e}")
    else:
        print(f"\n(replica check skipped: {rep['reason']})")

    consistency = {}
    for arm in (HEADLINE, "mlp_mem"):
        consistency[arm] = {}
        for by in ("date_id", "seconds_in_bucket", "stock_id"):
            bd = E.breakdown(scored, preds[arm], by)
            consistency[arm][by] = {
                "mean_improvement_bps": float(bd["improvement_bps"].mean()),
                "share_of_groups_better": float((bd["improvement_bps"] > 0).mean()),
                "worst_bps": float(bd["improvement_bps"].min()),
                "best_bps": float(bd["improvement_bps"].max()),
            }
    print(f"\nby seconds_in_bucket ({HEADLINE}):")
    print(E.breakdown(scored, preds[HEADLINE], "seconds_in_bucket")
          .to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    for arm in (HEADLINE, "mlp_mem"):
        for by in ("date_id", "stock_id"):
            c = consistency[arm][by]
            print(f"{arm} by {by}: improvement over zero, bps — mean {c['mean_improvement_bps']:+.4f}, "
                  f"better on {c['share_of_groups_better']:.1%} of groups, "
                  f"worst {c['worst_bps']:+.4f}, best {c['best_bps']:+.4f}")

    # Paired within fold: the only comparison that survives the 1.3 bps fold-to-fold swing.
    fold_cols = [c for c in ft.columns if c.startswith("fold")]
    paired = {
        "blend_forward_minus_lgbm_mem": (ft.loc["lgbm_mem", fold_cols] - ft.loc[HEADLINE, fold_cols]).tolist(),
        "mlp_mem_minus_lgbm_mem": (ft.loc["lgbm_mem", fold_cols] - ft.loc["mlp_mem", fold_cols]).tolist(),
    }
    print("\nblend_forward vs lgbm_mem, per fold (bps, + = blend better):",
          " ".join(f"{d:+.4f}" for d in paired["blend_forward_minus_lgbm_mem"]))

    print("\n--- mlp_mem early stopping, per fold ---")
    for i, s in enumerate(arms["mlp"]["mlp_training"]):
        print(f"fold {i}: best epoch {s['best_epoch']} of {s['epochs_run']} run, "
              f"inner holdout dates {s['inner_val_dates'][0]}..{s['inner_val_dates'][1]} "
              f"({s['inner_val_dates'][2]}d), best inner MAE "
              f"{s['history'][s['best_epoch']]['inner_val_mae']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "preset": cfg.name,
        "fixture": cfg.fixture.name,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items()},
        "device": arms["mlp"]["device"],
        "torch_version": arms["mlp"]["torch_version"],
        "arms": {"parallel": not args.serial,
                 **{arm: {"seconds": m["seconds"]} for arm, m in arms.items()}},
        "lgbm_params": boosted.LGBM_PARAMS,
        "mlp_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in neural.MLP_PARAMS.items()},
        "headline": HEADLINE,
        "feature_columns": {"row": list(F.FEATURE_NAMES), "all": list(F2.ALL_NAMES)},
        "runtime_seconds": round(time.time() - t0, 1),
        "folds": splits.describe(folds).to_dict("records"),
        "per_fold_mae": per_fold,
        "fold_table": ft.reset_index().to_dict("records"),
        "paired_by_fold": paired,
        "scorecard": sc.to_dict("records"),
        "replica_check": rep,
        "blend_weights": fwd["weights"],
        "consistency": consistency,
        "by_seconds": E.breakdown(scored, preds[HEADLINE], "seconds_in_bucket").to_dict("records"),
        "mlp_training": arms["mlp"]["mlp_training"],
        "importance_mean": arms["lgbm"]["importance_mean"],
    }
    args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    shown = args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out
    print(f"\nwrote {shown}   ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
