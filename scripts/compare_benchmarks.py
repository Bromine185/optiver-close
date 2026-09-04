#!/usr/bin/env python
"""Benchmarks: this repository's offline numbers beside the published competition results.

    python scripts/compare_benchmarks.py                    # -> reports/benchmark_comparison.md
    python scripts/compare_benchmarks.py --out some/path.md

Reads `reports/benchmarks.json` (sourced, cited leaderboard numbers — see
BENCHMARKS.md for where each came from), `reports/phase2_lgbm.json`, and
`reports/phase3_ensemble.json` if it exists, and renders one table.

The table has one comparable column and several that are not. MAE in bps is
not comparable across periods: predict-zero scores 6.385 on this repository's
300 scored dates and about 5.40 on the public leaderboard's period, and no
model moves a number by anything like that gap. The comparable statistic is
improvement over predict-zero ON THE SAME ROWS, which is why every row carries
the zero MAE of its own row set, and why rows whose zero MAE is unknown get no
improvement at all rather than a guessed one. Rows from this repository are
labelled as what they are — offline purged CV, dates 181..480 — and never as a
leaderboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

BENCHMARKS = REPO / "reports" / "benchmarks.json"
PHASE2 = REPO / "reports" / "phase2_lgbm.json"
PHASE3 = REPO / "reports" / "phase3_ensemble.json"

#: This repository's rows, by report and model name, with the label the table shows.
#: Ablation arms are left out: they are measurements of feature families, not models.
REPO_ARMS = {
    "phase2": [
        ("zero", "predict-zero (the floor)"),
        ("ridge", "ridge, 14 row-wise features"),
        ("lgbm_row", "LightGBM, 14 row-wise features"),
        ("ridge_mem", "ridge, 31 features (+memory)"),
        ("lgbm_mem", "LightGBM, 31 features (+memory)"),
    ],
    "phase3": [
        ("mlp_mem", "MLP + stock embedding, 31 features"),
        ("blend_fixed", "0.5 LightGBM + 0.5 MLP (fixed)"),
        ("blend_forward", "LightGBM + MLP, weight fitted forward"),
    ],
}

REPO_PERIOD = "offline purged CV, dates 181..480 (this repo)"
LB_PERIOD = {
    "public": "public leaderboard (hidden dates after train, scored to 2023-12-20)",
    "private": "private leaderboard (forecasting period, 2024)",
    "cv": "author's own CV (split not described)",
}


def load_benchmarks(path: Path = BENCHMARKS) -> dict:
    return json.loads(path.read_text())


def improvement(mae: float, zero: float | None) -> tuple[float, float]:
    """(bps, percent) over predict-zero on the same rows; NaN when the zero MAE is unknown."""
    if zero is None or not np.isfinite(zero):
        return np.nan, np.nan
    return zero - mae, 100.0 * (zero - mae) / zero


def repo_rows(report: Path, arms: list[tuple[str, str]], phase: str) -> list[dict]:
    if not report.exists():
        return []
    rep = json.loads(report.read_text())
    if rep.get("preset") != "FULL":
        # SMOKE numbers are never results (CLAUDE.md, non-negotiable #6).
        return []
    sc = {r["model"]: r for r in rep["scorecard"]}
    zero = sc["zero"]["mae_bps"]
    rows = []
    for model, label in arms:
        if model not in sc:
            continue
        bps, pct = improvement(sc[model]["mae_bps"], zero)
        rows.append({
            "entry": label,
            "period": REPO_PERIOD,
            "mae_bps": sc[model]["mae_bps"],
            "zero_mae_bps": zero,
            "improvement_bps": bps,
            "improvement_pct": pct,
            "source": f"reports/{report.name} ({phase}, `{model}`)",
        })
    return rows


def published_rows(bench: dict) -> list[dict]:
    zeros = {k: (v or {}).get("mae_bps") for k, v in bench["zero_baseline"].items()}
    rows = []
    for e in bench["entries"]:
        zero = zeros.get(e["leaderboard"])
        bps, pct = improvement(e["mae_bps"], zero)
        rows.append({
            "entry": e["name"],
            "period": LB_PERIOD.get(e["leaderboard"], e["leaderboard"]),
            "mae_bps": e["mae_bps"],
            "zero_mae_bps": zero if zero is not None else np.nan,
            "improvement_bps": bps,
            "improvement_pct": pct,
            "source": f"{e['source_url']} ({e['source_type']})",
        })
    return rows


def build_table(bench: dict, phase2: Path = PHASE2, phase3: Path = PHASE3) -> pd.DataFrame:
    rows = (published_rows(bench)
            + repo_rows(phase2, REPO_ARMS["phase2"], "Phase 2")
            + repo_rows(phase3, REPO_ARMS["phase3"], "Phase 3"))
    df = pd.DataFrame(rows)
    # Sort by the comparable column; rows with no comparable number go last, by MAE.
    return df.sort_values(["improvement_pct", "mae_bps"], ascending=[False, True],
                          na_position="last").reset_index(drop=True)


def render_markdown(df: pd.DataFrame, bench: dict, phase3_present: bool) -> str:
    def f(x, nd):
        return "—" if not np.isfinite(x) else f"{x:.{nd}f}"

    lines = [
        "# Benchmark comparison",
        "",
        f"Rendered by `python scripts/compare_benchmarks.py` from `reports/benchmarks.json` "
        f"(retrieved {bench['retrieved']}), `reports/phase2_lgbm.json`"
        + (" and `reports/phase3_ensemble.json`." if phase3_present else
           "; `reports/phase3_ensemble.json` was not present."),
        "",
        "Improvement is over predict-zero **on the same rows**, and is the only column that",
        "compares across periods. Rows with no published zero baseline for their period get",
        "no improvement rather than a guessed one. Why MAE itself does not compare, and what",
        "this table does not license anyone to say, is in `BENCHMARKS.md`.",
        "",
        "| entry | period | MAE (bps) | zero MAE, same rows | improvement (bps) | improvement (%) | source |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in df.itertuples(index=False):
        lines.append(
            f"| {r.entry} | {r.period} | {f(r.mae_bps, 4)} | {f(r.zero_mae_bps, 3)} | "
            f"{f(r.improvement_bps, 4)} | {f(r.improvement_pct, 2)} | {r.source} |"
        )
    lines += ["", "## Not sourced", ""]
    lines += [f"* {s}" for s in bench.get("not_sourced", [])]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", type=Path, default=BENCHMARKS)
    ap.add_argument("--phase2", type=Path, default=PHASE2)
    ap.add_argument("--phase3", type=Path, default=PHASE3)
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "benchmark_comparison.md")
    args = ap.parse_args()

    bench = load_benchmarks(args.benchmarks)
    df = build_table(bench, args.phase2, args.phase3)

    print(f"=== benchmarks  {len(bench['entries'])} published entries, "
          f"{int((df['period'] == REPO_PERIOD).sum())} rows from this repo ===\n")
    shown = df.drop(columns=["source"]).copy()
    shown["entry"] = shown["entry"].str.slice(0, 60)
    shown["period"] = shown["period"].str.slice(0, 44)
    print(shown.to_string(index=False, float_format=lambda x: f"{x:9.4f}", na_rep="—"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(df, bench, args.phase3.exists()))
    rel = args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out
    print(f"\nwrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
