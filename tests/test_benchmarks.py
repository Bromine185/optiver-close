"""The published-results file and the script that puts this repo's numbers beside it.

Nothing here scores anything. These pin what makes the comparison honest: every
published number has a source and a period, no improvement is ever computed
against a zero baseline that was not itself published, and the rows that come
from this repository are labelled as offline CV, not as a leaderboard.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO / "reports" / "benchmarks.json"
SCRIPT = REPO / "scripts" / "compare_benchmarks.py"

LEADERBOARDS = {"public", "private", "cv"}


@pytest.fixture(scope="module")
def bench():
    return json.loads(BENCHMARKS.read_text())


@pytest.fixture(scope="module")
def cb():
    spec = importlib.util.spec_from_file_location("compare_benchmarks", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_entry_is_sourced_and_finite(bench):
    assert bench["entries"], "no published entries at all"
    for e in bench["entries"]:
        for key in ("name", "mae_bps", "leaderboard", "model_family", "source_url", "source_type", "retrieved", "notes"):
            assert key in e, f"{e.get('name')!r} lacks {key}"
        assert isinstance(e["mae_bps"], (int, float)) and math.isfinite(e["mae_bps"])
        # Everything in this competition scored between 5 and 7 bps; anything else is a typo.
        assert 5.0 < e["mae_bps"] < 7.0, e["name"]
        assert e["leaderboard"] in LEADERBOARDS, e["name"]
        assert e["source_url"].startswith("https://"), e["name"]
        assert e["source_type"] in {"primary", "secondary"}, e["name"]


def test_zero_baselines_are_declared_for_every_leaderboard_type(bench):
    """Present or explicitly null — never missing, so an absent baseline is a stated fact."""
    zb = bench["zero_baseline"]
    assert set(zb) == LEADERBOARDS
    for lb, v in zb.items():
        if v is not None:
            assert math.isfinite(v["mae_bps"]) and v["source_url"].startswith("https://"), lb
    used = {e["leaderboard"] for e in bench["entries"]}
    # A leaderboard type with entries but no zero baseline must say so in not_sourced.
    for lb in used:
        if zb[lb] is None:
            assert any(lb in s for s in bench["not_sourced"]), f"{lb} has entries, no baseline, no note"


def test_improvement_is_never_computed_without_a_published_zero(cb, bench):
    df = cb.build_table(bench, phase2=REPO / "nonexistent.json", phase3=REPO / "nonexistent.json")
    assert len(df) == len(bench["entries"])
    for r in df.itertuples(index=False):
        if not np.isfinite(r.zero_mae_bps):
            assert not np.isfinite(r.improvement_pct) and not np.isfinite(r.improvement_bps)
        else:
            assert np.isfinite(r.improvement_pct)
    # The public all-zeros score is 5.40 and the mid-competition leader 5.3070: 1.72%.
    lead = df[df["entry"].str.startswith("1st place, mid-competition")].iloc[0]
    assert lead["improvement_pct"] == pytest.approx(100 * (5.40 - 5.3070) / 5.40, abs=1e-9)


def test_repo_rows_are_labelled_as_offline_cv_and_come_from_full_reports(cb, bench, tmp_path):
    df = cb.build_table(bench)  # committed phase2 report (FULL); phase3 if present
    repo = df[df["period"] == cb.REPO_PERIOD]
    assert "lgbm_mem" in " ".join(repo["source"]), "the Phase 2 headline is missing"
    assert "leaderboard" not in cb.REPO_PERIOD
    assert np.allclose(repo["zero_mae_bps"], 6.38518, atol=1e-4)
    # A SMOKE report must contribute nothing, however it is named.
    smoke = tmp_path / "phase2_lgbm.json"
    smoke.write_text(json.dumps({"preset": "SMOKE", "scorecard": [{"model": "zero", "mae_bps": 1.0},
                                                                    {"model": "lgbm_mem", "mae_bps": 0.9}]}))
    assert cb.repo_rows(smoke, cb.REPO_ARMS["phase2"], "Phase 2") == []


def test_cli_writes_the_markdown(tmp_path):
    out = tmp_path / "benchmark_comparison.md"
    r = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], capture_output=True, text=True,
                       cwd=REPO)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "| entry | period |" in text
    assert "offline purged CV, dates 181..480 (this repo)" in text
    assert "LightGBM, 31 features (+memory)" in text
    assert "## Not sourced" in text
