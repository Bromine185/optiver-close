"""MAE against hand-computed cases, and the breakdowns that back RESEARCH.md."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optiver import evaluate as E


def test_mae_hand_computed():
    """|1-0| + |-2-0| + |3.5-0| + |0-1| = 1 + 2 + 3.5 + 1 = 7.5, over 4 -> 1.875."""
    y = np.array([1.0, -2.0, 3.5, 0.0])
    p = np.array([0.0, 0.0, 0.0, 1.0])
    assert E.mae(y, p) == pytest.approx(1.875)


def test_mae_of_zero_is_the_mean_absolute_target():
    y = np.array([3.0, -4.0, 0.5])
    assert E.mae(y, np.zeros(3)) == pytest.approx(np.abs(y).mean())
    assert E.mae(y, np.zeros(3)) == pytest.approx(7.5 / 3)


def test_null_targets_are_excluded_not_scored_as_zero():
    """A model predicting 0 would otherwise be credited with a perfect hit on
    every unlabelled row, which is exactly backwards."""
    y = np.array([1.0, np.nan, 3.0])
    assert E.mae(y, np.zeros(3)) == pytest.approx(2.0)      # (1 + 3) / 2, not (1 + 0 + 3) / 3


def test_a_non_finite_prediction_on_a_labelled_row_raises():
    """Skipping it would score this model on 2 rows while its rivals are scored on
    3, and the dropped row is exactly the one it could not handle."""
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([0.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="non-finite prediction"):
        E.mae(y, p)


def test_a_non_finite_prediction_on_an_unlabelled_row_is_harmless():
    """The row is not scored either way, so there is nothing to hide."""
    y = np.array([1.0, np.nan, 3.0])
    assert E.mae(y, np.array([0.0, np.nan, 0.0])) == pytest.approx(2.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        E.mae(np.zeros(3), np.zeros(4))


def test_nothing_to_score_raises():
    with pytest.raises(ValueError, match="no labelled rows"):
        E.mae(np.array([np.nan, np.nan]), np.zeros(2))


def test_median_beats_mean_under_mae():
    """The property that makes predict-zero strong here: MAE's optimal constant
    is the median, and an index-relative target has a median near zero even when
    a few enormous tails drag the mean around."""
    y = np.array([-1.0, 0.0, 1.0, 2.0, 100.0])
    assert E.mae(y, np.full(5, np.median(y))) < E.mae(y, np.full(5, y.mean()))


def test_breakdown_reports_each_groups_own_zero_baseline():
    df = pd.DataFrame({"target": [1.0, 3.0, -10.0, 10.0], "stock_id": [0, 0, 1, 1]})
    out = E.breakdown(df, np.zeros(4), "stock_id").set_index("stock_id")
    assert out.loc[0, "mae"] == pytest.approx(2.0)
    assert out.loc[1, "mae"] == pytest.approx(10.0)
    assert (out["improvement_bps"] == 0).all()          # predicting zero cannot beat zero
    assert out.loc[0, "n"] == 2


def test_breakdown_improvement_is_signed_against_zero():
    df = pd.DataFrame({"target": [2.0, 2.0], "seconds_in_bucket": [0, 0]})
    better = E.breakdown(df, np.full(2, 1.0), "seconds_in_bucket")
    worse = E.breakdown(df, np.full(2, -5.0), "seconds_in_bucket")
    assert better["improvement_bps"].iloc[0] == pytest.approx(1.0)
    assert worse["improvement_bps"].iloc[0] == pytest.approx(-5.0)


def test_scorecard_sorts_best_first_and_measures_against_zero():
    df = pd.DataFrame({"target": [1.0, -3.0, 2.0]})
    sc = E.scorecard({"bad": np.full(3, 5.0), "zero": np.zeros(3)}, df).set_index("model")
    assert sc.index.tolist() == ["zero", "bad"]
    assert sc.loc["zero", "vs_zero_bps"] == pytest.approx(0.0)
    assert sc.loc["bad", "vs_zero_bps"] < 0


def test_fold_table_pairs_each_model_against_zero_within_fold():
    """Fold-to-fold MAE varies by ~1 bps on this data while model differences are
    ~0.06 bps, so an unpaired comparison measures the calendar."""
    per_fold = [
        {"fold": 0, "model": "zero", "mae_bps": 7.0},
        {"fold": 1, "model": "zero", "mae_bps": 5.0},
        {"fold": 0, "model": "m", "mae_bps": 6.9},
        {"fold": 1, "model": "m", "mae_bps": 4.9},
    ]
    t = E.fold_table(per_fold)
    assert t.loc["m", "mean_vs_zero"] == pytest.approx(0.1)
    assert t.index[0] == "m"


def test_describe_target_reports_the_numbers_the_log_quotes(smoke_df):
    d = E.describe_target(smoke_df["target"])
    assert d["n"] == int(smoke_df["target"].notna().sum())
    assert d["mean_abs"] == pytest.approx(E.mae(smoke_df["target"], np.zeros(len(smoke_df))))
    assert abs(d["median"]) < 1.0          # index-relative: near zero by construction
