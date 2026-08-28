"""The baselines and the CV runner: do they measure what they claim to measure."""

from __future__ import annotations

import numpy as np
import pytest

from optiver import baselines as B
from optiver import config as C
from optiver import data as D
from optiver import evaluate as E
from optiver import splits
from optiver.seeding import fork


@pytest.fixture(scope="module")
def cv(smoke_cfg):
    df = D.drop_null_targets(D.load(smoke_cfg))
    return B.run_cv(df, smoke_cfg, verbose=False), smoke_cfg


def test_every_model_is_scored_on_every_fold(cv):
    res, cfg = cv
    names = {m.name for m in B.default_models(cfg)}
    seen = {(r["fold"], r["model"]) for r in res["per_fold"]}
    assert seen == {(f, n) for f in range(cfg.n_folds) for n in names}


def test_out_of_fold_predictions_exist_only_on_validation_rows(cv):
    """The check that would catch a runner writing predictions into training rows
    and then scoring them — a fold table that looks great and means nothing."""
    res, cfg = cv
    df = res["df"]
    val_dates = np.concatenate([f.val_dates for f in res["folds"]])
    on_val = df["date_id"].isin(val_dates).to_numpy()
    for name, pred in res["oof"].items():
        assert np.isfinite(pred[on_val]).all(), name
        assert not np.isfinite(pred[~on_val]).any(), name


def test_zero_baseline_reproduces_the_mean_absolute_target_exactly(cv):
    """Ties the metric implementation to the baseline implementation: if either
    drifts, the floor every other number is quoted against moves silently.

    The reference is computed in float64. Summing 17,600 float32 targets instead
    gives 6.2102957 against 6.2102961 — a disagreement in the 8th digit, from
    accumulation rather than storage. It is far below anything that matters here
    (model differences are ~0.06 bps), but it is why `evaluate.mae` upcasts
    before summing rather than trusting the fixture's dtype."""
    res, cfg = cv
    df = res["df"]
    for f in res["folds"]:
        _, va = splits.fold_masks(df, f)
        want = np.abs(df.loc[va, "target"].to_numpy(np.float64)).mean()
        got = next(r["mae_bps"] for r in res["per_fold"] if r["fold"] == f.index and r["model"] == "zero")
        assert got == pytest.approx(want, rel=1e-12)


def test_the_unshrunk_carry_is_much_worse_than_predicting_zero(cv):
    """A recorded null result, asserted so it cannot quietly stop being true.
    Cross-auction autocorrelation of the target is 0.027; carrying the previous
    day's value forward at full weight adds roughly sqrt(2) times the noise."""
    res, _ = cv
    scored = res["df"].loc[res["scored_mask"]]
    preds = {k: v[res["scored_mask"]] for k, v in res["oof"].items()}
    assert E.mae(scored["target"], preds["carry_raw"]) > 1.2 * E.mae(
        scored["target"], preds["zero"]
    )


def test_the_shrinkage_the_carry_learns_is_tiny(cv):
    """If the fitted shrink came back near 1.0, either the target really does
    carry (it does not) or the fit is looking at the wrong column."""
    res, cfg = cv
    df = res["df"]
    f = res["folds"][0]
    tr, _ = splits.fold_masks(df, f)
    m = B.Carry(shrink=None).fit(df.loc[tr], None, cfg)
    assert 0.0 <= m.fitted_shrink < 0.2


def test_carry_falls_back_to_zero_where_there_is_no_previous_auction(cv):
    res, cfg = cv
    df = res["df"]
    first = df["date_id"].min()
    m = B.Carry(shrink=1.0)
    p = m.predict(df[df["date_id"] == first], None)
    assert np.all(p == 0.0)


def test_ridge_is_fitted_only_on_training_rows(cv, smoke_cfg):
    """Fit on the first fold's training dates, then confirm the model is unchanged
    by anything in validation: refitting on train alone must give the same
    coefficients as the CV run produced."""
    res, cfg = cv
    df, X = res["df"], None
    from optiver import features as F

    X = F.build(df)
    f = res["folds"][0]
    tr, _ = splits.fold_masks(df, f)
    a = B.RidgeMicro(alpha=cfg.ridge_alpha).fit(df.loc[tr], X.loc[tr], cfg).coefficients()
    b = res["coefficients"][0]
    assert np.allclose(a.reindex(b.index).to_numpy(), b.to_numpy())


def test_mae_optimal_scale_recovers_a_known_scale():
    rng = fork("mae-scale-test")
    x = rng.normal(size=20_000)
    assert B._mae_optimal_scale(0.5 * x, x) == pytest.approx(0.5, abs=0.01)
    # a prediction uncorrelated with the truth should be shrunk toward nothing
    assert B._mae_optimal_scale(rng.normal(size=20_000), x) < 0.1


def test_mae_optimal_scale_collapses_a_degenerate_prediction():
    assert B._mae_optimal_scale(np.array([1.0, 2.0]), np.zeros(2)) == 0.0


def test_ridge_coefficients_are_named_and_ordered_by_magnitude(cv):
    res, _ = cv
    from optiver import features as F

    c = res["coefficients"][0]
    assert set(c.index) == set(F.FEATURE_NAMES)
    assert (c.abs().to_numpy()[:-1] >= c.abs().to_numpy()[1:]).all()
