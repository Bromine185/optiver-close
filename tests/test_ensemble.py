"""The forward blend: the weight for fold k is fitted on folds < k and nothing else."""

from __future__ import annotations

import numpy as np
import pytest

from optiver import config as C, data as D, ensemble, splits


@pytest.fixture(scope="module")
def frame():
    cfg = C.get_config("SMOKE")
    df = D.drop_null_targets(D.load(cfg))
    folds = splits.make_folds(D.date_ids(df), cfg)
    return cfg, df, folds


def _val_masks(df, folds):
    return {f.index: splits.fold_masks(df, f)[1] for f in folds}


def test_fold_zero_takes_the_prior_and_later_folds_fit_on_earlier_rows_only(frame):
    """`a` is exact on every fold's rows EXCEPT its own; `b` is exact on the
    current fold only. A weight that peeked at the fold it scores would pick
    `b` (weight_a = 0); a forward weight sees only earlier folds, where `a` is
    perfect, and must pick `a` (weight_a = 1)."""
    cfg, df, folds = frame
    y = df["target"].to_numpy(np.float64)
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=5.0, size=len(y))
    va = _val_masks(df, folds)

    a = np.full(len(y), np.nan)
    b = np.full(len(y), np.nan)
    for k, m in va.items():
        a[m] = y[m] + noise[m]      # noisy on its own fold ...
        b[m] = y[m]                 # ... where b is exact
        for j, mj in va.items():
            if j < k:
                a[mj] = y[mj]       # a is exact on every EARLIER fold
                b[mj] = y[mj] + noise[mj]

    out = ensemble.blend_forward(a, b, df, folds, prior=0.5)
    w = {row["fold"]: row for row in out["weights"]}
    assert w[0]["fitted"] is False and w[0]["weight_a"] == 0.5 and w[0]["fit_dates"] is None
    for k in range(1, cfg.n_folds):
        assert w[k]["fitted"] is True
        assert w[k]["weight_a"] == 1.0, "the weight looked at the fold it scores"
        assert w[k]["fit_dates"][1] < folds[k].val_date_min
    # And the blend on fold 0 is the prior applied literally.
    m0 = va[0]
    np.testing.assert_allclose(out["pred"][m0], 0.5 * a[m0] + 0.5 * b[m0])


def test_blend_is_nan_exactly_outside_the_validation_blocks(frame):
    _, df, folds = frame
    y = df["target"].to_numpy(np.float64)
    out = ensemble.blend_forward(y, y, df, folds)
    any_val = np.zeros(len(y), dtype=bool)
    for m in _val_masks(df, folds).values():
        any_val |= m
    assert np.isfinite(out["pred"][any_val]).all()
    assert np.isnan(out["pred"][~any_val]).all()


def test_blending_a_vector_with_itself_is_the_identity(frame):
    _, df, folds = frame
    p = np.random.default_rng(1).normal(size=len(df))
    out = ensemble.blend_forward(p, p, df, folds)
    ok = np.isfinite(out["pred"])
    np.testing.assert_allclose(out["pred"][ok], p[ok])
    np.testing.assert_allclose(ensemble.blend_fixed(p, p, 0.3), p)


def test_weights_are_convex_and_the_grid_finds_a_known_optimum():
    rng = np.random.default_rng(2)
    y = rng.normal(size=20_000)
    a = y + rng.normal(scale=1.0, size=y.size)
    b = y + rng.normal(scale=1.0, size=y.size)
    w = ensemble.mae_optimal_weight(y, a, b)
    assert 0.0 <= w <= 1.0
    # Independent, equally noisy inputs: the optimum is 0.5 to grid resolution.
    assert abs(w - 0.5) <= 0.05
    with pytest.raises(ValueError):
        ensemble.blend_fixed(a, b, 1.5)
    with pytest.raises(ValueError):
        ensemble.mae_optimal_weight(np.array([]), np.array([]), np.array([]))
