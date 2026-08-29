"""The LightGBM model: protocol, determinism, and honest failure.

Not tests of model QUALITY — SMOKE numbers are never results (non-negotiable
#6). These pin the properties that make a FULL number believable: the same fit
twice is bit-identical, the model plugs into the same `run_cv` that scored
Phase 1, and its predictions vary with its inputs rather than collapsing to a
constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiver import baselines, boosted, config as C, data as D
from optiver import features as F, features2 as F2


@pytest.fixture(scope="module")
def df():
    return D.drop_null_targets(D.add_revealed_target(D.load(C.get_config("SMOKE"))))


@pytest.fixture(scope="module")
def X(df):
    return F2.build_all(df)


@pytest.fixture(scope="module")
def halves(df):
    dates = np.sort(df["date_id"].unique())
    cut = dates[len(dates) // 2]
    return df[df["date_id"] <= cut], df[df["date_id"] > cut]


def test_same_fit_twice_is_bit_identical(df, X, halves):
    """deterministic=True + fixed seed + pinned num_threads, verified not trusted."""
    tr, va = halves
    cfg = C.get_config("SMOKE")
    preds = []
    for _ in range(2):
        m = boosted.LightGBMMae(name="lgbm_mem", columns=tuple(F2.ALL_NAMES))
        m.fit(tr, X.loc[tr.index], cfg)
        preds.append(m.predict(va, X.loc[va.index]))
    np.testing.assert_array_equal(preds[0], preds[1])


def test_predictions_are_finite_and_not_constant(df, X, halves):
    tr, va = halves
    m = boosted.LightGBMMae(name="lgbm_mem", columns=tuple(F2.ALL_NAMES))
    m.fit(tr, X.loc[tr.index], C.get_config("SMOKE"))
    p = m.predict(va, X.loc[va.index])
    assert np.isfinite(p).all()
    assert np.std(p) > 0, "a constant prediction means the trees learned nothing at all"


def test_feature_importance_is_a_distribution_over_its_columns(df, X, halves):
    tr, _ = halves
    cols = tuple(F.FEATURE_NAMES)
    m = boosted.LightGBMMae(name="lgbm_row", columns=cols)
    m.fit(tr, X.loc[tr.index], C.get_config("SMOKE"))
    imp = m.feature_importance()
    assert tuple(sorted(imp.index)) == tuple(sorted(cols))
    assert imp.sum() == pytest.approx(1.0)
    assert (imp >= 0).all()


def test_column_subset_is_respected(df, X, halves):
    """The ablation arms depend on `columns` actually restricting the model:
    perturbing an EXCLUDED column must not move a single prediction."""
    tr, va = halves
    cols = tuple(c for c in F2.ALL_NAMES if c != "wap_ret_1b_bps")
    m = boosted.LightGBMMae(name="ablate", columns=cols)
    m.fit(tr, X.loc[tr.index], C.get_config("SMOKE"))
    Xva = X.loc[va.index]
    p1 = m.predict(va, Xva)
    Xmut = Xva.copy()
    Xmut["wap_ret_1b_bps"] = 999.0
    p2 = m.predict(va, Xmut)
    np.testing.assert_array_equal(p1, p2)


def test_lgbm_runs_inside_the_phase1_harness(df):
    """One SMOKE pass of run_cv with the Phase 2 builder and a small lgbm:
    the same folds, masks and scorecard machinery Phase 1's numbers came from.
    Params are shrunk for speed only — the code path is the full one."""
    cfg = C.get_config("SMOKE")
    small = dict(boosted.LGBM_PARAMS, n_estimators=20, num_leaves=15)
    models = [
        baselines.Zero(),
        boosted.LightGBMMae(name="lgbm_mem", columns=tuple(F2.ALL_NAMES), params=small),
    ]
    res = baselines.run_cv(df, cfg, models, feature_builder=F2.build_all, verbose=False)
    assert set(res["oof"]) == {"zero", "lgbm_mem"}
    scored = res["scored_mask"]
    assert scored.any()
    assert np.isfinite(res["oof"]["lgbm_mem"][scored]).all()
    assert "lgbm_mem" in res["importances"] and len(res["importances"]["lgbm_mem"]) == cfg.n_folds
