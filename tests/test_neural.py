"""The Phase 3 MLP: protocol, determinism, and the inner holdout.

Not tests of model QUALITY — SMOKE numbers are never results (non-negotiable
#6). These pin what makes a FULL number believable: the same fit twice is
bit-identical on CPU, the model plugs into the same `run_cv` that scored
Phases 1 and 2, its predictions vary with its inputs, and early stopping
looks only at dates that are inside the training window and embargoed from
the rest of it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from optiver import baselines, config as C, data as D, neural  # noqa: E402
from optiver import features2 as F2  # noqa: E402

SKIP2 = tuple(F2.INDICATOR2_NAMES) + tuple(F2.BOUNDED2_NAMES)
#: Shrunk for speed only — the code path is the full one.
SMALL = dict(neural.MLP_PARAMS, hidden=(32, 16), max_epochs=3, batch_size=2048)


def small_mlp(name: str = "mlp_mem", **over) -> neural.MlpMae:
    return neural.MlpMae(name=name, columns=tuple(F2.ALL_NAMES), extra_skip=SKIP2,
                         params=dict(SMALL, **over), device="cpu")


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


def test_same_fit_twice_is_bit_identical_on_cpu(df, X, halves):
    """Init, batch order and dropout all flow from one labelled seed, inside fork_rng."""
    tr, va = halves
    cfg = C.get_config("SMOKE")
    preds = []
    for _ in range(2):
        m = small_mlp()
        m.fit(tr, X.loc[tr.index], cfg)
        preds.append(m.predict(va, X.loc[va.index]))
    np.testing.assert_array_equal(preds[0], preds[1])


def test_fit_leaves_the_global_torch_rng_where_it_found_it(df, X, halves):
    """fork_rng, verified: a draw after the fit equals the draw that would have happened without it."""
    tr, _ = halves
    torch.manual_seed(12345)
    before = torch.rand(3)
    torch.manual_seed(12345)
    small_mlp().fit(tr, X.loc[tr.index], C.get_config("SMOKE"))
    after = torch.rand(3)
    assert torch.equal(before, after)


def test_predictions_are_finite_and_not_constant(df, X, halves):
    tr, va = halves
    m = small_mlp()
    m.fit(tr, X.loc[tr.index], C.get_config("SMOKE"))
    p = m.predict(va, X.loc[va.index])
    assert p.shape == (len(va),)
    assert np.isfinite(p).all()
    assert np.std(p) > 0, "a constant prediction means the network learned nothing at all"


def test_column_subset_is_respected(df, X, halves):
    """Perturbing an EXCLUDED column must not move a single prediction."""
    tr, va = halves
    cols = tuple(c for c in F2.ALL_NAMES if c != "wap_ret_1b_bps")
    m = neural.MlpMae(name="ablate", columns=cols, extra_skip=SKIP2, params=dict(SMALL), device="cpu")
    m.fit(tr, X.loc[tr.index], C.get_config("SMOKE"))
    Xva = X.loc[va.index]
    p1 = m.predict(va, Xva)
    Xmut = Xva.copy()
    Xmut["wap_ret_1b_bps"] = 999.0
    p2 = m.predict(va, Xmut)
    np.testing.assert_array_equal(p1, p2)


def test_inner_holdout_is_inside_training_dates_and_embargoed(df, X, halves):
    """Early stopping never sees the fold's validation dates: the inner block is
    the tail of the TRAINING dates, and an embargo separates it from the inner
    training dates — by position, so a holed date axis counts dates, not ids."""
    tr, _ = halves
    cfg = C.get_config("SMOKE")
    m = small_mlp()
    m.fit(tr, X.loc[tr.index], cfg)
    train_dates = np.sort(tr["date_id"].unique())
    inner_tr, inner_va = m.inner_train_dates, m.inner_val_dates
    assert set(inner_tr) <= set(train_dates) and set(inner_va) <= set(train_dates)
    assert not set(inner_tr) & set(inner_va)
    assert inner_tr.max() < inner_va.min()
    pos = {d: i for i, d in enumerate(train_dates)}
    assert pos[inner_va.min()] - pos[inner_tr.max()] - 1 == cfg.embargo_dates
    assert len(inner_va) == max(1, round(SMALL["inner_val_frac"] * len(train_dates)))


def test_early_stopping_keeps_the_best_inner_epoch(df, X, halves):
    """The model that predicts is the checkpoint with the lowest inner-holdout MAE,
    and the history records every epoch that ran."""
    tr, _ = halves
    m = small_mlp()
    m.fit(tr, X.loc[tr.index], C.get_config("SMOKE"))
    hist = m.history
    assert 1 <= len(hist) <= SMALL["max_epochs"]
    assert m.best_epoch == int(np.argmin([h["inner_val_mae"] for h in hist]))
    s = m.training_summary()
    assert s["device"] == "cpu" and s["epochs_run"] == len(hist)


def test_too_few_training_dates_raises_rather_than_silently_skipping_the_holdout(df, X):
    """A frame too small for an embargoed inner block is an error, not a toy branch."""
    dates = np.sort(df["date_id"].unique())[:3]
    tiny = df[df["date_id"].isin(dates)]
    with pytest.raises(ValueError, match="inner validation"):
        small_mlp().fit(tiny, X.loc[tiny.index], C.get_config("SMOKE"))


def test_mlp_runs_inside_the_phase1_harness(df):
    """One SMOKE pass of run_cv with the Phase 2 builder and a small MLP: the
    same folds, masks and scorecard machinery every earlier number came from."""
    cfg = C.get_config("SMOKE")
    models = [baselines.Zero(), small_mlp()]
    res = baselines.run_cv(df, cfg, models, feature_builder=F2.build_all, verbose=False)
    assert set(res["oof"]) == {"zero", "mlp_mem"}
    scored = res["scored_mask"]
    assert scored.any()
    assert np.isfinite(res["oof"]["mlp_mem"][scored]).all()
