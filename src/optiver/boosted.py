"""Phase 2 model: LightGBM optimising MAE directly.

One model class, deliberately un-tuned. Every hyperparameter below was fixed a
priori at the value any practitioner would name first, and none was moved after
seeing a validation number. That is not modesty, it is the same discipline as
Phase 1's floor: the first gradient-boosting number in the log must be one that
tuning cannot have flattered, or it is the ceiling of a search rather than a
measurement. Tuning, if it ever happens, needs a nested split inside the
training dates and is a separately-logged step.

Why the objective is the whole point: Phase 1's ridge minimises squared error
against an MAE metric, on a target with excess kurtosis 22.6, and needed two
train-only corrections (target winsorisation, an MAE-optimal rescale) to bridge
the mismatch. `objective="l1"` makes the tree growth itself chase the metric, so
BOTH corrections fall away here — no clip, no rescale, no standardisation
(trees are invariant to monotone feature scaling), no feature winsorisation
(splits are order statistics; a 4.4-million-bps outlier lands in the same leaf
as a 400-bps one).

Determinism: LightGBM is reproducible given `deterministic=True`, a fixed seed,
a fixed `force_col_wise`, and a FIXED THREAD COUNT — the histogram reduction
order changes with threads, so `num_threads` is pinned rather than left to the
machine. The seed derives from `seeding._label_seed`, per non-negotiable #5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .seeding import _label_seed

#: Fixed a priori — see the module docstring. Recorded verbatim in the report.
LGBM_PARAMS = {
    "objective": "l1",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 200,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 8,
    "verbose": -1,
}


@dataclass
class LightGBMMae:
    """LightGBM on a named column subset. Fits `baselines.run_cv`'s model protocol.

    `columns` is what makes the 2x2 experiment one class: the same model on
    `features.FEATURE_NAMES` isolates the model-class gain over ridge, and on
    `features2.ALL_NAMES` adds the memory features on top.
    """

    name: str = "lgbm"
    columns: tuple[str, ...] = ()
    params: dict = field(default_factory=lambda: dict(LGBM_PARAMS))
    model: object = field(default=None, repr=False)

    def fit(self, df: pd.DataFrame, X: pd.DataFrame, cfg: Config) -> "LightGBMMae":
        import lightgbm as lgb

        y = df["target"].to_numpy(np.float64)
        ok = np.isfinite(y)
        # float32 is enough for tree splits (LightGBM histograms the values
        # anyway) and halves the copy of a 4M x 31 matrix.
        Z = X[list(self.columns)].to_numpy(np.float32)
        self.model = lgb.LGBMRegressor(
            **self.params, random_state=_label_seed(f"lgbm-{self.name}") % (2**31 - 1)
        )
        self.model.fit(Z[ok], y[ok])
        return self

    def predict(self, df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[list(self.columns)].to_numpy(np.float32))

    def feature_importance(self) -> pd.Series:
        """Total gain per feature, normalised to sum to 1 within the model.

        Gain, not split count: a feature used once at the root can matter more
        than one used in a thousand leaf-level splits, and split counts reward
        exactly the high-cardinality noise columns trees love to overuse.
        """
        gain = self.model.booster_.feature_importance(importance_type="gain")
        s = pd.Series(gain, index=list(self.columns), dtype=np.float64)
        total = s.sum()
        return (s / total if total > 0 else s).sort_values(ascending=False)
