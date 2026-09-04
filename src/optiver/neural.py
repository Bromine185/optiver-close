"""Phase 3 model: a multilayer perceptron with a stock embedding, optimising MAE.

Why a second model class at all. Phase 2 established that the signal is
nonlinear (LightGBM took +0.038 bps over ridge on identical columns) and that
the trees and the memory features interact. An MLP asks a different question
of the same 31 columns: whether a smooth function of them, plus a per-stock
learned vector, extracts anything the trees leave on the table. The honest
answer is measured by the blend, not by the MLP alone — see `ensemble.py`.

The embedding is the one input the trees never see. `stock_id` is not a
feature in Phase 1 or Phase 2: a 200-level categorical would let the trees
memorise per-stock target scale, which the cross-sectional family already
carries in a form that transfers across dates. Here the stock enters as an
8-dimensional learned vector concatenated to the standardised features, so the
network can learn a per-stock *shaping* of the same inputs rather than a
per-stock level. Whether that is worth anything is a Phase 3 measurement.

Discipline inherited from Phase 2, unchanged:

* Hyperparameters are fixed a priori (`MLP_PARAMS`) at the values any
  practitioner would name first, and none was moved after seeing a validation
  number. The first neural number in the log is a measurement, not the argmax
  of a search.
* The objective is the metric: L1 loss, no target clip, no rescale.
* Early stopping is the one choice the training data makes for itself, and it
  is made on an INNER holdout carved from the training dates — the last
  `inner_val_frac` of them by position, separated from the inner training
  dates by the same embargo the outer harness uses — never on the fold's
  validation dates. The fold sees the model once, at prediction time.
* Features are winsorised and standardised on training rows only, with
  exactly the Phase 1 machinery (`features.quantile_bounds`, the same skip
  lists), so the MLP's inputs are the ridge's inputs and nothing else.

Determinism. Every draw the model makes — initialisation, batch order,
dropout — comes from a stream seeded by `seeding._label_seed`, inside a
`torch.random.fork_rng` context so the global CPU RNG is left as it was found.
On CPU, with a pinned thread count, two fits are bit-identical and
`tests/test_neural.py` asserts it. On CUDA and MPS they are not (reductions
are not order-stable on either), so a FULL run records its device, and a
number produced on a GPU is reproducible to float noise rather than to the bit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import features as F
from .config import Config, N_STOCKS
from .seeding import _label_seed, torch_generator

#: Fixed a priori — see the module docstring. Recorded verbatim in the report.
MLP_PARAMS = {
    "embedding_dim": 8,
    "hidden": (256, 128, 64),
    "dropout": 0.1,
    "batch_size": 4096,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "max_epochs": 20,
    "patience": 3,
    #: Share of the TRAINING dates (by position, latest last) held out for
    #: early stopping. Embargoed from the inner training dates by
    #: `cfg.embargo_dates`, exactly as the outer folds are.
    "inner_val_frac": 0.1,
    #: Pinned for the same reason LightGBM's is: reduction order on CPU
    #: changes with the thread count, and a seed alone does not fix it.
    "num_threads": 8,
}


def resolve_device(requested: str | None = None) -> str:
    """cuda if present, else mps, else cpu — unless the caller says otherwise."""
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_net(n_dense: int, n_stocks: int, embedding_dim: int, hidden: tuple[int, ...], dropout: float):
    """Build the network. Defined inside a function so the module imports without torch."""
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(n_stocks, embedding_dim)
            layers: list[nn.Module] = []
            width = n_dense + embedding_dim
            for h in hidden:
                layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
                width = h
            layers.append(nn.Linear(width, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor, stock: torch.Tensor) -> torch.Tensor:
            return self.mlp(torch.cat([x, self.embedding(stock)], dim=1)).squeeze(1)

    return Net()


@dataclass
class MlpMae:
    """MLP with a stock embedding on a named column subset. Fits `baselines.run_cv`'s protocol.

    `columns` / `extra_skip` mirror `RidgeMicro`: the same 31-column matrix
    with the same winsorisation skip list, so this model differs from the
    ridge in its function class and nothing about its inputs.
    """

    name: str = "mlp"
    columns: tuple[str, ...] = ()
    extra_skip: tuple[str, ...] = ()
    params: dict = field(default_factory=lambda: dict(MLP_PARAMS))
    #: None = `resolve_device()`. Tests pin "cpu" for bit-identity.
    device: str | None = None

    # fitted state
    bounds: dict = field(default_factory=dict, repr=False)
    mu: np.ndarray = field(default=None, repr=False)
    sd: np.ndarray = field(default=None, repr=False)
    model: object = field(default=None, repr=False)
    fitted_device: str = ""
    history: list[dict] = field(default_factory=list, repr=False)
    best_epoch: int = -1
    inner_train_dates: np.ndarray = field(default=None, repr=False)
    inner_val_dates: np.ndarray = field(default=None, repr=False)

    # --- inputs -----------------------------------------------------------

    def _design(self, X: pd.DataFrame, *, fit: bool) -> np.ndarray:
        Xs = X[list(self.columns)]
        if fit:
            self.fitted_columns = tuple(Xs.columns)
            self.bounds = F.quantile_bounds(Xs, extra_skip=self.extra_skip)
        Z = F.clip_outliers(Xs, self.bounds).to_numpy(np.float64)
        if fit:
            self.mu = Z.mean(axis=0)
            sd = Z.std(axis=0)
            # A constant column (an indicator that never fires in this fold)
            # keeps scale 1 and becomes a constant input the bias absorbs.
            self.sd = np.where(sd > 0, sd, 1.0)
        return ((Z - self.mu) / self.sd).astype(np.float32)

    @staticmethod
    def _stock_index(df: pd.DataFrame) -> np.ndarray:
        s = df["stock_id"].to_numpy(np.int64)
        if s.min() < 0 or s.max() >= N_STOCKS:
            raise ValueError(f"stock_id outside 0..{N_STOCKS - 1}: {s.min()}..{s.max()}")
        return s

    def _inner_split(self, dates: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
        """Last `inner_val_frac` of the training dates, by POSITION, embargoed.

        By position rather than by date_id arithmetic for the reason
        `splits.Fold.embargoed_dates` gives: the date axis may have holes
        (SMOKE's is 8 blocks of 8), and an embargo is a count of dates removed,
        not a difference of two integers.
        """
        uniq = np.sort(np.unique(dates))
        n_val = max(1, int(round(self.params["inner_val_frac"] * len(uniq))))
        cut = len(uniq) - n_val - cfg.embargo_dates
        if cut < 1:
            raise ValueError(
                f"{len(uniq)} training dates cannot hold an inner validation block of "
                f"{n_val} plus an embargo of {cfg.embargo_dates}"
            )
        self.inner_train_dates, self.inner_val_dates = uniq[:cut], uniq[-n_val:]
        return np.isin(dates, self.inner_train_dates), np.isin(dates, self.inner_val_dates)

    # --- protocol -----------------------------------------------------------

    def fit(self, df: pd.DataFrame, X: pd.DataFrame, cfg: Config) -> "MlpMae":
        import torch

        p = self.params
        torch.set_num_threads(int(p["num_threads"]))
        self.fitted_device = resolve_device(self.device)
        dev = torch.device(self.fitted_device)

        Z = self._design(X, fit=True)
        y = df["target"].to_numpy(np.float64)
        ok = np.isfinite(y)
        stock = self._stock_index(df)
        inner_tr, inner_va = self._inner_split(df["date_id"].to_numpy(), cfg)
        tr_idx = np.flatnonzero(inner_tr & ok)
        va_idx = np.flatnonzero(inner_va & ok)

        Xt = torch.from_numpy(Z).to(dev)
        St = torch.from_numpy(stock).to(dev)
        yt = torch.from_numpy(np.nan_to_num(y, nan=0.0).astype(np.float32)).to(dev)
        va_t = torch.from_numpy(va_idx).to(dev)
        tr_t = torch.from_numpy(tr_idx)  # stays on CPU: the permutation is drawn there
        batches = torch_generator(f"mlp-{self.name}-batches")
        bs = int(p["batch_size"])

        seed = _label_seed(f"mlp-{self.name}") % (2**63 - 1)
        # fork_rng(devices=[]) saves and restores the CPU generator only. The
        # manual_seed inside also seeds CUDA/MPS, whose state is NOT restored —
        # documented rather than hidden; the module docstring says why GPU
        # runs are float-close and not bit-identical anyway.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = _make_net(
                n_dense=Z.shape[1], n_stocks=N_STOCKS, embedding_dim=int(p["embedding_dim"]),
                hidden=tuple(p["hidden"]), dropout=float(p["dropout"]),
            ).to(dev)
            opt = torch.optim.Adam(
                model.parameters(), lr=float(p["learning_rate"]), weight_decay=float(p["weight_decay"])
            )

            self.history, best_mae, best_state, bad = [], np.inf, None, 0
            for epoch in range(int(p["max_epochs"])):
                model.train()
                perm = tr_t[torch.randperm(len(tr_t), generator=batches)].to(dev)
                total = torch.zeros((), device=dev)
                for start in range(0, len(perm), bs):
                    b = perm[start:start + bs]
                    opt.zero_grad(set_to_none=True)
                    loss = (model(Xt[b], St[b]) - yt[b]).abs().mean()
                    loss.backward()
                    opt.step()
                    total += loss.detach() * len(b)
                train_mae = float(total.item() / len(perm))
                inner_mae = float(self._mae(model, Xt, St, yt, va_t))
                self.history.append({"epoch": epoch, "train_mae": train_mae, "inner_val_mae": inner_mae})
                if inner_mae < best_mae:
                    best_mae, bad = inner_mae, 0
                    best_state = copy.deepcopy(model.state_dict())
                    self.best_epoch = epoch
                else:
                    bad += 1
                    if bad >= int(p["patience"]):
                        break
            model.load_state_dict(best_state)

        model.eval()
        self.model = model
        return self

    @staticmethod
    def _forward(model, Xt, St, idx, chunk: int = 1 << 16):
        import torch

        out = torch.empty(len(idx), dtype=torch.float32, device=Xt.device)
        with torch.no_grad():
            for start in range(0, len(idx), chunk):
                b = idx[start:start + chunk]
                out[start:start + chunk] = model(Xt[b], St[b])
        return out

    def _mae(self, model, Xt, St, yt, idx) -> float:
        model.eval()
        pred = self._forward(model, Xt, St, idx)
        return float((pred - yt[idx]).abs().mean().item())

    def predict(self, df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        import torch

        dev = torch.device(self.fitted_device)
        Z = self._design(X, fit=False)
        Xt = torch.from_numpy(Z).to(dev)
        St = torch.from_numpy(self._stock_index(df)).to(dev)
        idx = torch.arange(len(Z), device=dev)
        return self._forward(self.model, Xt, St, idx).cpu().numpy().astype(np.float64)

    def training_summary(self) -> dict:
        """What the report records per fold: where early stopping landed, and on what."""
        return {
            "device": self.fitted_device,
            "best_epoch": int(self.best_epoch),
            "epochs_run": len(self.history),
            "history": list(self.history),
            "inner_train_dates": [int(self.inner_train_dates.min()), int(self.inner_train_dates.max()),
                                  int(len(self.inner_train_dates))],
            "inner_val_dates": [int(self.inner_val_dates.min()), int(self.inner_val_dates.max()),
                                int(len(self.inner_val_dates))],
        }
