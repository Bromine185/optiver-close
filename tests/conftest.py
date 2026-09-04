"""Shared fixtures. Everything here runs off the COMMITTED smoke parquet.

No test may depend on `data/fixtures/train.parquet` — it is 130 MB, gitignored,
and absent on a fresh clone. Tests that genuinely need the full fixture are
marked `full_fixture` and skipped when it is missing, so the suite is green on a
machine that has never seen the raw Kaggle download.
"""

from __future__ import annotations

import os

import pytest

# torch and LightGBM each carry their own OpenMP runtime, and on macOS one
# process holding both segfaults or deadlocks depending on which initialised
# first (scripts/run_phase3.py has the measurements). The neural tests therefore
# run in a child interpreter that never loads lightgbm: test_neural.py is left
# out of this collection and executed by test_neural_isolated.py, which sets
# the variable below so the child collects it.
collect_ignore = [] if os.environ.get("OPTIVER_NEURAL_INPROC") else ["test_neural.py"]

from optiver import config as C
from optiver import data as D


@pytest.fixture(scope="session")
def smoke_cfg():
    return C.get_config("SMOKE")


@pytest.fixture(scope="session")
def smoke_df(smoke_cfg):
    return D.load(smoke_cfg)


@pytest.fixture(scope="session")
def manifest():
    return D.load_manifest()
