"""Run tests/test_neural.py in a child interpreter — see conftest.py for why."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

# find_spec, not importorskip: importing torch HERE would load its OpenMP
# runtime into the parent pytest process, which is the exact thing this file
# exists to avoid (the LightGBM tests collected alongside would then crash).
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level=True)

HERE = Path(__file__).resolve().parent


def test_neural_suite_passes_in_its_own_process():
    env = dict(os.environ, OPTIVER_NEURAL_INPROC="1")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(HERE / "test_neural.py"), "-p", "no:cacheprovider"],
        cwd=HERE.parent, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"neural tests failed in the child process:\n{proc.stdout}\n{proc.stderr}"
    assert " passed" in proc.stdout, proc.stdout
