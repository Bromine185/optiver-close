"""The Colab notebooks: guards against what a Colab auto-save can quietly revert.

A notebook saved from the Colab UI is an older copy of itself with outputs
attached. Twice now that has put the textual manifest diff back in place of
the structural check (323ac45 -> 41d91bc; 8e7c7f4 -> 1b0a7fb). This test
makes the third time a red suite instead of a refused rebuild on Colab.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from optiver.config import REPO

NOTEBOOKS = sorted((REPO / "notebooks").glob("colab_phase*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_manifest_cell_uses_the_structural_check(path: Path):
    nb = json.loads(path.read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    manifest_cells = [s for s in code if "manifest" in s and ("MANIFEST MISMATCH" in s or "M.check(" in s)]
    assert len(manifest_cells) == 1, f"{path.name}: expected exactly one manifest-check cell"
    cell = manifest_cells[0]
    assert "from optiver import manifest" in cell and "M.check(" in cell, (
        f"{path.name}: the manifest cell is the textual diff again — a Colab save reverted it; "
        f"re-apply the optiver.manifest cell")
    assert "NOISE = (" not in cell


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_never_installs_torch(path: Path):
    nb = json.loads(path.read_text())
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        for line in src.splitlines():
            # an actual install invocation (`!pip install torch`, `["pip", "install", "torch..."]`),
            # not prose that mentions one.
            if re.search(r'(pip3?\s+install|"pip",\s*"install")[^#]*\btorch', line):
                raise AssertionError(f"{path.name}: {line.strip()!r} would replace Colab's CUDA torch")


def test_three_phases_are_present():
    assert [p.stem for p in NOTEBOOKS] == ["colab_phase1", "colab_phase2", "colab_phase3"]
