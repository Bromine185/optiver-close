#!/usr/bin/env python
"""Reproduction report: the rebuilt manifest against the committed one.

    python scripts/check_manifest.py            # data/fixtures/manifest.json vs git HEAD
    python scripts/check_manifest.py --ref v1   # against another commit

Exit 0 when every recorded statement about the data still holds; exit 1 with
the list of changed values otherwise. `optiver.manifest` says what is compared
and why floats get a tolerance. The notebooks call the same function in-kernel
so a mismatch halts the notebook rather than scrolling past.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from optiver import config as C, manifest as M  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="HEAD", help="git ref holding the committed manifest")
    ap.add_argument("--rebuilt", type=Path, default=C.MANIFEST, help="the manifest to check")
    args = ap.parse_args()
    report = M.check(REPO, ref=args.ref, rebuilt=args.rebuilt)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
