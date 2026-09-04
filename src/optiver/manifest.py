"""The reproduction check: a rebuilt manifest against the committed one.

`scripts/build_fixture.py` records what it built in `data/fixtures/manifest.json`
— row counts, dtypes, null tallies, coverage, the float32 gate's verdict, the
target's moments. That file is committed, so a rebuild on any machine can be
checked against it, and the check is the reproduction report: if every
recorded statement about the data still holds, the fixture was reproduced.

Why the check is structural and not a text diff. Three classes of difference
between two honest builds are expected and mean nothing:

* when it was built and by which library versions (`built_at_utc`,
  `build_seconds`, `versions`, `mtime_utc`);
* how zstd packed the parquet — `sha256` and `bytes` move with pyarrow's
  version, not with the data;
* **last-ulp floating-point noise** in the summary statistics, because a
  different numpy or CPU reduces 5.2 M-row moments in a different order. The
  first Colab rebuild disagreed with the Mac's `excess_kurtosis` at 2e-15
  relative, and a line-based diff read that as content drift.

So integers, strings and booleans must match **exactly** — those are the
counts, the dtypes, the gate — and floats must match to `FLOAT_REL_TOL`,
six orders of magnitude looser than reduction-order noise and six orders
tighter than any real change to the data. One mismatch means the rebuild did
not reproduce the fixture and nothing downstream of it should be trusted.

This module exists because the check once lived only in a notebook cell, and
a notebook saved from the Colab UI reverted it to the text diff without
anyone noticing. Three notebooks now call one function, and the function has
tests.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import MANIFEST, REPO

#: Not content: when it was built, by what, and how the bytes were packed.
#: Everything else in the manifest is a statement about the data.
NOISE_KEYS = frozenset({
    "built_at_utc", "build_seconds", "mtime_utc", "smoke_rebuilt_at_utc",
    "versions", "sha256", "bytes",
})
FLOAT_REL_TOL = 1e-9
FLOAT_ABS_TOL = 1e-12


@dataclass
class Comparison:
    mismatches: list[str] = field(default_factory=list)
    #: Largest relative deviation among floats that PASSED — the noise actually observed.
    worst_float_rel: float = 0.0
    n_compared: int = 0

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def summary(self, limit: int = 20) -> str:
        if self.mismatches:
            shown = "\n".join(self.mismatches[:limit])
            more = f"\n... and {len(self.mismatches) - limit} more" if len(self.mismatches) > limit else ""
            return f"MANIFEST MISMATCH — {len(self.mismatches)} recorded value(s) changed:\n{shown}{more}"
        return (
            f"manifest reproduced: {self.n_compared} recorded values checked — every count, dtype, "
            f"null tally, coverage number and gate verdict exact; worst float deviation "
            f"{self.worst_float_rel:.2e} relative (reduction-order noise; tolerance {FLOAT_REL_TOL:.0e})"
        )


def compare(committed: dict, rebuilt: dict) -> Comparison:
    """Walk both manifests. Ints/strings/bools exact; floats to FLOAT_REL_TOL; NOISE_KEYS skipped."""
    out = Comparison()

    def walk(a, b, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k in NOISE_KEYS:
                    continue
                if k not in a or k not in b:
                    out.mismatches.append(f"{path}.{k}: present in only one manifest")
                else:
                    walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.mismatches.append(f"{path}: length {len(a)} vs {len(b)}")
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, f"{path}[{i}]")
        elif _is_float(a) or _is_float(b):
            out.n_compared += 1
            if not _is_number(a) or not _is_number(b):
                out.mismatches.append(f"{path}: {a!r} vs {b!r}")
            elif math.isnan(a) and math.isnan(b):
                return
            elif not math.isclose(a, b, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL):
                out.mismatches.append(f"{path}: {a!r} vs {b!r}")
            elif a != b:
                denom = max(abs(a), abs(b))
                out.worst_float_rel = max(out.worst_float_rel, abs(a - b) / denom if denom else 0.0)
        else:  # ints, strings, bools, None: exact. bool is an int in Python; `!=` keeps them apart from 0/1? No — 1 == True. Compare types too.
            out.n_compared += 1
            if type(a) is not type(b) or a != b:
                out.mismatches.append(f"{path}: {a!r} vs {b!r}")

    walk(committed, rebuilt, "")
    return out


def _is_float(x) -> bool:
    return isinstance(x, float)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def committed_manifest(repo: Path = REPO, ref: str = "HEAD", rel: str = "data/fixtures/manifest.json") -> dict:
    """The manifest as committed at `ref`, via git — never the working copy."""
    proc = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{rel}"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git show {ref}:{rel} failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def check(repo: Path = REPO, *, ref: str = "HEAD", rebuilt: Path = MANIFEST) -> Comparison:
    """Compare the manifest on disk (the rebuild) against the one committed at `ref`."""
    return compare(committed_manifest(repo, ref), json.loads(Path(rebuilt).read_text()))
