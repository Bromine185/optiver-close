"""Seeded, forkable randomness.

Ported from the sibling `diff_model` project, itself ported from `3d-night`. The
point of forking by *label* rather than by counter: re-running one component
must not perturb another's draws. If the fixture subsample and a model's
initialisation share one global generator, adding a single extra draw in the
subsampler silently changes the model, and the run stops being reproducible in
the way that matters.

    rng = fork("smoke-stocks")     # same stream every time, regardless of
    rng = fork("ridge-init")       # what else ran first

The torch half of the sibling's version is deliberately absent: this project has
no torch dependency (Phase 1 is ridge and arithmetic). If a later phase adds a
neural model, port `torch_generator` across verbatim rather than reaching for
`torch.manual_seed` at the call site.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

#: Every stream in the project derives from this. Change it and every number moves.
ROOT_SEED = 20260828


def _label_seed(label: str, root: int = ROOT_SEED) -> int:
    """Derive a stable 63-bit seed from a text label.

    BLAKE2b rather than Python's `hash()` because the latter is salted per
    process (PYTHONHASHSEED) and would produce a different stream on every run —
    the exact failure this module exists to prevent.
    """
    h = hashlib.blake2b(f"{root}:{label}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") >> 1


def fork(label: str, root: int = ROOT_SEED) -> np.random.Generator:
    """An independent numpy Generator for this label."""
    return np.random.default_rng(_label_seed(label, root))


def seed_everything(label: str = "global", root: int = ROOT_SEED) -> int:
    """Pin every global RNG. Returns the seed used, so it can be logged.

    A backstop for third-party code that draws from the global stream (sklearn's
    `solver="sag"`, for instance). Project code should call `fork` and pass the
    generator explicitly; anything that reaches for `np.random.*` directly is a
    defect, not something this function excuses.
    """
    seed = _label_seed(label, root) % (2**31 - 1)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
