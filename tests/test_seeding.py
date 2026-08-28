"""Determinism. If these fail, no other result in the repo is reproducible."""

from __future__ import annotations

import numpy as np

from optiver import seeding


def test_same_label_gives_the_same_stream():
    assert np.array_equal(seeding.fork("x").random(64), seeding.fork("x").random(64))


def test_different_labels_give_different_streams():
    a = seeding.fork("train-shuffle").random(64)
    b = seeding.fork("sampler").random(64)
    assert not np.allclose(a, b)


def test_one_stream_is_unaffected_by_another_being_drawn_from():
    """The whole reason for forking by label rather than by counter.

    Draw a reference stream. Then create both generators, drain a different
    number of values from the first, and check the second is untouched. With a
    shared global generator this fails immediately, and every 'reproducible' run
    that adds a debug print silently changes.
    """
    reference = seeding.fork("b").random(32)
    rng_a, rng_b = seeding.fork("a"), seeding.fork("b")
    rng_a.random(1_000)
    assert np.array_equal(rng_b.random(32), reference)


def test_label_seed_is_pinned():
    """Pinned values, not just self-consistency.

    The committed smoke fixture's stock sample came from `fork("smoke-stocks")`.
    If ROOT_SEED or the hash construction changes, the fixture on disk no longer
    matches the code that claims to build it, and this test is the only thing
    that notices.
    """
    assert seeding.ROOT_SEED == 20260828
    assert seeding._label_seed("smoke-stocks") == 6047035715707184594
    assert seeding._label_seed("global") == 600227426126492982


def test_label_seed_is_not_process_salted():
    """BLAKE2b, not Python's hash(). Recomputed in a subprocess with a different
    PYTHONHASHSEED; a salted hash would return a different number."""
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from optiver.seeding import _label_seed; print(_label_seed('smoke-stocks'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(seeding.__file__).rsplit("/src/", 1)[0],
        env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
    )
    assert out.stdout.strip() == "6047035715707184594", out.stderr


def test_seed_everything_returns_a_stable_seed():
    assert seeding.seed_everything("phase1") == seeding.seed_everything("phase1")
