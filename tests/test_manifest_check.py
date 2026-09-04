"""The reproduction check: what it forgives, what it refuses.

Runs against the committed manifest (available on a fresh clone) and edited
copies of it. The one property that matters: reduction-order noise passes and
a changed count does not, with six orders of magnitude between them.
"""

from __future__ import annotations

import copy
import json

import pytest

from optiver import config as C, manifest as M


@pytest.fixture(scope="module")
def committed():
    return json.loads(C.MANIFEST.read_text())


def _edit(m: dict, path: list, value):
    m = copy.deepcopy(m)
    node = m
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return m


def test_a_manifest_reproduces_itself(committed):
    r = M.compare(committed, committed)
    assert r.ok and r.worst_float_rel == 0.0 and r.n_compared > 50


def test_last_ulp_float_noise_is_forgiven_and_reported(committed):
    """The Colab rebuild: excess kurtosis off by ~2e-15 relative."""
    k = committed["target"]["excess_kurtosis"]
    r = M.compare(committed, _edit(committed, ["target", "excess_kurtosis"], k * (1 + 2e-15)))
    assert r.ok
    assert 0 < r.worst_float_rel < 1e-14


def test_a_real_float_change_is_refused(committed):
    k = committed["target"]["excess_kurtosis"]
    r = M.compare(committed, _edit(committed, ["target", "excess_kurtosis"], k * (1 + 1e-6)))
    assert not r.ok and ".target.excess_kurtosis" in r.mismatches[0]


def test_a_changed_count_is_refused_exactly(committed):
    r = M.compare(committed, _edit(committed, ["rows"], committed["rows"] - 1))
    assert not r.ok and r.mismatches == [f".rows: {committed['rows']!r} vs {committed['rows'] - 1!r}"]


def test_a_changed_dtype_or_gate_verdict_is_refused(committed):
    r = M.compare(committed, _edit(committed, ["dtypes", "target"], "float64"))
    assert not r.ok and ".dtypes.target" in r.mismatches[0]
    bools = _bool_paths(committed)
    assert bools, "the manifest records its checks as booleans; none found"
    for path in bools:
        m = _edit(committed, path, not _get(committed, path))
        r = M.compare(committed, m)
        assert not r.ok and ".".join([""] + [str(k) for k in path]) in r.mismatches[0]


def test_a_missing_or_extra_key_is_refused(committed):
    m = copy.deepcopy(committed)
    del m["rows"]
    r = M.compare(committed, m)
    assert not r.ok and ".rows: present in only one manifest" in r.mismatches


def test_noise_keys_are_ignored(committed):
    m = copy.deepcopy(committed)
    m["built_at_utc"] = "1970-01-01T00:00:00+00:00"
    m["build_seconds"] = 1e9
    m["versions"] = {"pandas": "0.0", "numpy": "0.0", "python": "0.0"}
    touched = 0
    for block in m.values():
        if isinstance(block, dict):
            for k in ("sha256", "bytes", "mtime_utc"):
                if k in block:
                    block[k] = "changed" if k != "bytes" else 1
                    touched += 1
    assert touched >= 2, "expected sha256/bytes/mtime_utc somewhere in the manifest"
    assert M.compare(committed, m).ok


def test_bool_and_int_are_not_interchangeable():
    assert not M.compare({"a": True}, {"a": 1}).ok, "True == 1 in Python; the check must not agree"
    assert M.compare({"a": 1}, {"a": 1.0}).ok, "an int and an equal float are the same recorded value"
    assert M.compare({"a": 1.0}, {"a": 1.0 + 1e-12}).ok
    assert not M.compare({"a": "x"}, {"a": 1.0}).ok
    assert not M.compare({"a": None}, {"a": 0}).ok


def test_check_against_head_passes_on_a_clean_checkout():
    """The committed manifest against itself via git — the notebook's exact call."""
    r = M.check(C.REPO)
    assert r.ok, r.summary()
    assert "manifest reproduced" in r.summary()


def _bool_paths(m: dict, prefix=()) -> list[list]:
    out = []
    for k, v in m.items():
        if isinstance(v, bool):
            out.append(list(prefix) + [k])
        elif isinstance(v, dict):
            out.extend(_bool_paths(v, prefix + (k,)))
    return out


def _get(m, path):
    for k in path:
        m = m[k]
    return m
