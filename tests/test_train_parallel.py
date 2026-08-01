"""Tests for scripts/train_parallel.py's resumable-eval-log loading.

Regression coverage for a real crash: pointing --out at a path that already
had a log.json from train_scale.py (a different schema -- "gen"/"hands"
rather than "elapsed_s"/"samples") raised KeyError on log[-1]["samples"]
before any training happened. _load_resumable_log should recognize the
schema mismatch and start fresh instead of trusting the file on disk.
"""

import importlib.util
import json
import os
import sys

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "train_parallel.py")


@pytest.fixture(scope="module")
def tp():
    """Import train_parallel.py by path, matching test_train_pattern.py's
    pattern for scripts/ modules that aren't part of an installed package."""
    spec = importlib.util.spec_from_file_location("train_parallel", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_missing_log_starts_fresh(tp, tmp_path):
    log, elapsed, samples = tp._load_resumable_log(str(tmp_path / "nope.log.json"))
    assert log == [] and elapsed == 0 and samples == 0


def test_empty_log_starts_fresh(tp, tmp_path):
    p = tmp_path / "empty.log.json"
    p.write_text("[]")
    log, elapsed, samples = tp._load_resumable_log(str(p))
    assert log == [] and elapsed == 0 and samples == 0


def test_corrupt_json_starts_fresh(tp, tmp_path):
    p = tmp_path / "bad.log.json"
    p.write_text("{not json")
    log, elapsed, samples = tp._load_resumable_log(str(p))
    assert log == [] and elapsed == 0 and samples == 0


def test_train_scale_schema_starts_fresh_instead_of_crashing(tp, tmp_path, capsys):
    """The exact regression: a train_scale.py-produced log at this --out
    path used to raise KeyError here."""
    p = tmp_path / "reused.log.json"
    p.write_text(json.dumps([{
        "gen": 1, "hands": 25, "elapsed_s": 42, "buffer": 100,
        "vs_random": 0.1, "win_random": 0.5, "vs_rule": -0.1, "win_rule": 0.4,
        "policy_loss": 0.5, "value_loss": 0.1, "top_clusters": [],
    }]))
    log, elapsed, samples = tp._load_resumable_log(str(p))
    assert log == [] and elapsed == 0 and samples == 0
    assert "doesn't look like a train_parallel.py log" in capsys.readouterr().out


def test_own_schema_resumes_with_offsets(tp, tmp_path):
    p = tmp_path / "own.log.json"
    entries = [
        {"elapsed_s": 60, "samples": 500, "hands_est": 38, "buffer": 200,
         "vs_random": 0.2, "win_random": 0.6, "vs_rule": 0.0, "win_rule": 0.5,
         "top_clusters": []},
        {"elapsed_s": 120, "samples": 950, "hands_est": 73, "buffer": 300,
         "vs_random": 0.25, "win_random": 0.62, "vs_rule": 0.02, "win_rule": 0.51,
         "top_clusters": []},
    ]
    p.write_text(json.dumps(entries))
    log, elapsed, samples = tp._load_resumable_log(str(p))
    assert log == entries
    assert elapsed == 120 and samples == 950
