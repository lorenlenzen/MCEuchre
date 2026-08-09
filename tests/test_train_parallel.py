"""Tests for scripts/train_parallel.py's resumable-eval-log loading,
periodic evaluation, and actor shutdown.

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
from queue import Empty

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


# --- _evaluate: named opponents, extensible past random/rule ----------------

def test_evaluate_returns_random_and_rule_by_default(tp):
    from rebel.networks import PolicyValueNet
    net = PolicyValueNet()
    results = tp._evaluate(net, hands=6, seed=0)
    assert set(results) == {"random", "rule"}
    for diff, win in results.values():
        assert isinstance(diff, float)
        assert 0.0 <= win <= 1.0


def test_evaluate_includes_extra_opponents(tp):
    """--diagnostic-checkpoint's mechanism: an extra named opponent factory
    gets folded into the same result dict alongside random/rule, so the
    eval-loop and log-entry code doesn't need to special-case it."""
    from rebel.evaluate import RandomAgent
    from rebel.networks import PolicyValueNet
    net = PolicyValueNet()
    results = tp._evaluate(net, hands=6, seed=0,
                           extra_opponents={"diagnostic": RandomAgent})
    assert set(results) == {"random", "rule", "diagnostic"}


def test_evaluate_extra_opponents_can_override_random_or_rule(tp):
    """dict.update semantics: a caller-supplied "random" or "rule" key would
    replace the built-in one rather than erroring -- not the intended use,
    but worth pinning down since it's a natural consequence of the
    implementation and silently swapping the sanity-floor opponent would be
    an easy mistake to make unnoticed."""
    from rebel.evaluate import RandomAgent
    from rebel.networks import PolicyValueNet
    net = PolicyValueNet()
    results = tp._evaluate(net, hands=6, seed=0,
                           extra_opponents={"rule": RandomAgent})
    assert set(results) == {"random", "rule"}


# --- _shutdown_actors: robust to a second Ctrl-C arriving mid-cleanup ------

class _FakeActor:
    def __init__(self):
        self.alive = True
        self.terminated = False

    def join(self, timeout=None):
        pass  # never notices stop_flag in time -- simulates a stuck actor

    def is_alive(self):
        return self.alive and not self.terminated

    def terminate(self):
        self.terminated = True
        self.alive = False


class _FakeValue:
    def __init__(self, value=0):
        self.value = value


class _FakeQueue:
    def get(self, timeout=None):
        raise Empty


def test_shutdown_actors_terminates_everyone(tp):
    actors = [_FakeActor() for _ in range(6)]
    tp._shutdown_actors(actors, _FakeValue(), _FakeQueue())
    assert all(a.terminated for a in actors)
    assert _FakeValue().value == 0  # sanity: fixture itself unaffected


def test_shutdown_actors_terminates_everyone_despite_second_ctrl_c(tp):
    """Regression test for the actual reported bug: Ctrl-C during training
    wasn't stopping all actors. Root cause was the old shutdown code doing
    `for a in actors: a.join(timeout=3); if alive: terminate()` as ONE
    loop -- a second KeyboardInterrupt (an impatient response to how long
    that serial per-actor wait could take) propagated straight out,
    abandoning whichever actors the loop hadn't reached yet. This
    reproduces that interrupt deterministically (no real signals, no real
    processes) and asserts the fixed _shutdown_actors always terminates
    every actor regardless of when the interrupt lands."""
    actors = [_FakeActor() for _ in range(14)]
    real_join = _FakeActor.join
    calls = {"n": 0}

    def flaky_join(self, timeout=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_join(self, timeout=timeout)

    _FakeActor.join = flaky_join
    try:
        tp._shutdown_actors(actors, _FakeValue(), _FakeQueue())
    finally:
        _FakeActor.join = real_join

    assert all(a.terminated for a in actors), (
        f"only {sum(a.terminated for a in actors)}/{len(actors)} terminated "
        f"-- the second-Ctrl-C bug is back")
