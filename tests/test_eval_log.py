"""Tests for rebel/eval_log.py -- per-script, resumable eval logs.

Regression coverage for two real defects:

* train_scale.py started a fresh list and dumped it over `<out>.log.json`,
  so re-running it on the same --out silently discarded the prior run's
  entries -- and, since train_parallel.py wrote to that same filename, an
  overnight parallel run's entire history too.
* train_parallel.py already defended against reading a foreign schema (it
  crashed on one once), but only in that one direction.

Splitting the filenames removes the collision; `required_keys` stays as a
second line of defence for legacy files.
"""

import json

import pytest

from rebel.eval_log import (legacy_log_path, load_resumable_log, log_path_for)

PARALLEL_KEYS = ("elapsed_s", "samples")
SCALE_KEYS = ("gen", "hands", "elapsed_s")


def _write(path, entries):
    json.dump(entries, open(path, "w"))
    return str(path)


def _scale_entry(gen=1, hands=25, elapsed_s=10):
    return {"gen": gen, "hands": hands, "elapsed_s": elapsed_s,
            "vs_random": 0.5, "win_random": 0.6}


def _parallel_entry(elapsed_s=100, samples=1000):
    return {"elapsed_s": elapsed_s, "samples": samples, "hands_est": 76}


# --- path naming ----------------------------------------------------------

def test_scripts_get_distinct_paths():
    """The collision fix itself: two scripts pointed at one --out must not
    write to the same file."""
    assert log_path_for("ckpt/run", "scale") != log_path_for("ckpt/run", "parallel")
    assert log_path_for("ckpt/run", "scale") == "ckpt/run.scale.log.json"
    assert legacy_log_path("ckpt/run") == "ckpt/run.log.json"


# --- starting fresh -------------------------------------------------------

def test_missing_file_starts_fresh(tmp_path):
    assert load_resumable_log(str(tmp_path / "nope.json"), SCALE_KEYS) == []


@pytest.mark.parametrize("content", ["", "not json{", "[]", "{}", '"a string"'])
def test_empty_or_corrupt_starts_fresh(tmp_path, content):
    p = tmp_path / "log.json"
    p.write_text(content)
    assert load_resumable_log(str(p), SCALE_KEYS) == []


def test_foreign_schema_starts_fresh_with_warning(tmp_path, capsys):
    """A train_parallel log must not be resumed as a train_scale one --
    reading it would KeyError on a missing 'gen' mid-run."""
    p = _write(tmp_path / "log.json", [_parallel_entry()])
    assert load_resumable_log(p, SCALE_KEYS, script="train_scale.py") == []
    assert "doesn't look like a train_scale.py log" in capsys.readouterr().out


def test_foreign_schema_detected_in_both_directions(tmp_path, capsys):
    p = _write(tmp_path / "log.json", [_scale_entry()])
    assert load_resumable_log(p, PARALLEL_KEYS, script="train_parallel.py") == []
    assert "train_parallel.py" in capsys.readouterr().out


def test_non_dict_entries_start_fresh(tmp_path):
    p = _write(tmp_path / "log.json", [1, 2, 3])
    assert load_resumable_log(p, SCALE_KEYS) == []


# --- resuming -------------------------------------------------------------

def test_own_schema_resumes(tmp_path):
    entries = [_scale_entry(gen=1), _scale_entry(gen=2, hands=50)]
    p = _write(tmp_path / "log.scale.log.json", entries)
    assert load_resumable_log(p, SCALE_KEYS) == entries


def test_legacy_path_is_adopted_when_new_path_absent(tmp_path, capsys):
    """Pre-split `<out>.log.json` histories must carry forward, not be
    orphaned by the rename."""
    entries = [_scale_entry(gen=3, hands=75)]
    legacy = _write(tmp_path / "run.log.json", entries)
    new = str(tmp_path / "run.scale.log.json")
    assert load_resumable_log(new, SCALE_KEYS, legacy_path=legacy) == entries
    assert "migrating from" in capsys.readouterr().out


def test_new_path_wins_over_legacy(tmp_path):
    """Once the new file exists the legacy one is ignored, so a stale
    pre-split log can't resurrect itself over newer history."""
    legacy = _write(tmp_path / "run.log.json", [_scale_entry(gen=99)])
    new_entries = [_scale_entry(gen=1)]
    new = _write(tmp_path / "run.scale.log.json", new_entries)
    assert load_resumable_log(new, SCALE_KEYS, legacy_path=legacy) == new_entries


def test_legacy_with_foreign_schema_is_not_adopted(tmp_path, capsys):
    """The migration path must apply the same schema check -- a legacy file
    written by the OTHER script is exactly the collision case."""
    legacy = _write(tmp_path / "run.log.json", [_parallel_entry()])
    new = str(tmp_path / "run.scale.log.json")
    assert load_resumable_log(new, SCALE_KEYS, legacy_path=legacy,
                              script="train_scale.py") == []
    assert "doesn't look like" in capsys.readouterr().out


# --- the offset arithmetic each script layers on top ----------------------

def test_scale_offsets_continue_the_series(tmp_path):
    """train_scale.py's resume: gen/hands/elapsed_s continue from the last
    entry rather than restarting at zero."""
    prior = [_scale_entry(gen=4, hands=100, elapsed_s=60)]
    p = _write(tmp_path / "run.scale.log.json", prior)
    log = load_resumable_log(p, SCALE_KEYS)
    gen_offset = log[-1]["gen"]
    hands_offset = log[-1]["hands"]
    elapsed_offset = log[-1]["elapsed_s"]
    # First entry of the next run (its own g=1, 25 hands, 12s elapsed).
    assert (gen_offset + 1, hands_offset + 25, elapsed_offset + 12) == (5, 125, 72)
