"""Resumable eval logs, shared by the training scripts.

Each training script keeps a JSON list of periodic evaluation entries so the
strength trend can be plotted across a whole training history rather than
one process's lifetime. Two hazards this module exists to handle, both of
which have actually bitten:

1. **Cross-script collision.** train_scale.py and train_parallel.py record
   genuinely different things -- generations/hands against wall-clock/sample
   counts -- so their entries are not one series and must not be spliced
   into one. They used to share a single `<out>.log.json` filename, so
   pointing both at the same --out crashed one of them (KeyError reading a
   key the other script never writes). Each script now writes its own
   `<out>.<name>.log.json`, which removes the collision rather than
   detecting it; `required_keys` remains as a second line of defence for
   legacy files and older formats.

2. **Silent history loss.** Starting a fresh list and dumping it over the
   old file discards everything from prior runs on the same --out --
   including a long overnight run's entire trend. Resuming means *appending*
   to what is already there, which is why the loaders below return the
   existing entries for the caller to extend rather than a bare file handle.

`legacy_path` migrates the pre-split `<out>.log.json` files: read once from
the old name when the new one does not exist yet, then write to the new name
from then on, so existing histories are carried forward instead of orphaned.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence


def log_path_for(out: str, name: str) -> str:
    """`<out>.<name>.log.json` -- the per-script eval-log path."""
    return f"{out}.{name}.log.json"


def legacy_log_path(out: str) -> str:
    """The pre-split shared path, read for migration but never written."""
    return f"{out}.log.json"


def load_resumable_log(path: str, required_keys: Sequence[str],
                       legacy_path: Optional[str] = None,
                       script: str = "this script") -> List[Dict[str, Any]]:
    """Existing entries at `path` to append to, or [] to start fresh.

    Returns [] -- starting a new log rather than resuming -- when the file
    is missing, empty, unparseable, or its last entry lacks any of
    `required_keys` (meaning it was written by a different script or an
    incompatible older version). The mismatch case warns rather than
    raising: a stale log should never take down a training run, but it also
    must not be silently trusted and then crash on a missing key mid-run.

    `legacy_path` is consulted only when `path` itself does not exist, and
    only its *contents* are adopted -- the caller still writes to `path`.
    """
    src = path
    if not os.path.exists(src):
        if legacy_path and os.path.exists(legacy_path):
            src = legacy_path
        else:
            return []
    try:
        existing = json.load(open(src))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(existing, list) or not existing:
        return []
    last = existing[-1]
    if not isinstance(last, dict) or not all(k in last for k in required_keys):
        print(f"warning: {src} exists but doesn't look like a {script} log "
              f"(last entry keys: "
              f"{sorted(last) if isinstance(last, dict) else type(last).__name__})"
              f" -- starting a fresh log instead of resuming it. Move or "
              f"rename the old file if you want to keep it.", flush=True)
        return []
    origin = f" (migrating from {src})" if src != path else ""
    print(f"resuming eval log from {src} ({len(existing)} entries)"
          f"{origin}", flush=True)
    return existing
