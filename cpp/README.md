# Building the C++ engine

## Prerequisite: MSVC toolchain

Visual Studio 2022 Build Tools with the C++ workload
(`Microsoft.VisualStudio.Workload.VCTools`), installed at
`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`.

**Known quirk on this machine**: `vswhere -requires
Microsoft.VisualStudio.Component.VC.Tools.x86.x64` returns nothing even
though the compiler is fully installed and works (`cl.exe` runs fine
directly) -- the component-registration metadata didn't get written
correctly, likely from an earlier interrupted install attempt. This means
`torch.utils.cpp_extension`'s own MSVC auto-detection (which relies on the
same vswhere query) **does not find the compiler**, and building fails with
`FileNotFoundError` / `subprocess.CalledProcessError: Command '['where',
'cl']'`. Re-running the VS Installer's `repair` may fix the registration
properly at some point, but the workaround below sidesteps it entirely and
is known-good.

**Workaround**: source `vcvars64.bat`'s environment directly (bypasses
vswhere-based detection) rather than relying on auto-detection. `.\cpp\
dev_env.ps1` does this -- dot-source it at the top of any PowerShell command
that builds or runs the extension:

```powershell
. .\cpp\dev_env.ps1
& $PY setup.py build_ext --inplace
```

## Environment gotchas specific to this setup (verify before assuming a build failure is a real bug)

- **PowerShell tool calls do not persist shell state between invocations.**
  Every single command needs `. .\cpp\dev_env.ps1` (or equivalent) at its
  start -- env vars set in a previous call are gone.
- **`cmd.exe /c "..."` invoked from the Bash tool does not work** -- Git
  Bash's MSYS layer mangles the `/c` flag (path-conversion quirk), silently
  causing `cmd.exe` to launch interactively instead of running the command,
  with no error, just empty output. Use PowerShell for anything that needs
  `cmd.exe`, or avoid `cmd.exe` in Bash entirely.
- **`ninja` needs `.venv\Scripts` on `PATH`** -- `pip install ninja` puts
  `ninja.exe` there, but `torch.utils.cpp_extension`'s ninja-availability
  check is a plain `subprocess.check_output(['ninja', '--version'])`, which
  fails silently if that directory isn't already on `PATH`. `dev_env.ps1`
  handles this.

## Verified working (this session)

A minimal pybind11 + LibTorch extension (`load_inline`, compile a function
and a tensor op, call both) built and ran successfully via `dev_env.ps1`'s
environment: MSVC compiled, ninja linked against `c10.lib`/`torch_cpu.lib`/
`torch.lib`/`torch_python.lib`, the resulting `.pyd` imported and ran
correctly in Python. This is the baseline the real build (`setup.py`) is
expected to reproduce at larger scale.

## What's ported

Engine (`engine.h/.cpp`), observation/infoset encoding (`infoset.h/.cpp`),
exact double-dummy solver + match equity (`solver.h/.cpp`,
`match_equity.h/.cpp`), belief/determinization (`belief.h/.cpp`), CFR
subgame search (`subgame.h/.cpp`), and `PolicyValueNet` as a real
`torch::nn::Module` (`network.h/.cpp`, bound via `torch::python::bind_module`
so it's a first-class `nn.Module` on the Python side, including real
LibTorch autograd -- `torch.optim` can train it directly, no separate
training loop needed). Each piece is differentially verified against the
unmodified, trusted pure-Python implementation in
`tests/test_cpp_equivalence.py` (bit-for-bit for the engine/observation/
solver/network-given-identical-weights; property-based, not RNG-identical,
for `sample_determinization`, since only Python stays the correctness oracle
and the C++ sampler only needs to be a correct/unbiased sampler -- see that
file's module docstring).

**Checkpoint interop** (`state_dict_()`/`load_state_dict_()` on
`cpp.PolicyValueNet`): `torch::python::bind_module` only gives the
*top-level* module full Python `nn.Module` machinery, so nested C++
submodules lack `_load_from_state_dict` and Python's generic (child-
recursing) `load_state_dict()`/`state_dict()` breaks on them. The two raw
methods walk `named_parameters()`/`named_buffers()` directly instead (pure
C++, unaffected by that gap) and exchange plain `name -> tensor` dicts that
`torch.save`/a plain Python `PolicyValueNet.load_state_dict()` both handle
normally -- verified round-trip exact (`tests/test_rebel_loop.py::
test_cpp_net_trains_via_rebel_trainer_and_checkpoint_interops`).

**Engine selector**: `ReBeLTrainer(engine="cpp")` (`rebel/train_rebel.py`)
routes the `self_play_hand()` hot loop -- state, observation, solver, CFR
search -- through the C++ path. `belief_model` depends on Python-only code
(`rebel/belief_model.py`) that wasn't ported, so `engine="cpp"` rejects it at
construction rather than silently falling back to Python for it.
`round2_seed_frac` and `value_ground_frac` both work fine with
`engine="cpp"`: `_biased_deal` never calls `rollout_value`, it only needed
engine-aware hand/up_card conversion (same pattern as `_cluster_key`); and
`_grounded_value_sample`'s cpp path uses `cpp_rollout_value`
(`rebel/train_rebel.py`) -- a Python-level mirror built on the already-bound
`mceuchre_cpp.solve_value`, not a new C++ port of `rebel/pimc.py`'s
`rollout_value` itself.
`net=` can be either a Python or C++ `PolicyValueNet`
independently of `engine=` (leaf evaluation calls `net(obs)` generically
either way); training the C++ net works via the ordinary `train_step`, no
special-casing needed. `scripts/train_parallel.py` and
`scripts/train_scale.py` both expose `--engine {python,cpp}`.

## Subgame boundary = phase boundary (ported)

`SubgameSolver::build` (`cpp/subgame.cpp`) mirrors `rebel/subgame.py`'s
`_build` exactly: a bidding-rooted solve (BID_ROUND_1/2, DEALER_DISCARD)
expands the whole auction regardless of `depth_limit_` -- it's short and
bounded on its own (<=4 round-1 + <=4 round-2 decisions before trump is set
or a real misdeal terminal) -- and the instant a child's phase becomes
`Play`, that child is an immediate leaf, never recursed into. This removed
`--bid-depth-limit` entirely (see `checkpoints/README.md`'s pre-redesign
notes for why it became pointless) and fixed the same structural
OrderUp/Call-vs-Pass asymmetry on the C++ path the Python fix targets:
previously OrderUp/Call reached real search-backed leaves within a few plies
while Pass got cut off deep in still-uncertain bidding, biased almost
entirely on an unverified net guess. Verified bit-for-bit identical to
Python (`tests/test_cpp_equivalence.py::test_subgame_solver_depth_limited_matches`,
a BID_ROUND_1-rooted depth-limited solve, `root_policy` matching to 1e-9).

## Measured payoff

Real wall-clock `self_play_hand()` time, single actor, production settings
(`--num-worlds 24 --cfr-iters 60 --depth-limit 6 --full-depth-cards 2
--stick-the-dealer`), 6 hands each, same seed:

| engine | mean s/hand |
|---|---|
| python | 49.97 |
| cpp    | 14.47 |

(Measured before the subgame-boundary fix above -- bidding-rooted solves are
now both correct *and* faster than they were here, so a fresh `self_play_hand()`
number would look better than this table on both engines; not re-measured
end-to-end yet.)

**~3.45x wall-clock speedup** -- real, but well below the ~10-30x figure
floated earlier in planning (explicitly caveated at the time as "a guess,
not a measurement"). The likely reason: the profiling that motivated this
port found `_cfr`'s tree recursion to be the hot path by call count (69M
calls), and that part is now fully C++ -- but leaf evaluation (the batched
`net(obs)` forward pass) was *already* running through optimized LibTorch/
ATen kernels even when called from pure-Python code, so porting the engine
around it doesn't speed that portion up, and it now makes up a
proportionally larger share of what's left. Confirming/improving this
further (e.g. quantifying the CFR-recursion-vs-net-forward split directly)
is future work, not done this session.
