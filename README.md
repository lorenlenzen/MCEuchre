# MCEuchre

A Euchre engine and a self-play AI built toward **expert-level play** using
**ReBeL** (Recursive Belief-based Learning) — the algorithm that combines
reinforcement learning with decision-time search for imperfect-information
games.

> **Why not just actor-critic?** Reactive policy methods plateau at moderate
> Euchre because expert play requires *reasoning about hidden cards at
> decision time* — inferring hands from the bidding and the cards played.
> ReBeL does CFR-based search over belief states with a neural value function,
> which is how bots reached superhuman play in poker. See
> [`docs/rebel_design.md`](docs/rebel_design.md) for the full rationale and
> roadmap.

## Status

A full, tested path from the rules engine to a running **ReBeL self-play
loop**, plus a **PIMC** search agent as a strong baseline.

| Piece | What it is | State |
|---|---|---|
| `euchre/` | Rules engine: cards, bowers, bidding, play, scoring | ✅ tested (left-bower-as-trump, lone marches, stick-the-dealer) |
| `euchre/infoset.py` | Information-set keys + observation tensors | ✅ tested |
| `rebel/solver.py` | Exact double-dummy solver (alpha-beta + move reduction) | ✅ verified vs brute force |
| `rebel/pimc.py` | PIMC search agent (sample worlds → solve → average) | ✅ tested |
| `rebel/public_belief_state.py` | Void-aware determinization / belief sampling | ✅ tested |
| `rebel/subgame.py` | Depth-limited CFR subgame solver (ReBeL's search core) | ✅ verified vs double-dummy |
| `rebel/train_rebel.py` | The ReBeL self-play loop (search → targets → train net) | ✅ runs & learns |
| `rebel/mccfr.py` | External-sampling MCCFR — tabular learner on the real game | ✅ runs |
| `rebel/networks.py` | PyTorch policy/value and PBS-value networks | ✅ tested |
| `rebel/evaluate.py` | Reference agents + head-to-head harness | ✅ tested |

The search/self-play hot paths are optimized (persistent CFR tree, batched leaf
evaluation, arithmetic `Card.id`, list-based regret matching): a self-play hand
at a useful search setting dropped from **~20s to ~0.45s (~44×)**, so thousands
of hands take minutes. The path to expert strength (more self-play, belief
refinement, team-play conventions) is in
[`docs/rebel_design.md`](docs/rebel_design.md).

## Quick start

```bash
pip install -r requirements.txt
pytest -q                 # fast suite (45 tests); add -m slow for search-heavy ones
python scripts/demo.py    # tour: baselines, solver, PIMC, CFR subgame, ReBeL loop
```

## How the ReBeL pieces fit

```
                 determinization (belief)         value/policy net
                        │                                │
   state ──► SubgameSolver: CFR over sampled worlds ─────┤ leaves valued by net
                        │                                │
                 root strategy + value ──► training targets ──► train net
                        │                                        (bootstraps leaves)
                   sample action ──► next state
```

## Layout

```
euchre/    # the game — no ML dependency, importable on its own
rebel/     # learning + search components (MCCFR, nets, belief, eval)
tests/     # engine, encoding, network, and belief consistency tests
docs/      # ReBeL design & roadmap
scripts/   # runnable demos
```

## Design highlights

- **Correctness first.** The engine is exhaustively tested, with special
  attention to the rule that trips up most implementations: the left bower
  (Jack of the same-color suit) counts as the second-highest trump and no
  longer follows its printed suit.
- **Relative-seat encoding.** Observations are built from the acting player's
  seat so the network sees the same strategic situation identically from any
  position.
- **Tabular MCCFR as ground truth.** It converges toward equilibrium on the
  real game (no abstraction) and supplies both a benchmark and warm-start
  targets for the ReBeL networks.
- **Void-aware belief sampling.** Determinizations respect hand sizes, played
  cards, and suits players have shown void of — the substrate for PIMC/ReBeL
  search.
```
