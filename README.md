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

This repo currently contains **Milestone 0: the foundation** — a correct,
fully-tested game engine plus the learning and search building blocks ReBeL
needs.

| Piece | What it is | State |
|---|---|---|
| `euchre/` | Rules engine: cards, bowers, bidding, play, scoring | ✅ tested (24-card deck, left-bower-as-trump, lone marches, stick-the-dealer) |
| `euchre/infoset.py` | Information-set keys + fixed-length observation tensors | ✅ tested |
| `rebel/mccfr.py` | External-sampling MCCFR — a tabular learner on the *real* game | ✅ runs |
| `rebel/public_belief_state.py` | Consistent determinization / belief sampling (void-aware) | ✅ tested |
| `rebel/networks.py` | PyTorch policy/value and PBS-value networks | ✅ forward-tested |
| `rebel/evaluate.py` | Reference agents + head-to-head evaluation harness | ✅ tested |

Next milestones (belief-state subgame solver → the ReBeL self-play loop) are
laid out in the design doc.

## Quick start

```bash
pip install -r requirements.txt
pytest -q                 # run the test suite (32 tests)
python scripts/demo.py    # train MCCFR briefly and measure strength
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
