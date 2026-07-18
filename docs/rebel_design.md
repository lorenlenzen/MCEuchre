# ReBeL for Euchre — Design & Roadmap

This document explains how the pieces in this repo fit together and the plan
for reaching expert-level play with **ReBeL** (Recursive Belief-based
Learning, Brown et al. 2020), the algorithm that unifies reinforcement
learning and search for imperfect-information games.

## Why ReBeL (and why the engine comes first)

Euchre is an **imperfect-information, partially-observable, 2-v-2 team**
trick-taking game. Reactive policy methods (like the actor-critic that
plateaued) can't do the decision-time reasoning experts rely on — inferring
hidden cards from the bidding and the cards played. ReBeL addresses this by
doing **CFR-based search at decision time** over *public belief states*, with
a neural network supplying leaf values. Its theoretical grounding (converges
toward a Nash equilibrium in two-player zero-sum settings, strong empirical
play in multiplayer/team settings) is why it's the target here.

Everything downstream trusts the rules engine. A single wrong rule — most
often the **left bower counting as trump** — silently corrupts every value
target. So the engine is built and exhaustively tested first.

## Core concepts

| Concept | Meaning in Euchre | Where |
|---|---|---|
| **World / history** | A full deal + all actions so far | `EuchreState` |
| **Information set** | What one player can observe (own hand + public history) | `infoset_key` |
| **Public state** | What *everyone* observes (bids, cards played, trump, up-card) | derivable from `EuchreState` minus hidden hands |
| **Public belief state (PBS)** | Public state + a probability distribution over each player's hidden hand | `rebel/public_belief_state.py` (WIP) |
| **Value** | Team point differential for the hand | `_utility` in `mccfr.py` |

## Components in this repo

```
euchre/            # the game (no ML dependency)
  cards.py         # deck, bowers, trick resolution  ✅ tested
  actions.py       # action types + flat index space ✅ tested
  game.py          # state machine, legal moves, scoring ✅ tested
  infoset.py       # infoset keys + observation tensors ✅ tested
rebel/
  mccfr.py         # external-sampling MCCFR (tabular ground truth) ✅ runs
  networks.py      # PolicyValueNet, PBSValueNet (PyTorch) ✅ forward-tested
  evaluate.py      # agents + head-to-head harness ✅ tested
  public_belief_state.py  # PBS construction + features  ⏳ next
  cfr_subgame.py          # depth-limited CFR-D subgame solver ⏳ next
  train_rebel.py          # the ReBeL self-play loop        ⏳ next
```

## Roadmap

### Milestone 0 — Foundation ✅ (this commit)
Correct engine, encodings, tabular MCCFR, evaluation harness, PyTorch nets.
MCCFR gives us a learner that improves on the *real* game and a way to
measure strength against baselines.

### Milestone 1 — Belief state & subgame solver
1. **PBS construction**: given a public state, represent the belief as a
   distribution over deals consistent with the public information (start with
   uniform over legal completions; refine with a learned belief net).
2. **Depth-limited subgame**: build the tree of public states from a root PBS
   down to a depth limit; at leaves, call the value net instead of recursing.
3. **CFR-D / linear CFR** inside the subgame to compute an equilibrium of the
   depth-limited game.

### Milestone 2 — The ReBeL loop
Following the paper's algorithm:
1. Start at the initial PBS.
2. Build a depth-limited subgame; run `T` iterations of CFR using the value
   net at leaves.
3. Sample an iteration `t ∈ [1, T]`; set the root policy to iteration `t`'s
   average strategy.
4. Add `(PBS, computed values)` to the value-net training set; add
   `(infoset, strategy)` to the policy-net set.
5. Sample a leaf PBS according to the strategy and recurse (self-play descent).
6. Periodically retrain the nets on the accumulated data.

Bootstrapping: the tabular MCCFR strategies from Milestone 0 provide value
and policy targets to warm-start the nets before full ReBeL self-play.

### Milestone 3 — Team-game correctness & strength
* Team subtleties: partners share reward but not information. Evaluate whether
  independent-per-player CFR suffices or whether a joint/correlated policy
  (TMECor-style) is needed for the calling and signaling conventions.
* Calling vs. play: consider separate network heads; the go-alone decision is
  high-variance and high-value.
* Keep policies stochastic (mixed strategies are optimal here).

### Milestone 4 — Evaluation & tuning
* Elo across a pool (random, rule-based, MCCFR, ReBeL checkpoints).
* **Local best response / exploitability** to quantify how far from optimal.
* Ablations: depth limit, CFR iterations, belief-net quality, self-play
  population.

## Design decisions & rationale

* **Per-hand episodes.** ReBeL treats one hand as the game; the value is the
  hand's point differential. Game-to-10 meta-strategy (e.g., risk adjustment
  when trailing) is a thin wrapper added later.
* **Relative-seat encoding.** Observations are encoded from the acting
  player's seat so the net generalizes across positions.
* **Tabular MCCFR as ground truth.** It solves the *real* game (no
  abstraction), so it both benchmarks and supplies training targets — the net
  is judged by how well it reproduces it.
* **Standard variant first.** Makers may go alone; defenders may not.
  Stick-the-dealer is a flag. Other variants are additive.
```
```
