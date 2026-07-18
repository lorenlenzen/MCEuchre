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
  game.py          # state machine, legal moves, scoring ✅ tested (fast clone)
  infoset.py       # infoset keys + observation tensors ✅ tested
rebel/
  mccfr.py         # external-sampling MCCFR (tabular ground truth) ✅ runs
  networks.py      # PolicyValueNet, PBSValueNet (PyTorch) ✅ tested
  evaluate.py      # agents + head-to-head harness ✅ tested
  solver.py        # double-dummy alpha-beta solver ✅ tested vs brute force
  pimc.py          # PIMC search agent ✅ tested
  public_belief_state.py  # void-aware determinization ✅ tested
  subgame.py       # depth-limited CFR subgame solver ✅ tested vs double-dummy
  train_rebel.py   # the ReBeL self-play loop ✅ runs & learns
```

## Roadmap

### Milestone 0 — Foundation ✅
Correct engine, encodings, tabular MCCFR, evaluation harness, PyTorch nets.
MCCFR gives us a learner that improves on the *real* game and a way to
measure strength against baselines.

### Milestone 0.5 — PIMC search ✅
`solver.py` is an exact double-dummy solver (alpha-beta + transposition table +
double-dummy move reduction), verified against brute-force minimax. `pimc.py`
samples worlds from the acting player's belief, solves each, and averages —
genuine decision-time reasoning about hidden cards, and a strong benchmark.

### Milestone 1 — Belief state & subgame solver ✅
1. **Belief / determinization** (`public_belief_state.py`): samples full deals
   consistent with the public information, respecting hand sizes, played cards,
   void inferences, and the up-card's (sometimes uncertain) location. Fixing
   the acting player's hand guarantees their real infoset is covered.
2. **Depth-limited subgame** (`subgame.py`): treats "nature picks a world" as a
   chance root and runs vanilla CFR over the resulting game, sharing regret
   across worlds via infoset keys. A depth limit cuts the tree; leaf values come
   from the value function (network) — or, with no limit, it solves to terminal
   (exact given the belief), verified to match double-dummy on single worlds.

### Milestone 2 — The ReBeL loop ✅
`train_rebel.py` implements the self-play cycle:
1. Play hands; at each decision run the depth-limited CFR subgame solver, with
   the current network valuing the leaves.
2. The solved root strategy is the policy played (sampled); it and the solved
   root value become training targets.
3. Train one `PolicyValueNet` — policy head via cross-entropy to the CFR
   strategies, value head via MSE to the CFR root values.
4. The improving value head sharpens the leaf estimates that feed the next
   round of search — ReBeL's bootstrap.
`ReBeLNetAgent` then plays from the trained policy head at inference speed, with
`CFRSearchAgent` available to layer search back on top.

> **Performance caveat.** The engine is pure Python, so a single self-play hand
> (a CFR solve per decision, network-valued leaves) takes ~20s at tiny
> settings. The loop is built to *run and learn* correctly; reaching expert
> strength needs far more self-play, which in turn needs a faster engine
> (vectorized/batched inference, or the hot paths in C). That optimization is
> the main lever remaining and is called out in Milestone 4.

### Milestone 3 — Team-game correctness & strength
* Team subtleties: partners share reward but not information. Evaluate whether
  independent-per-player CFR suffices or whether a joint/correlated policy
  (TMECor-style) is needed for the calling and signaling conventions.
* Calling vs. play: consider separate network heads; the go-alone decision is
  high-variance and high-value.
* Keep policies stochastic (mixed strategies are optimal here).

### Milestone 4 — Performance, evaluation & tuning
* **Speed (the current bottleneck):** batch network leaf-evaluations across a
  CFR sweep; cache/incrementalize `infoset_key` and `observation_tensor`; move
  the engine hot paths (`apply`, `legal_actions`, trick resolution) to a
  vectorized or compiled representation. This is what unlocks enough self-play
  to matter.
* Elo across a pool (random, rule-based, PIMC, MCCFR, ReBeL checkpoints).
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
