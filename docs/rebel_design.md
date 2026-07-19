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

### Performance

Profiling drove the optimization, not guesswork. A CFR solve started out
~53% in `infoset_key` (with `Card.id` doing an O(n) `list.index` on every one
of millions of calls), and self-play re-derived the entire subgame tree —
`infoset_key`, `apply`, `clone`, `legal_actions`, *and every network leaf
call* — on **every CFR iteration**, even though the tree is identical across
iterations. The fixes:

1. **`Card.id` → arithmetic** (precomputed rank index): removes the single
   hottest call.
2. **Persistent CFR tree** (`SubgameSolver`): expand the subgame once into
   cached nodes that hold a shared `_Info`; iterations then do pure
   regret-matching arithmetic. This takes `infoset_key`/`apply`/`clone` (and
   leaf values) out of the per-iteration loop.
3. **De-NumPy'd regret matching**: infosets have ≤ ~6 actions, where NumPy's
   per-call overhead is pure cost, so the accumulators are plain Python lists.
4. **Batched leaf evaluation** (`batch_value_fn`): every depth-limit leaf in a
   solve is valued in **one** network forward pass instead of one-per-leaf.

Measured result: a full-depth 3-card CFR solve went **~16s → ~2.8s**, and a
self-play hand at a useful search setting (8 worlds, 20 CFR iters, depth 4)
went **~20s → ~0.45s (~44×)** — enough that thousands of self-play hands are a
matter of minutes rather than hours. The correctness suite is unchanged (the
optimized solver still matches double-dummy and brute force).

#### Cross-worlds vectorization: attempted, and why it doesn't pay off

The obvious next idea is to vectorize CFR across the belief worlds. It was
investigated and **does not work for a sampled belief** — a result worth
recording so it isn't re-attempted:

* Branching the tree *per world* diverges immediately: after one ply, 20
  sampled worlds already present 13 distinct legal-action sets; by depth 4 a
  depth-limited subgame has 911 nodes but 892 infosets, only **one** shared
  across nodes. Nothing to bundle.
* Re-organizing into a **public tree** (branch on the played card, pool all
  consistent deals per node — `rebel/range_cfr.py`, the DeepStack structure)
  is correct (M=1 reproduces double-dummy; root value matches the scalar
  solver) but still doesn't vectorize: a player's strategy is per information
  set (exact hand), so deals must be grouped by hand, and randomly sampled
  deals almost never share an exact opponent hand. Measured at depth 4 / 80
  deals: 885 nodes, 2927 groups, **average group size 1.2**. NumPy over
  size-1 arrays is slower than the lean scalar recursion.

The only formulation that truly vectorizes is the **dense enumerated range**:
carry a belief vector over *all* possible hands (not a sample) and
matrix-multiply strategies against it, with explicit **card-removal** for joint
consistency. Full joint enumeration is intractable mid-game (>300k consistent
deals), so it needs per-player marginal ranges plus a 4-player + kitty
card-removal correction — a genuine research effort (DeepStack did this for
heads-up poker; the 4-player partnership + kitty extension is novel). That is
the real path if this lever is pursued.

Lower-risk headroom that *does* apply: cache/incrementalize
`observation_tensor`, and — given tree search's tiny, unaligned per-op data —
move the engine hot paths to a compiled representation (bitboards in
C/Cython), a mechanical multiplier on top of the structural fixes above.

### Milestone 3 — Team-game correctness & strength ✅ (first cut)

Implemented:

* **Belief refinement conditioned on the bidding** (`rebel/belief_model.py`).
  The uniform determinizer ignores what the *bidding* revealed; a player who
  ordered up or called a suit almost certainly holds strong trump. A monotonic
  soft model of calling behaviour reweights each sampled deal by how well it
  explains the observed bids (order-ups, passes, and going alone), and the
  weights flow into the CFR chance-reach and PIMC averaging. Verified to
  sharpen the belief in the right direction: across sampled positions the
  maker's reconstructed hand carries **+0.76** more trump strength under the
  weighted belief than under the uniform one (higher in 100% of positions).
  Opt-in via a `belief_model` argument on `PIMCAgent`, `SubgameSolver`,
  `CFRSearchAgent`, and `ReBeLTrainer`.
* **Separate calling / play network heads** (`PolicyValueNet`). The flat action
  space splits into card plays `[0,24)` and bidding/discard `[24,59)`, each with
  its own output head so the two very different decision types specialise.
  Going alone is first-class in the action space (`OrderUp(alone)`,
  `Call(alone)`).
* **Stochastic play** (`ReBeLNetAgent` temperature). Optimal play here is a
  mixed strategy; temperature keeps the policy from collapsing to a
  deterministic, exploitable one.

Team-correlation note (deliberate scoping): partners share reward (utility is
the team point differential, verified by test) but not information, so they can
only coordinate hidden-information conventions through public actions. The
solver uses **independent per-player CFR**, which finds an equilibrium but does
not by itself develop optimal *signalling* conventions — that needs a
correlated formulation (a team maxmin / TMECor solve), which is a known,
larger research step and is not implemented here.

#### Original Milestone 3 notes
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
