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

* **Separate calling / play network heads** (`PolicyValueNet`). The flat action
  space splits into card plays `[0,24)` and bidding/discard `[24,59)`, each with
  its own output head so the two very different decision types specialise.
  Going alone is first-class in the action space (`OrderUp(alone)`,
  `Call(alone)`).
* **Stochastic play** (`ReBeLNetAgent` temperature). Optimal play here is a
  mixed strategy; temperature keeps the policy from collapsing to a
  deterministic, exploitable one.

Team-correlation note: partners share reward (utility is the team point
differential, verified by test) but not information, so they can only coordinate
hidden-information conventions through public actions. The main pipeline uses
**independent per-player CFR**, which finds an equilibrium but does not by
itself develop optimal *signalling* conventions.

**Correlated team play (TMECor)** — the fix for that — is implemented as a
validated reference in `rebel/tmecor.py`. A team playing with correlation is a
single *coordinator* whose pure strategies are the team's joint deterministic
plans (one action per team infoset); TMECor is a Nash equilibrium of the
zero-sum game coordinator-A vs coordinator-B, which we solve with regret
matching (no LP dependency). It is validated on a canonical coordination game
where correlation strictly helps — **TMECor reaches the correlated optimum 0
while independent CFR is trapped at −1** (analytic best-independent / TME is
−0.5, so correlation is worth +0.5 over the best product strategy) — and it
respects the provable inequality **TMECor ≥ Nash** on real Euchre endgames.

Scope, honestly (as with `range_cfr`): enumerating joint pure strategies is
exponential in the number of team infosets, so this is exact only for small
subgames (endgames / toys) and is wired into nothing. Scaling it to full games
needs the compact team representations from the literature (column generation
over best-response oracles, or TB-DAG CFR) — a genuine research step. The
reference here makes the technique concrete and quantifies the correlation gap.

### Milestone 3.5 — Suit-agnostic relational network ✅

**Diagnosis.** Live training exposed a specific, confirmed failure: round-2
bidding (which suit to call) would pick a suit the hand held *zero* cards of
("Call Clubs" on a 3-hearts hand), and got *worse* the longer self-play ran.
Isolated the cause via direct comparison rather than guesswork:

* An exact double-dummy `rollout_value` oracle and the actual `SubgameSolver`
  CFR search (both net-leaf and exact-grounded-leaf) all agreed the correct
  call was Hearts, the search producing it with **98% confidence**. So the
  regret minimization itself was not broken.
* The trained net's **value head**, from the same shared trunk, also ranked
  the suits correctly. Only the **policy head's** suit choice was inverted.
* Measuring the policy head's greedy call across 200 real round-2 states
  showed a **hand-independent skew** (48% of all calls were Clubs,
  regardless of hand) — not a broken head (its weight rows were genuinely
  differentiated, cosine similarity 0.28–0.47, not collapsed), but an
  **anchor** riding on top of real hand-dependent variation. The anchor was
  demonstrably arbitrary and mobile: a checkpoint from earlier the same run
  favored **Hearts** (mean call-logit −1.87) while the later one favored
  **Clubs** (−1.88) — training had made it *worse*, not better.
* Root cause: `euchre/actions.py`'s Call-suit outputs sit at fixed absolute
  indices (Call Clubs = 48 … Call Spades = 51) with **no shared parameters**
  between suits, and only receive gradient from round-2 samples — roughly
  **0.03% of the training signal** (round 2 itself is ~0.67% of hands, and
  each hand yields ~13 total samples across all decision types). Four
  independent sub-problems, each starved of data, is exactly the setup for
  an unconverged, drifting per-suit offset.

**Fix: make the net suit-agnostic by construction**, not by more data.
Two coordinated rewrites:

* **`euchre/infoset.py` — trump-relative encoding.** `observation_tensor`
  now encodes everything by *role* relative to a reference suit R (trump
  once set, else the up-card's suit during bidding) via `same_color_suit`:
  `role0` = R, `role1` = its same-color partner ("next"), `role2`/`role3` =
  the two off-color suits ("green") — **both greens carry the identical role
  tag**. Layout: a suit-independent global block, 4 per-suit blocks (kept at
  *absolute* slots so a shared tower still maps 1:1 to each output index, but
  their *contents* — role tag, as-if-trump holdings, effective plain
  holdings, per-relative-seat played counts — are role-relative), and 24
  per-card feature blocks. `OBS_SIZE` grew from 249 to **394**.
  `infoset_key` (the tabular CFR regret-table key) is untouched — a single
  `SubgameSolver` solve is always a concrete, fixed-suit situation, so
  cross-suit generalization is the net's job alone.
* **`rebel/networks.py` — relational `PolicyValueNet`.** A shared
  suit-encoder MLP runs over each of the 4 role-relative suit blocks
  (identical weights ⇒ the two greens are handled by *the same parameters*,
  not just similar ones); a symmetric mean-pool over the 4 embeddings feeds
  a context trunk (permutation-invariant, preserving green symmetry). A
  **shared make-trump scorer** — `(suit embedding, context) → (score,
  score-alone)` — produces *every* OrderUp and Call logit: round 1's
  OrderUp routes through the reference-role suit, round 2's Call through
  whichever role each suit carries. This unification is a deliberate
  sample-efficiency fix, not just a symmetry one — round 1's abundant
  order-up gradient now trains the exact weights that score round-2 suit
  calls, directly attacking the data sparsity that let the anchor form.
  Shared per-card scorers likewise produce every Play and Discard logit.
  External interface unchanged: `forward(obs) -> (logits[59], value)`, so
  `SubgameSolver`, `ReBeLTrainer`, and every agent/script are untouched.

**Verification** (`tests/test_suit_symmetry.py`, 4 tests, all passing):

| check | result |
|---|---|
| Encoding equivariant under the 8-element color-preserving suit-relabeling group | **exact 0.0** error |
| Network equivariant under the same group | **1.5e-08** (float32 precision) |
| Green-swap: the two green suits' Call logits swap exactly | verified |
| Symmetry survives real gradient steps (30 training steps) | **1.2e-07** error |

The Clubs/Hearts anchor is now **mathematically impossible**, not merely
trained away — green1 and green2 are provably the same function. Full suite:
75/75 passing. Throughput: ~17.4k `observation_tensor` calls/sec, ~42k
states/sec batched net forward (measured on this machine) — heavier than the
prior absolute encoding but not the search hot-path's bottleneck.

**Consequence:** this is an architecture + encoding change, so existing
checkpoints (`rebel_hq*.pt`) are incompatible and training must restart from
scratch. See [`checkpoints/README.md`](../checkpoints/README.md) for the
warm-start approach used to reduce that cost.

### Milestone 3.6 — Match equity (score-conditioned value + CFR search) ✅

Closed a gap flagged since early this session (see "Known gaps," below,
prior to this milestone): every hand was trained and evaluated in total
isolation from the race-to-10 match score. Real Euchre strategy is
score-dependent — analogous to backgammon match equity or poker tournament
ICM — e.g. a team one point from winning should strongly prefer a safe
sure-thing over a high-variance loner attempt with equal or better *raw*
expected points.

**Key insight that shrank the scope.** Euchre's scoring has a structural
property: *exactly one team scores per non-misdeal hand*. For any monotonic
equity table (non-decreasing in own score, non-increasing in the
opponent's — true of any correctly-built table), this guarantees the
*ordinal* ranking of the 7 discrete hand outcomes under equity always matches
their raw-point ranking, for a fixed starting score — provable by
transitivity through the table, independent of its specific shape. That
equivalence holds for *pure/deterministic* comparisons (`solve_value`'s
double-dummy minimax, `rollout_value`'s `DEALER_DISCARD` enumeration — full
information, no hidden-info mixing), so **their internal search stays
completely raw-point-based, transposition table untouched** — this session's
exact-solve performance work is fully preserved. It does *not* hold for
comparisons of *expectations over mixed strategies* — exactly what CFR's
regret matching does — because a saturating equity function can flip which
option is better even when two options tie on raw expected points (the whole
point of match equity: risk-shaping). So the real work was narrower than
"rewrite every solver": convert to equity units only at the two boundaries
where search compares expectations under uncertainty.

**Components:**
* **`rebel/match_equity.py`** (new) — `fit_hand_outcome_distribution()`
  empirically measures the 7 discrete single-hand outcome frequencies via
  `RuleBasedAgent` self-play (dealer-alternating, hence team-symmetric in
  expectation — explicitly symmetrized post-fit since a *finite* empirical
  sample won't measure exact symmetry even though the underlying process
  has it, turning `win_prob(0,0)==0.5` into a provable invariant rather than
  an approximate one). `build_equity_table()` value-iterates
  `E(a,b)` = team0's match win probability to a fixed point over the
  resulting absorbing Markov chain (misdeal is a genuine self-loop, handled
  by iteration rather than analytic elimination). `MatchEquityModel` wraps
  the table with `.equity_delta()` (a hand outcome → team0-signed
  win-probability delta — zero-sum by construction, so it plugs directly
  into the existing `util = [diff if team_of(p)==0 else -diff ...]` pattern)
  and `.sample_score()` (draws a realistic starting score for self-play,
  weighted by exact forward-visitation mass, not uniformly over the grid).
* **`euchre/game.py`** — `EuchreState` gains `team0_score`/`team1_score`
  (default 0, so every existing single-hand caller is unaffected).
* **`euchre/infoset.py`** — `infoset_key` gets a team-relative `sc{mine},
  {theirs}` component (so MCCFR's regret table shares statistics between
  mirror-image situations — a team0 player up 7-4 and a team1 player up 7-4
  are strategically identical); `observation_tensor`'s global block gets two
  new score features. `OBS_SIZE` grew again, 394 → **396**. Zero changes
  needed in `rebel/networks.py` — score is global-block-only, and the
  suit-agnostic architecture's context trunk already takes `GLOBAL_DIM` from
  the imported constant.
* **`rebel/subgame.py` / `rebel/mccfr.py`** — an optional `equity_model`
  param (default `None`, byte-identical prior behavior); when set, CFR's
  *terminal* utility becomes an equity delta instead of raw points. Score is
  fixed for a whole hand, so it's read once (`SubgameSolver`, from the root)
  or straight off the terminal state itself (`MCCFRTrainer._utility`, no
  separate threading needed).
* **`rebel/pimc.py`** (`rollout_value`) — gains optional
  `team0_score`/`team1_score`/`equity_model` params; when given, converts
  its raw double-dummy result to equity units *once*, at the return
  boundary. The internal recursive solve (`_rollout_value_raw`) is untouched
  — same ordinal-equivalence argument.
* **`rebel/train_rebel.py`** — `ReBeLTrainer` gains `equity_model`;
  `_fresh_deal` samples a realistic score per hand when set, and passes it
  through to `SubgameSolver` and `_grounded_value_sample`'s `rollout_value`
  call. `scripts/train_parallel.py`/`train_scale.py`/`recalibrate_value.py`/
  `warm_start_value.py` all take `--match-equity-table` (on by default,
  loading the precomputed table) / `--no-match-equity` (opt out).

**Verification** (`tests/test_match_equity.py`, 16 tests, all passing):
table sanity (`win_prob(0,0)` exactly 0.5, boundaries, full-grid
monotonicity, complementarity to float precision); `infoset_key` correctly
distinguishes and team-relativizes score; `equity_model=None` produces
byte-identical output to the pre-feature behavior in `SubgameSolver`,
`MCCFRTrainer`, and `rollout_value`; and the core behavioral claim —
at a score one point from winning, a certain +1 strictly beats a 50/50
gamble between +2 and +0 under equity despite an exactly tied raw expected
value (1.0 both) — proven directly from monotonicity, not just observed.
End-to-end smoke-tested through the real multiprocess actor pipeline
(`train_parallel.py`, 1 actor, ~433 hands, no errors).

**Consequence:** another `OBS_SIZE` change, so checkpoints built on the
suit-agnostic-only encoding (`rebel_sa_warm.pt`, `rebel_sa.pt`) also need
regenerating/retraining under this milestone.

#### Original Milestone 3 notes
* Team subtleties: partners share reward but not information. Evaluate whether
  independent-per-player CFR suffices or whether a joint/correlated policy
  (TMECor-style) is needed for the calling and signaling conventions.
* Calling vs. play: consider separate network heads; the go-alone decision is
  high-variance and high-value.
* Keep policies stochastic (mixed strategies are optimal here).

### Milestone 4 — Performance, evaluation & tuning

* **Evaluation ladder ✅** (`rebel/ladder.py`, `scripts/ladder.py`). A
  round-robin tournament that fits **Bradley-Terry / Elo** ratings from the
  pairwise results (the MLE model behind Elo), with a leaderboard reporting
  Elo, win rate, average margin, and bootstrap confidence intervals. Seat bias
  is cancelled by alternating orientation *decoupled from* the dealer rotation
  (an early version accidentally kept one agent on the dealing team every hand —
  self-play flushed it out at +0.5 margin, now ~0). This makes "expert"
  measurable: strength is Elo separation from the field, and from a strong
  searcher (PIMC) in particular.
* **Exploitability (future):** exact best response is intractable here; a
  local-best-response (LBR) lower bound — a searcher that best-responds to a
  fixed agent using the double-dummy solver over the induced belief — is the
  natural next measurement.
* **Speed:** already largely addressed (persistent CFR tree, batched leaf
  evaluation, arithmetic `Card.id`); the remaining lever is compiling the
  engine hot paths (see the performance section above).
* Ablations to run on the ladder: depth limit, CFR iterations, self-play
  population.

### Known gaps (not yet scheduled)

* ~~**No score/match-equity awareness.**~~ **Resolved — Milestone 3.6.**
  `EuchreState`/observation now carry score, and `SubgameSolver`/
  `MCCFRTrainer`/`rollout_value` convert to match-equity units when an
  `equity_model` is supplied (opt-in via `--match-equity-table`, on by
  default in the training scripts). One corner deliberately left alone:
  `PIMCAgent` (a benchmark agent, not the trained pipeline) still defaults
  to raw points — `rollout_value`'s new params are there if that's wanted
  later, but its per-world cross-averaging would itself need equity
  conversion for full consistency, not just passing the params through.
  The hand-level ladder benchmarks (`rebel/evaluate.py`) remain
  intentionally score-blind (0-0 every hand) — isolated single-hand
  comparisons, not match play, so a neutral score is the right default
  there, not a gap.
* **Stick-the-dealer defaults off.** `ReBeLTrainer` now takes a
  `stick_the_dealer` param (plumbed through `--stick-the-dealer` in both
  `scripts/train_parallel.py` and `scripts/train_scale.py`, including the
  periodic eval calls), but it defaults to `False`, matching
  `EuchreState.new_hand()`'s own default. `rebel_hq.pt` was trained entirely
  with it off, so it has never actually seen a forced-call decision in round
  2 — a full pass-out just misdeals the hand (`reward=(0,0)`) instead. Any
  checkpoint trained so far is untested (and probably weak) in rule sets
  where stick-the-dealer is on; turning the flag on for future runs starts
  exercising it, but past training doesn't retroactively cover it.

## Design decisions & rationale

* **Per-hand episodes.** ReBeL treats one hand as the game; the value is the
  hand's point differential. Game-to-10 meta-strategy (e.g., risk adjustment
  when trailing) is a thin wrapper added later -- see "Known gaps" above.
* **Relative-seat encoding.** Observations are encoded from the acting
  player's seat so the net generalizes across positions.
* **Relative-suit (role) encoding.** Suits are encoded and scored by role
  relative to the trump/up-card suit (reference / next / green), with shared
  per-suit and per-card network towers, so the net generalizes across suits
  the same way it already does across seats — see Milestone 3.5. Absolute
  suit identity (`Suit.CLUBS` etc.) exists only in `EuchreState`/actions, never
  as a learned parameter.
* **Tabular MCCFR as ground truth.** It solves the *real* game (no
  abstraction), so it both benchmarks and supplies training targets — the net
  is judged by how well it reproduces it.
* **Standard variant first.** Makers may go alone; defenders may not.
  Stick-the-dealer is a flag. Other variants are additive.
```
```
