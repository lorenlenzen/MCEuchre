# ReBeL high-quality training checkpoint

## ⚠️ Every checkpoint here is stale (encoding changed twice)

Diagnosed this session: the trained policy head carried an arbitrary,
hand-independent per-suit anchor in round-2 bidding (it would "Call Clubs"
holding zero clubs; the anchor *drifted* — Hearts on one checkpoint, Clubs on
a later one from the same run) because `observation_tensor` encoded suits in
fixed absolute slots with no shared parameters, and round-2 samples are only
~0.03% of the training signal. The fix (see "Milestone 3.5" in
[`docs/rebel_design.md`](../docs/rebel_design.md)) re-encodes observations
relative to the trump/up-card suit and makes `PolicyValueNet` a relational
net with shared per-suit/per-card towers, so the two off-color suits are
provably handled by identical weights (verified to float precision, not
approximately). `OBS_SIZE` changed **249 → 394**.

Then Milestone 3.6 (match equity — see below) added score to the observation
too: `OBS_SIZE` changed **again, 394 → 396**. So `rebel_hq*.pt`
(pre-3.5) *and* `rebel_sa*.pt` (3.5 but pre-3.6) are **both incompatible**
with the current code — loading either fails on a shape mismatch. Train
fresh; don't `--resume` from either family.

## Speeding up a fresh run: value warm-start + match equity

Random-init leaf values are noise, and `SubgameSolver`'s depth-limited CFR
search reads leaf values from the net for everything beyond the search
depth — so the first self-play hands waste effort on CFR targets built from
garbage values. `scripts/warm_start_value.py` pre-grounds the value head (and
the shared trunk it sits on) against **exact** `rollout_value` double-dummy
outcomes before self-play starts, using the same safe pattern proven by
`scripts/recalibrate_value.py` this session (pure value MSE, zero policy
loss, train/val split, early stopping) — it cannot reproduce the *other*
warm-start attempt's failure (`warm_start_bidding.py` regressed the quiz
3/6→1/6 by imitating a heuristic *policy* with class imbalance and no
validation) because it never touches a policy target at all. Policy heads
stay at random init; self-play + CFR train those, same as before.

Match equity (Milestone 3.6) is **on by default** in both scripts below —
they load `rebel/match_equity_table.json`, building it first if it doesn't
exist yet:

```
python scripts/build_match_equity_table.py --out rebel/match_equity_table.json
python scripts/warm_start_value.py --out checkpoints/rebel_sa_warm --samples 4000
python scripts/train_parallel.py --resume checkpoints/rebel_sa_warm.pt \
  --actors 7 --minutes 900 --num-worlds 24 --cfr-iters 60 --depth-limit 6 \
  --full-depth-cards 3 --value-ground-frac 0.1 \
  --out checkpoints/rebel_sa
```

Pass `--no-match-equity` to either script to train raw-point/score-blind
instead (e.g. for an ablation comparison against a match-equity run).

`--full-depth-cards 3` (up from 2): measured not significantly slower than 2
after the subgame-boundary fix -- Euchre's follow-suit rule keeps the actual
legal-play branching low late in a trick regardless of hand size, and
`solve_value`'s transposition table catches most of the repeated subtrees on
top of that. One more card of *exact* endgame grounding for roughly the same
cost is a clean win, for the same reason `--value-ground-frac` matters: it's
real, non-circular signal the value head can't get by training on its own
bootstrapped guesses.

`--value-ground-frac` (see Milestone-3.5-adjacent work this session) mixes a
steady trickle of the same kind of grounded sample into every live training
batch, so the value head's calibration doesn't silently drift back the way it
did before — the original failure this whole investigation traced back to.

## Actor-conditioned leaf values, and `--bid-exact-frac`

Diagnosed after a ~60,000-hand plateau: the bidding **search** was scoring
*worse* than the policy head it trains (5/15 vs 7/15 on the buildable bid quiz
questions), and confidently so — 0.98–1.00 on its errors. The head was
faithfully learning a broken search, so more hands actively made it worse.

Root cause was the leaf value's *perspective*, not depth. `SubgameSolver`
expands the auction fully and cuts the instant a child enters `PLAY`, so
`--depth-limit`/`--full-depth-cards` do not apply to bidding at all — every
bid leaf is one value-net call. Those calls used `observation_tensor(s,
s.current_player)`: the opening **leader's** infoset, which does not contain
the bidder's hand. The estimate therefore averaged away the actor's own cards,
the information a bid decision turns on. Measured: leaf value moved by exactly
`0.000` across the dealer's six discards, and varied per world with sd
0.02–0.08 against a true sd of 0.10–0.17 (correlation 0.11–0.70, one
*negative*). Alone suffered most — its value depends far more on the maker's
exact holding — so CFR overvalued alone by +0.13…+0.22 equity while not-alone
stayed within ±0.06.

Leaves are now scored from the **searching actor's** seat
(`ReBeLTrainer._value_fn_for`). Grounding captures the bidder's observation to
match, and it now varies which seat bids: bidding opens left of the dealer,
who is *also* the opening leader, so the old code only ever grounded the one
seat where the two coincide — 200/200 byte-identical samples between the two
perspectives before this. On the pre-existing checkpoint this alone takes
net-leaf bid1 search from 3/9 to 6/9 and removes every spurious alone, though
that net was trained on the old perspective, so the honest test is after a
re-ground.

`--bid-exact-frac` solves a fraction of round-1 bid decisions with **every**
leaf valued by exact double-dummy — all-or-nothing per solve, never mixed
inside one tree (mixing would recreate the estimator asymmetry the
phase-boundary fix removed). Unlike `--value-ground-frac`, which is value-only,
these supervise the bid **policy** head against ground truth — the only thing
in the pipeline that does. Cost at production settings: `1.0` with
`--bid-exact-worlds 4` is 2.3x baseline, so `0.05` is ≈1.06x.

Keep it modest. Double-dummy hands the defense perfect information, so it
leans conservative: exact leaves fixed all seven over-aggressive quiz cases
but introduced three over-passive ones. `--bid2-exact-frac` is separate and
off by default purely on cost (a bid2 solve is ~2000 leaves/world against
bid1's ~48); a bid1 solve already contains the whole round-2 auction
internally, so `--bid-exact-frac` alone still grounds round-2 reasoning.

`scripts/bid_search_diagnostic.py` is the tool for all of this — `--mode
compare` (head vs net-leaf vs exact-leaf search), `--mode gaps` (per-action
CFR value against exact truth), `--mode leaves` (per-world leaf
discrimination). Use it rather than the quiz alone, which reads the policy
head only and so cannot tell a bad search from a badly-learned one.

---

## Below: pre-redesign notes (kept for reference)

Written for `rebel_hq.pt`, before Milestone 3.5. `--samples-per-step`
behavior is unchanged and the `--resume checkpoints/rebel_hq.pt` example
itself is stale per the warning above. **`--bid-depth-limit` itself is gone
now** -- a later fix moved the CFR subgame boundary to the phase boundary
(bidding-rooted solves expand the whole auction and cut exactly when trump
gets fixed, instead of sharing a flat ply budget with real card play), which
made a separate bidding depth pointless: any bidding-rooted solve behaves
identically regardless of what `--bid-depth-limit` was set to, once it's set
at all. The solve-time benchmarks below are kept purely as a historical
record of the old flat-ply-budget cost curve; they don't describe current
behavior (bidding-rooted solves are now both correct *and* faster than any
row in that table -- ~35ms/decision at num-worlds 24, cfr-iters 60,
depth-limit 6, cpp engine).

Warm-start with:
```
python scripts/train_scale.py --resume checkpoints/rebel_hq.pt \
  --num-worlds 24 --cfr-iters 60 --depth-limit 6 --bid-depth-limit 7 \
  --full-depth-cards 2 \
  --generations 300 --hands-per-gen 8 --train-steps 25 --eval-every 3 --eval-hands 200
```
Config: 24 worlds, depth 6 for play, exact full-depth for the last 2 tricks,
belief on.

`--bid-depth-limit` gives BID_ROUND_1/2 and DEALER_DISCARD decisions more
real CFR lookahead than PLAY decisions get: bidding is the furthest any
decision sits from the trainer's only exact solves (the last
`--full-depth-cards` tricks), so at a shared depth limit it leans almost
entirely on the value net's guess of how the rest of the hand plays out.

Benchmarked single-decision solve time (num-worlds 24, cfr-iters 60, first
BID_ROUND_1 decision of a fresh hand, isolated -- no other job competing for
CPU):

| bid_depth_limit | solve time |
|---|---|
| 6 (previous baseline, uniform with play) | 4.43s |
| 7 | 14.03s |
| 8 | 38.75s |
| 9 | did not finish in 90s |

Cost grows ~3x per additional ply (subgame tree size scales with
branching-factor^depth). **`12` was tried first and is far past usable** --
it stalled a live 7-actor run for 17+ minutes without completing a single
hand. `7` is the current recommendation: one real ply of extra exact
lookahead for a bounded ~3x cost per bidding decision. `8` is a known,
survivable but much heavier option (39s/decision) if `7` turns out
insufficient; don't go to `9`+ without a further benchmark.

`scripts/train_parallel.py` also takes `--samples-per-step` (replaces the
old flat `--train-steps`): training steps now scale with how much new
self-play data has actually arrived rather than firing a fixed count every
loop tick, which at low actor throughput was resampling each replay-buffer
entry hundreds of times before eviction.

## Known gaps before investing in a long run

See "Known gaps" in [`docs/rebel_design.md`](../docs/rebel_design.md) for
the full detail. In short, this checkpoint (`rebel_hq.pt`, pre-3.5/3.6) was
trained with:
* **no game-score awareness** -- resolved by Milestone 3.6 (match equity),
  but only for checkpoints trained *after* that change; this one predates it
  and its bidding thresholds do no risk adjustment when trailing or ahead.
* **stick-the-dealer off** -- `--stick-the-dealer` now exists on both
  `train_parallel.py` and `train_scale.py` (plumbed through `ReBeLTrainer`
  and into periodic eval too), but it defaults off and `rebel_hq.pt` so far
  was trained entirely without it, so it has never actually seen a
  forced-call decision. Turning the flag on going forward starts training
  it; it does not retroactively fix what's already in this checkpoint.

Committed periodically for durability against container restarts.
