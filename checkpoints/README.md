# ReBeL high-quality training checkpoint

## ⚠️ `rebel_hq*.pt` / `rebel_hq_valuefix.pt` are stale (encoding changed)

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
approximately). `OBS_SIZE` changed (249 → 394) and the network internals
changed, so **every existing checkpoint here is incompatible** with the
current code — loading one will fail on a shape mismatch. Train fresh; don't
`--resume` from `rebel_hq*.pt`.

## Speeding up a fresh run: value warm-start

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

```
python scripts/warm_start_value.py --out checkpoints/rebel_sa_warm --samples 4000
python scripts/train_parallel.py --resume checkpoints/rebel_sa_warm.pt \
  --actors 7 --minutes 900 --num-worlds 24 --cfr-iters 60 --depth-limit 6 \
  --bid-depth-limit 6 --full-depth-cards 2 --value-ground-frac 0.1 \
  --out checkpoints/rebel_sa
```

`--value-ground-frac` (see Milestone-3.5-adjacent work this session) mixes a
steady trickle of the same kind of grounded sample into every live training
batch, so the value head's calibration doesn't silently drift back the way it
did before — the original failure this whole investigation traced back to.

---

## Below: pre-redesign notes (kept for reference)

Written for `rebel_hq.pt`, before Milestone 3.5. The bid-depth-limit solve-time
benchmarks are still architecturally relevant (they measure CFR tree cost, not
network internals) and `--samples-per-step` behavior is unchanged; the
`--resume checkpoints/rebel_hq.pt` example itself is stale per the warning
above.

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
the full detail. In short, this checkpoint's bidding net was trained with:
* **no game-score awareness** -- every hand is isolated from the race-to-10
  context, so its bidding thresholds can't do risk adjustment when trailing
  or ahead. Still an open gap, needs real state/loop changes (see the design
  doc), not just a flag.
* **stick-the-dealer off** -- `--stick-the-dealer` now exists on both
  `train_parallel.py` and `train_scale.py` (plumbed through `ReBeLTrainer`
  and into periodic eval too), but it defaults off and `rebel_hq.pt` so far
  was trained entirely without it, so it has never actually seen a
  forced-call decision. Turning the flag on going forward starts training
  it; it does not retroactively fix what's already in this checkpoint.

Committed periodically for durability against container restarts.
