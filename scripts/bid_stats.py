"""Per-seat bidding-behaviour report for a trained checkpoint.

Answers the questions the match-equity table structurally cannot -- how
often does this net order up, call next, call green, and how often does it
go alone doing any of those -- broken out by seat, because first seat and
the dealer face genuinely different decisions on the same cards.

Motivating case: alone-inflation is a documented past failure mode in this
repo (the exact-leaf bid measurement that flipped "four spurious alones",
and the round-2 gap where "every alone option outranks its same-suit
not-alone twin"). outcome_dist can't see it -- a failed loner that still
takes three tricks scores 1, exactly like an ordinary make -- so the call
has to be counted where it happens.

    python scripts/bid_stats.py --net checkpoints/rebel_test3.pt --hands 3000

Self-play by default (the same agent in all four seats), which is what you
want when measuring an agent's own tendencies rather than a matchup.
"""

import argparse
import json
import os
import random
import sys

# Single-threaded torch, set before the import, exactly as train_parallel.py
# does for its actors. This is batch-1 policy-head inference on a small net:
# per-op thread fan-out/sync costs far more than the arithmetic it
# parallelizes, and the penalty explodes when the machine is already busy --
# measured 40 eval hands at 0.52s single-threaded against 62s at the default
# 16 threads while a 14-actor training run held every core. Diagnostics like
# this are exactly what you run DURING training, so the guard matters.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

from euchre.game import EuchreState
from rebel.bid_stats import BidCounter
from rebel.networks import PolicyValueNet
from rebel.train_rebel import ReBeLNetAgent


def collect(agent, hands: int, seed: int = 0,
            stick_the_dealer: bool = False) -> BidCounter:
    """Play `hands` self-play hands, tallying every bidding decision.

    Mirrors rebel/evaluate.py's play_hand loop (same dealer rotation) but
    observes each action before applying it, which play_hand doesn't expose.
    Uses the Python engine throughout: this is policy-head inference only,
    no search, so the C++ hot path buys nothing here.
    """
    counter = BidCounter()
    rng = random.Random(seed)
    for h in range(hands):
        state = EuchreState.new_hand(
            dealer=h % 4, stick_the_dealer=stick_the_dealer).deal(rng)
        while not state.is_terminal():
            action = agent.act(state, rng)
            counter.record(state, action, engine="python")
            state = state.apply(action)
    return counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="checkpoint .pt to measure")
    ap.add_argument("--hands", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stick-the-dealer", action="store_true",
                    help="match how the checkpoint was trained -- with this "
                         "off a full pass-out misdeals instead of forcing "
                         "the dealer to call, which changes round-2 rates.")
    ap.add_argument("--sampled", dest="greedy", action="store_false",
                    default=True,
                    help="sample from the policy instead of acting greedily. "
                         "Greedy (the default) reports the agent's modal "
                         "choice; sampled reports the mixed strategy it "
                         "actually plays in self-play.")
    ap.add_argument("--json-out", type=str, default=None,
                    help="also write the per-seat rows as JSON")
    args = ap.parse_args()

    net = PolicyValueNet()
    net.load_state_dict(torch.load(args.net, map_location="cpu"))
    net.eval()
    agent = ReBeLNetAgent(net, greedy=args.greedy)

    print(f"net: {args.net}  hands: {args.hands}  "
          f"greedy: {args.greedy}  stick_the_dealer: {args.stick_the_dealer}",
          flush=True)
    counter = collect(agent, args.hands, seed=args.seed,
                      stick_the_dealer=args.stick_the_dealer)
    print(counter.format_table())

    rows = counter.as_rows()
    alone_r1 = sum(r["r1_orderup_alone"] for r in rows)
    made_r1 = sum(r["r1_orderup"] + r["r1_orderup_alone"] for r in rows)
    alone_r2 = sum(r["r2_next_alone"] + r["r2_green_alone"] for r in rows)
    made_r2 = sum(r["r2_next"] + r["r2_next_alone"]
                  + r["r2_green"] + r["r2_green_alone"] for r in rows)
    print(f"\nalone share of calls -- round 1: "
          f"{alone_r1}/{made_r1} ({alone_r1 / made_r1:.1%})" if made_r1
          else "\nalone share of calls -- round 1: no calls")
    print(f"alone share of calls -- round 2: "
          f"{alone_r2}/{made_r2} ({alone_r2 / made_r2:.1%})" if made_r2
          else "alone share of calls -- round 2: no calls")

    if args.json_out:
        json.dump(rows, open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
