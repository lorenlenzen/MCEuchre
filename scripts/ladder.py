"""Run the evaluation ladder over a pool of agents and print a leaderboard.

    python scripts/ladder.py            # fast pool (random, rule-based, MCCFR)
    python scripts/ladder.py --pimc     # add PIMC search (much slower)

Elo is anchored so the field averages 1500. Search agents are slow, so keep
``--hands`` modest when they are in the pool.
"""

import argparse
import time

from rebel.ladder import AgentSpec, evaluate_ladder, format_leaderboard
from rebel.evaluate import (
    RandomAgent, RuleBasedAgent, PointCountAgent, MCCFRAgent,
)
from rebel.mccfr import MCCFRTrainer
from rebel.pimc import PIMCAgent
from rebel.belief_model import BiddingBeliefModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=200,
                    help="hands per pairing")
    ap.add_argument("--pimc", action="store_true",
                    help="include PIMC search agents (slow)")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="bootstrap resamples for Elo CIs (0 to skip)")
    args = ap.parse_args()

    print("Preparing agents...")
    mccfr = MCCFRTrainer(seed=0)
    mccfr.train(iterations=200)  # weak, but a distinct rung

    specs = [
        AgentSpec("random", RandomAgent),
        AgentSpec("rule_based", RuleBasedAgent),
        AgentSpec("point_count", PointCountAgent),
        AgentSpec("mccfr_200it", lambda: MCCFRAgent(mccfr)),
    ]
    if args.pimc:
        specs.append(AgentSpec(
            "pimc", lambda: PIMCAgent(worlds=8, call_worlds=4)))
        specs.append(AgentSpec(
            "pimc+belief", lambda: PIMCAgent(
                worlds=8, call_worlds=4,
                belief_model=BiddingBeliefModel())))

    print(f"Round robin: {len(specs)} agents, {args.hands} hands/pair "
          f"({len(specs) * (len(specs) - 1) // 2} pairings)")
    t0 = time.time()
    standings = evaluate_ladder(specs, hands_per_pair=args.hands, seed=0,
                                bootstrap=args.bootstrap)
    print(f"\nDone in {time.time() - t0:.0f}s\n")
    print(format_leaderboard(standings))


if __name__ == "__main__":
    main()
