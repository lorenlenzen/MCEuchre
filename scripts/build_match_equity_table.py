"""One-off precompute: fit the empirical single-hand outcome distribution and
build the match-equity table from it, saving both to a JSON file that
training scripts load at startup (cheap; not recomputed per process).

    python scripts/build_match_equity_table.py --hands 20000 \
        --out rebel/match_equity_table.json
"""

import argparse

from rebel.match_equity import (
    MatchEquityModel, build_equity_table, fit_hand_outcome_distribution,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=20000,
                    help="hands to play (RuleBasedAgent vs RuleBasedAgent) "
                         "to fit the single-hand outcome distribution")
    ap.add_argument("--target", type=int, default=10,
                    help="race-to-target match score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="rebel/match_equity_table.json")
    args = ap.parse_args()

    print(f"fitting hand-outcome distribution over {args.hands} hands "
          f"(RuleBasedAgent vs RuleBasedAgent)...", flush=True)
    dist = fit_hand_outcome_distribution(hands=args.hands, seed=args.seed)
    print("outcome distribution (team0_pts, team1_pts): probability", flush=True)
    for outcome, prob in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {outcome}: {prob:.4f}", flush=True)

    print(f"\nvalue-iterating the {args.target}x{args.target} equity table...",
          flush=True)
    table = build_equity_table(dist, target=args.target)
    model = MatchEquityModel(table, dist)

    print(f"\nsanity: E(0,0)={model.win_prob(0,0):.4f} "
          f"(should be exactly 0.5)", flush=True)
    print(f"        E({args.target-1},0)={model.win_prob(args.target-1,0):.4f} "
          f"E(0,{args.target-1})={model.win_prob(0,args.target-1):.4f}", flush=True)

    model.save(args.out)
    print(f"\nsaved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
