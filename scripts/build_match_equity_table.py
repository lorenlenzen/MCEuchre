"""One-off precompute: fit the empirical single-hand outcome distribution and
build the match-equity table from it, saving both to a JSON file that
training scripts load at startup (cheap; not recomputed per process).

    python scripts/build_match_equity_table.py --hands 20000 \
        --out rebel/match_equity_table.json

By default fits against RuleBasedAgent vs RuleBasedAgent (a fixed heuristic,
cheap and always available). Pass --checkpoint to fit against your own
trained net instead (ReBeLNetAgent, policy head only -- no decision-time
search, since that would be far too slow across thousands of hands) for a
more accurate outcome distribution once you have a checkpoint worth trusting:

    python scripts/build_match_equity_table.py --hands 20000 \
        --checkpoint checkpoints/rebel_sa.pt --out rebel/match_equity_table.json
"""

import argparse

from rebel.match_equity import (
    MatchEquityModel, build_equity_table, fit_hand_outcome_distribution,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=20000,
                    help="hands to play to fit the single-hand outcome "
                         "distribution")
    ap.add_argument("--target", type=int, default=10,
                    help="race-to-target match score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="rebel/match_equity_table.json")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="fit against this trained checkpoint (ReBeLNetAgent, "
                         "greedy) instead of the RuleBasedAgent default -- "
                         "both seats use the same checkpoint, matching how "
                         "self-play/deployment actually plays.")
    ap.add_argument("--greedy", dest="greedy", action="store_true", default=True,
                    help="(--checkpoint only) act greedily (default)")
    ap.add_argument("--sampled", dest="greedy", action="store_false",
                    help="(--checkpoint only) sample from the policy instead "
                         "of acting greedily -- more variance, closer to "
                         "self-play's own mixed-strategy behavior")
    args = ap.parse_args()

    if args.checkpoint:
        import torch
        from rebel.networks import PolicyValueNet
        from rebel.train_rebel import ReBeLNetAgent

        net = PolicyValueNet()
        net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        net.eval()
        agent_factory = lambda: ReBeLNetAgent(net, greedy=args.greedy)
        print(f"fitting hand-outcome distribution over {args.hands} hands "
              f"({args.checkpoint} vs itself, greedy={args.greedy})...",
              flush=True)
    else:
        agent_factory = None  # fit_hand_outcome_distribution's RuleBasedAgent default
        print(f"fitting hand-outcome distribution over {args.hands} hands "
              f"(RuleBasedAgent vs RuleBasedAgent)...", flush=True)

    dist = fit_hand_outcome_distribution(
        hands=args.hands, seed=args.seed,
        **({"agent_factory": agent_factory} if agent_factory else {}))
    print("outcome distribution (dealing team's pts, other team's pts): probability",
          flush=True)
    for outcome, prob in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {outcome}: {prob:.4f}", flush=True)

    print(f"\nvalue-iterating the {args.target}x{args.target} dealer-relative "
          f"equity table...", flush=True)
    table = build_equity_table(dist, target=args.target)
    model = MatchEquityModel(table, dist)

    print(f"\nsanity: Ed(0,0)={model.win_prob(0, 0, True):.4f}  "
          f"Eo(0,0)={model.win_prob(0, 0, False):.4f}  "
          f"(dealing team's real edge at an even score -- these should sum to "
          f"~1.0000 but are no longer exactly 0.5/0.5; that symmetry was an "
          f"artifact of the old team0/team1-symmetrized table)", flush=True)
    print(f"        Ed({args.target-1},0)={model.win_prob(args.target-1, 0, True):.4f} "
          f"Ed(0,{args.target-1})={model.win_prob(0, args.target-1, True):.4f}",
          flush=True)

    model.save(args.out)
    print(f"\nsaved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
