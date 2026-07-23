"""Train the ReBeL loop at scale with belief refinement on, tracking strength.

Runs self-play with the bidding-conditioned belief model, and every few
generations evaluates the (inference-only) net agent against the baselines so
we get a learning curve. Checkpoints the net and a JSON log.

    python scripts/train_scale.py --generations 40 --hands-per-gen 25
"""

import argparse
import json
import time

import torch

from rebel.train_rebel import ReBeLTrainer, ReBeLNetAgent
from rebel.belief_model import BiddingBeliefModel
from rebel.evaluate import evaluate, RandomAgent, RuleBasedAgent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--hands-per-gen", type=int, default=25)
    ap.add_argument("--train-steps", type=int, default=15)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-hands", type=int, default=300)
    ap.add_argument("--num-worlds", type=int, default=6)
    ap.add_argument("--cfr-iters", type=int, default=12)
    ap.add_argument("--depth-limit", type=int, default=4)
    ap.add_argument("--bid-depth-limit", type=int, default=None,
                    help="deeper depth limit for bidding-phase decisions "
                         "(BID_ROUND_1/2, DEALER_DISCARD); defaults to "
                         "--depth-limit if unset")
    ap.add_argument("--full-depth-cards", type=int, default=0,
                    help="solve to terminal when <= this many cards remain")
    ap.add_argument("--stick-the-dealer", action="store_true",
                    help="force the dealer to call in round 2 instead of "
                         "letting a full pass-out misdeal the hand")
    ap.add_argument("--grad-clip-norm", type=float, default=5.0,
                    help="cap the gradient norm of any single train_step; a "
                         "safety net against an outsized update from an "
                         "occasional high-loss batch, not a tuning knob.")
    ap.add_argument("--round2-seed-frac", type=float, default=0.0,
                    help="fraction of self-play hands dealt with the "
                         "up-card's suit biased weak for all four hands; "
                         "round 1 still plays out for real, this just "
                         "raises the odds it naturally resolves into "
                         "round 2, oversampling the rarest, "
                         "slowest-training bidding decision type.")
    ap.add_argument("--value-ground-frac", type=float, default=0.0,
                    help="fraction of self-play hands (in addition to, not "
                         "instead of, the normal hand) that also generate "
                         "one exact rollout_value-grounded post-call "
                         "sample, supervising value_loss only -- a "
                         "continuous anchor against the value head "
                         "re-drifting the way it was found to have "
                         "(overestimating post-call outcomes by roughly "
                         "half a point to a full point) this session.")
    ap.add_argument("--out", type=str, default="rebel_scale")
    ap.add_argument("--resume", type=str, default=None,
                    help="checkpoint .pt to warm-start the net from")
    args = ap.parse_args()

    trainer = ReBeLTrainer(
        num_worlds=args.num_worlds, cfr_iterations=args.cfr_iters,
        depth_limit=args.depth_limit, bid_depth_limit=args.bid_depth_limit,
        full_depth_cards=args.full_depth_cards,
        stick_the_dealer=args.stick_the_dealer,
        grad_clip_norm=args.grad_clip_norm,
        round2_seed_frac=args.round2_seed_frac,
        value_ground_frac=args.value_ground_frac,
        belief_model=BiddingBeliefModel(), lr=1e-3, seed=0)
    if args.resume:
        trainer.net.load_state_dict(torch.load(args.resume))
        print(f"resumed from {args.resume}")

    log = []
    total_hands = 0
    t0 = time.time()
    stats = {"policy_loss": 0.0, "value_loss": 0.0}
    for g in range(1, args.generations + 1):
        for _ in range(args.hands_per_gen):
            trainer.self_play_hand()
        total_hands += args.hands_per_gen
        for _ in range(args.train_steps):
            stats = trainer.train_step(batch_size=128)

        if g % args.eval_every == 0 or g == args.generations:
            def net_agent():
                return ReBeLNetAgent(trainer.net, greedy=True)
            s_rand = evaluate(net_agent, RandomAgent,
                              hands=args.eval_hands, seed=100 + g,
                              stick_the_dealer=args.stick_the_dealer)
            s_rule = evaluate(net_agent, RuleBasedAgent,
                              hands=args.eval_hands, seed=200 + g,
                              stick_the_dealer=args.stick_the_dealer)
            top_clusters = trainer.cluster_stats()
            entry = {
                "gen": g, "hands": total_hands, "buffer": len(trainer.buffer),
                "elapsed_s": round(time.time() - t0),
                "policy_loss": round(stats["policy_loss"], 4),
                "value_loss": round(stats["value_loss"], 4),
                "vs_random": round(s_rand["team0_mean_point_diff"], 3),
                "win_random": round(s_rand["team0_win_rate"], 3),
                "vs_rule": round(s_rule["team0_mean_point_diff"], 3),
                "win_rule": round(s_rule["team0_win_rate"], 3),
                "top_clusters": top_clusters,
            }
            log.append(entry)
            print(f"gen {g:>3} | hands {total_hands:>5} | {entry['elapsed_s']:>4}s "
                  f"| ploss {entry['policy_loss']:.3f} vloss {entry['value_loss']:.3f} "
                  f"| vs random {entry['vs_random']:+.3f} ({entry['win_random']:.2f}) "
                  f"| vs rule {entry['vs_rule']:+.3f} ({entry['win_rule']:.2f})",
                  flush=True)
            if top_clusters:
                tc = ", ".join(f"{r['key']}:{r['sample_share']:.0%}"
                              for r in top_clusters)
                print(f"       top clusters (sample share): {tc}", flush=True)
            torch.save(trainer.net.state_dict(), args.out + ".pt")
            json.dump(log, open(args.out + ".log.json", "w"), indent=2)

    print(f"\nDone: {total_hands} hands in {time.time() - t0:.0f}s. "
          f"Checkpoint: {args.out}.pt")


if __name__ == "__main__":
    main()
