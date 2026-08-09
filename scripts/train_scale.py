"""Train the ReBeL loop at scale, tracking strength.

Runs self-play, and every few generations evaluates the (inference-only) net
agent against the baselines so we get a learning curve. Checkpoints the net
and a JSON log.

    python scripts/train_scale.py --generations 40 --hands-per-gen 25
"""

import argparse
import json
import os
import random
import sys
import time

import torch

from rebel.train_rebel import ReBeLTrainer, ReBeLNetAgent
from rebel.evaluate import evaluate, RandomAgent, RuleBasedAgent
from rebel.match_equity import MatchEquityModel
from rebel.plr import PLRBuffer, deal_to_spec, score_samples, spec_to_state
from rebel.eval_log import legacy_log_path, load_resumable_log, log_path_for

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_pattern import (constrained_deal, parse_require,  # noqa: E402
                           parse_score, parse_up_rank, parse_seat)


def _pattern_deal_fn(trainer, patterns, void_up, score, up_ranks, seat, max_tries):
    """--require's ReBeLTrainer.deal_fn: a fresh, engine-appropriate,
    pattern-matching BID_ROUND_1 deal, drawn with `trainer.rng` -- the same
    RNG self_play_hand's own randomness (world sampling, action sampling)
    already uses, so the whole run stays under one seeded stream.

    Always phase="bid1" and the returned `passes` are discarded (never
    applied): unlike train_pattern.py's one-shot leaf generation, this deal
    feeds an ordinary self_play_hand() call, which walks the WHOLE hand for
    real starting from round-1's first decision -- every seat trains on it,
    not just whichever seat holds the pattern.

    constrained_deal PLACES the required cards rather than rejection-
    sampling for them, so max_tries only needs to cover void/up-rank
    exhaustion edge cases (see that function's docstring), not the
    pattern's natural rarity -- a satisfiable pattern should succeed within
    a handful of tries, not thousands.

    `seat` (--seat) pins which bidding-order position gets the pattern-
    holding hand (see parse_seat); it does NOT change where self_play_hand
    starts playing the hand from -- that's still round-1's first decision
    regardless, same as any other deal. It only constrains who ends up
    holding the pattern by the time their own turn comes around."""
    def deal_fn():
        for _ in range(max_tries):
            drawn = constrained_deal(trainer, patterns, "bid1", trainer.rng,
                                     void_up=void_up, score=score,
                                     up_ranks=up_ranks, seat=seat)
            if drawn is not None:
                return drawn[0]
        raise RuntimeError(
            f"--require couldn't be satisfied in {max_tries} tries -- try "
            f"raising --require-max-tries, or check --void-up/--up-rank "
            f"aren't contradictory with the pattern")
    return deal_fn


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
    ap.add_argument("--bid-exact-frac", type=float, default=0.0,
                    help="fraction of round-1 bidding decisions solved with "
                         "every leaf valued by exact double-dummy instead of "
                         "the value net (all-or-nothing per solve). Unlike "
                         "--value-ground-frac these supervise the bid POLICY "
                         "head against ground truth. Keep modest (~0.05): "
                         "double-dummy gives the defense perfect information "
                         "and so leans conservative.")
    ap.add_argument("--bid2-exact-frac", type=float, default=0.0,
                    help="same for round-2-rooted decisions; separate knob "
                         "because a bid2 solve is ~2000 leaves/world against "
                         "bid1's ~48.")
    ap.add_argument("--bid-exact-worlds", type=int, default=None,
                    help="belief worlds for exact-leaf solves only (defaults "
                         "to --num-worlds).")
    ap.add_argument("--play-exact-frac", type=float, default=0.0,
                    help="fraction of card-play decisions solved with "
                         "exact double-dummy leaves; see "
                         "train_parallel.py's flag for the measurement "
                         "that says leads are the weak spot.")
    ap.add_argument("--play-exact-all-positions",
                    dest="play_exact_lead_only", action="store_false",
                    default=True,
                    help="apply it at every trick position, not just leads.")
    ap.add_argument("--play-exact-worlds", type=int, default=None,
                    help="belief worlds for play-exact solves only (defaults "
                         "to --num-worlds) -- kept separate from "
                         "--bid-exact-worlds; see train_parallel.py's flag "
                         "for the bug that shared knob caused.")
    ap.add_argument("--belief-weight-frac", type=float, default=0.0,
                    help="fraction of bidding-phase net-leaf solves that "
                         "importance-weight sampled worlds by the net's own "
                         "pass-sequence probability instead of sampling them "
                         "uniformly; see train_parallel.py's flag for the "
                         "full rationale. cpp engine only. Off by default.")
    ap.add_argument("--require", type=str, default=None,
                    help="run ordinary self-play (every seat's real decision "
                         "gets a genuine solve, unchanged, from round-1 "
                         "bidding onward) but ONLY from deals where the "
                         "acting-to-be seat's hand matches this structural "
                         "pattern -- same <rank><relsuit> syntax as "
                         "scripts/train_pattern.py's --require (e.g. "
                         "'JU,JN,JG,Jg' for all four jacks). Unlike "
                         "train_pattern.py this doesn't fast-forward to one "
                         "decision or freeze the trunk -- it's the same "
                         "training loop as an unconstrained run, just with "
                         "every deal drawn from the pattern instead of "
                         "uniformly at random, so every seat trains on it, "
                         "not just the pattern-holder. --round2-seed-frac "
                         "and --value-ground-frac both still apply on top "
                         "(they call the same overridden dealer). Off "
                         "(uniform dealing) by default.")
    ap.add_argument("--void-up", action="store_true",
                    help="(--require only) the pattern-holding hand must "
                         "also be void in the up-card's (effective) suit.")
    ap.add_argument("--score", type=str, default=None,
                    help="(--require only) pin the match score as "
                         "MINE,THEIRS for every pattern-matching deal (e.g. "
                         "'8,9'), from the pattern-holder's side. Without "
                         "this the score is sampled from the equity model "
                         "as usual (or stays 0-0 with --no-match-equity).")
    ap.add_argument("--up-rank", type=str, default=None,
                    help="(--require only) pin the up card's rank, e.g. "
                         "'J' or 'T,J,Q'; see scripts/train_pattern.py's "
                         "flag of the same name.")
    ap.add_argument("--seat", type=str, default=None,
                    choices=["first", "second", "third", "dealer"],
                    help="(--require only) pin which bidding-order position "
                         "(first/second/third/dealer, bidding opens left of "
                         "the dealer who acts last -- scripts/quiz_eval.py's "
                         "seat convention) gets the pattern-holding hand. "
                         "Does NOT change where self-play starts playing "
                         "the hand from -- still round-1's first decision "
                         "regardless, same as any deal. Without it the "
                         "position is uniformly random.")
    ap.add_argument("--plr-replay-prob", type=float, default=0.0,
                    help="enable Prioritized Level Replay: probability that "
                         "a self-play hand REPLAYS a stored high-loss deal "
                         "instead of dealing a fresh one (0 = off, the "
                         "default). Deals are scored by the same per-sample "
                         "loss cluster_priority already uses, so this is the "
                         "per-deal version of the existing cluster-level "
                         "prioritization -- the granularity clusters can't "
                         "express. Replay re-runs the real CFR solves, so "
                         "targets are always fresh, never stale stored ones. "
                         "This is also the dial on distribution shift: the "
                         "buffer only ever holds deals that occurred "
                         "naturally, but replaying them still over-weights "
                         "hard hands relative to a true uniform shuffle. "
                         "0.3-0.5 is the usual range; 1.0 would train almost "
                         "entirely on replays and drift furthest.")
    ap.add_argument("--plr-capacity", type=int, default=250,
                    help="how many scored deals the buffer holds. Size it "
                         "to expected hands-per-actor: roughly 10-20%% of "
                         "them, so it fills in the first ~quarter of the run "
                         "and then turns over. Too large and it never fills, "
                         "so no eviction pressure ever develops (at 29 "
                         "hands/actor/hour, 250 needs ~14h); too small and "
                         "each stored deal gets replayed many times, risking "
                         "overfitting to a handful of hands (at 159 "
                         "hands/actor/hour, 50 gives ~17 replays each). "
                         "Note --plr-min-score-ratio already keeps the "
                         "buffer selective before it fills, so an unfilled "
                         "buffer costs eviction pressure, not selection.")
    ap.add_argument("--plr-temperature", type=float, default=1.0,
                    help="(PLR) rank-prioritization temperature; weight is "
                         "(1/rank)^(1/T). Lower = greedier toward the "
                         "highest-loss deals.")
    ap.add_argument("--plr-staleness-coef", type=float, default=0.3,
                    help="(PLR) fraction of sampling weight given to how "
                         "long ago a deal was last replayed. Entries have no "
                         "TTL -- a stored score only refreshes when that "
                         "deal is sampled again -- so this controls how fast "
                         "the buffer re-measures itself. Measured coverage "
                         "over one full buffer's worth of replays: 38%% of "
                         "entries at 0.1, 58%% at 0.5. Default 0.3 (the "
                         "published algorithm uses ~0.1, over far smaller "
                         "level sets).")
    ap.add_argument("--plr-min-score-ratio", type=float, default=1.0,
                    help="(PLR) admission gate: a NEW deal must score at "
                         "least this multiple of the current typical loss "
                         "to earn a buffer slot. 1.0 = only hands harder "
                         "than average get stored, which is what makes the "
                         "buffer a struggle-finder rather than a cache of "
                         "recent hands. 0.0 restores textbook PLR "
                         "(unconditional admission until full) -- but a "
                         "404-hand run over 14 actors leaves each buffer "
                         "2.9%% full, so nothing would ever be selected on "
                         "difficulty at all.")
    ap.add_argument("--plr-dump-top", type=int, default=25,
                    help="(PLR) how many of the hardest stored deals to "
                         "write to <out>.plr.json, rendered readably with "
                         "hands keyed by seat position and suits labelled "
                         "by role (U/N/G/g) so a recurring shape converts "
                         "straight into a --require pattern.")
    ap.add_argument("--require-max-tries", type=int, default=100,
                    help="(--require only) retries per hand before giving "
                         "up and raising -- constrained_deal PLACES the "
                         "pattern rather than rejection-sampling for it, so "
                         "this only needs to cover void/up-rank exhaustion "
                         "edge cases, not the pattern's natural rarity.")
    ap.add_argument("--fresh-optimizer", action="store_true",
                    help="ignore the resumed checkpoint's sibling .opt.pt and "
                         "start Adam from zero. Worth it after a change to "
                         "WHAT the value head predicts -- see "
                         "train_parallel.py's flag for the rationale.")
    ap.add_argument("--match-equity-table", type=str,
                    default="rebel/match_equity_table.json",
                    help="path to the precomputed match-equity table (see "
                         "scripts/build_match_equity_table.py). See "
                         "train_parallel.py's --match-equity-table for what "
                         "this changes.")
    ap.add_argument("--no-match-equity", action="store_true",
                    help="disable match-equity awareness -- raw point-"
                         "differential targets at a fixed 0-0 score.")
    ap.add_argument("--out", type=str, default="rebel_scale")
    ap.add_argument("--resume", type=str, default=None,
                    help="checkpoint .pt to warm-start the net from")
    ap.add_argument("--engine", choices=["python", "cpp"], default="python",
                    help="self-play engine. 'cpp' uses the compiled "
                         "hot-path port (cpp/, see cpp/README.md; must be "
                         "built first: python setup.py build_ext --inplace), "
                         "differentially verified bit-for-bit against the "
                         "Python path in tests/test_cpp_equivalence.py. "
                         "--value-ground-frac and --round2-seed-frac both "
                         "work fine with --engine cpp -- value_ground_frac's "
                         "cpp path uses cpp_rollout_value "
                         "(rebel/train_rebel.py), a Python-level mirror "
                         "built on the already-bound mceuchre_cpp.solve_value, "
                         "not a new C++ port.")
    args = ap.parse_args()

    # --void-up/--score/--up-rank/--seat place cards / fix state at deal
    # time, which only --require's constructor does -- without it there's
    # no pattern-matching deal for them to constrain.
    for flag, val in (("--void-up", args.void_up), ("--score", args.score),
                      ("--up-rank", args.up_rank), ("--seat", args.seat)):
        if val and args.require is None:
            print(f"error: {flag} needs --require")
            sys.exit(1)
    require_patterns = require_score = require_up_ranks = require_seat = None
    if args.require is not None:
        require_patterns = parse_require(args.require)
        require_score = parse_score(args.score) if args.score else None
        require_up_ranks = parse_up_rank(args.up_rank) if args.up_rank else None
        require_seat = parse_seat(args.seat) if args.seat else None
        desc = f"--require {args.require}"
        if args.void_up:
            desc += ", void in up"
        if args.score:
            desc += f", at {args.score}"
        if args.up_rank:
            desc += f", up-card rank in {{{args.up_rank}}}"
        if args.seat:
            desc += f", seat={args.seat}"
        print(f"pattern-constrained dealing: {desc}")

    print(f"self-play engine: {args.engine}")

    equity_model = None
    if not args.no_match_equity:
        if not os.path.exists(args.match_equity_table):
            print(f"error: --match-equity-table {args.match_equity_table!r} "
                  f"not found. Build it first:\n"
                  f"    python scripts/build_match_equity_table.py "
                  f"--out {args.match_equity_table}\n"
                  f"or pass --no-match-equity to train without it.")
            sys.exit(1)
        equity_model = MatchEquityModel.load(args.match_equity_table)
        print(f"match equity: on ({args.match_equity_table})")
    else:
        print("match equity: off (--no-match-equity)")

    trainer = ReBeLTrainer(
        num_worlds=args.num_worlds, cfr_iterations=args.cfr_iters,
        depth_limit=args.depth_limit,
        full_depth_cards=args.full_depth_cards,
        stick_the_dealer=args.stick_the_dealer, equity_model=equity_model,
        grad_clip_norm=args.grad_clip_norm,
        round2_seed_frac=args.round2_seed_frac,
        value_ground_frac=args.value_ground_frac,
        bid_exact_frac=args.bid_exact_frac,
        bid2_exact_frac=args.bid2_exact_frac,
        bid_exact_worlds=args.bid_exact_worlds,
        play_exact_frac=args.play_exact_frac,
        play_exact_lead_only=args.play_exact_lead_only,
        play_exact_worlds=args.play_exact_worlds,
        belief_weight_frac=args.belief_weight_frac,
        engine=args.engine, lr=1e-3, seed=0)
    if require_patterns is not None:
        trainer.deal_fn = _pattern_deal_fn(
            trainer, require_patterns, args.void_up, require_score,
            require_up_ranks, require_seat, args.require_max_tries)

    # PLR layers ON TOP of whatever dealer is already in place: a non-replay
    # hand falls through to `base_deal` -- the pattern dealer above when
    # --require is set, the built-in uniform one otherwise -- so --plr and
    # --require compose (replay the hardest hands *within* the pattern)
    # rather than one silently overriding the other.
    plr = None
    if args.plr_replay_prob > 0.0:
        plr = PLRBuffer(capacity=args.plr_capacity,
                        replay_prob=args.plr_replay_prob,
                        temperature=args.plr_temperature,
                        staleness_coef=args.plr_staleness_coef,
                        min_score_ratio=args.plr_min_score_ratio,
                        rng=random.Random(1234))
        base_deal = trainer.deal_fn or trainer._default_deal

        def plr_deal_fn():
            if plr.should_replay():
                return spec_to_state(plr.sample(), trainer)
            return base_deal()

        trainer.deal_fn = plr_deal_fn
        print(f"PLR: on (replay_prob={args.plr_replay_prob}, "
              f"capacity={args.plr_capacity}, T={args.plr_temperature}, "
              f"staleness={args.plr_staleness_coef}); hardest deals -> "
              f"{args.out}.plr.json")
        if args.plr_replay_prob >= 0.9:
            print(f"  WARNING: --plr-replay-prob {args.plr_replay_prob} "
                  f"starves the buffer -- new deals only enter on non-replay "
                  f"hands, so at 1.0 the first hand is replayed forever "
                  f"(1 distinct deal over 300 hands, measured). Use <=0.5 "
                  f"unless you specifically want that.", flush=True)

    if args.resume:
        trainer.net.load_state_dict(torch.load(args.resume))
        print(f"resumed from {args.resume}")
        # Adam's per-parameter momentum/variance state, checkpointed
        # alongside the weights as a sibling <resume-without-.pt>.opt.pt
        # file (not embedded in the same file, so quiz_eval.py and every
        # other plain-state_dict loader is unaffected) -- see
        # train_parallel.py's resume block for the full rationale. Missing
        # sidecar (older checkpoints) just means starting Adam fresh, same
        # as before this existed.
        opt_path = os.path.splitext(args.resume)[0] + ".opt.pt"
        if args.fresh_optimizer:
            print(f"ignoring optimizer state at {opt_path} (--fresh-optimizer)")
        elif os.path.exists(opt_path):
            trainer.opt.load_state_dict(torch.load(opt_path))
            print(f"resumed optimizer state from {opt_path}")
        else:
            print(f"no optimizer state at {opt_path} -- starting Adam fresh")

    # Resume this script's own eval log rather than clobbering it. Starting
    # a fresh list and dumping it over the old file discarded every prior
    # run's entries on the same --out -- and, back when both scripts shared
    # one `<out>.log.json`, a train_parallel.py run's history too. The
    # strength trend only means anything plotted across the whole training
    # history, so new entries are appended, with the prior run's final
    # gen/hands/elapsed carried forward as offsets so the x-axes stay
    # continuous instead of restarting at zero.
    log_path = log_path_for(args.out, "scale")
    log = load_resumable_log(log_path, ("gen", "hands", "elapsed_s"),
                             legacy_path=legacy_log_path(args.out),
                             script="train_scale.py")
    gen_offset = log[-1]["gen"] if log else 0
    hands_offset = log[-1]["hands"] if log else 0
    elapsed_offset = log[-1]["elapsed_s"] if log else 0
    total_hands = 0
    t0 = time.time()
    stats = {"policy_loss": 0.0, "value_loss": 0.0}
    for g in range(1, args.generations + 1):
        for _ in range(args.hands_per_gen):
            trainer.self_play_hand()
            if plr is not None:
                # Score the deal that was just played -- fresh or replayed
                # alike. Re-scoring a replay is the point, not redundancy:
                # its stored score was measured against an older net, and
                # refreshing it is what lets a deal the net has since
                # learned fall out of the buffer.
                plr.update(deal_to_spec(trainer.last_hand_state, args.engine),
                           score_samples(trainer.net, trainer.last_hand_samples))
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
            entry = {
                "gen": gen_offset + g,
                "hands": hands_offset + total_hands,
                "buffer": len(trainer.buffer),
                "elapsed_s": elapsed_offset + round(time.time() - t0),
                "policy_loss": round(stats["policy_loss"], 4),
                "value_loss": round(stats["value_loss"], 4),
                "vs_random": round(s_rand["team0_mean_point_diff"], 3),
                "win_random": round(s_rand["team0_win_rate"], 3),
                "vs_rule": round(s_rule["team0_mean_point_diff"], 3),
                "win_rule": round(s_rule["team0_win_rate"], 3),
            }
            if plr is not None:
                st = plr.stats()
                entry["plr"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in st.items()}
            log.append(entry)
            line = (f"gen {entry['gen']:>3} | hands {entry['hands']:>5} "
                    f"| {entry['elapsed_s']:>4}s "
                    f"| ploss {entry['policy_loss']:.3f} vloss {entry['value_loss']:.3f} "
                    f"| vs random {entry['vs_random']:+.3f} ({entry['win_random']:.2f}) "
                    f"| vs rule {entry['vs_rule']:+.3f} ({entry['win_rule']:.2f})")
            if plr is not None:
                st = entry["plr"]
                line += (f" | plr {st['size']}/{args.plr_capacity} "
                         f"replayed {st['replayed']} "
                         f"score {st['score_min']:.2f}-{st['score_max']:.2f}")
            print(line, flush=True)
            torch.save(trainer.net.state_dict(), args.out + ".pt")
            torch.save(trainer.opt.state_dict(), args.out + ".opt.pt")
            json.dump(log, open(log_path, "w"), indent=2)
            if plr is not None:
                # The deals themselves, not just counts: this is the "what
                # kind of hands need more training" artifact.
                plr.dump(args.out + ".plr.json", n=args.plr_dump_top)

    print(f"\nDone: {total_hands} hands in {time.time() - t0:.0f}s. "
          f"Checkpoint: {args.out}.pt")


if __name__ == "__main__":
    main()
