"""Parallel ReBeL self-play training (multiprocessing actor-learner).

Self-play is CPU-bound pure-Python CFR, so it parallelizes across cores with
processes (threads are blocked by the GIL). Architecture:

* N **actor** processes each play hands with a snapshot of the current net for
  CFR leaf values and push (obs, policy, value) samples to a shared queue.
* The **learner** (main process) pulls samples into a replay buffer, does
  gradient steps, and republishes updated weights to the actors periodically.

Speedup is ~linear in cores (minus the learner and coordination overhead): a
handful here, tens on a big server. It does NOT change per-target quality --
same worlds/depth/belief config as the single-process trainer.

    python scripts/train_parallel.py --actors 3 --minutes 30 \
        --num-worlds 24 --cfr-iters 60 --depth-limit 6 --full-depth-cards 3 \
        --resume checkpoints/rebel_hq.pt --out /path/rebel_par

Runs for --minutes (bursts fit this ephemeral container), checkpointing and
evaluating periodically. Resume with --resume to continue.
"""

import argparse
import json
import os
import queue
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

import multiprocessing as mp  # noqa: E402


def _atomic_save(state_dict, path, retries=20, delay=0.5):
    tmp = path + ".tmp"
    torch.save(state_dict, tmp)
    # os.replace is atomic on POSIX regardless of readers, but on Windows it
    # maps to MoveFileEx and can raise PermissionError (WinError 5) if an
    # actor process has `path` open via torch.load() at this exact instant --
    # Python's default file-open sharing mode doesn't include
    # FILE_SHARE_DELETE. Usually that window is only as long as one
    # torch.load call (~tens of ms), but this has twice now been observed to
    # outlast even a 2s retry budget (8 x 0.25s) -- almost certainly a
    # background AV/EDR scan grabbing the file rather than an actor, since
    # those can hold a lock for seconds under load. 20 x 0.5s = 10s covers
    # that without meaningfully affecting cadence (publishes are already
    # only every --publish-secs).
    #
    # If it's STILL locked after 10s, don't let a non-critical periodic save
    # take down a multi-hour unattended run: warn and skip this cycle rather
    # than raising. Actors just keep the previous weights (or the eval
    # checkpoint stays one cycle stale) until the next successful save --
    # the buffer isn't checkpointed either way (see resume's cold-buffer
    # note), so a crash here is a strictly worse outcome than a skipped
    # publish.
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                print(f"[_atomic_save] giving up on {path} after "
                      f"{retries} attempts ({retries * delay:.0f}s) -- "
                      f"still locked, skipping this save", flush=True)
                return
            time.sleep(delay)


def _load_resumable_log(log_path, legacy_path=None):
    """(log, elapsed_offset, samples_offset) for continuing an eval log at
    `log_path`, or ([], 0, 0) if there's nothing usable to resume.

    Schema/collision handling lives in rebel/eval_log.py (shared with
    train_scale.py); this wrapper just reads the offsets this script's own
    x-axis needs off the last entry."""
    from rebel.eval_log import load_resumable_log
    existing = load_resumable_log(log_path, ("elapsed_s", "samples"),
                                  legacy_path=legacy_path,
                                  script="train_parallel.py")
    if not existing:
        return [], 0, 0
    return existing, existing[-1]["elapsed_s"], existing[-1]["samples"]


def _shutdown_actors(actors, stop_flag, samples_q, drain_secs=2, join_secs=0.5):
    """Stop every actor process, robust to a SECOND Ctrl-C arriving while
    this is still running (an impatient response to how long this used to
    take -- see below). The unconditional terminate pass at the end always
    runs no matter what happens in the try block above it, or how many
    times it's interrupted.

    Previously this was one `for a in actors: a.join(timeout=3); if
    alive: terminate()` loop -- serial, so a large --actors count could
    take up to actors*3s of silence before the LAST actor even got a
    terminate() attempt, and a second KeyboardInterrupt during that loop
    propagated straight out, abandoning whichever actors it hadn't reached
    yet: never joined, never terminated. Exactly "not all actors stop."
    Splitting the graceful wait from the terminate pass fixes both: the
    per-actor wait budget is much shorter (bounding the silent period
    actors-many times over), and terminate() below runs for every actor
    regardless of what happened above or how many times it was
    interrupted -- see test_train_parallel.py's regression test, which
    reproduces the old failure with a fake actor list and asserts the new
    shape always terminates all of them.
    """
    stop_flag.value = 1
    try:
        t_end = time.time() + drain_secs  # let actors blocked on put() exit
        while time.time() < t_end:
            try:
                samples_q.get(timeout=0.2)
            except queue.Empty:
                break
        for a in actors:
            a.join(timeout=join_secs)
    except KeyboardInterrupt:
        pass
    for a in actors:
        if a.is_alive():
            a.terminate()
    for a in actors:
        a.join(timeout=2)


def actor_loop(actor_id, cfg, weights_path, version, samples_q, stop_flag):
    """Play hands forever, pushing samples; reload weights when the learner
    bumps the version."""
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    from rebel.train_rebel import ReBeLTrainer
    from rebel.networks import PolicyValueNet

    # Each actor loads its own MatchEquityModel from the shared path rather
    # than the parent constructing one and pickling it across the process
    # boundary -- cfg stays plain/picklable (a string, like weights_path),
    # matching how actors already load their own net from a path.
    equity_model = None
    if cfg["equity_table"] is not None:
        from rebel.match_equity import MatchEquityModel
        equity_model = MatchEquityModel.load(cfg["equity_table"])

    engine = cfg.get("engine", "python")

    t = ReBeLTrainer(
        net=PolicyValueNet(), num_worlds=cfg["worlds"],
        cfr_iterations=cfg["iters"], depth_limit=cfg["depth"],
        full_depth_cards=cfg["fdc"],
        stick_the_dealer=cfg["stick"], round2_seed_frac=cfg["round2_seed"],
        value_ground_frac=cfg["value_ground"],
        bid_exact_frac=cfg["bid_exact"], bid2_exact_frac=cfg["bid2_exact"],
        bid_exact_worlds=cfg["bid_exact_worlds"],
        play_exact_frac=cfg["play_exact"],
        play_exact_lead_only=cfg["play_exact_lead_only"],
        play_exact_worlds=cfg["play_exact_worlds"],
        belief_weight_frac=cfg["belief_weight"],
        equity_model=equity_model,
        engine=engine,
        seed=1000 * actor_id + int(time.time()) % 997)

    # PLR buffers are PER ACTOR, not shared. A shared buffer would have to
    # be a multiprocessing.Manager proxy, and every should_replay/sample
    # call would then be an IPC round-trip plus a pickle of the sampling
    # weights over up to --plr-capacity entries -- on the per-hand hot path,
    # across every actor. Independent buffers keep that cost at zero and fit
    # how actors already work (own trainer, own RNG, own weight snapshot).
    # The learner still benefits from all of them, since every actor's
    # replayed hands flow into the same sample queue.
    #
    # Consequence worth knowing: buffers live only as long as the process,
    # so a resumed run re-discovers its hard deals from scratch. Deals are
    # cheap to re-find (a few hundred hands refills a buffer) and the
    # alternative -- checkpointing 14 buffers and reconciling them on
    # resume -- costs far more than it saves.
    plr = None
    if cfg.get("plr_replay_prob", 0.0) > 0.0:
        import random as _random

        from rebel.plr import (PLRBuffer, deal_to_spec, score_samples,
                               spec_to_state)
        plr = PLRBuffer(capacity=cfg["plr_capacity"],
                        replay_prob=cfg["plr_replay_prob"],
                        temperature=cfg["plr_temperature"],
                        staleness_coef=cfg["plr_staleness_coef"],
                        rng=_random.Random(9000 + actor_id))
        base_deal = t.deal_fn or t._default_deal

        def plr_deal_fn():
            if plr.should_replay():
                return spec_to_state(plr.sample(), t)
            return base_deal()

        t.deal_fn = plr_deal_fn

    hands_played = 0
    local_v = -1
    while not stop_flag.value:
        if version.value != local_v:
            try:
                t.net.load_state_dict(torch.load(weights_path,
                                                 map_location="cpu"))
                local_v = version.value
            except Exception:
                pass
        t.buffer.clear()
        try:
            t.self_play_hand()
        except Exception:
            continue
        if plr is not None:
            try:
                plr.update(deal_to_spec(t.last_hand_state, engine),
                           score_samples(t.net, t.last_hand_samples))
            except Exception:
                pass  # scoring must never take down an actor
            hands_played += 1
            # Only actor 0 reports, and only occasionally: 14 actors each
            # logging their own buffer would drown the learner's eval lines.
            if actor_id == 0 and hands_played % 250 == 0:
                s = plr.stats()
                print(f"  [plr actor0] {s['size']}/{cfg['plr_capacity']} deals, "
                      f"{s['replayed']} replayed, score "
                      f"{s['score_min']:.2f}-{s['score_max']:.2f} "
                      f"(typical loss {s['typical']:.3f})", flush=True)
        for s in t.buffer:
            while not stop_flag.value:
                try:
                    samples_q.put(s, timeout=1.0)
                    break
                except queue.Full:
                    continue
        t.buffer.clear()


def _evaluate(net, hands, seed, stick_the_dealer=False, extra_opponents=None):
    """{name: (mean_point_diff, win_rate)} for the live net (greedy) against
    RandomAgent, RuleBasedAgent, and any `extra_opponents` factories.

    Random/rule stay as a sanity floor (did training break outright?), but
    both saturate once a net is decent -- RandomAgent has no strategy at all
    and RuleBasedAgent is a fixed, simple heuristic (a flat trump-count
    threshold, never goes alone), so a strong net's win rate against either
    plateaus near its ceiling long before the net stops improving.
    `extra_opponents` is for a frozen past checkpoint (see
    --diagnostic-checkpoint): since it's pulled from the same skill
    distribution as the net being trained, its win rate keeps discriminating
    real improvement well past the point where random/rule stop moving."""
    from rebel.train_rebel import ReBeLNetAgent
    from rebel.evaluate import evaluate, RandomAgent, RuleBasedAgent
    def agent():
        return ReBeLNetAgent(net, greedy=True)
    opponents = {"random": RandomAgent, "rule": RuleBasedAgent}
    opponents.update(extra_opponents or {})
    out = {}
    for i, (name, opp_factory) in enumerate(opponents.items()):
        r = evaluate(agent, opp_factory, hands=hands, seed=seed + i,
                    stick_the_dealer=stick_the_dealer)
        out[name] = (r["team0_mean_point_diff"], r["team0_win_rate"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actors", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--num-worlds", type=int, default=24)
    ap.add_argument("--cfr-iters", type=int, default=60)
    ap.add_argument("--depth-limit", type=int, default=6)
    ap.add_argument("--full-depth-cards", type=int, default=3)
    ap.add_argument("--stick-the-dealer", action="store_true",
                    help="force the dealer to call in round 2 instead of "
                         "letting a full pass-out misdeal the hand. Off by "
                         "default, which means self-play never trains the "
                         "forced-call decision at all.")
    ap.add_argument("--round2-seed-frac", type=float, default=0.0,
                    help="fraction of self-play hands dealt with the "
                         "up-card's suit biased weak for all four hands, "
                         "instead of a normal deal. Round 1 still plays "
                         "out for real (every player's actual decision, "
                         "every sample recorded) -- this only raises the "
                         "odds it naturally resolves all-pass into round "
                         "2, which otherwise only arises in ~5%% of hands "
                         "(measured), making it the slowest-training "
                         "bidding decision type despite being cheap to "
                         "solve. Off by default.")
    ap.add_argument("--value-ground-frac", type=float, default=0.0,
                    help="fraction of self-play hands that also generate "
                         "one extra exact rollout_value-grounded post-call "
                         "sample (in addition to the normal hand), "
                         "supervising value_loss only, never policy_loss. "
                         "A continuous, low-weight anchor against the "
                         "value head drifting the way it was found to have "
                         "this session (overestimating post-call outcomes "
                         "by roughly half a point to a full point on "
                         "average, since nothing outside the last "
                         "--full-depth-cards tricks ever checks its leaf "
                         "estimates against reality) -- see "
                         "scripts/recalibrate_value.py for the one-shot "
                         "version of the same fix. Off by default.")
    ap.add_argument("--bid-exact-frac", type=float, default=0.0,
                    help="fraction of round-1 BIDDING decisions solved with "
                         "EVERY leaf valued by exact double-dummy instead of "
                         "the value net (all-or-nothing per solve, never "
                         "mixed within one tree). Unlike --value-ground-frac, "
                         "which is value-only, these samples supervise the "
                         "bid POLICY head against ground truth -- the only "
                         "thing in the pipeline that does. Measured on the "
                         "bid1 quiz: net-leaf search 3/9, exact-leaf search "
                         "6/9, fixing every over-aggressive call. Keep it "
                         "modest (~0.05): double-dummy hands the defense "
                         "perfect information, so it leans conservative, and "
                         "the same measurement introduced three over-passive "
                         "errors. Off by default.")
    ap.add_argument("--bid2-exact-frac", type=float, default=0.0,
                    help="same as --bid-exact-frac but for round-2-rooted "
                         "decisions. Separate knob purely on cost: a bid2 "
                         "solve is ~2000 leaves/world against bid1's ~48. A "
                         "bid1 solve already contains the whole round-2 "
                         "auction internally, so --bid-exact-frac alone still "
                         "grounds round-2 reasoning. Off by default.")
    ap.add_argument("--bid-exact-worlds", type=int, default=None,
                    help="belief worlds for exact-leaf solves only (defaults "
                         "to --num-worlds). Lower is usually right: each leaf "
                         "costs a double-dummy solve rather than a slice of "
                         "one batched forward pass.")
    ap.add_argument("--play-exact-frac", type=float, default=0.0,
                    help="fraction of card-PLAY decisions solved with every "
                         "leaf valued by exact double-dummy. Aimed, not "
                         "blanket: scoring every legal card exactly over real "
                         "play decisions, only ~18%% have a genuine choice at "
                         "all, and on those the search picks the best card "
                         "90-95%% of the time from 2nd/3rd/4th seat but only "
                         "60%% when LEADING -- against a 54%% random-legal "
                         "baseline, and below the policy head's 69%%. A search "
                         "worse than the head is manufacturing bad targets, "
                         "the same signature as the bidding bug. Off by "
                         "default.")
    ap.add_argument("--play-exact-all-positions", dest="play_exact_lead_only",
                    action="store_false", default=True,
                    help="apply --play-exact-frac at every position in the "
                         "trick rather than leads only. Off by default "
                         "because 2nd/3rd/4th are already at 90-95%% and "
                         "would just pay the cost.")
    ap.add_argument("--play-exact-worlds", type=int, default=None,
                    help="belief worlds for play-exact solves only (defaults "
                         "to --num-worlds). Separate from --bid-exact-worlds "
                         "on purpose -- they used to share one knob, so "
                         "setting --bid-exact-worlds low (to keep bid solves "
                         "cheap) silently starved play-exact leads of belief "
                         "coverage too, producing high-variance targets that "
                         "never converged over a full 13-hour run.")
    ap.add_argument("--belief-weight-frac", type=float, default=0.0,
                    help="fraction of BIDDING-phase (BidRound1/BidRound2) "
                         "net-leaf solves that importance-weight sampled "
                         "worlds by the net's OWN probability for the pass "
                         "sequence actually observed getting there, instead "
                         "of sampling them uniformly (cpp.SubgameSolver's "
                         "belief_weighted; see cpp/belief.cpp's "
                         "sample_weighted_worlds and docs/rebel_design.md's "
                         "planned net-native self-play section). Replaces "
                         "the deleted rebel/belief_model.py heuristic with "
                         "the net's own belief. Unlike --bid-exact-frac this "
                         "is always cheap (one extra batched forward pass "
                         "per solve, not a double-dummy search) -- the "
                         "fraction is a rollout-risk knob, not a cost one: "
                         "an undertrained bidding policy makes for a noisy "
                         "belief signal early on. cpp engine only. Off by "
                         "default.")
    ap.add_argument("--plr-replay-prob", type=float, default=0.0,
                    help="enable Prioritized Level Replay: probability that "
                         "a self-play hand REPLAYS a stored high-loss deal "
                         "instead of dealing a fresh one (0 = off, the "
                         "default). Deals are scored by the same per-sample "
                         "loss cluster_priority already uses, giving the "
                         "per-DEAL granularity clusters can't express; "
                         "replay re-runs the real CFR solves, so targets "
                         "stay fresh rather than being stale stored ones. "
                         "Each actor keeps its OWN buffer (no cross-process "
                         "sharing -- see actor_loop) and buffers are not "
                         "checkpointed, so a resumed run re-discovers its "
                         "hard deals. This is also the distribution-shift "
                         "dial: the buffer only ever holds deals that "
                         "occurred naturally, but replaying them still "
                         "over-weights hard hands against a true uniform "
                         "shuffle. 0.3-0.5 is the usual range.")
    ap.add_argument("--plr-capacity", type=int, default=1000,
                    help="(PLR, per actor) how many scored deals each "
                         "actor's buffer holds. A fresh deal displaces the "
                         "weakest stored one only if it scores higher.")
    ap.add_argument("--plr-temperature", type=float, default=1.0,
                    help="(PLR) rank-prioritization temperature; weight is "
                         "(1/rank)^(1/T). Lower = greedier toward the "
                         "highest-loss deals.")
    ap.add_argument("--plr-staleness-coef", type=float, default=0.3,
                    help="(PLR) fraction of sampling weight given to how "
                         "long ago a deal was last replayed. Entries have "
                         "no TTL -- a stored score only refreshes when that "
                         "deal is sampled again -- so this sets how fast a "
                         "buffer re-measures itself. Measured coverage over "
                         "one buffer's worth of replays: 38%% of entries at "
                         "0.1, 58%% at 0.5.")
    ap.add_argument("--fresh-optimizer", action="store_true",
                    help="ignore the resumed checkpoint's sibling .opt.pt and "
                         "start Adam from zero state. Worth it after a change "
                         "to WHAT the value head predicts (e.g. the "
                         "actor-conditioned leaf perspective): Adam's "
                         "second-moment estimates are a preconditioner fitted "
                         "to the old gradient distribution, and pairing a "
                         "stale, small variance estimate with the large "
                         "gradients a newly-seen input region produces gives "
                         "oversized steps exactly when the net is least "
                         "stable. Cheap insurance -- the moments rebuild in "
                         "~1000 steps (beta2=0.999), noise against a "
                         "multi-hour run.")
    ap.add_argument("--match-equity-table", type=str,
                    default="rebel/match_equity_table.json",
                    help="path to the precomputed match-equity table (see "
                         "scripts/build_match_equity_table.py). When loaded, "
                         "self-play samples a realistic starting match score "
                         "per hand (euchre/game.py's team0_score/team1_score) "
                         "and CFR's own terminal utilities become equity-"
                         "aware, not just the value head's regression target "
                         "-- see rebel/match_equity.py and docs/rebel_design.md.")
    ap.add_argument("--no-match-equity", action="store_true",
                    help="disable match-equity awareness entirely -- every "
                         "hand trains raw point-differential targets at a "
                         "fixed 0-0 score, the behavior before this feature.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0,
                    help="cap the gradient norm of any single train_step; a "
                         "safety net against an outsized update from an "
                         "occasional high-loss batch, not a tuning knob.")
    ap.add_argument("--samples-per-step", type=float, default=32.0,
                    help="target new samples generated per training step; "
                         "controls the replay ratio (= batch-size / "
                         "samples-per-step, ~4x by default). Replaces a "
                         "flat --train-steps-every-cycle, which at low "
                         "actor throughput was resampling each buffer "
                         "entry ~hundreds of times before eviction.")
    ap.add_argument("--max-train-steps", type=int, default=8,
                    help="safety cap on training steps taken in a single "
                         "main-loop cycle, in case a burst of samples "
                         "arrives at once.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--min-buffer", type=int, default=256)
    ap.add_argument("--publish-secs", type=float, default=20.0)
    ap.add_argument("--eval-secs", type=float, default=120.0)
    ap.add_argument("--eval-hands", type=int, default=200)
    ap.add_argument("--diagnostic-checkpoint", type=str, default=None,
                    help="also evaluate the live net (greedy) against a "
                         "FROZEN net loaded once from this checkpoint, "
                         "logged as vs_diagnostic/win_diagnostic. Random and "
                         "rule are a sanity floor, not a progress signal --  "
                         "both saturate once a net clears their (low, fixed) "
                         "ceiling, e.g. RuleBasedAgent never goes alone. A "
                         "past checkpoint from the same skill distribution "
                         "keeps discriminating real improvement well past "
                         "that point. Point this at a stable snapshot (e.g. "
                         "a copy you don't keep training), not the file "
                         "you're actively resuming from -- that one keeps "
                         "moving underneath you.")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--out", type=str, default="rebel_par")
    ap.add_argument("--engine", choices=["python", "cpp"], default="python",
                    help="actor self-play engine. 'cpp' uses the compiled "
                         "hot-path port (cpp/, see cpp/README.md; must be "
                         "built first: python setup.py build_ext --inplace) "
                         "for the engine/observation/solver/CFR-search hot "
                         "loop -- differentially verified bit-for-bit "
                         "against the Python path in "
                         "tests/test_cpp_equivalence.py. --value-ground-frac and "
                         "--round2-seed-frac both work fine with --engine "
                         "cpp -- value_ground_frac's cpp path uses "
                         "cpp_rollout_value (rebel/train_rebel.py), a "
                         "Python-level mirror built on the already-bound "
                         "mceuchre_cpp.solve_value, not a new C++ port. The "
                         "learner process (this one) always uses the Python "
                         "engine regardless -- it only owns the "
                         "buffer/train_step, never self-play.")
    args = ap.parse_args()

    from rebel.train_rebel import ReBeLTrainer, ReBeLNetAgent
    from rebel.networks import PolicyValueNet

    print(f"actor engine: {args.engine}", flush=True)

    # Loaded ONCE, frozen for the whole run -- re-loading it every eval would
    # just re-read the same file, but binding it here (rather than inside
    # _evaluate) makes it explicit that this net never changes, unlike `net`.
    extra_opponents = {}
    if args.diagnostic_checkpoint:
        diag_net = PolicyValueNet()
        diag_net.load_state_dict(torch.load(args.diagnostic_checkpoint,
                                            map_location="cpu"))
        diag_net.eval()
        extra_opponents["diagnostic"] = lambda: ReBeLNetAgent(diag_net, greedy=True)
        print(f"diagnostic opponent: {args.diagnostic_checkpoint} (frozen)",
              flush=True)

    equity_table_path = None
    if not args.no_match_equity:
        equity_table_path = args.match_equity_table
        if not os.path.exists(equity_table_path):
            print(f"error: --match-equity-table {equity_table_path!r} not "
                  f"found. Build it first:\n"
                  f"    python scripts/build_match_equity_table.py "
                  f"--out {equity_table_path}\n"
                  f"or pass --no-match-equity to train without it.",
                  flush=True)
            sys.exit(1)
        print(f"match equity: on ({equity_table_path})", flush=True)
    else:
        print("match equity: off (--no-match-equity)", flush=True)

    net = PolicyValueNet()
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"resumed from {args.resume}", flush=True)
    # The learner reuses ReBeLTrainer purely for its buffer + train_step.
    learner = ReBeLTrainer(net=net, lr=args.lr, grad_clip_norm=args.grad_clip_norm)
    if args.resume:
        # Adam's per-parameter momentum/variance state is checkpointed
        # alongside the weights (a sibling <resume-without-.pt>.opt.pt file,
        # not embedded in the same file, so every other loader of a plain
        # .pt checkpoint -- quiz_eval.py, export_web_model.py, the
        # recalibration scripts -- is unaffected). Without this, every
        # resume restarted Adam from zero state, which is known to produce
        # less-calibrated, noisier early updates until those running
        # averages re-stabilize -- on top of the replay buffer itself also
        # starting cold (never checkpointed; that part is unavoidable
        # without saving the buffer too, not attempted here). Missing
        # sidecar (older checkpoints, or ones produced by a script that
        # never had an optimizer) just means starting Adam fresh, same as
        # before this existed.
        opt_path = os.path.splitext(args.resume)[0] + ".opt.pt"
        if args.fresh_optimizer:
            print(f"ignoring optimizer state at {opt_path} "
                  f"(--fresh-optimizer)", flush=True)
        elif os.path.exists(opt_path):
            learner.opt.load_state_dict(torch.load(opt_path, map_location="cpu"))
            print(f"resumed optimizer state from {opt_path}", flush=True)
        else:
            print(f"no optimizer state at {opt_path} -- starting Adam fresh",
                  flush=True)

    # 'fork' is fast/low-overhead but Linux-only; Windows has only 'spawn',
    # and 'fork' is unsafe with torch on macOS -- so use fork only on Linux.
    # MCEUCHRE_MP_METHOD overrides (e.g. force 'spawn').
    forced = os.environ.get("MCEUCHRE_MP_METHOD")
    if forced:
        ctx = mp.get_context(forced)
    elif sys.platform.startswith("linux") and "fork" in mp.get_all_start_methods():
        ctx = mp.get_context("fork")
    else:
        ctx = mp.get_context("spawn")
    print(f"multiprocessing start method: {ctx.get_start_method()}", flush=True)
    weights_path = args.out + ".weights.pt"
    _atomic_save(net.state_dict(), weights_path)
    version = ctx.Value("i", 1)
    stop_flag = ctx.Value("i", 0)
    samples_q = ctx.Queue(maxsize=4000)
    cfg = {"worlds": args.num_worlds, "iters": args.cfr_iters,
           "depth": args.depth_limit,
           "fdc": args.full_depth_cards, "stick": args.stick_the_dealer,
           "round2_seed": args.round2_seed_frac,
           "value_ground": args.value_ground_frac,
           "bid_exact": args.bid_exact_frac,
           "bid2_exact": args.bid2_exact_frac,
           "bid_exact_worlds": args.bid_exact_worlds,
           "play_exact": args.play_exact_frac,
           "play_exact_lead_only": args.play_exact_lead_only,
           "play_exact_worlds": args.play_exact_worlds,
           "belief_weight": args.belief_weight_frac,
           "plr_replay_prob": args.plr_replay_prob,
           "plr_capacity": args.plr_capacity,
           "plr_temperature": args.plr_temperature,
           "plr_staleness_coef": args.plr_staleness_coef,
           "equity_table": equity_table_path,
           "engine": args.engine}

    actors = [ctx.Process(target=actor_loop,
                          args=(i, cfg, weights_path, version, samples_q,
                                stop_flag))
              for i in range(args.actors)]
    for a in actors:
        a.start()
    if args.plr_replay_prob > 0.0:
        print(f"PLR: on (replay_prob={args.plr_replay_prob}, "
              f"capacity={args.plr_capacity}/actor, T={args.plr_temperature}, "
              f"staleness={args.plr_staleness_coef}); per-actor buffers, "
              f"not checkpointed", flush=True)
    print(f"started {args.actors} actors; running {args.minutes:.0f} min",
          flush=True)

    start = time.time()
    deadline = start + args.minutes * 60
    total = 0
    last_pub = start
    last_eval = start
    # Resuming (--resume, same --out) restarts this process's own clocks/
    # counters from zero, but the eval trend only means something plotted
    # across the whole training history -- so if a log from a prior run of
    # this --out is on disk, keep appending to it instead of overwriting,
    # carrying its last elapsed_s/samples forward as an offset so the new
    # entries' x-axis stays continuous instead of jumping back to 0.
    from rebel.eval_log import legacy_log_path, log_path_for
    log_path = log_path_for(args.out, "parallel")
    log, elapsed_offset, samples_offset = _load_resumable_log(
        log_path, legacy_path=legacy_log_path(args.out))
    # Accumulates new samples between training steps. Steps are taken at a
    # rate proportional to fresh data (samples_per_step) rather than a fixed
    # count every cycle -- at low actor throughput, a flat step count per
    # ~0.5s loop tick was resampling each buffer entry hundreds of times
    # before eviction (e.g. 4 steps x 128 batch every ~0.5s against ~3.5
    # new samples/s measured in practice is a ~290x replay ratio), which
    # trains heavily on a small, constantly-stale window instead of
    # tracking fresh CFR targets as the net (and thus the targets
    # themselves) moves.
    step_credit = 0.0
    try:
        while time.time() < deadline:
            drained = 0
            try:
                while drained < 512:
                    s = samples_q.get(timeout=0.5)
                    learner._store(s)
                    total += 1
                    drained += 1
            except queue.Empty:
                pass
            step_credit += drained

            if len(learner.buffer) >= args.min_buffer:
                stats = {}
                steps = min(int(step_credit // args.samples_per_step),
                           args.max_train_steps)
                step_credit -= steps * args.samples_per_step
                for _ in range(steps):
                    stats = learner.train_step(args.batch_size)

            now = time.time()
            if now - last_pub >= args.publish_secs:
                _atomic_save(net.state_dict(), weights_path)
                version.value += 1
                last_pub = now

            if now - last_eval >= args.eval_secs and total >= args.min_buffer:
                results = _evaluate(net, args.eval_hands, seed=100 + len(log),
                                    stick_the_dealer=args.stick_the_dealer,
                                    extra_opponents=extra_opponents)
                cum_samples = samples_offset + total
                hands_est = cum_samples // 13  # ~13 samples/hand
                entry = {"elapsed_s": elapsed_offset + round(now - start),
                         "samples": cum_samples,
                         "hands_est": hands_est, "buffer": len(learner.buffer)}
                for name, (diff, win) in results.items():
                    entry[f"vs_{name}"] = round(diff, 3)
                    entry[f"win_{name}"] = round(win, 3)
                log.append(entry)
                vr, wr = results["random"]
                vu, wu = results["rule"]
                line = (f"  {entry['elapsed_s']:>4}s | samples {total:>6} "
                       f"(~{hands_est} hands) | vs random {vr:+.3f} ({wr:.2f}) "
                       f"| vs rule {vu:+.3f} ({wu:.2f})")
                if "diagnostic" in results:
                    vd, wd = results["diagnostic"]
                    line += f" | vs diagnostic {vd:+.3f} ({wd:.2f})"
                print(line, flush=True)
                _atomic_save(net.state_dict(), args.out + ".pt")
                _atomic_save(learner.opt.state_dict(), args.out + ".opt.pt")
                json.dump(log, open(log_path, "w"), indent=2)
                last_eval = now
    finally:
        _shutdown_actors(actors, stop_flag, samples_q)
        _atomic_save(net.state_dict(), args.out + ".pt")
        _atomic_save(learner.opt.state_dict(), args.out + ".opt.pt")
        json.dump(log, open(log_path, "w"), indent=2)

    rate = total / max(time.time() - start, 1)
    print(f"done: {total} samples (~{total // 13} hands) in "
          f"{time.time() - start:.0f}s = {rate:.1f} samples/s; "
          f"checkpoint {args.out}.pt", flush=True)


if __name__ == "__main__":
    main()
