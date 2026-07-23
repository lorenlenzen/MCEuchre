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
        --num-worlds 24 --cfr-iters 60 --depth-limit 6 --full-depth-cards 2 \
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


def _atomic_save(net, path, retries=20, delay=0.5):
    tmp = path + ".tmp"
    torch.save(net.state_dict(), tmp)
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
    # the buffer and optimizer state aren't checkpointed either way, so a
    # crash here is a strictly worse outcome than a skipped publish.
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


def actor_loop(actor_id, cfg, weights_path, version, samples_q, stop_flag):
    """Play hands forever, pushing samples; reload weights when the learner
    bumps the version."""
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    from rebel.train_rebel import ReBeLTrainer
    from rebel.networks import PolicyValueNet
    from rebel.belief_model import BiddingBeliefModel

    t = ReBeLTrainer(
        net=PolicyValueNet(), num_worlds=cfg["worlds"],
        cfr_iterations=cfg["iters"], depth_limit=cfg["depth"],
        bid_depth_limit=cfg["bid_depth"],
        full_depth_cards=cfg["fdc"], belief_model=BiddingBeliefModel(),
        stick_the_dealer=cfg["stick"], round2_seed_frac=cfg["round2_seed"],
        value_ground_frac=cfg["value_ground"],
        seed=1000 * actor_id + int(time.time()) % 997)
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
        for s in t.buffer:
            while not stop_flag.value:
                try:
                    samples_q.put(s, timeout=1.0)
                    break
                except queue.Full:
                    continue
        t.buffer.clear()


def _evaluate(net, hands, seed, stick_the_dealer=False):
    from rebel.train_rebel import ReBeLNetAgent
    from rebel.evaluate import evaluate, RandomAgent, RuleBasedAgent
    def agent():
        return ReBeLNetAgent(net, greedy=True)
    r = evaluate(agent, RandomAgent, hands=hands, seed=seed,
                stick_the_dealer=stick_the_dealer)
    u = evaluate(agent, RuleBasedAgent, hands=hands, seed=seed + 1,
                stick_the_dealer=stick_the_dealer)
    return r["team0_mean_point_diff"], r["team0_win_rate"], \
        u["team0_mean_point_diff"], u["team0_win_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actors", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--num-worlds", type=int, default=24)
    ap.add_argument("--cfr-iters", type=int, default=60)
    ap.add_argument("--depth-limit", type=int, default=6)
    ap.add_argument("--bid-depth-limit", type=int, default=None,
                    help="deeper depth limit for bidding-phase decisions "
                         "(BID_ROUND_1/2, DEALER_DISCARD); defaults to "
                         "--depth-limit if unset. Bidding is the furthest "
                         "any decision sits from the trainer's only exact "
                         "solves (the last --full-depth-cards tricks), so "
                         "it benefits from more real lookahead before "
                         "falling back to the value net.")
    ap.add_argument("--full-depth-cards", type=int, default=2)
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
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--out", type=str, default="rebel_par")
    args = ap.parse_args()

    from rebel.train_rebel import ReBeLTrainer
    from rebel.networks import PolicyValueNet

    net = PolicyValueNet()
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"resumed from {args.resume}", flush=True)
    # The learner reuses ReBeLTrainer purely for its buffer + train_step.
    learner = ReBeLTrainer(net=net, lr=args.lr, grad_clip_norm=args.grad_clip_norm)

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
    _atomic_save(net, weights_path)
    version = ctx.Value("i", 1)
    stop_flag = ctx.Value("i", 0)
    samples_q = ctx.Queue(maxsize=4000)
    cfg = {"worlds": args.num_worlds, "iters": args.cfr_iters,
           "depth": args.depth_limit, "bid_depth": args.bid_depth_limit,
           "fdc": args.full_depth_cards, "stick": args.stick_the_dealer,
           "round2_seed": args.round2_seed_frac,
           "value_ground": args.value_ground_frac}

    actors = [ctx.Process(target=actor_loop,
                          args=(i, cfg, weights_path, version, samples_q,
                                stop_flag))
              for i in range(args.actors)]
    for a in actors:
        a.start()
    print(f"started {args.actors} actors; running {args.minutes:.0f} min",
          flush=True)

    start = time.time()
    deadline = start + args.minutes * 60
    total = 0
    last_pub = start
    last_eval = start
    log = []
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
                _atomic_save(net, weights_path)
                version.value += 1
                last_pub = now

            if now - last_eval >= args.eval_secs and total >= args.min_buffer:
                vr, wr, vu, wu = _evaluate(net, args.eval_hands,
                                           seed=100 + len(log),
                                           stick_the_dealer=args.stick_the_dealer)
                hands_est = total // 13  # ~13 samples/hand
                top_clusters = learner.cluster_stats()
                entry = {"elapsed_s": round(now - start), "samples": total,
                         "hands_est": hands_est, "buffer": len(learner.buffer),
                         "vs_random": round(vr, 3), "win_random": round(wr, 3),
                         "vs_rule": round(vu, 3), "win_rule": round(wu, 3),
                         "top_clusters": top_clusters}
                log.append(entry)
                print(f"  {entry['elapsed_s']:>4}s | samples {total:>6} "
                      f"(~{hands_est} hands) | vs random {vr:+.3f} ({wr:.2f}) "
                      f"| vs rule {vu:+.3f} ({wu:.2f})", flush=True)
                if top_clusters:
                    tc = ", ".join(f"{r['key']}:{r['sample_share']:.0%}"
                                  for r in top_clusters)
                    print(f"       top clusters (sample share): {tc}", flush=True)
                _atomic_save(net, args.out + ".pt")
                json.dump(log, open(args.out + ".log.json", "w"), indent=2)
                last_eval = now
    finally:
        stop_flag.value = 1
        # Drain so actors blocked on put() can exit.
        t_end = time.time() + 3
        while time.time() < t_end:
            try:
                samples_q.get(timeout=0.2)
            except queue.Empty:
                break
        for a in actors:
            a.join(timeout=3)
            if a.is_alive():
                a.terminate()
        _atomic_save(net, args.out + ".pt")
        json.dump(log, open(args.out + ".log.json", "w"), indent=2)

    rate = total / max(time.time() - start, 1)
    print(f"done: {total} samples (~{total // 13} hands) in "
          f"{time.time() - start:.0f}s = {rate:.1f} samples/s; "
          f"checkpoint {args.out}.pt", flush=True)


if __name__ == "__main__":
    main()
