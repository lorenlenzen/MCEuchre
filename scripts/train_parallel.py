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


def _atomic_save(net, path):
    tmp = path + ".tmp"
    torch.save(net.state_dict(), tmp)
    os.replace(tmp, path)


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
        full_depth_cards=cfg["fdc"], belief_model=BiddingBeliefModel(),
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


def _evaluate(net, hands, seed):
    from rebel.train_rebel import ReBeLNetAgent
    from rebel.evaluate import evaluate, RandomAgent, RuleBasedAgent
    def agent():
        return ReBeLNetAgent(net, greedy=True)
    r = evaluate(agent, RandomAgent, hands=hands, seed=seed)
    u = evaluate(agent, RuleBasedAgent, hands=hands, seed=seed + 1)
    return r["team0_mean_point_diff"], r["team0_win_rate"], \
        u["team0_mean_point_diff"], u["team0_win_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actors", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--num-worlds", type=int, default=24)
    ap.add_argument("--cfr-iters", type=int, default=60)
    ap.add_argument("--depth-limit", type=int, default=6)
    ap.add_argument("--full-depth-cards", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-steps", type=int, default=4)
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
    learner = ReBeLTrainer(net=net, lr=args.lr)

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
           "depth": args.depth_limit, "fdc": args.full_depth_cards}

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

            if len(learner.buffer) >= args.min_buffer:
                stats = {}
                for _ in range(args.train_steps):
                    stats = learner.train_step(args.batch_size)

            now = time.time()
            if now - last_pub >= args.publish_secs:
                _atomic_save(net, weights_path)
                version.value += 1
                last_pub = now

            if now - last_eval >= args.eval_secs and total >= args.min_buffer:
                vr, wr, vu, wu = _evaluate(net, args.eval_hands,
                                           seed=100 + len(log))
                hands_est = total // 13  # ~13 samples/hand
                entry = {"elapsed_s": round(now - start), "samples": total,
                         "hands_est": hands_est, "buffer": len(learner.buffer),
                         "vs_random": round(vr, 3), "win_random": round(wr, 3),
                         "vs_rule": round(vu, 3), "win_rule": round(wu, 3)}
                log.append(entry)
                print(f"  {entry['elapsed_s']:>4}s | samples {total:>6} "
                      f"(~{hands_est} hands) | vs random {vr:+.3f} ({wr:.2f}) "
                      f"| vs rule {vu:+.3f} ({wu:.2f})", flush=True)
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
