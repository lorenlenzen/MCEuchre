"""Targeted value-head grounding: find which bidding-strength clusters the
current value head is most wrong about, then retrain the value head with
each cluster weighted by how wrong it currently is -- so training time
concentrates on the actual weak spots instead of a uniform pass.

This exists because manually writing quiz questions to find weaknesses (see
docs/euchre_quiz.json) doesn't scale, but training directly on a small
curated/labeled set risks exactly the "strategic collapse" failure this
repo already hit once: checkpoints/README.md documents warm_start_bidding.py
regressing quiz accuracy 3/6 -> 1/6 by imitating a heuristic *policy*
directly (hard labels, full weight, no validation). This script never
touches the policy head or a hard-label policy target -- like
recalibrate_value.py before it, it is value-only (Sample.supervise_policy is
always False here), MSE against exact rollout_value double-dummy ground
truth, with held-out validation. The only thing "targeted" adds on top of
recalibrate_value.py's blanket approach is a per-cluster loss weight, so a
weak cluster gets more effective training signal without ever overriding
what the self-play/CFR process decides the policy should do.

Clusters are the same hand-strength buckets ReBeLTrainer._cluster_key already
assigns real bidding decisions (bid1/bid2, bucketed by PointCountAgent
score) -- see ReBeLTrainer._grounded_value_sample, whose cluster_key is now
("bid1_ground"|"bid2_ground", bucket) instead of one flat ("value_ground",)
key, specifically so this kind of per-bucket diagnosis is possible.

Supports --engine cpp (mceuchre_cpp.solve_value via cpp_rollout_value, see
rebel/train_rebel.py) for substantially faster sample generation than
--engine python, now that ReBeLTrainer's value_ground_frac path itself
works under the cpp engine.

    python scripts/targeted_value_ground.py --resume checkpoints/rebel_sa.pt \
        --out checkpoints/rebel_sa_ground --engine cpp --samples 4000
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

from rebel.match_equity import MatchEquityModel
from rebel.networks import PolicyValueNet
from rebel.train_rebel import ReBeLTrainer


def generate_samples(trainer: ReBeLTrainer, n: int, round2_frac: float):
    """n grounded Sample objects (skipping the rare None from a round-2 walk
    that doesn't land on a callable state -- see _grounded_value_sample)."""
    trainer._VALUE_GROUND_ROUND2_FRAC = round2_frac
    out = []
    tries = 0
    max_tries = n * 20
    while len(out) < n and tries < max_tries:
        tries += 1
        s = trainer._grounded_value_sample()
        if s is not None:
            out.append(s)
    if len(out) < n:
        print(f"warning: only generated {len(out)}/{n} samples in {max_tries} tries",
              flush=True)
    return out


def cluster_mse(net, samples):
    """{cluster_key: (mse, count)} against the net's CURRENT value head."""
    by_key = {}
    for s in samples:
        by_key.setdefault(s.cluster_key, []).append(s)
    out = {}
    for k, group in by_key.items():
        obs = torch.from_numpy(np.stack([s.obs for s in group]))
        target = torch.tensor([s.value for s in group], dtype=torch.float32)
        with torch.no_grad():
            _, value = net(obs)
        mse = float(((value - target) ** 2).mean())
        out[k] = (mse, len(group))
    return out


def print_cluster_table(title, stats, top_n):
    rows = sorted(stats.items(), key=lambda kv: -kv[1][0])[:top_n]
    print(f"  {title}")
    print(f"    {'cluster':<22}{'count':>7}{'mse':>10}")
    for key, (mse, count) in rows:
        print(f"    {str(key):<22}{count:>7}{mse:>10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default="checkpoints/rebel_sa.pt")
    ap.add_argument("--out", type=str, default="checkpoints/rebel_sa_ground")
    ap.add_argument("--engine", choices=["python", "cpp"], default="cpp")
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--round2-frac", type=float, default=0.3,
                    help="fraction of generated samples that are round-2 "
                         "calls rather than round-1 order-ups -- matches "
                         "ReBeLTrainer._VALUE_GROUND_ROUND2_FRAC's default "
                         "over-representation of round 2 relative to its "
                         "natural ~1% self-play frequency.")
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3,
                    help="stop if val MSE hasn't improved for this many epochs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=10,
                    help="how many weakest clusters to report/track")
    ap.add_argument("--weight-power", type=float, default=1.0,
                    help="cluster loss weight = clip(mse/median_mse, "
                         "min, max) ** this. 0 disables targeting "
                         "(reduces to recalibrate_value.py's uniform pass); "
                         "higher concentrates more aggressively on the "
                         "worst clusters.")
    ap.add_argument("--min-weight", type=float, default=0.2)
    ap.add_argument("--max-weight", type=float, default=5.0)
    ap.add_argument("--match-equity-table", type=str,
                    default="rebel/match_equity_table.json")
    ap.add_argument("--no-match-equity", action="store_true")
    args = ap.parse_args()

    equity_model = None
    if not args.no_match_equity:
        if not os.path.exists(args.match_equity_table):
            print(f"error: --match-equity-table {args.match_equity_table!r} "
                  f"not found. Build it first:\n"
                  f"    python scripts/build_match_equity_table.py "
                  f"--out {args.match_equity_table}\n"
                  f"or pass --no-match-equity to run without it.")
            sys.exit(1)
        equity_model = MatchEquityModel.load(args.match_equity_table)
        print(f"match equity: on ({args.match_equity_table})", flush=True)
    else:
        print("match equity: off (--no-match-equity)", flush=True)

    net = PolicyValueNet()
    net.load_state_dict(torch.load(args.resume, map_location="cpu"))
    print(f"resumed from {args.resume}", flush=True)

    trainer = ReBeLTrainer(net=net, engine=args.engine, equity_model=equity_model,
                           seed=args.seed)

    print(f"generating {args.samples} grounded samples "
          f"(engine={args.engine}, round2_frac={args.round2_frac})...", flush=True)
    samples = generate_samples(trainer, args.samples, args.round2_frac)

    print("\nweakest clusters BEFORE training (current value head vs. exact "
          "double-dummy ground truth):", flush=True)
    before_stats = cluster_mse(net, samples)
    print_cluster_table("all generated clusters", before_stats, args.top_n)
    overall_mse_before = sum(mse * n for mse, n in before_stats.values()) / len(samples)
    print(f"  overall mse: {overall_mse_before:.3f}", flush=True)

    # Per-cluster training weight from the diagnosis above: how much worse
    # than typical is this cluster right now. Computed ONCE, before training,
    # from the same held-out-agnostic pass above (a slight optimism, like
    # recalibrate_value.py's single val split, not iteratively re-diagnosed
    # per epoch) -- simple and stable rather than a moving target.
    med_mse = sorted(mse for mse, _ in before_stats.values())[len(before_stats) // 2]
    cluster_weight = {
        k: float(np.clip((mse / max(med_mse, 1e-6)) ** args.weight_power,
                         args.min_weight, args.max_weight))
        for k, (mse, _) in before_stats.items()
    }

    rng = random.Random(args.seed + 2)
    rng.shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    val, train = samples[:n_val], samples[n_val:]
    print(f"\n{len(train)} train / {len(val)} val", flush=True)

    def batch_tensors(batch):
        obs = torch.from_numpy(np.stack([s.obs for s in batch]))
        target = torch.tensor([s.value for s in batch], dtype=torch.float32)
        weight = torch.tensor([cluster_weight.get(s.cluster_key, 1.0) for s in batch],
                              dtype=torch.float32)
        return obs, target, weight

    def val_mse():
        obs, target, _ = batch_tensors(val)
        with torch.no_grad():
            _, value = net(obs)
        return float(((value - target) ** 2).mean())

    best_val_mse = val_mse()
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    stale = 0
    # only the value head's parameters get gradients from this loss, but the
    # trunk is shared, so use a real optimizer over all params with a loss
    # that has ZERO policy term -- policy logits get no gradient signal here.
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(1, args.max_epochs + 1):
        rng.shuffle(train)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, len(train), args.batch_size):
            batch = train[i:i + args.batch_size]
            obs, target, weight = batch_tensors(batch)
            _, value = net(obs)
            per_sample = (value - target) ** 2
            loss = (per_sample * weight).sum() / weight.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1

        vmse = val_mse()
        print(f"epoch {epoch}/{args.max_epochs}: train_weighted_mse={ep_loss/n_batches:.3f} "
              f"val_mse={vmse:.3f}", flush=True)

        if vmse < best_val_mse - 1e-4:
            best_val_mse = vmse
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"no val improvement for {args.patience} epochs, stopping",
                      flush=True)
                break

    net.load_state_dict(best_state)

    print("\nweakest clusters AFTER training (best-val checkpoint):", flush=True)
    after_stats = cluster_mse(net, samples)
    print_cluster_table("all generated clusters", after_stats, args.top_n)
    overall_mse_after = sum(mse * n for mse, n in after_stats.values()) / len(samples)
    print(f"  overall mse: {overall_mse_after:.3f} (was {overall_mse_before:.3f})",
          flush=True)

    print(f"\n  {'cluster':<22}{'before':>10}{'after':>10}{'weight':>9}")
    for k, _ in sorted(before_stats.items(), key=lambda kv: -kv[1][0])[:args.top_n]:
        b = before_stats[k][0]
        a = after_stats.get(k, (float('nan'), 0))[0]
        print(f"  {str(k):<22}{b:>10.3f}{a:>10.3f}{cluster_weight.get(k, 1.0):>9.2f}")

    torch.save(net.state_dict(), args.out + ".pt")
    print(f"\nsaved to {args.out}.pt", flush=True)


if __name__ == "__main__":
    main()
