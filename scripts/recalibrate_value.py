"""Recalibrate the value head against real, grounded leaf values.

Diagnosed this session: the trained value net overestimates the outcome of
post-call states (order-up, call) by a full point on average (mean net
prediction -0.273 vs. exact ground truth -1.313 on random deals -- bias
+1.040). Since bidding's CFR targets get their leaf values from this same
net for anything beyond the real search depth, the bid_head's dramatic
over-calling (57.4% greedy call rate vs. PointCountAgent's 11.8% on the same
hands) looks like a rational response to a badly inflated value signal, not
an independently broken policy.

This retrains ONLY the value head against exact rollout_value ground truth
on post-call states -- no policy loss at all, unlike the earlier
warm_start_bidding.py attempt, whose failure was specifically in policy
imitation (class imbalance, too many epochs, no validation), not value
grounding (which trained smoothly there too). A held-out validation slice
checks the bias actually shrinks instead of trusting a fixed epoch count.

    python scripts/recalibrate_value.py --resume checkpoints/rebel_hq.pt \
        --out checkpoints/rebel_hq_valuefix --samples 2000
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

from euchre.actions import Call, OrderUp, Pass
from euchre.cards import Suit
from euchre.game import Phase, team_of
from euchre.infoset import observation_tensor
from rebel.match_equity import MatchEquityModel
from rebel.networks import PolicyValueNet
from rebel.pimc import rollout_value
from rebel.train_rebel import ReBeLTrainer


def build_samples(n, round2_frac, seed, equity_model=None):
    """(obs, value_target) pairs for post-call states -- exactly the kind of
    leaf `batch_value_fn_from_net` gets asked to score during CFR search.
    `equity_model` set: targets are match win-probability deltas at a
    realistic sampled score (unit-consistent with a match-equity-aware live
    training run); unset (default): raw point differential at a fixed 0-0
    score, exactly this function's original behavior."""
    helper = ReBeLTrainer(seed=seed, equity_model=equity_model)  # only for _fresh_deal
    rng = random.Random(seed + 1)
    out = []
    while len(out) < n:
        state = helper._fresh_deal()
        if rng.random() < round2_frac:
            # walk to round 2 via 4 genuine passes -- BID_ROUND_1 always has
            # Pass as a legal option regardless of position, so this always
            # transitions cleanly (same mechanism verified earlier this
            # session for the now-removed _seed_round2_hand).
            for _ in range(4):
                state = state.apply(Pass())
            if state.phase != Phase.BID_ROUND_2:
                continue  # defensive; should be unreachable
            calls = [a for a in state.legal_actions()
                    if isinstance(a, Call) and not a.alone]
            if not calls:
                continue  # defensive; should be unreachable
            nxt = state.apply(rng.choice(calls))
        else:
            nxt = state.apply(OrderUp(alone=False))

        # exact -- all 4 hands already known; nxt carries whatever score
        # _fresh_deal sampled (apply()/clone() preserve it).
        v0 = rollout_value(nxt, team0_score=nxt.team0_score,
                           team1_score=nxt.team1_score,
                           equity_model=equity_model)
        leaf_player = nxt.current_player
        target = v0 if team_of(leaf_player) == 0 else -v0
        obs = observation_tensor(nxt, leaf_player)
        out.append((obs, target))
    return out


def mse_and_bias(net, samples):
    obs = torch.from_numpy(np.stack([s[0] for s in samples]))
    target = torch.tensor([s[1] for s in samples], dtype=torch.float32)
    with torch.no_grad():
        _, value = net(obs)
    err = value - target
    return float((err ** 2).mean()), float(err.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default="checkpoints/rebel_hq.pt")
    ap.add_argument("--out", type=str, default="checkpoints/rebel_hq_valuefix")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--round2-frac", type=float, default=0.3)
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3,
                    help="stop if val MSE hasn't improved for this many epochs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match-equity-table", type=str,
                    default="rebel/match_equity_table.json",
                    help="path to the precomputed match-equity table; "
                         "targets become win-probability deltas at a "
                         "realistic sampled score instead of raw points, "
                         "unit-consistent with a match-equity-aware live run.")
    ap.add_argument("--no-match-equity", action="store_true",
                    help="raw point-differential targets at a fixed 0-0 "
                         "score, this script's original behavior.")
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

    print(f"generating {args.samples} post-call value samples "
          f"(round2_frac={args.round2_frac})...", flush=True)
    samples = build_samples(args.samples, args.round2_frac, args.seed,
                            equity_model=equity_model)
    rng = random.Random(args.seed + 2)
    rng.shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    val, train = samples[:n_val], samples[n_val:]
    print(f"{len(train)} train / {len(val)} val", flush=True)

    val_mse0, val_bias0 = mse_and_bias(net, val)
    print(f"before: val_mse={val_mse0:.3f} val_bias={val_bias0:+.3f}", flush=True)

    # only the value head's parameters get gradients from this loss, but the
    # trunk is shared, so use a real optimizer over all params with a loss
    # that has ZERO policy term -- policy logits get no gradient signal here.
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best_val_mse = val_mse0
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    stale = 0

    for epoch in range(1, args.max_epochs + 1):
        rng.shuffle(train)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, len(train), args.batch_size):
            batch = train[i:i + args.batch_size]
            obs = torch.from_numpy(np.stack([b[0] for b in batch]))
            target = torch.tensor([b[1] for b in batch], dtype=torch.float32)
            _, value = net(obs)
            loss = ((value - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1

        val_mse, val_bias = mse_and_bias(net, val)
        print(f"epoch {epoch}/{args.max_epochs}: train_mse={ep_loss/n_batches:.3f} "
              f"val_mse={val_mse:.3f} val_bias={val_bias:+.3f}", flush=True)

        if val_mse < best_val_mse - 1e-4:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"no val improvement for {args.patience} epochs, stopping",
                      flush=True)
                break

    net.load_state_dict(best_state)
    final_mse, final_bias = mse_and_bias(net, val)
    print(f"final (best-val checkpoint): val_mse={final_mse:.3f} "
          f"val_bias={final_bias:+.3f}  (was {val_mse0:.3f} / {val_bias0:+.3f})",
          flush=True)

    torch.save(net.state_dict(), args.out + ".pt")
    print(f"saved to {args.out}.pt", flush=True)


if __name__ == "__main__":
    main()
