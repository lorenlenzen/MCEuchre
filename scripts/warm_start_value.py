"""Warm-start a freshly initialized net's value head (+ shared trunk) against
exact ground truth, before self-play begins.

Why this helps: SubgameSolver's depth-limited CFR search reads leaf values
from the net for everything beyond the search depth. At random init those
leaf values are noise, so the very first self-play hands produce CFR targets
dominated by that noise -- training has to first discover "the value head is
garbage" before it can start improving anything. Pre-grounding the value head
(and the shared trunk/context it sits on) against real double-dummy outcomes
gives self-play a much better starting point for free, since rollout_value
(rebel/pimc.py) is an *exact* solve, not a network guess.

This reuses the exact pattern proven safe in scripts/recalibrate_value.py this
session (which fixed a real, measured value-head bias: it overestimated
post-call outcomes by +0.6 to +1.0 points on average): PURE value MSE loss,
zero policy loss, train/val split, early stopping. That discipline is what
distinguishes this from the warm_start_bidding.py attempt that regressed the
quiz score (3/6 -> 1/6) by imitating a heuristic policy with class imbalance
and no validation -- this script never touches policy targets at all, so it
cannot reproduce that failure mode.

Broadened beyond recalibrate_value.py's post-call-only sampling: with
`--deep-frac`, after reaching the post-call state, some samples continue
playing random legal actions forward a random number of plies before grounding
-- covering the value head across the whole hand depth (early/mid/late-hand
PLAY states), not just the immediate post-bid leaf, since PLAY is the large
majority of what self-play actually generates (~81-84%, measured via
ReBeLTrainer.cluster_stats() during live training). rollout_value gets cheaper
the deeper into the hand it's called (near-free with 1-2 cards left), so this
costs little extra wall-clock for a lot of extra coverage.

Only the value head + shared trunk/context get warm-started this way; the
policy heads (make_trump/play/discard/pass) stay at random init and are left
entirely to self-play + CFR, exactly as before. Suit-relabeling symmetry is
architectural (rebel/networks.py's shared per-suit/per-card towers), so it is
preserved regardless of what this script trains -- verified in "Verify"
below, not just assumed.

    python scripts/warm_start_value.py --out checkpoints/rebel_sa_warm --samples 4000
"""

import argparse
import random

import numpy as np
import torch

from euchre.actions import Call, OrderUp, Pass
from euchre.game import Phase, team_of
from euchre.infoset import observation_tensor
from rebel.networks import PolicyValueNet
from rebel.pimc import rollout_value
from rebel.train_rebel import ReBeLTrainer


def _land_post_call(helper: "ReBeLTrainer", rng: random.Random, round2_frac: float):
    """Deal, then apply a real bid (round 1 or, with round2_frac odds, a
    round-2 call reached via genuine passes) -- same construction proven in
    recalibrate_value.py's build_samples. Returns None on the rare defensive
    case a round-2 walk doesn't land on a callable state."""
    state = helper._fresh_deal()
    if rng.random() < round2_frac:
        for _ in range(4):
            state = state.apply(Pass())
        if state.phase != Phase.BID_ROUND_2:
            return None
        calls = [a for a in state.legal_actions()
                if isinstance(a, Call) and not a.alone]
        if not calls:
            return None
        return state.apply(rng.choice(calls))
    return state.apply(OrderUp(alone=False))


def build_samples(n, round2_frac, deep_frac, max_deep_plies, seed):
    helper = ReBeLTrainer(seed=seed)  # only for _fresh_deal
    rng = random.Random(seed + 1)
    out = []
    while len(out) < n:
        nxt = _land_post_call(helper, rng, round2_frac)
        if nxt is None:
            continue
        if rng.random() < deep_frac:
            plies = rng.randint(1, max_deep_plies)
            for _ in range(plies):
                if nxt.is_terminal():
                    break
                nxt = nxt.apply(rng.choice(nxt.legal_actions()))
            if nxt.is_terminal():
                continue  # rollout_value needs a decision state; skip

        v0 = rollout_value(nxt)  # exact -- all 4 hands already known
        leaf_player = nxt.current_player
        target = v0 if team_of(leaf_player) == 0 else -v0
        obs = observation_tensor(nxt, leaf_player)
        out.append((obs, target))
        if len(out) % 500 == 0:
            print(f"  ...{len(out)}/{n} samples", flush=True)
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
    ap.add_argument("--resume", type=str, default=None,
                    help="warm-start FROM an existing suit-agnostic "
                         "checkpoint instead of a fresh random init "
                         "(architecture must match the current networks.py)")
    ap.add_argument("--out", type=str, default="checkpoints/rebel_sa_warm")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--round2-frac", type=float, default=0.3)
    ap.add_argument("--deep-frac", type=float, default=0.5,
                    help="fraction of samples that keep playing random legal "
                         "actions forward past the post-call leaf before "
                         "grounding, for value coverage across the whole "
                         "hand depth rather than just the immediate leaf")
    ap.add_argument("--max-deep-plies", type=int, default=16)
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net = PolicyValueNet()
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"resumed from {args.resume}", flush=True)
    else:
        print("starting from a fresh random-init net", flush=True)

    print(f"generating {args.samples} value-grounding samples "
          f"(round2_frac={args.round2_frac}, deep_frac={args.deep_frac})...",
          flush=True)
    samples = build_samples(args.samples, args.round2_frac, args.deep_frac,
                            args.max_deep_plies, args.seed)
    rng = random.Random(args.seed + 2)
    rng.shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    val, train = samples[:n_val], samples[n_val:]
    print(f"{len(train)} train / {len(val)} val", flush=True)

    val_mse0, val_bias0 = mse_and_bias(net, val)
    print(f"before: val_mse={val_mse0:.3f} val_bias={val_bias0:+.3f}", flush=True)

    # Pure value loss -- policy heads get zero gradient from this script.
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
