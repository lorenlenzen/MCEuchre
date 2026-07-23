"""Warm-start the bidding heads from PointCountAgent, then let self-play
refine from there.

PointCountAgent (rebel/evaluate.py) already beats RuleBasedAgent and gets the
clear-cut bidding cases right (weak trump-heavy -> pass, bowers+ace -> alone)
that the self-play-trained net currently gets wrong on the same cases. This
clones its policy as a supervised warm-start for BID_ROUND_1/2 decisions --
cheap (no CFR solving, no persistent regret tables, no depth-limited search
at all) -- and pairs a real double-dummy value (via rollout_value, a single
exact solve, all four hands already known from a real deal) with the
heuristic's chosen action wherever it recommends a concrete call.

Pass recommendations get policy supervision only. Valuing a pass would
require solving the next player's decision -- the same problem PIMC/
PointCountAgent already sidestep by treating pass as a fixed threshold
rather than a solved value, rather than inventing a fabricated number here.

    python scripts/warm_start_bidding.py --resume checkpoints/rebel_hq.pt \
        --out checkpoints/rebel_hq_warmstart --samples 4000
"""

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from euchre.actions import NUM_ACTIONS, Pass, action_to_index
from euchre.game import Phase, team_of
from euchre.infoset import observation_tensor
from rebel.evaluate import PointCountAgent
from rebel.networks import PolicyValueNet
from rebel.pimc import rollout_value
from rebel.train_rebel import ReBeLTrainer, legal_mask

POLICY_SMOOTH = 0.9  # weight on the heuristic's chosen action; rest spread
                     # over the other legal actions -- avoids training toward
                     # the same kind of overconfident spike we're trying to
                     # move away from.


def build_samples(n, round2_frac, seed):
    """Yield (obs, mask, policy_target, value_target_or_None)."""
    helper = ReBeLTrainer(seed=seed)  # only used for _fresh_deal /
                                       # _seed_round2_hand -- no CFR here
    agent = PointCountAgent()
    rng = random.Random(seed + 1)
    out = []
    tries = 0
    while len(out) < n and tries < n * 4:
        tries += 1
        if rng.random() < round2_frac:
            state = helper._seed_round2_hand()
        else:
            state = helper._fresh_deal()  # always first-to-act, bids_seen=0

        if state.phase not in (Phase.BID_ROUND_1, Phase.BID_ROUND_2):
            continue
        legal = state.legal_actions()
        if len(legal) <= 1:
            continue
        player = state.current_player

        chosen = agent.act(state, rng)
        obs = observation_tensor(state, player)
        mask = legal_mask(state)

        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        smooth = (1.0 - POLICY_SMOOTH) / (len(legal) - 1)
        for a in legal:
            policy[action_to_index(a)] = smooth
        policy[action_to_index(chosen)] = POLICY_SMOOTH

        value = None
        if not isinstance(chosen, Pass):
            nxt = state.apply(chosen)
            v0 = rollout_value(nxt)  # exact -- all 4 hands already known
            value = v0 if team_of(player) == 0 else -v0

        out.append((obs, mask, policy, value))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default="checkpoints/rebel_hq.pt")
    ap.add_argument("--out", type=str, default="checkpoints/rebel_hq_warmstart")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--round2-frac", type=float, default=0.5,
                    help="fraction of samples seeded at BID_ROUND_2 rather "
                         "than a normal round-1 deal -- round 2 is the "
                         "specifically diagnosed weak spot, so it's "
                         "deliberately over-represented relative to its "
                         "~5%% natural frequency, not sampled proportionally.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net = PolicyValueNet()
    net.load_state_dict(torch.load(args.resume, map_location="cpu"))
    print(f"resumed from {args.resume}", flush=True)

    print(f"generating {args.samples} distillation samples "
          f"(round2_frac={args.round2_frac})...", flush=True)
    samples = build_samples(args.samples, args.round2_frac, args.seed)
    n_valued = sum(1 for s in samples if s[3] is not None)
    print(f"got {len(samples)} samples, {n_valued} with a real value target "
          f"({len(samples) - n_valued} pass-only, policy supervision only)",
          flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = random.Random(args.seed + 2)

    for epoch in range(1, args.epochs + 1):
        order = list(range(len(samples)))
        rng.shuffle(order)
        ep_policy_loss = ep_value_loss = 0.0
        n_batches = 0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            batch = [samples[j] for j in idx]

            obs = torch.from_numpy(np.stack([b[0] for b in batch]))
            mask = torch.from_numpy(np.stack([b[1] for b in batch]))
            target_p = torch.from_numpy(np.stack([b[2] for b in batch]))
            has_value = np.array([b[3] is not None for b in batch])
            target_v = torch.tensor(
                [b[3] if b[3] is not None else 0.0 for b in batch],
                dtype=torch.float32)
            value_weight = torch.from_numpy(has_value.astype(np.float32))

            logits, value = net(obs)
            logits = logits.masked_fill(~mask, float("-inf"))
            logp = F.log_softmax(logits, dim=-1)
            logp = torch.where(mask, logp, torch.zeros_like(logp))
            policy_loss = -(target_p * logp).sum(dim=-1).mean()

            per_value_loss = (value - target_v) ** 2 * value_weight
            n_valued_batch = value_weight.sum().clamp(min=1.0)
            value_loss = per_value_loss.sum() / n_valued_batch

            loss = policy_loss + value_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip_norm)
            opt.step()

            ep_policy_loss += float(policy_loss.item())
            ep_value_loss += float(value_loss.item())
            n_batches += 1

        print(f"epoch {epoch}/{args.epochs}: "
              f"policy_loss={ep_policy_loss/n_batches:.4f} "
              f"value_loss={ep_value_loss/n_batches:.4f}", flush=True)

    torch.save(net.state_dict(), args.out + ".pt")
    print(f"saved warm-started checkpoint to {args.out}.pt", flush=True)


if __name__ == "__main__":
    main()
