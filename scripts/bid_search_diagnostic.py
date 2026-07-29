"""Is the bidding SEARCH better than the policy head, and does it agree with
ground truth?

Built while diagnosing the ~60k-hand plateau, and kept because it answers the
one question the quiz alone cannot: quiz_eval.py reads the policy head
directly (no search), so a bad score there is ambiguous between "the search
produces bad targets" and "the head failed to learn good ones". Running both
side by side separates them. The measurement that started this: policy head
7/15, real search 5/15 on the buildable bid questions, at 0.98-1.00
confidence on its errors -- the head was faithfully learning a broken search,
so more training could only make it worse.

Three modes:

  --mode compare   policy head vs net-leaf search vs exact-leaf search, per
                   question. Exact-leaf is the ground-truth reference: every
                   phase-boundary leaf solved by double-dummy instead of the
                   value net.
  --mode gaps      per-root-action, CFR's value against exact double-dummy on
                   the SAME sampled worlds. This is what localises a bias to
                   one branch -- it showed alone inflated by +0.13..+0.22
                   equity while not-alone sat within +/-0.06.
  --mode leaves    per-world leaf-value discrimination: how much the net's
                   leaf estimate actually varies with the world, against how
                   much the truth varies. Shows whether leaves carry real
                   information or just a conditional mean.

    python scripts/bid_search_diagnostic.py --resume checkpoints/rebel_sa_ground.pt \
        --mode compare
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quiz_eval import build_state_any, describe, answer_action  # noqa: E402

from euchre.actions import action_to_index, index_to_action  # noqa: E402
from euchre.game import Phase, team_of  # noqa: E402
from euchre.infoset import observation_tensor  # noqa: E402
from rebel.match_equity import MatchEquityModel  # noqa: E402
from rebel.networks import PolicyValueNet  # noqa: E402
from rebel.pimc import resolve_dealer_discard, rollout_value  # noqa: E402
from rebel.subgame import SubgameSolver  # noqa: E402
from rebel.train_rebel import ReBeLTrainer, legal_mask  # noqa: E402


def load(args):
    equity_model = None
    if not args.no_match_equity:
        if not os.path.exists(args.match_equity_table):
            print(f"error: --match-equity-table {args.match_equity_table!r} not found")
            sys.exit(1)
        equity_model = MatchEquityModel.load(args.match_equity_table)
    net = PolicyValueNet()
    net.load_state_dict(torch.load(args.resume, map_location="cpu"))
    net.eval()
    trainer = ReBeLTrainer(net=net, equity_model=equity_model, seed=args.seed)
    qs = [q for q in json.load(open(args.quiz))["questions"]
          if q.get("phase") in args.phases.split(",") and q.get("buildable")]
    return equity_model, net, trainer, qs


def exact_leaf_fn(equity_model):
    """Every leaf by exact double-dummy -- all-or-nothing, matching how
    ReBeLTrainer._exact_leaf_fn is used (never mixed with net leaves inside
    one tree, which would recreate the estimator asymmetry the
    phase-boundary fix removed)."""
    memo = {}

    def fn(states):
        out = []
        for s in states:
            if s.is_terminal():
                r = s.returns()
                out.append(equity_model.equity_delta(
                    s.team0_score, s.team1_score,
                    team_of(s.dealer) == 0, r[0], r[1])
                    if equity_model is not None else float(r[0] - r[1]))
            else:
                out.append(rollout_value(s, memo, team0_score=s.team0_score,
                                         team1_score=s.team1_score,
                                         equity_model=equity_model))
        return out
    return fn


def solve(state, actor, vfn, equity_model, worlds, iters, seed):
    s = SubgameSolver(state, actor, num_worlds=worlds, iterations=iters,
                      depth_limit=6, batch_value_fn=vfn,
                      equity_model=equity_model, rng=random.Random(seed))
    s.run()
    return s


def head_policy(net, state, actor):
    with torch.no_grad():
        d = net.policy(
            torch.from_numpy(observation_tensor(state, actor)).unsqueeze(0),
            torch.from_numpy(legal_mask(state)).unsqueeze(0)).squeeze(0).numpy()
    return d


def mode_compare(args, equity_model, net, trainer, qs):
    fmt = lambda i: describe(index_to_action(i))
    print(f"{'Q':>3}  {'answer':<15}{'HEAD':<15}{'NET-leaf':<17}"
          f"{'EXACT-leaf':<17}")
    ok_h = ok_n = ok_e = 0
    for q in qs:
        st, P = build_state_any(q)
        hi = int(np.argmax(head_policy(net, st, P)))
        res = {}
        for tag, vfn in (("net", trainer._value_fn_for(P)),
                         ("exact", exact_leaf_fn(equity_model))):
            w = args.worlds if tag == "net" else args.exact_worlds
            pol = solve(st, P, vfn, equity_model, w, args.iters,
                        args.seed).root_policy()
            a = max(pol, key=pol.get)
            res[tag] = (action_to_index(a), pol[a])
        want = action_to_index(answer_action(q))
        ok_h += hi == want
        ok_n += res["net"][0] == want
        ok_e += res["exact"][0] == want
        print(f"{q['id']:>3}  {describe(answer_action(q)):<15}{fmt(hi):<15}"
              f"{fmt(res['net'][0]) + f' {res['net'][1]:.2f}':<17}"
              f"{fmt(res['exact'][0]) + f' {res['exact'][1]:.2f}':<17}")
    n = len(qs)
    print(f"\nhead {ok_h}/{n}   net-leaf search {ok_n}/{n}   "
          f"exact-leaf search {ok_e}/{n}")
    if ok_n < ok_h:
        print("\nNOTE: the search is scoring WORSE than the policy head. The "
              "head is trained to imitate this search, so its targets are the "
              "problem -- more training makes this worse, not better.")


def mode_gaps(args, equity_model, net, trainer, qs):
    for q in qs:
        st, P = build_state_any(q)
        s = solve(st, P, trainer._value_fn_for(P), equity_model,
                  args.worlds, args.iters, args.seed)
        info = s.infosets[s.root_key]
        avg, sgn = info.average(), (1.0 if team_of(P) == 0 else -1.0)
        print(f"\n=== Q{q['id']}  answer={describe(answer_action(q))}  "
              f"score={q.get('score')}  (actor-signed) ===")
        print(f"{'action':<16}{'CFR':>10}{'exact':>10}{'gap':>9}{'policy':>9}")
        for j, a in enumerate(info.actions):
            cfr = sum(w * s._expected_value(root.children[j])[0]
                      for root, w in zip(s.roots, s.weights)) * sgn
            ex, ok = 0.0, True
            for world, w in zip(s.worlds, s.weights):
                nxt = world.apply(a)
                if nxt.phase == Phase.DEALER_DISCARD:
                    nxt, _ = resolve_dealer_discard(nxt)
                if nxt.is_terminal():
                    r = nxt.returns()
                    v0 = equity_model.equity_delta(
                        world.team0_score, world.team1_score,
                        team_of(world.dealer) == 0, r[0], r[1])
                elif nxt.phase == Phase.PLAY:
                    v0 = rollout_value(nxt, team0_score=nxt.team0_score,
                                       team1_score=nxt.team1_score,
                                       equity_model=equity_model)
                else:
                    ok = False   # Pass has no double-dummy truth: its value
                    break        # depends on the others' bidding policy
                ex += w * v0
            if ok:
                ex *= sgn
                print(f"{describe(a):<16}{cfr:>+10.3f}{ex:>+10.3f}"
                      f"{cfr - ex:>+9.3f}{avg[j]:>9.2f}")
            else:
                print(f"{describe(a):<16}{cfr:>+10.3f}{'--':>10}{'--':>9}"
                      f"{avg[j]:>9.2f}")


def mode_leaves(args, equity_model, net, trainer, qs):
    print(f"{'Q':>3} {'action':<15}{'net mean':>10}{'net sd':>8}"
          f"{'exact mean':>12}{'exact sd':>10}{'corr':>7}")
    for q in qs:
        st, P = build_state_any(q)
        s = solve(st, P, trainer._value_fn_for(P), equity_model,
                  args.worlds, args.iters, args.seed)
        sgn = 1.0 if team_of(P) == 0 else -1.0
        for a in s.infosets[s.root_key].actions:
            nets, exacts = [], []
            for world in s.worlds:
                nxt = world.apply(a)
                if nxt.phase == Phase.DEALER_DISCARD:
                    nxt, _ = resolve_dealer_discard(nxt)
                if nxt.phase != Phase.PLAY or nxt.is_terminal():
                    nets = None
                    break
                nets.append(trainer._value_fn_for(P)([nxt])[0] * sgn)
                exacts.append(rollout_value(
                    nxt, team0_score=nxt.team0_score,
                    team1_score=nxt.team1_score,
                    equity_model=equity_model) * sgn)
            if not nets:
                continue
            n_, e_ = np.array(nets), np.array(exacts)
            c = (np.corrcoef(n_, e_)[0, 1]
                 if n_.std() > 1e-9 and e_.std() > 1e-9 else float("nan"))
            print(f"{q['id']:>3} {describe(a):<15}{n_.mean():>+10.3f}"
                  f"{n_.std():>8.3f}{e_.mean():>+12.3f}{e_.std():>10.3f}{c:>7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default="checkpoints/rebel_sa_ground.pt")
    ap.add_argument("--mode", choices=["compare", "gaps", "leaves"],
                    default="compare")
    ap.add_argument("--quiz", type=str, default="docs/euchre_quiz.json")
    ap.add_argument("--phases", type=str, default="bid1",
                    help="comma-separated quiz phases to include "
                         "(bid1,bid2). bid2-rooted solves are ~40x more "
                         "leaves per world, so --mode compare on them is slow.")
    ap.add_argument("--worlds", type=int, default=24)
    ap.add_argument("--exact-worlds", type=int, default=8,
                    help="worlds for exact-leaf solves (each leaf is a "
                         "double-dummy solve, not a slice of a batched "
                         "forward pass)")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match-equity-table", type=str,
                    default="rebel/match_equity_table.json")
    ap.add_argument("--no-match-equity", action="store_true")
    args = ap.parse_args()

    equity_model, net, trainer, qs = load(args)
    print(f"{args.resume}  mode={args.mode} worlds={args.worlds} "
          f"iters={args.iters}  {len(qs)} [{args.phases}] questions\n")
    {"compare": mode_compare, "gaps": mode_gaps,
     "leaves": mode_leaves}[args.mode](args, equity_model, net, trainer, qs)


if __name__ == "__main__":
    main()
