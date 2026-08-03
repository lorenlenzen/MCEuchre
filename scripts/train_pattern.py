"""Correct a *pattern* of positions, not one hand.

The tempting move when the agent misplays a specific hand is to train on that
hand. Don't. This net is suit-agnostic with shared per-suit/per-card towers on
a shared trunk, so there is no such thing as a local update -- one position's
gradient moves weights serving every position. The repo has already paid for
this lesson twice: warm_start_bidding.py took the quiz 3/6 -> 1/6 by imitating
a heuristic policy on specific positions, and this session
targeted_value_ground.py cost 3 quiz questions (12/27 -> 9/27) from trunk
drift alone, over 20,000 samples with a held-out split and early stopping.

So this script takes the flagged hand only as a *specification*: it reads off
which hand-strength cluster the position belongs to (the same
ReBeLTrainer._cluster_key bucket prioritized replay already uses), then
generates many FRESH deals landing in that same cluster and trains on those.
Many distinct positions sharing the flagged one's structure, rather than one
position repeated -- corrective without memorizing.

Targets come from an exact-leaf CFR solve: every phase-boundary leaf valued by
double-dummy instead of the value net, all-or-nothing per solve (see
ReBeLTrainer._exact_leaf_fn). That matters because it makes the target
non-circular -- unlike ordinary self-play targets, it cannot inherit the value
head's own bias, which is the whole failure this session traced. It also means
the POLICY target is trustworthy here, so unlike recalibrate_value.py /
targeted_value_ground.py this script does train the policy head.

Three guardrails, all on by default:
  * the trunk is frozen (--no-freeze-trunk to disable), so only output heads
    move -- see PolicyValueNet.head_parameters;
  * a held-out split with early stopping on the pattern's own loss;
  * a quiz score before and after, because the pattern's loss going down tells
    you nothing about what the update cost everywhere else. That check is what
    caught the regression above.

    python scripts/train_pattern.py --resume checkpoints/rebel_sa_ground.pt \
        --out checkpoints/rebel_sa_pat --quiz-id 28 --samples 300 --engine cpp
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quiz_eval import build_state_any, score_net  # noqa: E402

from euchre.actions import NUM_ACTIONS, Pass, action_to_index  # noqa: E402
from euchre.cards import (Card, Rank, Suit, SUITS, effective_suit,  # noqa: E402
                          same_color_suit)
from euchre.game import EuchreState, Phase, team_of  # noqa: E402
from euchre.infoset import observation_tensor  # noqa: E402
from rebel.match_equity import MatchEquityModel  # noqa: E402
from rebel.networks import PolicyValueNet  # noqa: E402
from rebel.subgame import SubgameSolver  # noqa: E402
from rebel.train_rebel import (ReBeLTrainer, Sample, cpp_legal_mask,  # noqa: E402
                               legal_mask)


def parse_cluster(text):
    """"bid1:4" -> ("bid1", 4)."""
    phase, _, bucket = text.partition(":")
    if phase not in ("bid1", "bid2") or not bucket.isdigit():
        raise ValueError(f"--cluster must look like bid1:4, got {text!r}")
    return (phase, int(bucket))


def cluster_of_quiz_question(trainer, quiz_path, qid):
    q = next((q for q in json.load(open(quiz_path))["questions"]
              if q["id"] == qid), None)
    if q is None:
        raise SystemExit(f"no quiz question with id {qid}")
    if q.get("phase") not in ("bid1", "bid2"):
        raise SystemExit(f"Q{qid} is phase {q.get('phase')!r}; this script "
                         f"handles bid1/bid2 patterns only (those are the "
                         f"decisions exact-leaf solves are affordable for)")
    state, actor = build_state_any(q)
    # _cluster_key wants this trainer's engine's state type; the quiz builder
    # produces Python states, so read the bucket with a Python-engine helper
    # regardless of which engine will generate samples. The bucket is a pure
    # function of the hand and up-card, so it transfers.
    key = ReBeLTrainer(seed=0)._cluster_key(state, actor)
    return (key[0], key[1]), q


_RANKS = {"9": Rank.NINE, "T": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN,
          "K": Rank.KING, "A": Rank.ACE}
# The four suit-relative labels a --require token's suit half can use,
# instead of an absolute suit -- the net is suit-agnostic (see this module's
# docstring), so a structural pattern like "right + left bower" should be
# expressible without pinning it to a physical suit. U is the up-card's own
# suit; N is the other suit of the same color (holds the left bower); G/g are
# the two off-color suits, kept distinct (rather than one shared letter) so a
# pattern can ask for one card from each. Case matters only for G vs g.
_RELSUITS = ("U", "N", "G", "g")
_ALL_CARDS = [Card(s, r) for s in Suit for r in Rank]


def parse_require(text):
    """"JU,J*,*g" -> a list of (token, rank_char, relsuit_char) patterns.

    Each token is <rank><relsuit>, either half of which may be `*`: `JU` is
    the jack of the up-card's suit (the right bower), `J*` any jack, `*g` any
    card of the second off-color suit, `**` any card. Patterns must match
    DISTINCT cards, so `J*,J*` means two different jacks. Suits are resolved
    to physical cards per-deal by ``_relsuit_map`` once the up-card's suit is
    chosen, since U/N/G/g name a role relative to the up-card, not a fixed
    suit.
    """
    out = []
    for raw in (t.strip() for t in text.split(",") if t.strip()):
        if len(raw) != 2:
            raise ValueError(f"--require token {raw!r} must be 2 characters "
                             f"(<rank><relsuit>), e.g. JU, J*, *g, **")
        r, s = raw[0].upper(), raw[1]
        if r != "*" and r not in _RANKS:
            raise ValueError(f"bad rank {r!r} in {raw!r}; use one of "
                             f"{''.join(_RANKS)} or *")
        if s.upper() in ("U", "N"):
            s = s.upper()
        elif s not in ("G", "g", "*"):
            raise ValueError(f"bad suit {s!r} in {raw!r}; use one of "
                             f"U, N, G, g or *")
        out.append((f"{r}{s}", r, s))
    if len(out) > 5:
        raise ValueError(f"--require has {len(out)} patterns but a hand holds "
                         f"only 5 cards")
    return out


def relsuit_map(up_suit, rng):
    """Map U/N/G/g to physical suits for one deal, given the up-card's suit.

    U is the up-card's suit and N (same color, holds the left bower) follows
    from it deterministically; the two off-color suits are functionally
    interchangeable, so which one is G vs g is randomized per deal rather
    than fixed, so a pattern using both (e.g. one green ace each) doesn't
    always land on the same physical pair of suits.
    """
    next_suit = same_color_suit(up_suit)
    greens = [s for s in SUITS if s not in (up_suit, next_suit)]
    rng.shuffle(greens)
    return {"U": up_suit, "N": next_suit, "G": greens[0], "g": greens[1]}


def resolve_patterns(patterns, suit_map):
    """[(tok, rank_char, relsuit_char), ...] -> [(tok, candidate_cards), ...]
    for one deal's concrete U/N/G/g -> suit assignment."""
    out = []
    for tok, r, s in patterns:
        cands = [c for c in _ALL_CARDS
                 if (r == "*" or c.rank == _RANKS[r])
                 and (s == "*" or c.suit == suit_map[s])]
        out.append((tok, cands))
    return out


def choose_required(patterns, rng, attempts=20):
    """One distinct card per pattern, or None if they can't all be satisfied.

    Greedy over the most-constrained pattern first, which for hand-sized
    problems (<=5 patterns over 24 cards) essentially always succeeds; the
    retry loop covers the rare unlucky ordering rather than doing real
    matching."""
    for _ in range(attempts):
        chosen, used = [], set()
        for _tok, cands in sorted(patterns, key=lambda p: len(p[1])):
            free = [c for c in cands if c not in used]
            if not free:
                break
            pick = rng.choice(free)
            used.add(pick)
            chosen.append(pick)
        if len(chosen) == len(patterns):
            return chosen
    return None


def parse_score(text):
    """"9,6" -> (9, 6), read as (my score, their score) from the ACTING
    player's side, not team0's."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"--score must look like 9,6 (mine,theirs), "
                         f"got {text!r}")
    return int(parts[0]), int(parts[1])


def parse_up_rank(text):
    """"J" or "T,J,Q" -> a set of Ranks the up card is allowed to be.

    The up card's suit is always U (--require's relative suits are defined
    relative to it); this pins its rank the same way --require pins hand
    cards. Comma-separated for a set rather than one rank so e.g. "any
    bower-adjacent up-card" (T,J,Q) is expressible.
    """
    out = set()
    for tok in (t.strip().upper() for t in text.split(",") if t.strip()):
        if tok not in _RANKS:
            raise ValueError(f"--up-rank token {tok!r} must be one of "
                             f"{''.join(_RANKS)}")
        out.add(_RANKS[tok])
    return out


def _forbidden_for_void_up(up):
    """Cards the acting hand may not hold, to be void in the up-card's suit.

    The only void that matters for a round-1 decision: void there means
    ordering up leaves you with ZERO trump. Evaluated by EFFECTIVE suit with
    the up-card's suit as trump, so it excludes the left bower too -- it is a
    trump void, not merely an absence of that printed suit.
    """
    return {c for c in _ALL_CARDS if effective_suit(c, up.suit) == up.suit}


def constrained_deal(trainer, patterns, phase, rng, void_up=False, score=None,
                     up_ranks=None):
    """A deal in which the seat about to act holds a card for every pattern,
    is void in the up-card's suit if asked, and (optionally) faces a pinned
    match score.

    Placed rather than waited for: rejection-sampling a tight structural
    pattern is hopeless at this scale -- all four jacks is 0.047% of hands per
    seat, ~640,000 draws for 300 samples -- and it also isn't what you want,
    since every accepted deal would still be one arbitrary deal. Here the
    constrained parts are fixed and *everything else* varies: the rest of the
    hand, the up-card's rank (unless --up-rank pins it), the dealer, all
    three opponents' hands, and the score unless pinned. Returns
    (state, passes_to_apply), or None if the draw failed.

    `patterns` uses --require's U/N/G/g relative suits, which name a role
    relative to the up-card rather than a fixed suit -- so, unlike the old
    absolute-suit scheme, the up-card's own SUIT has to be chosen before the
    patterns can be resolved to physical cards at all (its rank is drawn
    after, same as before, from `up_ranks` or any rank not already placed).
    That suit choice is retried along with everything else: a bad one (e.g.
    --up-rank exhausted by --require cards of that same suit) is a property
    of the whole draw, not something fixable in place. Retries a bounded
    number of times rather than looping forever; the caller counts failures
    against --max-tries.
    """
    passes = rng.randint(0, 3) if phase == "bid1" else 4 + rng.randint(0, 3)
    actor = rng.randint(0, 3)
    # Seat the dealer so `actor` is the one to act after exactly `passes`
    # passes: bidding opens at dealer+1 and round 2 reopens there too.
    dealer = (actor - 1 - passes) % 4

    hands = up = kitty = None
    for _ in range(40):
        up_suit = rng.choice(SUITS)
        suit_map = relsuit_map(up_suit, rng)
        resolved = resolve_patterns(patterns, suit_map) if patterns else []
        required = choose_required(resolved, rng) if resolved else []
        if required is None:
            continue
        req = set(required)

        rank_pool = [rk for rk in (up_ranks if up_ranks else list(_RANKS.values()))
                    if Card(up_suit, rk) not in req]
        if not rank_pool:
            continue                      # --up-rank exhausted by --require
        cand_up = Card(up_suit, rng.choice(rank_pool))

        forbidden = _forbidden_for_void_up(cand_up) if void_up else set()
        if req & forbidden:
            continue                      # required card violates the void
        pool = [c for c in _ALL_CARDS if c != cand_up and c not in req]
        allowed = [c for c in pool if c not in forbidden]
        need = 5 - len(required)
        if len(allowed) < need:
            continue
        rng.shuffle(allowed)
        hand = list(required) + allowed[:need]
        rest = [c for c in pool if c not in set(hand)]
        rng.shuffle(rest)
        hands = [None] * 4
        hands[actor] = hand
        i = 0
        for seat in range(4):
            if seat == actor:
                continue
            hands[seat] = rest[i:i + 5]
            i += 5
        up, kitty = cand_up, rest[i:i + 3]
        break
    if hands is None:
        return None

    team0_score = team1_score = 0
    if score is not None:
        mine, theirs = score
        team0_score, team1_score = ((mine, theirs) if team_of(actor) == 0
                                    else (theirs, mine))
    elif trainer.equity_model is not None:
        dealer_score, other_score = trainer.equity_model.sample_score(rng)
        if team_of(dealer) == 0:
            team0_score, team1_score = dealer_score, other_score
        else:
            team0_score, team1_score = other_score, dealer_score

    if trainer.engine == "cpp":
        cpp = trainer._cpp
        bitmasks = [sum(1 << c.id for c in h) for h in hands]
        state = cpp.EuchreState.new_hand(
            dealer=dealer, stick_the_dealer=trainer.stick_the_dealer,
            team0_score=team0_score, team1_score=team1_score
        # hands are bitmasks (Hand), kitty is a plain vector of CardIds
        ).deal_from(bitmasks, up.id, [c.id for c in kitty])
    else:
        state = EuchreState.new_hand(
            dealer=dealer, stick_the_dealer=trainer.stick_the_dealer,
            team0_score=team0_score, team1_score=team1_score
        ).deal_from(hands, up, kitty)
    return state, passes


def apply_passes(trainer, state, passes):
    cpp = trainer._cpp
    is_pass = ((lambda a: a.kind == cpp.ActionKind.Pass) if cpp is not None
               else (lambda a: isinstance(a, Pass)))
    for _ in range(passes):
        p = next((a for a in state.legal_actions() if is_pass(a)), None)
        if p is None:
            return None          # stick-the-dealer: passing isn't legal here
        state = state.apply(p)
        if state.is_terminal():
            return None          # misdeal
    return state


def walk_to_phase(trainer, state, phase, rng):
    """A fresh deal advanced to a random seat's bid1 (0-3 passes) or bid2
    decision (4 passes, then 0-3 more). Varying the seat matters: bidding
    opens left of the dealer, so always acting immediately would sample one
    seat out of four and, at a post-call leaf, the one seat that happens to
    also be the opening leader."""
    cpp = trainer._cpp
    if cpp is not None:
        is_pass = lambda a: a.kind == cpp.ActionKind.Pass
        want = cpp.Phase.BidRound1 if phase == "bid1" else cpp.Phase.BidRound2
    else:
        is_pass = lambda a: isinstance(a, Pass)
        want = Phase.BID_ROUND_1 if phase == "bid1" else Phase.BID_ROUND_2

    for _ in range(rng.randint(0, 3) if phase == "bid1" else 4 + rng.randint(0, 3)):
        p = next((a for a in state.legal_actions() if is_pass(a)), None)
        if p is None:
            return None          # stick-the-dealer: passing isn't legal here
        state = state.apply(p)
        if state.is_terminal():
            return None          # misdeal
    return state if state.phase == want else None


def generate(trainer, cluster, n, worlds, iters, max_tries, rng,
             patterns=None, phase="bid1", void_up=False, score=None,
             up_ranks=None):
    """n exact-leaf-solved Samples matching the requested pattern.

    Two selection modes. `cluster` rejection-samples until the deal lands in a
    hand-strength bucket -- fine for buckets that occur several percent of the
    time. `patterns` (--require) instead *places* the required cards, which is
    the only workable route for structural patterns: they don't correspond to
    a single cluster at all (all four jacks spreads across buckets 5-8
    depending on the up-card suit) and they're far too rare to wait for.
    """
    out, tries, t0 = [], 0, time.time()
    while len(out) < n and tries < max_tries:
        tries += 1
        if patterns is not None:
            drawn = constrained_deal(trainer, patterns, phase, rng,
                                     void_up=void_up, score=score,
                                     up_ranks=up_ranks)
            if drawn is None:
                continue
            state = apply_passes(trainer, *drawn)
        else:
            state = walk_to_phase(trainer, trainer._fresh_deal(), cluster[0], rng)
        if state is None:
            continue
        actor = state.current_player
        if patterns is None:
            key = trainer._cluster_key(state, actor)
            if (key[0], key[1]) != cluster:
                continue
        solver = (trainer._cpp.SubgameSolver(
                      state, actor, worlds, iters, trainer.depth_limit,
                      trainer._exact_leaf_fn(), trainer._cpp_equity_model,
                      rng.getrandbits(63))
                  if trainer.engine == "cpp" else
                  SubgameSolver(state, actor, num_worlds=worlds,
                                iterations=iters,
                                depth_limit=trainer.depth_limit,
                                batch_value_fn=trainer._exact_leaf_fn(),
                                equity_model=trainer.equity_model, rng=rng))
        solver.run()
        policy = solver.root_policy()
        target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if trainer.engine == "cpp":
            for idx, p in policy.items():
                target[idx] = p
            obs = np.asarray(trainer._cpp.observation_tensor(state, actor))
            mask = cpp_legal_mask(state)
        else:
            for a, p in policy.items():
                target[action_to_index(a)] = p
            obs = observation_tensor(state, actor)
            mask = legal_mask(state)
        root_val = solver.root_value()
        out.append(Sample(
            obs=obs, mask=mask, policy=target,
            value=root_val if team_of(actor) == 0 else -root_val,
            cluster_key=((trainer._cluster_key(state, actor)[:2] + ("exact",))
                         if patterns is not None else cluster + ("exact",))))
        if len(out) % 25 == 0:
            print(f"  ...{len(out)}/{n} samples "
                  f"({tries} deals scanned, {time.time() - t0:.0f}s)", flush=True)
    return out, tries


def batch_loss(net, batch):
    obs = torch.from_numpy(np.stack([s.obs for s in batch]))
    mask = torch.from_numpy(np.stack([s.mask for s in batch]))
    tgt_p = torch.from_numpy(np.stack([s.policy for s in batch]))
    tgt_v = torch.tensor([s.value for s in batch], dtype=torch.float32)
    logits, value = net(obs)
    logp = F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    # Zero the illegal entries so the target's 0 * (-inf) doesn't become NaN
    # -- same guard as ReBeLTrainer.train_step, and omitting it here made
    # every policy_loss NaN while the value loss looked perfectly healthy.
    logp = torch.where(mask, logp, torch.zeros_like(logp))
    policy_loss = -(tgt_p * logp).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, tgt_v)
    return policy_loss, value_loss


def quiz_score(net, quiz_path):
    """Collateral check: what did this update cost everywhere else? The
    pattern's own loss going down says nothing about that, and this is the
    check that caught targeted_value_ground.py's 12/27 -> 9/27 regression.

    Delegates to quiz_eval.score_net so this number is the same one
    scripts/quiz_eval.py prints -- a collateral check that disagreed with the
    tool you actually read would be worse than none."""
    return score_net(net, json.load(open(quiz_path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--quiz-id", type=int, default=None,
                    help="take the target pattern from this quiz question's "
                         "hand-strength cluster")
    ap.add_argument("--cluster", type=str, default=None,
                    help="target cluster directly, e.g. bid1:4 (alternative "
                         "to --quiz-id)")
    ap.add_argument("--require", type=str, default=None,
                    help="target a STRUCTURAL pattern instead of a strength "
                         "bucket: a comma-separated list of <rank><relsuit> "
                         "tokens the acting hand must contain, either half "
                         "of which may be '*'. Suits are relative to the "
                         "up-card, not physical, since the net is "
                         "suit-agnostic: U = the up-card's own suit, N = the "
                         "other suit of the same color (holds the left "
                         "bower), G/g = the two off-color suits (kept "
                         "distinct so a pattern can ask for one from each). "
                         "JU = the right bower, J* = any jack, *g = any card "
                         "of the second off-color suit, ** = any card; "
                         "repeats must match distinct cards, so 'J*,J*' is "
                         "two different jacks. Use this when the thing you "
                         "want isn't a cluster at all -- all four jacks "
                         "spreads across buckets 5-8 depending on the "
                         "up-card suit, and at 0.047%% of hands per seat no "
                         "amount of rejection sampling will find them. The "
                         "required cards are placed; the rest of the hand, "
                         "the up-card's rank (--up-rank to pin it), the "
                         "dealer, the score and all three opponents' hands "
                         "still vary.")
    ap.add_argument("--up-rank", type=str, default=None,
                    help="pin the up card's rank: a rank letter or "
                         "comma-separated set, e.g. 'J' or 'T,J,Q'. Needs "
                         "--require (its U/N/G/g suits fix the up-card's "
                         "suit; this fixes its rank the same way --require "
                         "fixes hand cards). Without it the rank is any "
                         "rank not already placed by --require.")
    ap.add_argument("--phase", choices=["bid1", "bid2"], default="bid1",
                    help="which decision to solve, for --require (with "
                         "--quiz-id/--cluster the phase comes from those).")
    ap.add_argument("--void-up", action="store_true",
                    help="the acting hand must be VOID in the up-card's "
                         "suit -- ordering up would leave zero trump. "
                         "Evaluated by effective suit, so it excludes the "
                         "left bower too rather than just the printed suit. "
                         "Combines with --require; use --require '**' if "
                         "you only want a void.")
    ap.add_argument("--score", type=str, default=None,
                    help="pin the match score as MINE,THEIRS from the acting "
                         "player's side, e.g. '9,6'. Without this the score "
                         "is sampled per deal from the equity model. Needed "
                         "for score-dependent patterns like the 9-6 donation, "
                         "where the right play exists only at that score.")
    ap.add_argument("--quiz", type=str, default="docs/euchre_quiz.json")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--max-tries", type=int, default=200000)
    ap.add_argument("--engine", choices=["python", "cpp"], default="cpp")
    ap.add_argument("--exact-worlds", type=int, default=4,
                    help="belief worlds per exact-leaf solve. Low on purpose: "
                         "every leaf is a double-dummy solve, not a slice of "
                         "one batched forward pass.")
    ap.add_argument("--cfr-iters", type=int, default=60)
    ap.add_argument("--depth-limit", type=int, default=6)
    ap.add_argument("--no-freeze-trunk", action="store_true",
                    help="also update the shared suit-encoder/context trunk. "
                         "Off by default for a reason -- see this module's "
                         "docstring; a targeted correction that moves the "
                         "trunk stops being targeted.")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match-equity-table", type=str,
                    default="rebel/match_equity_table.json")
    ap.add_argument("--no-match-equity", action="store_true")
    ap.add_argument("--no-quiz-check", action="store_true")
    args = ap.parse_args()

    given = [n for n, v in (("--quiz-id", args.quiz_id),
                            ("--cluster", args.cluster),
                            ("--require", args.require)) if v is not None]
    if len(given) != 1:
        raise SystemExit("give exactly one of --quiz-id, --cluster or "
                         f"--require (got {given or 'none'})")
    # --void-up, --score and --up-rank place cards / fix state at deal time,
    # which only the --require constructor does; cluster selection reaches
    # its positions by rejection sampling and has no way to impose any of
    # them.
    for flag, val in (("--void-up", args.void_up), ("--score", args.score),
                      ("--up-rank", args.up_rank)):
        if val and args.require is None:
            raise SystemExit(f"{flag} needs --require (it constrains how the "
                             f"deal is BUILT; --quiz-id/--cluster sample "
                             f"existing deals instead). Use --require '**' if "
                             f"you want no card requirement.")

    equity_model = None
    if not args.no_match_equity:
        if not os.path.exists(args.match_equity_table):
            raise SystemExit(f"--match-equity-table {args.match_equity_table!r} "
                             f"not found; build it or pass --no-match-equity")
        equity_model = MatchEquityModel.load(args.match_equity_table)

    net = PolicyValueNet()
    net.load_state_dict(torch.load(args.resume, map_location="cpu"))
    print(f"resumed from {args.resume}", flush=True)

    trainer = ReBeLTrainer(net=net, engine=args.engine, equity_model=equity_model,
                           depth_limit=args.depth_limit, seed=args.seed)
    rng = random.Random(args.seed + 1)
    patterns, cluster = None, None
    score = parse_score(args.score) if args.score else None
    up_ranks = parse_up_rank(args.up_rank) if args.up_rank else None
    if score is not None:
        target = getattr(equity_model, "target", 10) or 10
        if max(score) >= target:
            raise SystemExit(f"--score {args.score}: a team already at "
                             f"{target} has won; the hand would be moot")
    if args.require is not None:
        patterns = parse_require(args.require)
        # Fail loudly and immediately on an impossible ask ("J*" five times --
        # there are only four jacks) rather than spinning to --max-tries and
        # reporting an empty result that looks like bad luck. Any fixed
        # up-suit works for this check: each U/N/G/g letter always names
        # exactly one physical suit (6 cards) regardless of which one, so
        # feasibility doesn't depend on the actual per-deal assignment.
        canonical_map = relsuit_map(Suit.CLUBS, random.Random(0))
        if choose_required(resolve_patterns(patterns, canonical_map), rng) is None:
            raise SystemExit(
                f"--require {args.require!r} can't be satisfied: no five "
                f"distinct cards match those patterns simultaneously")
        desc = f"{args.phase} hands containing {', '.join(t for t, _, _ in patterns)}"
        if args.void_up:
            desc += ", void in up"
        if score is not None:
            desc += f", at {score[0]}-{score[1]}"
        if up_ranks:
            desc += f", up-card rank in {{{','.join(r.symbol for r in up_ranks)}}}"
        print(f"pattern: {desc}", flush=True)
    elif args.quiz_id is not None:
        cluster, q = cluster_of_quiz_question(trainer, args.quiz, args.quiz_id)
        print(f"pattern from quiz Q{args.quiz_id}: cluster {cluster}\n"
              f"  {q.get('text', '')[:100]}", flush=True)
    else:
        cluster = parse_cluster(args.cluster)
        print(f"pattern: cluster {cluster}", flush=True)

    phase = args.phase if patterns is not None else cluster[0]
    print(f"\ngenerating {args.samples} exact-leaf samples "
          f"(engine={args.engine}, worlds={args.exact_worlds})...", flush=True)
    samples, tries = generate(trainer, cluster, args.samples, args.exact_worlds,
                              args.cfr_iters, args.max_tries, rng,
                              patterns=patterns, phase=phase, void_up=args.void_up,
                              score=score, up_ranks=up_ranks)
    if not samples:
        raise SystemExit(f"no usable deals in {tries} tries -- "
                         + ("stick-the-dealer may be blocking the passes "
                            "needed to reach that seat" if patterns is not None
                            else f"is cluster {cluster} reachable?"))
    print(f"{len(samples)} samples from {tries} deals "
          f"({len(samples)/tries:.1%} acceptance)", flush=True)
    if len(samples) < args.samples:
        print(f"warning: wanted {args.samples}; a thin cluster gives a "
              f"smaller, noisier correction", flush=True)

    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_frac))
    val, train = samples[:n_val], samples[n_val:]
    print(f"{len(train)} train / {len(val)} val", flush=True)

    def val_loss():
        with torch.no_grad():
            pl, vl = batch_loss(net, val)
        return float(pl), float(vl)

    quiz_before = None
    if not args.no_quiz_check:
        quiz_before = quiz_score(net, args.quiz)
    pl0, vl0 = val_loss()
    print(f"\nbefore: val policy_loss={pl0:.4f} value_loss={vl0:.4f}"
          + (f"   quiz {quiz_before[0]}/{quiz_before[1]}" if quiz_before else ""),
          flush=True)

    freeze = not args.no_freeze_trunk
    params = list(net.head_parameters() if freeze else net.parameters())
    n_train_p = sum(p.numel() for p in params)
    n_all_p = sum(p.numel() for p in net.parameters())
    print(f"trunk {'FROZEN' if freeze else 'trainable'}: updating "
          f"{n_train_p}/{n_all_p} parameters ({n_train_p/n_all_p:.0%})",
          flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)

    best = pl0 + vl0
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    stale = 0
    for epoch in range(1, args.max_epochs + 1):
        rng.shuffle(train)
        tot, nb = 0.0, 0
        for i in range(0, len(train), args.batch_size):
            pl, vl = batch_loss(net, train[i:i + args.batch_size])
            loss = pl + vl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        vpl, vvl = val_loss()
        print(f"epoch {epoch}/{args.max_epochs}: train={tot/max(nb,1):.4f} "
              f"val policy={vpl:.4f} value={vvl:.4f}", flush=True)
        if vpl + vvl < best - 1e-5:
            best, stale = vpl + vvl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"no val improvement for {args.patience} epochs, stopping",
                      flush=True)
                break

    net.load_state_dict(best_state)
    pl1, vl1 = val_loss()
    print(f"\nafter (best-val): val policy_loss={pl1:.4f} (was {pl0:.4f}) "
          f"value_loss={vl1:.4f} (was {vl0:.4f})", flush=True)
    if quiz_before is not None:
        qa = quiz_score(net, args.quiz)
        print(f"quiz: {qa[0]}/{qa[1]}  (was {quiz_before[0]}/{quiz_before[1]})",
              flush=True)
        if qa[0] < quiz_before[0]:
            print("  WARNING: the pattern improved but the quiz got WORSE. "
                  "That is the collateral this script exists to catch -- "
                  "prefer the old checkpoint, or retry with fewer epochs / a "
                  "lower --lr.", flush=True)

    torch.save(net.state_dict(), args.out + ".pt")
    print(f"\nsaved to {args.out}.pt", flush=True)


if __name__ == "__main__":
    main()
