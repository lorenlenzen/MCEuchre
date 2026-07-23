"""Feed the Ohio Euchre quiz's bidding questions into a trained ReBeL net.

Loads docs/euchre_quiz.json, and for every buildable bidding question
(phase bid1/bid2) constructs a *legal* EuchreState at the exact decision
point -- the answerer's real hand in their seat, the right up-card /
turned-down suit, and the correct number of prior passes applied so the
bidding bookkeeping matches -- then reports the net's greedy action next to
the quiz's answer.

    python scripts/quiz_eval.py --net checkpoints/rebel_hq.pt

IMPORTANT: the observation encoding (euchre/infoset.py) has no game-score
channel, and stick-the-dealer scenarios were never trained. Questions whose
answer depends on either are flagged; treat their pass/fail as uninformative
about the net.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from euchre.cards import Card, Suit, Rank
from euchre.game import EuchreState, Phase
from euchre.actions import Pass, OrderUp, Call, action_to_index, NUM_ACTIONS
from euchre.infoset import observation_tensor
from rebel.networks import PolicyValueNet
from rebel.train_rebel import legal_mask

_SUIT = {"C": Suit.CLUBS, "D": Suit.DIAMONDS, "H": Suit.HEARTS, "S": Suit.SPADES}
_RANK = {"9": Rank.NINE, "T": Rank.TEN, "J": Rank.JACK,
         "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE}
# acting order at the table: first acts 1st, dealer acts 4th
_SEAT_ORDER = {"first": 1, "second": 2, "third": 3, "dealer": 4}


def card(s):
    return Card(_SUIT[s[1]], _RANK[s[0]])


def build_state(q):
    """Construct the EuchreState at q's bidding decision, with the answerer
    seated at absolute seat 0."""
    P = 0
    a = _SEAT_ORDER[q["seat"]]          # 1..4 acting order of the answerer
    dealer = (P - a) % 4                # so the answerer sits `a`-th after dealer
    my_hand = [card(c) for c in q["hand"]]

    # The up-card stays visible to the net throughout bidding -- trump isn't
    # set until someone actually orders up or calls, and observation_tensor's
    # show_up flag is keyed on `trump is None`, true for all of round 1 AND
    # round 2. So the *exact* turned-down card matters (e.g. a turned-down
    # ace signals a weak dealer; a turned-down bower denies a specific card
    # outright), not just its suit -- use the real card when the quiz states
    # one, only falling back to an arbitrary unheld card when it doesn't.
    if q["phase"] == "bid1":
        up = card(q["up_card"])
    elif "turned_down_card" in q:
        up = card(q["turned_down_card"])
        assert up not in my_hand, f"q{q['id']}: turned_down_card is in the answerer's own hand"
    else:
        td = _SUIT[q["turned_down"]]
        up = next(Card(td, r) for r in Rank
                  if Card(td, r) not in my_hand)

    # Fill the other three hands + 3-card kitty from the unseen pool.
    known = set(my_hand) | {up}
    unseen = [c for c in (Card(s, r) for s in Suit for r in Rank)
              if c not in known]
    hands = [None, None, None, None]
    hands[P] = my_hand
    idx = 0
    for seat in range(4):
        if seat == P:
            continue
        hands[seat] = unseen[idx:idx + 5]
        idx += 5
    kitty = unseen[idx:idx + 3]

    st = EuchreState.new_hand(dealer=dealer,
                              stick_the_dealer=q.get("stick", False))
    st = st.deal_from(hands, up, kitty)

    # Drive to the decision by applying the prior passes.
    n_pass = (a - 1) if q["phase"] == "bid1" else (4 + (a - 1))
    for _ in range(n_pass):
        assert any(isinstance(x, Pass) for x in st.legal_actions()), \
            "expected Pass to be legal while advancing the bidding"
        st = st.apply(Pass())

    # Validate we landed where we intended.
    want_phase = Phase.BID_ROUND_1 if q["phase"] == "bid1" else Phase.BID_ROUND_2
    assert st.phase == want_phase, f"q{q['id']}: phase {st.phase} != {want_phase}"
    assert st.current_player == P, f"q{q['id']}: current {st.current_player} != {P}"
    return st, P


def answer_action(q):
    aa = q["answer_action"]
    if aa["type"] == "pass":
        return Pass()
    if aa["type"] == "orderup":
        return OrderUp(alone=aa["alone"])
    return Call(_SUIT[aa["suit"]], alone=aa["alone"])


def describe(action):
    if isinstance(action, Pass):
        return "Pass"
    if isinstance(action, OrderUp):
        return "OrderUp" + (" alone" if action.alone else "")
    if isinstance(action, Call):
        return f"Call {action.suit.name[0]}" + (" alone" if action.alone else "")
    return str(action)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="checkpoints/rebel_hq.pt")
    ap.add_argument("--quiz", default="docs/euchre_quiz.json")
    args = ap.parse_args()

    net = PolicyValueNet()
    net.load_state_dict(torch.load(args.net, map_location="cpu"))
    net.eval()

    quiz = json.load(open(args.quiz))
    bidding = [q for q in quiz["questions"]
               if q.get("buildable") and q["phase"] in ("bid1", "bid2")]

    print(f"net: {args.net}   bidding questions: {len(bidding)}\n")
    header = (f"{'Q':>2} {'seat':>6} {'phase':>5} {'score':>7} "
              f"{'quiz answer':>14} {'net greedy':>12} {'ok':>3}  flags")
    print(header)
    print("-" * len(header))

    fair_total = fair_correct = 0
    total_ok = 0
    for q in bidding:
        st, P = build_state(q)
        obs = torch.from_numpy(observation_tensor(st, P)).unsqueeze(0)
        mask = torch.from_numpy(legal_mask(st)).unsqueeze(0)
        with torch.no_grad():
            dist = net.policy(obs, mask).squeeze(0).numpy()

        legal = st.legal_actions()
        best = max(legal, key=lambda act: dist[action_to_index(act)])
        want = answer_action(q)
        ok = (action_to_index(best) == action_to_index(want))
        total_ok += int(ok)

        flags = []
        if q.get("score_dependent"):
            flags.append("SCORE")
        if q.get("stick"):
            flags.append("STICK")
        if q.get("seat_assumed"):
            flags.append("SEAT?")
        fair = not flags
        if fair:
            fair_total += 1
            fair_correct += int(ok)

        sc = f"{q['score'][0]}-{q['score'][1]}" if "score" in q else ""
        print(f"{q['id']:>2} {q['seat']:>6} {q['phase']:>5} {sc:>7} "
              f"{describe(want):>14} {describe(best):>12} "
              f"{'Y' if ok else '.':>3}  {','.join(flags)}")

    print()
    print(f"overall: {total_ok}/{len(bidding)} match the quiz answer")
    print(f"fair subset (no SCORE/STICK/SEAT? flags): "
          f"{fair_correct}/{fair_total} match")


if __name__ == "__main__":
    main()
