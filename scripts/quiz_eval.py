"""Feed the Ohio Euchre quiz's questions into a trained ReBeL net.

Loads docs/euchre_quiz.json, and for every buildable question (phase
bid1/bid2/discard/play) constructs a *legal* EuchreState at the exact
decision point -- the answerer's real hand in their seat, the right
up-card/turned-down suit, correct bidding bookkeeping, and (for discard/play)
a real replay of the narrated pickup/discard and trick history via
build_discard_state/build_play_state -- then reports the net's greedy action
next to the quiz's answer. Q21 is the one question in the quiz that can't be
built at all (no hand given, options are abstract suit names rather than
concrete cards); everything else is buildable, either exactly (from stated
cards) or with a documented decent default for information the quiz's prose
doesn't state (see each play question's "note" field, and build_play_state's
docstring for why those defaults are decision-irrelevant).

    python scripts/quiz_eval.py --net checkpoints/rebel_hq.pt

Questions with a "score" field are built at that exact match score (see
build_state: [my_team, their_team], my_team always answered from seat 0) --
observation_tensor has carried a score channel since the match-equity work
(euchre/infoset.py). SCORE-flagged questions are still excluded from the
"fair" pass-rate subset below (their correct answer depends on score, so
they're a different kind of test than the hand-strength-only questions the
fair subset targets) but are tallied separately so you can see match-equity
awareness in isolation. STICK-flagged questions depend on the net having
been trained with stick_the_dealer=True -- true for some checkpoints, not
others; treat their pass/fail as uninformative unless you know your
checkpoint was trained that way.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

import random

from euchre.cards import Card, Suit, Rank, effective_suit
from euchre.game import EuchreState, Phase
from euchre.actions import Pass, OrderUp, Call, Discard, Play, action_to_index, NUM_ACTIONS
from euchre.infoset import observation_tensor
from rebel.networks import PolicyValueNet
from rebel.train_rebel import legal_mask

_SUIT = {"C": Suit.CLUBS, "D": Suit.DIAMONDS, "H": Suit.HEARTS, "S": Suit.SPADES}
_RANK = {"9": Rank.NINE, "T": Rank.TEN, "J": Rank.JACK,
         "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE}
_RANK_STR = {v: k for k, v in _RANK.items()}
_SUIT_STR = {v: k for k, v in _SUIT.items()}


def card_str(c):
    """ASCII card format matching the quiz's own convention (e.g. "QD") --
    Card.__str__ uses unicode suit symbols, which fail to encode on the
    default Windows console codepage."""
    return f"{_RANK_STR[c.rank]}{_SUIT_STR[c.suit]}"
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

    team0_score, team1_score = q.get("score", (0, 0))
    st = EuchreState.new_hand(dealer=dealer,
                              stick_the_dealer=q.get("stick", False),
                              team0_score=team0_score, team1_score=team1_score)
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


def build_discard_state(q):
    """Construct the EuchreState at q's DEALER_DISCARD decision -- the
    dealer, who always discards regardless of who ordered/called, seated at
    absolute seat 0. Whoever picked up (q["orderer"]: "dealer" self-orders
    in round 1, or "partner" orders the dealer up) only changes how many
    prior passes reach that point -- the dealer always ends up holding all 6
    cards and choosing the discard."""
    P = 0  # dealer
    up = card(q["up_card"])
    full_hand = [card(c) for c in q["hand"]]
    assert up in full_hand, f"q{q['id']}: up_card not in the post-pickup hand"
    my_hand = [c for c in full_hand if c != up]
    assert len(my_hand) == 5, f"q{q['id']}: expected 5 cards before pickup"

    known = set(full_hand)
    unseen = [c for c in (Card(s, r) for s in Suit for r in Rank) if c not in known]
    hands = [None, None, None, None]
    hands[P] = my_hand
    idx = 0
    for seat in range(1, 4):
        hands[seat] = unseen[idx:idx + 5]
        idx += 5
    kitty = unseen[idx:idx + 3]

    team0_score, team1_score = q.get("score", (0, 0))
    st = EuchreState.new_hand(dealer=P, stick_the_dealer=q.get("stick", False),
                              team0_score=team0_score, team1_score=team1_score)
    st = st.deal_from(hands, up, kitty)

    orderer = q.get("orderer", "dealer")
    alone = q.get("alone", False)
    if orderer == "dealer":
        n_pass = 3  # seats 1,2,3 all pass; dealer (4th to act) orders up self
    elif orderer == "partner":
        n_pass = 1  # seat 1 passes, then the partner (seat 2) orders up
    else:
        raise ValueError(f"q{q['id']}: unknown orderer {orderer!r}")
    for _ in range(n_pass):
        st = st.apply(Pass())
    st = st.apply(OrderUp(alone=alone))

    assert st.phase == Phase.DEALER_DISCARD, f"q{q['id']}: phase {st.phase}"
    assert st.current_player == P, f"q{q['id']}: current {st.current_player} != {P}"
    return st, P


def _abs_seat(dealer, bidding_pos):
    """bidding-order position (1-4, dealer=4) -> absolute seat, with the
    dealer at absolute seat `dealer` and the answerer always at absolute
    seat 0 (build_play_state's own convention, matching build_state's)."""
    return (dealer + bidding_pos) % 4


def build_play_state(q):
    """Construct the EuchreState at q's PLAY decision (a lead or a mid-trick
    play), replaying the exact sequence of prior plays the quiz describes.

    "S1"/"S2"/"S3"/"partner"/"dealer" in the quiz's original text are all
    just aliases for a fixed bidding-order seat position 1-4 (dealer=4) --
    the same convention the "seat" field already uses -- verified by
    cross-checking narrated trick winners (e.g. Q9's "S2 wins AC") against
    the actual trick_winner rules; see docs/euchre_quiz.json's `tricks`/
    `current_trick` fields for the resolved positions.

    Only cards the quiz actually narrates are hand-specified (q["full_hand"]
    for the answerer, q["accept"]["up_card"]/["turned_down_card"], and each
    named seat+card in q["tricks"]/q["current_trick"]). Everything else --
    unnarrated opponent holdings, the dealer's discard when not otherwise
    pinned by a later required play -- is filled deterministically (seeded
    by question id, so reruns are reproducible) from the remaining unseen
    pool. This is safe because PLAY-phase observation_tensor never depends
    on hidden opponent hands, and the up-card's exact identity stops
    mattering the moment trump is set (observation_tensor's show_up flag is
    keyed on `trump is None`, always false during PLAY) -- confirmed by
    reading euchre/infoset.py directly, not assumed.

    Each trick in q["tricks"] is {"plays": [{"seat": n, "card": "XY"}, ...],
    "winner": n (optional)}; the seat+winner are asserted against the
    engine's own turn order / trick_winner resolution at replay time, so any
    mistake in the manual seat-position reasoning above raises immediately
    rather than silently building a wrong-but-legal state."""
    P = 0
    a = _SEAT_ORDER[q["seat"]]
    dealer = (P - a) % 4

    full_hand = [card(c) for c in q["full_hand"]]
    assert len(full_hand) == 5, f"q{q['id']}: full_hand must have 5 cards"

    accept = q["accept"]
    up = card(accept["up_card"] if accept["round"] == 1 else accept["turned_down_card"])

    tricks = q.get("tricks", [])
    current_trick = q.get("current_trick", [])
    all_plays = [p for t in tricks for p in t["plays"]] + current_trick

    required = {1: [], 2: [], 3: [], 4: []}
    for p in all_plays:
        required[p["seat"]].append(card(p["card"]))

    known = set(full_hand) | {up}
    for cards_ in required.values():
        known |= set(cards_)

    # A play that doesn't match its trick's led effective-suit is only legal
    # if that seat was void in it -- true by construction for a real quiz
    # narrative, but NOT automatically true for the arbitrary filler cards
    # that round out that seat's hand below. Without this, a randomly-filled
    # card of the led suit could make the narrated (off-suit) play illegal
    # when replayed. trump_suit is known once accept round/suit is read.
    trump_suit = up.suit if accept["round"] == 1 else _SUIT[accept["suit"]]
    void_suits = {1: set(), 2: set(), 3: set(), 4: set()}
    trick_play_lists = [t["plays"] for t in tricks] + ([current_trick] if current_trick else [])
    for plays in trick_play_lists:
        led_suit = effective_suit(card(plays[0]["card"]), trump_suit)
        for p in plays[1:]:
            if effective_suit(card(p["card"]), trump_suit) != led_suit:
                void_suits[p["seat"]].add(led_suit)

    rng = random.Random(1_000_000 + q["id"])
    unseen = [c for c in (Card(s, r) for s in Suit for r in Rank) if c not in known]
    rng.shuffle(unseen)

    hands = [None, None, None, None]
    hands[P] = full_hand
    pool = list(unseen)
    # Most-constrained seats (largest avoid-suit set) fill first, so an
    # unconstrained seat can't greedily consume the very cards a void-suit
    # constrained seat actually needs to avoid -- with seats processed in a
    # fixed 1..4 order, that starvation is exactly what happened here.
    fill_order = sorted((p for p in (1, 2, 3, 4) if p != a),
                        key=lambda p: -len(void_suits[p]))
    for pos in fill_order:
        cards_ = list(required[pos])
        avoid = void_suits[pos]
        while len(cards_) < 5:
            idx = next((i for i, c in enumerate(pool)
                       if effective_suit(c, trump_suit) not in avoid), None)
            assert idx is not None, (
                f"q{q['id']}: no eligible unseen card left for bidding-seat "
                f"{pos} avoiding {avoid} -- pool exhausted of legal options")
            cards_.append(pool.pop(idx))
        hands[_abs_seat(dealer, pos)] = cards_
    assert len(pool) >= 3, f"q{q['id']}: not enough unseen cards for a 3-card kitty"
    kitty = pool[:3]

    team0_score, team1_score = q.get("score", (0, 0))
    st = EuchreState.new_hand(dealer=dealer, stick_the_dealer=q.get("stick", False),
                              team0_score=team0_score, team1_score=team1_score)
    st = st.deal_from(hands, up, kitty)

    orderer_seat = accept["orderer_seat"]
    alone = accept.get("alone", False)
    if accept["round"] == 1:
        for _ in range(orderer_seat - 1):
            st = st.apply(Pass())
        st = st.apply(OrderUp(alone=alone))
        assert st.phase == Phase.DEALER_DISCARD, f"q{q['id']}: phase {st.phase}"
        # Default: the dealer immediately discards back the very card they
        # picked up -- always legal, and safe unless the answerer IS the
        # dealer and needs to keep that specific card (then q["discard"]
        # names the real discard explicitly).
        discard_card = card(q.get("discard", accept["up_card"]))
        st = st.apply(Discard(discard_card))
    else:
        for _ in range(4):  # round 1 fully passes out before round 2 begins
            st = st.apply(Pass())
        for _ in range(orderer_seat - 1):
            st = st.apply(Pass())
        st = st.apply(Call(_SUIT[accept["suit"]], alone=alone))
    assert st.phase == Phase.PLAY, f"q{q['id']}: phase {st.phase} != PLAY"

    def _replay(play):
        nonlocal st
        expected = _abs_seat(dealer, play["seat"])
        assert st.current_player == expected, (
            f"q{q['id']}: expected bidding-seat {play['seat']} (absolute "
            f"{expected}) to act, engine says {st.current_player}")
        st = st.apply(Play(card(play["card"])))

    for trick in tricks:
        for play in trick["plays"]:
            _replay(play)
        if "winner" in trick:
            want = _abs_seat(dealer, trick["winner"])
            got = st.completed_tricks[-1][0]
            assert got == want, (
                f"q{q['id']}: trick winner mismatch -- narrative says "
                f"bidding-seat {trick['winner']} (absolute {want}), engine "
                f"says {got}")
    for play in current_trick:
        _replay(play)

    assert st.phase == Phase.PLAY, f"q{q['id']}: ended in {st.phase}, not PLAY"
    assert st.current_player == P, f"q{q['id']}: current {st.current_player} != {P}"
    return st, P


def build_state_any(q):
    if q["phase"] == "discard":
        return build_discard_state(q)
    if q["phase"] == "play":
        return build_play_state(q)
    return build_state(q)


def answer_action(q):
    aa = q["answer_action"]
    if aa["type"] == "pass":
        return Pass()
    if aa["type"] == "orderup":
        return OrderUp(alone=aa["alone"])
    if aa["type"] == "discard":
        return Discard(card(aa["card"]))
    if aa["type"] == "play":
        return Play(card(aa["card"]))
    return Call(_SUIT[aa["suit"]], alone=aa["alone"])


def describe(action):
    if isinstance(action, Pass):
        return "Pass"
    if isinstance(action, OrderUp):
        return "OrderUp" + (" alone" if action.alone else "")
    if isinstance(action, Call):
        return f"Call {action.suit.name[0]}" + (" alone" if action.alone else "")
    if isinstance(action, Discard):
        return f"Discard {card_str(action.card)}"
    if isinstance(action, Play):
        return f"Play {card_str(action.card)}"
    return str(action)


def print_distribution(net, q, equity_model=None):
    """Full per-action policy distribution for one quiz question, not just
    the greedy pick -- e.g. a near-uniform spread across suits (honest
    uncertainty from too little round-2 data) looks very different from a
    confident, hand-independent bias, even when both give the wrong greedy
    answer. Useful for telling those two failure modes apart."""
    st, P = build_state_any(q)
    obs = torch.from_numpy(observation_tensor(st, P)).unsqueeze(0)
    mask = torch.from_numpy(legal_mask(st)).unsqueeze(0)
    with torch.no_grad():
        dist = net.policy(obs, mask).squeeze(0).numpy()
    legal = sorted(st.legal_actions(),
                   key=lambda a: -dist[action_to_index(a)])
    print(f"Q{q['id']} ({q.get('seat', 'dealer')}, {q['phase']}, hand={q['hand']}, "
          f"up={card_str(st.up_card)}, quiz answer={describe(answer_action(q))}):")
    if equity_model is not None:
        # The answerer is always seated as team0 (build_state's convention),
        # so am_i_dealer is just "is team0 (seat 0) the one dealing". Deltas
        # cover the 4 point-value outcomes bidding decisions actually turn
        # on -- a euchre's penalty is always 2 regardless of alone, so
        # loner-euchred isn't distinct from a regular euchre and isn't shown
        # separately.
        from euchre.game import team_of
        team0_score, team1_score = q.get("score", (0, 0))
        am_i_dealer = team_of(st.dealer) == 0
        wp = equity_model.win_prob(team0_score, team1_score, am_i_dealer)
        d_make = equity_model.equity_delta(team0_score, team1_score, am_i_dealer, 1, 0)
        d_march = equity_model.equity_delta(team0_score, team1_score, am_i_dealer, 2, 0)
        d_loner = equity_model.equity_delta(team0_score, team1_score, am_i_dealer, 4, 0)
        d_euchre = equity_model.equity_delta(team0_score, team1_score, am_i_dealer, 0, 2)
        print(f"  match equity: score={team0_score}-{team1_score}  "
              f"{'I deal' if am_i_dealer else 'they deal'}  "
              f"my win prob={wp:.3f}  "
              f"delta(make +1)={d_make:+.3f}  delta(march +2)={d_march:+.3f}  "
              f"delta(loner march +4)={d_loner:+.3f}  "
              f"delta(euchred -2)={d_euchre:+.3f}")
    for a in legal:
        print(f"  {describe(a):14s} p={dist[action_to_index(a)]:.3f}")


def answerable_questions(quiz):
    return [q for q in quiz["questions"]
            if q.get("buildable")
            and q["phase"] in ("bid1", "bid2", "discard", "play")]


def predict_best(net, q):
    """(best, want, ok) for one question under greedy play.

    Picks the highest-scoring LEGAL action rather than argmax over the full
    59-wide vector -- shared with scripts/train_pattern.py's collateral check
    so the two can't quietly disagree about what the score is."""
    st, P = build_state_any(q)
    obs = torch.from_numpy(observation_tensor(st, P)).unsqueeze(0)
    mask = torch.from_numpy(legal_mask(st)).unsqueeze(0)
    with torch.no_grad():
        dist = net.policy(obs, mask).squeeze(0).numpy()
    best = max(st.legal_actions(), key=lambda act: dist[action_to_index(act)])
    want = answer_action(q)
    return best, want, action_to_index(best) == action_to_index(want)


def score_net(net, quiz):
    """(correct, total) over every answerable question."""
    qs = answerable_questions(quiz)
    return sum(predict_best(net, q)[2] for q in qs), len(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="checkpoints/rebel_hq.pt")
    ap.add_argument("--quiz", default="docs/euchre_quiz.json")
    ap.add_argument("--detail", type=int, action="append", default=[],
                    metavar="QID",
                    help="print the full policy distribution (not just the "
                         "greedy pick) for this question id; repeatable, "
                         "e.g. --detail 11 --detail 2")
    ap.add_argument("--equity-table", type=str,
                    default="rebel/match_equity_table.json",
                    help="match-equity table to show in --detail output "
                         "(win prob / equity deltas at the question's score). "
                         "Silently skipped if missing.")
    args = ap.parse_args()

    net = PolicyValueNet()
    net.load_state_dict(torch.load(args.net, map_location="cpu"))
    net.eval()

    equity_model = None
    if args.detail and args.equity_table and os.path.exists(args.equity_table):
        from rebel.match_equity import MatchEquityModel
        equity_model = MatchEquityModel.load(args.equity_table)

    quiz = json.load(open(args.quiz))
    answerable = answerable_questions(quiz)

    if args.detail:
        for qid in args.detail:
            q = next((x for x in answerable if x["id"] == qid), None)
            if q is None:
                print(f"Q{qid}: not a buildable question")
                continue
            print_distribution(net, q, equity_model)
        print()

    print(f"net: {args.net}   answerable questions: {len(answerable)}\n")
    header = (f"{'Q':>2} {'seat':>6} {'phase':>7} {'score':>7} "
              f"{'quiz answer':>14} {'net greedy':>12} {'ok':>3}  flags")
    print(header)
    print("-" * len(header))

    fair_total = fair_correct = 0
    score_total = score_correct = 0
    total_ok = 0
    for q in answerable:
        best, want, ok = predict_best(net, q)
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
        if q.get("score_dependent"):
            score_total += 1
            score_correct += int(ok)

        sc = f"{q['score'][0]}-{q['score'][1]}" if "score" in q else ""
        print(f"{q['id']:>2} {q.get('seat', 'dealer'):>6} {q['phase']:>7} {sc:>7} "
              f"{describe(want):>14} {describe(best):>12} "
              f"{'Y' if ok else '.':>3}  {','.join(flags)}")

    print()
    print(f"overall: {total_ok}/{len(answerable)} match the quiz answer")
    print(f"fair subset (no SCORE/STICK/SEAT? flags): "
          f"{fair_correct}/{fair_total} match")
    if score_total:
        print(f"score-dependent subset (built at their real quiz score, now "
              f"that observation_tensor carries it): "
              f"{score_correct}/{score_total} match")


if __name__ == "__main__":
    main()
