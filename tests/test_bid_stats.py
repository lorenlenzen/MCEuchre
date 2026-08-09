"""Tests for rebel/bid_stats.py -- per-seat bidding-behaviour counters.

The classifier has to get two things right or every number it reports is
wrong: the seat mapping (bidding-order position, dealer = 4) and the
next/green split (which is defined by the turned-down suit's COLOUR, not by
suit identity).
"""

import random

import pytest

from euchre.actions import Call, OrderUp, Pass, Play
from euchre.cards import Card, Suit, same_color_suit
from euchre.game import EuchreState, Phase
from rebel.bid_stats import (BidCounter, ROUND1_CATEGORIES, ROUND2_CATEGORIES,
                             SEAT_NAMES, classify)


def _dealt(dealer=0, up_id=0):
    """A deterministic dealt state with a chosen up-card."""
    deck = [c for c in range(24) if c != up_id]
    random.Random(0).shuffle(deck)
    hands = [[Card.from_id(c) for c in deck[i * 5:(i + 1) * 5]] for i in range(4)]
    kitty = [Card.from_id(c) for c in deck[20:23]]
    return EuchreState.new_hand(dealer=dealer).deal_from(
        hands, Card.from_id(up_id), kitty)


def _to_round2(state):
    """Four passes -> BID_ROUND_2, with turned_down set."""
    for _ in range(4):
        state = state.apply(Pass())
    assert state.phase == Phase.BID_ROUND_2
    return state


# --- seat mapping ---------------------------------------------------------

@pytest.mark.parametrize("dealer", range(4))
def test_seat_positions_are_bidding_order_with_dealer_last(dealer):
    """Bidding opens left of the dealer: that seat is 1, the dealer is 4 --
    quiz_eval.py's SEAT_ORDER convention, which --seat also uses."""
    state = _dealt(dealer=dealer)
    seen = []
    for _ in range(4):
        seat, cat = classify(state, Pass())
        seen.append(seat)
        assert cat == "r1_pass"
        state = state.apply(Pass())
    assert seen == [1, 2, 3, 4]


def test_dealer_seat_is_four():
    state = _dealt(dealer=2)
    # Dealer acts 4th, so skip three passes.
    for _ in range(3):
        state = state.apply(Pass())
    assert state.current_player == 2
    assert classify(state, Pass())[0] == 4


# --- round 1 categories ---------------------------------------------------

def test_round1_categories():
    state = _dealt(dealer=0)
    assert classify(state, Pass()) == (1, "r1_pass")
    assert classify(state, OrderUp(alone=False)) == (1, "r1_orderup")
    assert classify(state, OrderUp(alone=True)) == (1, "r1_orderup_alone")


# --- round 2 next/green split --------------------------------------------

@pytest.mark.parametrize("up_suit", list(Suit))
def test_next_is_the_same_colour_as_the_turned_down_suit(up_suit):
    """`next` must follow the turned-down suit's COLOUR for every up-card,
    not a hardcoded suit."""
    up_id = up_suit * 6  # the nine of that suit
    state = _to_round2(_dealt(dealer=0, up_id=up_id))
    assert state.turned_down == up_suit

    next_suit = same_color_suit(up_suit)
    assert classify(state, Call(next_suit, alone=False)) == (1, "r2_next")
    assert classify(state, Call(next_suit, alone=True)) == (1, "r2_next_alone")

    for green in [s for s in Suit if s not in (up_suit, next_suit)]:
        assert classify(state, Call(green, alone=False)) == (1, "r2_green")
        assert classify(state, Call(green, alone=True)) == (1, "r2_green_alone")


def test_round2_pass():
    state = _to_round2(_dealt(dealer=0))
    assert classify(state, Pass()) == (1, "r2_pass")


def test_next_and_green_partition_every_legal_round2_call():
    """The turned-down suit is illegal in round 2, so next + green must
    cover every legal call -- nothing may fall through unclassified."""
    state = _to_round2(_dealt(dealer=0))
    for action in state.legal_actions():
        assert classify(state, action) is not None, action


# --- non-bidding decisions ------------------------------------------------

def test_play_and_discard_decisions_are_not_counted():
    state = _dealt(dealer=0).apply(OrderUp(alone=False))
    assert state.phase == Phase.DEALER_DISCARD
    assert classify(state, state.legal_actions()[0]) is None
    state = state.apply(state.legal_actions()[0])
    assert state.phase == Phase.PLAY
    assert classify(state, Play(next(iter(state.hands[state.current_player])))) is None


# --- counter aggregation --------------------------------------------------

def test_rates_are_normalized_within_each_round():
    """Round 1 and round 2 have very different denominators (a seat only
    reaches round 2 if everyone passed round 1), so pooling them would make
    round-2 rates meaningless."""
    c = BidCounter()
    c.counts[(1, "r1_orderup")] = 30
    c.counts[(1, "r1_pass")] = 70
    c.counts[(1, "r2_next")] = 2
    c.counts[(1, "r2_pass")] = 2

    row = next(r for r in c.as_rows() if r["seat"] == "first")
    assert row["r1_decisions"] == 100 and row["r2_decisions"] == 4
    assert row["r1_orderup_rate"] == pytest.approx(0.30)
    assert row["r2_next_rate"] == pytest.approx(0.50)   # 2/4, not 2/104


def test_rates_are_zero_when_a_round_never_happened():
    c = BidCounter()
    c.counts[(1, "r1_pass")] = 5
    row = next(r for r in c.as_rows() if r["seat"] == "first")
    assert row["r2_decisions"] == 0
    assert all(row[f"{cat}_rate"] == 0.0 for cat in ROUND2_CATEGORIES)


def test_record_returns_whether_it_counted():
    c = BidCounter()
    state = _dealt(dealer=0)
    assert c.record(state, Pass()) is True
    played = state.apply(OrderUp(alone=False))
    assert c.record(played, played.legal_actions()[0]) is False
    assert sum(c.counts.values()) == 1


def test_all_rows_present_and_named():
    rows = BidCounter().as_rows()
    assert [r["seat"] for r in rows] == [SEAT_NAMES[i] for i in (1, 2, 3, 4)]
    for r in rows:
        for cat in ROUND1_CATEGORIES + ROUND2_CATEGORIES:
            assert cat in r and f"{cat}_rate" in r


# --- cpp engine parity ----------------------------------------------------

def test_cpp_engine_classifies_identically():
    """Same deal, same actions, both engines -> same (seat, category)."""
    import mceuchre_cpp as cpp

    deck = list(range(24))
    random.Random(3).shuffle(deck)
    py = EuchreState.new_hand(dealer=1).deal_from(
        [[Card.from_id(c) for c in deck[i * 5:(i + 1) * 5]] for i in range(4)],
        Card.from_id(deck[20]), [Card.from_id(c) for c in deck[21:24]])
    cp = cpp.EuchreState.new_hand(dealer=1, stick_the_dealer=False,
                                  team0_score=0, team1_score=0).deal_from_deck(deck)

    for _ in range(4):   # round 1: four passes
        assert (classify(py, Pass()) ==
                classify(cp, cpp.Action.pass_(), engine="cpp"))
        py, cp = py.apply(Pass()), cp.apply(cpp.Action.pass_())

    assert py.phase == Phase.BID_ROUND_2
    for suit in Suit:
        if suit == py.turned_down:
            continue
        for alone in (False, True):
            assert (classify(py, Call(suit, alone=alone)) ==
                    classify(cp, cpp.Action.call(int(suit), alone), engine="cpp"))
