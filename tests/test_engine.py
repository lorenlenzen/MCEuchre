"""Tests for the Euchre engine, with emphasis on the bower / trump rules."""

import random

import pytest

from euchre.cards import (
    Card, Suit, Rank, DECK, NUM_CARDS,
    is_left_bower, is_right_bower, is_trump, effective_suit,
    same_color_suit, card_strength, trick_winner,
)
from euchre.actions import (
    Pass, OrderUp, Call, Discard, Play,
    action_to_index, index_to_action, NUM_ACTIONS,
)
from euchre.game import EuchreState, Phase, team_of, partner_of


# --- Cards ------------------------------------------------------------------

def test_deck_is_24_unique_cards():
    assert len(DECK) == 24 == NUM_CARDS
    assert len(set(DECK)) == 24


def test_card_id_roundtrip():
    for c in DECK:
        assert Card.from_id(c.id) == c
    assert sorted(c.id for c in DECK) == list(range(24))


def test_same_color_pairs():
    assert same_color_suit(Suit.HEARTS) == Suit.DIAMONDS
    assert same_color_suit(Suit.DIAMONDS) == Suit.HEARTS
    assert same_color_suit(Suit.CLUBS) == Suit.SPADES
    assert same_color_suit(Suit.SPADES) == Suit.CLUBS


def test_right_and_left_bower():
    trump = Suit.HEARTS
    jh = Card(Suit.HEARTS, Rank.JACK)
    jd = Card(Suit.DIAMONDS, Rank.JACK)
    jc = Card(Suit.CLUBS, Rank.JACK)
    assert is_right_bower(jh, trump) and not is_left_bower(jh, trump)
    assert is_left_bower(jd, trump) and not is_right_bower(jd, trump)
    assert not is_trump(jc, trump)
    assert is_trump(jh, trump) and is_trump(jd, trump)


def test_left_bower_effective_suit_is_trump():
    trump = Suit.SPADES
    left = Card(Suit.CLUBS, Rank.JACK)  # same color as spades
    assert effective_suit(left, trump) == Suit.SPADES
    # A non-bower club is still a club.
    assert effective_suit(Card(Suit.CLUBS, Rank.ACE), trump) == Suit.CLUBS


def test_trump_ordering_right_beats_left_beats_ace():
    trump = Suit.HEARTS
    right = Card(Suit.HEARTS, Rank.JACK)
    left = Card(Suit.DIAMONDS, Rank.JACK)
    ace = Card(Suit.HEARTS, Rank.ACE)
    nine = Card(Suit.HEARTS, Rank.NINE)
    led = Suit.HEARTS
    assert (card_strength(right, trump, led) > card_strength(left, trump, led)
            > card_strength(ace, trump, led) > card_strength(nine, trump, led))


def test_trick_winner_trump_beats_led():
    trump = Suit.SPADES
    plays = [
        (0, Card(Suit.HEARTS, Rank.ACE)),   # leads hearts
        (1, Card(Suit.HEARTS, Rank.NINE)),
        (2, Card(Suit.SPADES, Rank.NINE)),  # trumps in
        (3, Card(Suit.HEARTS, Rank.KING)),
    ]
    assert trick_winner(plays, trump) == 2


def test_trick_winner_left_bower_counts_as_trump():
    trump = Suit.SPADES
    plays = [
        (0, Card(Suit.HEARTS, Rank.ACE)),
        (1, Card(Suit.CLUBS, Rank.JACK)),   # left bower -> trump, wins
        (2, Card(Suit.SPADES, Rank.NINE)),
        (3, Card(Suit.HEARTS, Rank.KING)),
    ]
    # left bower (strength 207) beats the nine of spades (202)
    assert trick_winner(plays, trump) == 1


def test_trick_winner_highest_of_led_when_no_trump():
    trump = Suit.SPADES
    plays = [
        (0, Card(Suit.HEARTS, Rank.TEN)),
        (1, Card(Suit.HEARTS, Rank.ACE)),   # highest heart
        (2, Card(Suit.CLUBS, Rank.KING)),   # off-suit, cannot win
        (3, Card(Suit.HEARTS, Rank.QUEEN)),
    ]
    assert trick_winner(plays, trump) == 1


# --- Action encoding --------------------------------------------------------

def test_action_index_roundtrip():
    actions = (
        [Play(c) for c in DECK]
        + [Discard(c) for c in DECK]
        + [Call(s, alone=False) for s in Suit]
        + [Call(s, alone=True) for s in Suit]
        + [OrderUp(False), OrderUp(True), Pass()]
    )
    seen = set()
    for a in actions:
        idx = action_to_index(a)
        assert 0 <= idx < NUM_ACTIONS
        assert idx not in seen, f"index collision at {a}"
        seen.add(idx)
        assert index_to_action(idx) == a
    assert len(seen) == NUM_ACTIONS


# --- Game flow --------------------------------------------------------------

def _fixed_deal():
    """Deterministic deal for reproducible flow tests."""
    state = EuchreState.new_hand(dealer=3)
    # Player hands (5 each), up-card, kitty(3) = 24 cards, all distinct.
    hands = [
        [Card(Suit.HEARTS, r) for r in
         (Rank.NINE, Rank.TEN, Rank.QUEEN, Rank.KING, Rank.ACE)],
        [Card(Suit.CLUBS, r) for r in
         (Rank.NINE, Rank.TEN, Rank.QUEEN, Rank.KING, Rank.ACE)],
        [Card(Suit.SPADES, r) for r in
         (Rank.NINE, Rank.TEN, Rank.QUEEN, Rank.KING, Rank.ACE)],
        [Card(Suit.DIAMONDS, r) for r in
         (Rank.NINE, Rank.TEN, Rank.QUEEN, Rank.KING, Rank.ACE)],
    ]
    up = Card(Suit.HEARTS, Rank.JACK)
    kitty = [Card(Suit.DIAMONDS, Rank.JACK),
             Card(Suit.CLUBS, Rank.JACK),
             Card(Suit.SPADES, Rank.JACK)]
    return state.deal_from(hands, up, kitty)


def test_deal_shapes():
    rng = random.Random(42)
    s = EuchreState.new_hand(dealer=0).deal(rng)
    assert s.phase == Phase.BID_ROUND_1
    assert s.current_player == 1  # left of dealer
    assert all(len(h) == 5 for h in s.hands)
    assert len(s.kitty) == 3 and s.up_card is not None
    all_cards = [c for h in s.hands for c in h] + s.kitty + [s.up_card]
    assert len(set(all_cards)) == 24


def test_round1_all_pass_moves_to_round2():
    s = _fixed_deal()
    for _ in range(4):
        assert s.phase == Phase.BID_ROUND_1
        s = s.apply(Pass())
    assert s.phase == Phase.BID_ROUND_2
    assert s.turned_down == Suit.HEARTS  # up-card was a heart
    assert s.current_player == 0  # left of dealer(3) again


def test_round2_cannot_call_turned_down_suit():
    s = _fixed_deal()
    for _ in range(4):
        s = s.apply(Pass())
    legal = s.legal_actions()
    called_suits = {a.suit for a in legal if isinstance(a, Call)}
    assert Suit.HEARTS not in called_suits
    assert called_suits == {Suit.CLUBS, Suit.DIAMONDS, Suit.SPADES}


def test_order_up_triggers_dealer_discard_and_pickup():
    s = _fixed_deal()
    s = s.apply(OrderUp(alone=False))  # player 0 orders up hearts
    assert s.trump == Suit.HEARTS
    assert s.maker == 0
    assert s.phase == Phase.DEALER_DISCARD
    assert s.current_player == 3  # dealer
    assert len(s.hands[3]) == 6  # picked up the up-card
    assert Card(Suit.HEARTS, Rank.JACK) in s.hands[3]
    # Dealer discards down to 5.
    s = s.apply(Discard(Card(Suit.DIAMONDS, Rank.NINE)))
    assert len(s.hands[3]) == 5
    assert s.phase == Phase.PLAY
    assert s.current_player == 0  # left of dealer leads


def test_full_hand_plays_out_and_scores():
    s = _fixed_deal()
    s = s.apply(OrderUp(alone=False))
    s = s.apply(Discard(Card(Suit.DIAMONDS, Rank.NINE)))
    rng = random.Random(0)
    guard = 0
    while not s.is_terminal():
        legal = s.legal_actions()
        assert legal, "non-terminal state must have legal actions"
        s = s.apply(rng.choice(legal))
        guard += 1
        assert guard < 100
    r0, r1 = s.returns()
    assert (r0, r1) in {(1, 0), (2, 0), (0, 2)}  # not alone
    assert sum(s.tricks_won) == 5


def test_must_follow_suit():
    s = _fixed_deal()
    s = s.apply(OrderUp(alone=False))
    s = s.apply(Discard(Card(Suit.DIAMONDS, Rank.NINE)))
    # Player 0 leads a heart; player 1 (all clubs) is void -> may play anything,
    # but if a player holds the led suit they must follow.
    s = s.apply(Play(Card(Suit.HEARTS, Rank.ACE)))
    # Player 1 has no hearts (all clubs) so all plays are legal.
    legal_cards = {a.card for a in s.legal_actions()}
    assert legal_cards == set(s.hands[1])


def test_going_alone_partner_sits_out():
    s = _fixed_deal()
    s = s.apply(OrderUp(alone=True))  # player 0 alone, partner 2 sits
    assert s.alone and s.lone_player == 0 and s.sitting == 2
    s = s.apply(Discard(Card(Suit.DIAMONDS, Rank.NINE)))
    # Play proceeds skipping player 2. Play a full trick and check 3 cards.
    order = []
    start = s.current_player
    for _ in range(3):
        p = s.current_player
        order.append(p)
        s = s.apply(s.legal_actions()[0])
    assert 2 not in order  # sitting player never acts
    assert len(order) == 3


def test_lone_march_scores_four():
    """Construct a hand the lone maker sweeps: build directly and force wins."""
    state = EuchreState.new_hand(dealer=3)
    # Player 0 gets the five biggest hearts-trump cards.
    p0 = [Card(Suit.HEARTS, Rank.JACK),   # right bower
          Card(Suit.DIAMONDS, Rank.JACK),  # left bower
          Card(Suit.HEARTS, Rank.ACE),
          Card(Suit.HEARTS, Rank.KING),
          Card(Suit.HEARTS, Rank.QUEEN)]
    others = [c for c in DECK if c not in p0]
    hands = [p0, others[0:5], others[5:10], others[10:15]]
    up = others[15]
    kitty = others[16:19]
    s = state.deal_from(hands, up, kitty)
    # Player 0 calls hearts alone (whatever the up-card, use round 2 if needed).
    # Route: p0 is left of dealer(3) => acts first in round 1.
    if up.suit == Suit.HEARTS:
        s = s.apply(OrderUp(alone=True))
        s = s.apply(Discard(s.hands[3][0]))
    else:
        # all pass round 1, then p0 calls hearts alone in round 2
        for _ in range(4):
            s = s.apply(Pass())
        s = s.apply(Call(Suit.HEARTS, alone=True))
    # p0 leads and always plays its highest; it holds the 5 strongest trumps.
    guard = 0
    while not s.is_terminal():
        legal = s.legal_actions()
        # lone player leads every trick with an unbeatable trump
        s = s.apply(legal[0])
        guard += 1
        assert guard < 100
    assert s.returns() == (4, 0)
    assert s.tricks_won[0] == 5


def test_teams_and_partners():
    assert team_of(0) == team_of(2) == 0
    assert team_of(1) == team_of(3) == 1
    assert partner_of(0) == 2 and partner_of(1) == 3


def test_clone_is_independent():
    s = _fixed_deal()
    s2 = s.apply(OrderUp(alone=False))
    assert s.phase == Phase.BID_ROUND_1  # original unchanged
    assert s2.phase == Phase.DEALER_DISCARD


def test_random_playthroughs_never_crash():
    rng = random.Random(123)
    for _ in range(200):
        s = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
        guard = 0
        while not s.is_terminal():
            legal = s.legal_actions()
            assert legal
            s = s.apply(rng.choice(legal))
            guard += 1
            assert guard < 200
        r = s.returns()
        assert sum(r) in (0, 1, 2, 4)  # 0 only on thrown-in no-stick hand
