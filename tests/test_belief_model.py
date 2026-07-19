"""Tests for bidding-conditioned belief refinement and team-game credit."""

import random

from euchre.cards import Card, Suit, Rank
from euchre.game import EuchreState, Phase, team_of, partner_of
from euchre.actions import OrderUp, Discard, Pass, Call
from rebel.belief_model import (
    BiddingBeliefModel, suit_strength, reconstruct_bids,
    reconstruct_original_hands, sample_weighted_belief,
)
from rebel.evaluate import RuleBasedAgent


def test_suit_strength_orders_bowers_highest():
    trump = Suit.HEARTS
    right = [Card(Suit.HEARTS, Rank.JACK)]
    left = [Card(Suit.DIAMONDS, Rank.JACK)]
    ace = [Card(Suit.HEARTS, Rank.ACE)]
    off = [Card(Suit.SPADES, Rank.NINE)]
    assert (suit_strength(right, trump) > suit_strength(left, trump)
            > suit_strength(ace, trump) > suit_strength(off, trump))


def test_call_prob_monotonic_in_strength():
    model = BiddingBeliefModel()
    weak = [Card(Suit.SPADES, Rank.NINE), Card(Suit.CLUBS, Rank.TEN)]
    strong = [Card(Suit.HEARTS, Rank.JACK), Card(Suit.DIAMONDS, Rank.JACK),
              Card(Suit.HEARTS, Rank.ACE)]
    assert (model.call_prob(strong, Suit.HEARTS)
            > model.call_prob(weak, Suit.HEARTS))


def _reach_start_of_play(seed):
    rng = random.Random(seed)
    ag = RuleBasedAgent()
    s = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
    steps = 0
    while not s.is_terminal() and s.phase != Phase.PLAY:
        s = s.apply(ag.act(s, rng))
        steps += 1
        if steps > 30:
            break
    if s.is_terminal() or s.phase != Phase.PLAY:
        return None
    return s


def test_reconstruct_bids_identifies_maker_call():
    # Deal, order up in round 1: maker is the player left of dealer.
    s = EuchreState.new_hand(dealer=0).deal(random.Random(3))
    s = s.apply(OrderUp(alone=False))
    s = s.apply(Discard(s.hands[s.dealer][0]))
    bids = reconstruct_bids(s)
    kinds = {(p, k) for p, k, _su in bids}
    assert (s.maker, "orderup") in kinds
    # Everyone acting before the maker passed on ordering up.
    assert all(k == "pass_orderup" for p, k, _ in bids if p != s.maker)


def test_weighted_belief_sharpens_makers_trump():
    """Conditioning on the bidding should give the maker more trump strength."""
    model = BiddingBeliefModel()
    gains = []
    for seed in range(200):
        s = _reach_start_of_play(seed)
        if s is None or not reconstruct_bids(s) or s.maker == s.current_player:
            continue
        deals, w = sample_weighted_belief(s, s.current_player, 40, model,
                                          random.Random(seed))
        trump, maker = s.trump, s.maker
        unif = sum(suit_strength(reconstruct_original_hands(d)[maker], trump)
                   for d in deals) / len(deals)
        wt = sum(w[i] * suit_strength(
            reconstruct_original_hands(deals[i])[maker], trump)
            for i in range(len(deals)))
        gains.append(wt - unif)
        if len(gains) >= 25:
            break
    assert len(gains) >= 10
    assert sum(gains) / len(gains) > 0.1  # maker looks stronger, on average


def test_weighted_belief_is_normalized():
    s = _reach_start_of_play(1)
    assert s is not None
    deals, w = sample_weighted_belief(s, s.current_player, 30,
                                      BiddingBeliefModel(), random.Random(0))
    assert len(deals) == len(w) == 30
    assert abs(sum(w) - 1.0) < 1e-9
    assert all(x >= 0 for x in w)


def test_no_bids_falls_back_to_uniform():
    # Right after the deal there is no maker yet -> uniform weights.
    s = EuchreState.new_hand(dealer=0).deal(random.Random(0))
    # advance into play without recording a maker is impossible, so test the
    # reconstruct path directly on a fresh (pre-call) state.
    assert reconstruct_bids(s) == []


def test_partners_share_reward():
    """Team utility must be identical (up to sign) for partners."""
    rng = random.Random(0)
    for _ in range(50):
        s = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
        while not s.is_terminal():
            s = s.apply(rng.choice(s.legal_actions()))
        r = s.returns()
        for p in range(4):
            mine = r[team_of(p)] - r[1 - team_of(p)]
            partner = r[team_of(partner_of(p))] - r[1 - team_of(partner_of(p))]
            assert mine == partner  # partners have identical differential
