"""Belief refinement conditioned on the bidding.

The uniform determinization sampler treats every deal consistent with the
public cards as equally likely. But the *bidding* is highly informative: a
player who ordered up or called a suit almost certainly holds strong trump in
it, and everyone who passed probably does not. Ignoring that makes the search
reason about opponents' hands that no sensible player would have bid the way
they did.

This module reweights a sampled belief by how well each deal explains the
observed bids, under a simple, monotonic model of how players call (stronger
hand in a suit -> more likely to make it trump; very strong -> more likely to
go alone). The weights feed the CFR chance-reach / PIMC averaging, so search
concentrates on plausible worlds.

Scope and approximation: refinement applies once trump is known (the maker has
been decided). Each player's *bidding-time* hand is reconstructed as their
current hand plus the cards they have since played; for the dealer after a
pickup this is their post-pickup hand, which is the right hand to judge their
order-up on anyway. The model is deliberately soft -- only the monotonicity
(more trump strength -> more likely to have bid for it) needs to be right for
the belief to sharpen in the correct direction.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from euchre.cards import (
    Card, Suit, Rank, SUITS,
    is_trump, is_right_bower, is_left_bower,
)
from euchre.game import EuchreState
from .public_belief_state import sample_determinization


def suit_strength(hand: Sequence[Card], suit: Suit) -> float:
    """A soft score for how much a hand wants ``suit`` to be trump."""
    s = 0.0
    for c in hand:
        if is_trump(c, suit):
            if is_right_bower(c, suit):
                s += 3.0
            elif is_left_bower(c, suit):
                s += 2.5
            elif c.rank == Rank.ACE:
                s += 2.0
            elif c.rank == Rank.KING:
                s += 1.5
            else:
                s += 1.0
        elif c.rank == Rank.ACE:
            s += 0.5  # off-suit ace is worth something
    return s


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class BiddingBeliefModel:
    """A monotonic soft model of calling behaviour."""

    def __init__(self, call_thresh: float = 3.0, temp: float = 1.2,
                 alone_thresh: float = 6.5, pickup_bonus: float = 1.0) -> None:
        self.call_thresh = call_thresh
        self.temp = temp
        self.alone_thresh = alone_thresh
        self.pickup_bonus = pickup_bonus

    def call_prob(self, hand: Sequence[Card], suit: Suit,
                  is_dealer: bool = False, pickup: bool = False) -> float:
        s = suit_strength(hand, suit)
        if pickup and is_dealer:
            s += self.pickup_bonus  # the dealer would gain the up-card
        return _sigmoid((s - self.call_thresh) / self.temp)

    def best_call_prob(self, hand: Sequence[Card], exclude: Suit) -> float:
        return max(self.call_prob(hand, S) for S in SUITS if S != exclude)

    def alone_prob(self, hand: Sequence[Card], suit: Suit) -> float:
        return _sigmoid((suit_strength(hand, suit) - self.alone_thresh)
                        / self.temp)


# -- reconstruction ----------------------------------------------------------

def reconstruct_original_hands(state: EuchreState) -> List[Set[Card]]:
    """Each player's bidding-time hand = current hand + cards they have played."""
    orig = [set(state.hands[p]) for p in range(4)]
    for _w, plays in state.completed_tricks:
        for seat, c in plays:
            orig[seat].add(c)
    for seat, c in state.current_trick:
        orig[seat].add(c)
    return orig


def reconstruct_bids(state: EuchreState) -> List[Tuple[int, str, Suit]]:
    """Recover the sequence of informative bids from the resolved public state.

    Returns (player, kind, suit) with kind one of: ``pass_orderup`` (declined
    to order up the up-card), ``orderup``, ``pass_call`` (declined to call any
    suit in round 2), ``call``.
    """
    if state.maker is None or state.up_card is None:
        return []
    dealer = state.dealer
    maker = state.maker
    up_suit = state.up_card.suit
    order = [(dealer + 1 + i) % 4 for i in range(4)]
    bids: List[Tuple[int, str, Suit]] = []

    if state.turned_down is None:
        # Maker ordered up in round 1; everyone before them passed.
        for q in order:
            if q == maker:
                bids.append((q, "orderup", up_suit))
                break
            bids.append((q, "pass_orderup", up_suit))
    else:
        # Everyone passed round 1, then the maker called in round 2.
        for q in order:
            bids.append((q, "pass_orderup", up_suit))
        for q in order:
            if q == maker:
                bids.append((q, "call", state.trump))
                break
            bids.append((q, "pass_call", state.turned_down))
    return bids


def deal_log_weight(hands: List[Set[Card]],
                    bids: List[Tuple[int, str, Suit]],
                    model: BiddingBeliefModel, alone: bool,
                    dealer: int) -> float:
    lw = 0.0
    for player, kind, suit in bids:
        h = hands[player]
        if kind == "orderup":
            p = model.call_prob(h, suit, is_dealer=(player == dealer),
                                pickup=True)
            p *= model.alone_prob(h, suit) if alone else \
                (1.0 - model.alone_prob(h, suit))
        elif kind == "pass_orderup":
            p = 1.0 - model.call_prob(h, suit, is_dealer=(player == dealer),
                                      pickup=True)
        elif kind == "call":
            p = model.call_prob(h, suit)
            p *= model.alone_prob(h, suit) if alone else \
                (1.0 - model.alone_prob(h, suit))
        else:  # pass_call
            p = 1.0 - model.best_call_prob(h, exclude=suit)
        lw += math.log(max(p, 1e-6))
    return lw


def sample_weighted_belief(state: EuchreState, actor: int, num_deals: int,
                           model: Optional[BiddingBeliefModel] = None,
                           rng: Optional[random.Random] = None
                           ) -> Tuple[List[EuchreState], List[float]]:
    """Sample ``num_deals`` determinizations and weight them by bid consistency.

    Returns (deals, weights) with weights summing to 1. Falls back to uniform
    weights when there is no bidding information to condition on.
    """
    rng = rng or random.Random()
    model = model or BiddingBeliefModel()
    bids = reconstruct_bids(state)
    deals = [sample_determinization(state, actor, rng) for _ in range(num_deals)]
    if not bids:
        w = 1.0 / num_deals
        return deals, [w] * num_deals

    log_w = [deal_log_weight(reconstruct_original_hands(d), bids, model,
                             state.alone, state.dealer)
             for d in deals]
    m = max(log_w)
    weights = [math.exp(l - m) for l in log_w]
    total = sum(weights)
    return deals, [x / total for x in weights]
