"""Public belief states and determinization.

A *determinization* is one full deal sampled consistently with everything a
player can observe: their own hand and all public information (cards played,
the up-card, void inferences from failures to follow suit). Sampling many
determinizations and solving each is the core of PIMC search and the raw
material for a ReBeL public belief state.

The sampler guarantees consistency:
* the querying player's hand is preserved exactly,
* every already-played card stays with the player who played it,
* the up-card is handled by how much its location is actually known:
  before a pickup it sits in its own (public) slot; after an order-up the
  opponents only know it is in the dealer's hand *or* was discarded to the
  kitty, so it is sampled between exactly those two destinations,
* opponents are never assigned a suit they have shown themselves void of.

Belief refinement with a learned network is a later milestone; the uniform
sampler here is the unbiased starting point.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set

from euchre.cards import Card, Suit, DECK, effective_suit
from euchre.game import EuchreState, Phase


def known_voids(state: EuchreState) -> Dict[int, Set[Suit]]:
    """Infer, from the play so far, which suits each seat cannot hold.

    A player who did not follow the led suit when they had the chance must be
    void in it (Euchre forces following suit). Uses effective suit, so the
    left bower is treated as trump.
    """
    voids: Dict[int, Set[Suit]] = {p: set() for p in range(4)}
    trump = state.trump
    if trump is None:
        return voids
    tricks = list(state.completed_tricks)
    if state.current_trick:
        tricks.append((-1, state.current_trick))
    for _winner, plays in tricks:
        if not plays:
            continue
        led = effective_suit(plays[0][1], trump)
        for seat, card in plays[1:]:
            if effective_suit(card, trump) != led:
                voids[seat].add(led)
    return voids


def _ordered_up(state: EuchreState) -> bool:
    """True if the up-card was accepted as trump in round 1 (dealer picks up)."""
    return (state.up_card is not None and state.trump is not None
            and state.maker is not None and state.turned_down is None)


def _pickup_happened(state: EuchreState) -> bool:
    """Back-compat alias: the up-card was ordered up and picked up."""
    return _ordered_up(state)


def sample_determinization(state: EuchreState, player: int,
                           rng: Optional[random.Random] = None,
                           max_tries: int = 400) -> EuchreState:
    """Sample a full-information ``EuchreState`` consistent with ``player``'s
    information. The returned state shares all public fields and the querying
    player's hand; hidden hands and kitty are a consistent random completion.
    """
    rng = rng or random
    voids = known_voids(state)
    trump = state.trump
    up = state.up_card
    ordered_up = _ordered_up(state)
    discard_done = len(state.kitty) == 4  # only a pickup discard grows kitty to 4

    # Cards whose location the player already knows for certain.
    known: Set[Card] = set(state.hands[player])
    for _w, plays in state.completed_tricks:
        known.update(c for _, c in plays)
    known.update(c for _, c in state.current_trick)

    # The up-card and kitty. Four cases:
    #  1. Not ordered up (bidding, or a round-2 call): the up-card sits in its
    #     own public slot (preserved by clone), never re-sampled; kitty hidden.
    #  2. Ordered up, querying player IS the dealer: the dealer knows their own
    #     hand and, once they have discarded, their discard (the last kitty
    #     card) -- so that kitty card is fixed.
    #  3. Ordered up, other player, before the discard: the up-card is known to
    #     be in the dealer's hand (it cannot be in the kitty yet).
    #  4. Ordered up, other player, after the discard: the up-card floats
    #     between the dealer's hand and the kitty (the possible discard).
    # ``up_constraint`` pins the up-card's allowed destinations for cases 3/4;
    # it stays in the unseen pool rather than being fixed as known.
    kitty_fixed: List[Card] = []
    up_constraint: Optional[dict] = None
    if up is not None:
        if not ordered_up:
            known.add(up)                                    # case 1
        elif player == state.dealer:                         # case 2
            known.add(up)
            if discard_done:
                discard = state.kitty[-1]
                known.add(discard)
                kitty_fixed.append(discard)
        elif not discard_done:                               # case 3
            up_constraint = {"seats": [state.dealer], "kitty": False}
        else:                                                # case 4
            up_constraint = {"seats": [state.dealer], "kitty": True}

    # Hidden slots to fill from the unseen pool.
    unseen = [c for c in DECK if c not in known]
    need = {p: len(state.hands[p]) for p in range(4)}
    need[player] = 0  # own hand already fixed
    kitty_target = len(state.kitty)
    kitty_slots = kitty_target - len(kitty_fixed)

    if sum(need.values()) + kitty_slots != len(unseen):
        raise RuntimeError("Determinization slot count mismatch "
                           f"(need={need}, kitty_slots={kitty_slots}, "
                           f"unseen={len(unseen)}).")

    for _ in range(max_tries):
        pool = list(unseen)
        rng.shuffle(pool)
        assign: Dict[int, List[Card]] = {p: list(state.hands[p]) if p == player
                                         else [] for p in range(4)}
        cur_need = dict(need)
        kitty: List[Card] = list(kitty_fixed)

        def destinations(card: Card) -> List[int]:
            """Seats with remaining room that may legally hold ``card``."""
            if up_constraint is not None and card == up:
                return [p for p in up_constraint["seats"] if cur_need[p] > 0]
            opts = []
            for p in range(4):
                if p == player or cur_need[p] <= 0:
                    continue
                if trump is not None and effective_suit(card, trump) in voids[p]:
                    continue
                opts.append(p)
            return opts

        def kitty_allowed(card: Card) -> bool:
            if up_constraint is not None and card == up:
                return up_constraint["kitty"]
            return True

        def place(i: int) -> bool:
            if i == len(pool):
                return (all(v == 0 for v in cur_need.values())
                        and len(kitty) == kitty_target)
            card = pool[i]
            opts = destinations(card)
            rng.shuffle(opts)
            for p in opts:
                assign[p].append(card)
                cur_need[p] -= 1
                if place(i + 1):
                    return True
                assign[p].pop()
                cur_need[p] += 1
            if kitty_allowed(card) and len(kitty) < kitty_target:
                kitty.append(card)
                if place(i + 1):
                    return True
                kitty.pop()
            return False

        if place(0):
            s = state.clone()
            s.hands = [assign[p] for p in range(4)]
            s.kitty = kitty
            return s

    raise RuntimeError("Could not sample a consistent determinization; "
                       "void constraints may be contradictory.")
