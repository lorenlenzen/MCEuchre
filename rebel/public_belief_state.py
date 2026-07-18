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


def _pickup_happened(state: EuchreState) -> bool:
    """True if the up-card was ordered up and the dealer picked it up."""
    return (state.up_card is not None and state.trump is not None
            and state.maker is not None and state.turned_down is None)


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
    pickup = _pickup_happened(state)

    # Cards whose location the player already knows for certain.
    known: Set[Card] = set(state.hands[player])
    for _w, plays in state.completed_tricks:
        known.update(c for _, c in plays)
    known.update(c for _, c in state.current_trick)

    # The up-card. Before any pickup it lives in its own public slot (preserved
    # by clone) and is never re-sampled. After a pickup, only the dealer knows
    # where it went; other players sample it between the dealer's hand and the
    # kitty (the possible discard).
    if up is not None and (not pickup or player == state.dealer):
        known.add(up)  # public slot, or dealer knows their own hand/discard
    up_is_floating = pickup and player != state.dealer and up is not None

    # Hidden slots to fill from the unseen pool.
    unseen = [c for c in DECK if c not in known]
    need = {p: len(state.hands[p]) for p in range(4)}
    need[player] = 0  # own hand already fixed
    kitty_target = len(state.kitty)

    if sum(need.values()) + kitty_target != len(unseen):
        raise RuntimeError("Determinization slot count mismatch "
                           f"(need={need}, kitty={kitty_target}, "
                           f"unseen={len(unseen)}).")

    for _ in range(max_tries):
        pool = list(unseen)
        rng.shuffle(pool)
        assign: Dict[int, List[Card]] = {p: list(state.hands[p]) if p == player
                                         else [] for p in range(4)}
        cur_need = dict(need)
        kitty: List[Card] = []

        def destinations(card: Card) -> List[int]:
            """Seats with remaining room that may legally hold ``card``."""
            if up_is_floating and card == up:
                # Floating up-card may only be in the dealer's hand (or kitty).
                return [state.dealer] if cur_need[state.dealer] > 0 else []
            opts = []
            for p in range(4):
                if p == player or cur_need[p] <= 0:
                    continue
                if trump is not None and effective_suit(card, trump) in voids[p]:
                    continue
                opts.append(p)
            return opts

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
            if len(kitty) < kitty_target:
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
