"""Information sets and observation encoding.

Two views of a state from one player's perspective:

* :func:`infoset_key` — a canonical string identifying the information set,
  used to key tabular CFR regret/strategy tables. Two histories a player
  cannot tell apart must produce the same key.
* :func:`observation_tensor` — a fixed-length float vector for neural
  networks. Positions are encoded *relative* to the acting player so the same
  strategic situation looks identical from any seat.

The hidden information (other players' hands, the kitty) is never included:
both views are strictly what ``player`` can legally observe.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .cards import (
    Card, Suit, Rank, RANKS, SUITS, NUM_CARDS, _RANK_INDEX,
    same_color_suit, effective_suit, is_trump, is_left_bower, is_right_bower,
)
from .game import EuchreState, Phase, team_of


def _rel(seat: int, me: int) -> int:
    """Seat expressed relative to the acting player (0 = me, 1 = left, ...)."""
    return (seat - me) % 4


def infoset_key(state: EuchreState, player: int) -> str:
    """Canonical string for the information set of ``player`` at ``state``."""
    my_score, their_score = (
        (state.team0_score, state.team1_score) if team_of(player) == 0
        else (state.team1_score, state.team0_score))
    parts: List[str] = [f"ph{state.phase.value}", f"d{_rel(state.dealer, player)}",
                        f"sc{my_score},{their_score}"]

    # Private hand (sorted by id for canonical order).
    hand_ids = sorted(c.id for c in state.hands[player])
    parts.append("h" + ",".join(map(str, hand_ids)))

    # Up-card is public during bidding.
    if state.phase in (Phase.BID_ROUND_1, Phase.BID_ROUND_2,
                       Phase.DEALER_DISCARD) and state.up_card is not None:
        parts.append(f"u{state.up_card.id}")
    if state.turned_down is not None:
        parts.append(f"td{int(state.turned_down)}")
    parts.append(f"b{state.bids_seen}")

    # Trump / maker once known.
    if state.trump is not None:
        parts.append(f"t{int(state.trump)}")
        parts.append(f"m{_rel(state.maker, player)}")
        parts.append(f"a{int(state.alone)}")

    # Public play history: completed tricks (winner + ordered plays) and the
    # current trick, all with relative seats.
    for winner, plays in state.completed_tricks:
        seq = ";".join(f"{_rel(pl, player)}:{c.id}" for pl, c in plays)
        parts.append(f"T{_rel(winner, player)}|{seq}")
    cur = ";".join(f"{_rel(pl, player)}:{c.id}" for pl, c in state.current_trick)
    parts.append(f"C{cur}")
    parts.append(f"w{state.tricks_won[0]},{state.tricks_won[1]}")
    return "/".join(parts)


# --- Fixed-length observation vector (suit-agnostic / trump-relative) --------
#
# Every suit-bearing feature is encoded by its ROLE relative to a reference
# suit R (= trump once set, else the up-card's suit during bidding), never by
# absolute suit identity. The two off-color ("green") suits share the single
# "green" role tag, so a network built on this encoding cannot develop a
# per-absolute-suit preference: relabeling the two greens (or any color-
# preserving suit permutation) permutes the observation's per-suit blocks
# without otherwise changing it. See docs/rebel_design.md and the plan.
#
# Layout: [ global block | 4 per-suit blocks (absolute slot order, role-
# relative contents) | 24 per-card feature blocks (card-id order) ]. The
# per-suit blocks stay in ABSOLUTE order so a shared network tower maps each
# to its own Call/Play output index by construction; only their *contents*
# are role-relative, which is what makes the tower suit-agnostic.

_PHASES = [Phase.BID_ROUND_1, Phase.BID_ROUND_2, Phase.DEALER_DISCARD,
           Phase.PLAY, Phase.TERMINAL]

# Race-to-target match score. A hand's starting score is always < this for
# both teams (the match ends, and no further hand is dealt, the instant a
# team reaches it), so team score / MATCH_TARGET always lands in [0, 1).
# rebel/match_equity.py imports this so the equity table and the observation
# encoding always agree on what "match" means.
MATCH_TARGET = 10

# Roles of a suit relative to the reference suit R.
_ROLE_REF, _ROLE_NEXT, _ROLE_GREEN = 0, 1, 2
N_ROLES = 3

# The 7 trump-rank "slots" for the as-if-this-suit-were-trump holdings, in
# strength order: right bower, left bower, then A/K/Q/10/9 of the suit.
_TRUMP_HOLDING_RANKS = [Rank.ACE, Rank.KING, Rank.QUEEN, Rank.TEN, Rank.NINE]

# --- segment dims ---
_GLOBAL = (
    len(_PHASES)   # phase one-hot
    + 4            # dealer, relative seat
    + 5            # maker: none + 4 relative seats
    + 1            # alone
    + 2            # tricks won: mine, theirs (/5)
    + 1            # am I on lead this trick?
    + 1            # bids_seen (/7)
    + 1            # up-card visible?
    + 6            # up-card rank one-hot (belongs to the reference suit)
    + 4            # led-card role this trick: none/ref/next/green
    + 2            # match score: mine, theirs (/MATCH_TARGET)
)  # = 32

_SUIT_BLOCK = (
    N_ROLES        # role one-hot (ref/next/green)
    + 1            # is the turned-down suit (illegal to call in round 2)
    + 1            # is the actual trump suit
    + 7            # as-if-trump holdings: RB, LB, A, K, Q, 10, 9
    + 1            # as-if-trump count (/5)
    + 6            # effective plain-suit holdings under actual trump (by rank)
    + 1            # void in this effective suit
    + 4            # cards of this effective suit played, per relative seat (/5)
    + 1            # cards of this suit seen so far (/6)
)  # = 25
NUM_SUITS = 4

_CARD_FEAT = (
    6              # rank one-hot
    + 1            # in my hand
    + 1            # is right bower (actual trump)
    + 1            # is left bower (actual trump)
    + 1            # is effective trump (actual trump)
    + 1            # has been played this hand
)  # = 11

GLOBAL_OFF = 0
SUIT_OFF = _GLOBAL
SUIT_BLOCK_DIM = _SUIT_BLOCK
GLOBAL_DIM = _GLOBAL
CARD_FEAT_DIM = _CARD_FEAT
CARD_OFF = _GLOBAL + NUM_SUITS * _SUIT_BLOCK
OBS_SIZE = _GLOBAL + NUM_SUITS * _SUIT_BLOCK + NUM_CARDS * _CARD_FEAT


def _reference_suit(state: EuchreState):
    """The suit everything is encoded relative to: trump once set, else the
    up-card's suit during bidding. None only in DEAL/TERMINAL (no decision)."""
    if state.trump is not None:
        return state.trump
    if state.up_card is not None:
        return state.up_card.suit
    return None


def _role_of(suit: Suit, ref) -> int:
    if ref is None:
        return _ROLE_GREEN  # no reference (DEAL/TERMINAL); role is unused
    if suit == ref:
        return _ROLE_REF
    if suit == same_color_suit(ref):
        return _ROLE_NEXT
    return _ROLE_GREEN


def observation_tensor(state: EuchreState, player: int) -> np.ndarray:
    v = np.zeros(OBS_SIZE, dtype=np.float32)
    hand = state.hands[player]
    hand_set = set(hand)
    trump = state.trump
    ref = _reference_suit(state)
    show_up = (state.up_card is not None and trump is None)

    # ---- global block ----
    o = GLOBAL_OFF
    v[o + _PHASES.index(state.phase)] = 1.0
    o += len(_PHASES)
    v[o + _rel(state.dealer, player)] = 1.0
    o += 4
    v[o + (0 if state.maker is None else _rel(state.maker, player) + 1)] = 1.0
    o += 5
    v[o] = 1.0 if state.alone else 0.0
    o += 1
    my_team = player % 2
    v[o] = state.tricks_won[my_team] / 5.0
    v[o + 1] = state.tricks_won[1 - my_team] / 5.0
    o += 2
    my_score = state.team0_score if my_team == 0 else state.team1_score
    their_score = state.team1_score if my_team == 0 else state.team0_score
    v[o] = my_score / MATCH_TARGET
    v[o + 1] = their_score / MATCH_TARGET
    o += 2
    v[o] = 1.0 if (state.phase == Phase.PLAY and not state.current_trick) else 0.0
    o += 1
    v[o] = state.bids_seen / 7.0
    o += 1
    v[o] = 1.0 if show_up else 0.0
    o += 1
    if show_up:
        v[o + _RANK_INDEX[state.up_card.rank]] = 1.0
    o += 6
    # led-card role this trick: index 0 = no led card, else role+1
    if state.current_trick:
        led_role = _role_of(effective_suit(state.current_trick[0][1], trump), ref)
        v[o + led_role + 1] = 1.0
    else:
        v[o] = 1.0
    o += 4
    assert o == SUIT_OFF

    # ---- per-suit blocks (absolute slot, role-relative contents) ----
    # Precompute public play info once.
    played_cards = []  # (rel_seat, card)
    for _winner, plays in state.completed_tricks:
        for seat, c in plays:
            played_cards.append((_rel(seat, player), c))
    for seat, c in state.current_trick:
        played_cards.append((_rel(seat, player), c))

    for s in SUITS:
        o = SUIT_OFF + int(s) * _SUIT_BLOCK
        v[o + _role_of(s, ref)] = 1.0
        o += N_ROLES
        v[o] = 1.0 if state.turned_down == s else 0.0
        o += 1
        v[o] = 1.0 if trump == s else 0.0
        o += 1
        # as-if-s-were-trump holdings: right bower, left bower, A, K, Q, 10, 9
        v[o] = 1.0 if Card(s, Rank.JACK) in hand_set else 0.0
        v[o + 1] = 1.0 if Card(same_color_suit(s), Rank.JACK) in hand_set else 0.0
        for i, r in enumerate(_TRUMP_HOLDING_RANKS):
            v[o + 2 + i] = 1.0 if Card(s, r) in hand_set else 0.0
        o += 7
        as_if_trump = sum(1 for c in hand if is_trump(c, s))
        v[o] = as_if_trump / 5.0
        o += 1
        # effective plain-suit holdings under ACTUAL trump (by rank)
        eff_here = [c for c in hand if effective_suit(c, trump) == s]
        for c in eff_here:
            v[o + _RANK_INDEX[c.rank]] = 1.0
        o += 6
        v[o] = 1.0 if not eff_here else 0.0
        o += 1
        # cards of this effective suit played, per relative seat (/5)
        for rel_seat, c in played_cards:
            if effective_suit(c, trump) == s:
                v[o + rel_seat] += 1.0 / 5.0
        o += 4
        # cards of this (effective) suit seen so far. Divisor 7 because the
        # effective trump suit spans 7 cards (its own 6 + the left bower); all
        # other effective suits are smaller, so this keeps the feature in
        # [0, 1] for every suit/role.
        seen = len(eff_here) + sum(1 for _rs, c in played_cards
                                   if effective_suit(c, trump) == s)
        if show_up and effective_suit(state.up_card, trump) == s:
            seen += 1
        v[o] = seen / 7.0
        o += 1

    # ---- per-card feature blocks (card-id order) ----
    played_set = {c for _rs, c in played_cards}
    for cid in range(NUM_CARDS):
        c = Card.from_id(cid)
        o = CARD_OFF + cid * _CARD_FEAT
        v[o + _RANK_INDEX[c.rank]] = 1.0
        o += 6
        v[o] = 1.0 if c in hand_set else 0.0
        o += 1
        v[o] = 1.0 if (trump is not None and is_right_bower(c, trump)) else 0.0
        o += 1
        v[o] = 1.0 if (trump is not None and is_left_bower(c, trump)) else 0.0
        o += 1
        v[o] = 1.0 if (trump is not None and is_trump(c, trump)) else 0.0
        o += 1
        v[o] = 1.0 if c in played_set else 0.0
        o += 1

    return v
