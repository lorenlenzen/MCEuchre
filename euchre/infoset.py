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

from .cards import Card, Suit, NUM_CARDS
from .game import EuchreState, Phase


def _rel(seat: int, me: int) -> int:
    """Seat expressed relative to the acting player (0 = me, 1 = left, ...)."""
    return (seat - me) % 4


def infoset_key(state: EuchreState, player: int) -> str:
    """Canonical string for the information set of ``player`` at ``state``."""
    parts: List[str] = [f"ph{state.phase.value}", f"d{_rel(state.dealer, player)}"]

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


# --- Fixed-length observation vector ----------------------------------------

_PHASES = [Phase.BID_ROUND_1, Phase.BID_ROUND_2, Phase.DEALER_DISCARD,
           Phase.PLAY, Phase.TERMINAL]

# Segment sizes, summed into OBS_SIZE.
_SEG = {
    "hand": NUM_CARDS,          # 24  own cards (multi-hot)
    "up": NUM_CARDS,            # 24  up-card one-hot (bidding only)
    "up_flag": 1,               #  1  up-card visible?
    "trump": 5,                 #  5  none + 4 suits
    "turned_down": 5,           #  5  none + 4 suits
    "phase": len(_PHASES),      #  5
    "dealer_rel": 4,            #  4
    "maker_rel": 5,             #  5  none + 4 relative seats
    "alone": 1,                 #  1
    "played_by_seat": 4 * NUM_CARDS,  # 96  cards played this hand, by rel seat
    "trick_by_seat": 4 * NUM_CARDS,   # 96  cards in current trick, by rel seat
    "tricks_won": 2,            #  2  (my team, their team) counts / 5
    "to_lead": 1,               #  1  am I on lead this trick?
}
OBS_SIZE = sum(_SEG.values())


def _suit_onehot(vec: np.ndarray, base: int, suit) -> None:
    # index 0 = none, 1..4 = suit
    vec[base + (0 if suit is None else int(suit) + 1)] = 1.0


def observation_tensor(state: EuchreState, player: int) -> np.ndarray:
    v = np.zeros(OBS_SIZE, dtype=np.float32)
    o = 0

    for c in state.hands[player]:
        v[o + c.id] = 1.0
    o += _SEG["hand"]

    show_up = (state.up_card is not None and state.trump is None)
    if show_up:
        v[o + state.up_card.id] = 1.0
    o += _SEG["up"]
    v[o] = 1.0 if show_up else 0.0
    o += _SEG["up_flag"]

    _suit_onehot(v, o, state.trump); o += _SEG["trump"]
    _suit_onehot(v, o, state.turned_down); o += _SEG["turned_down"]

    v[o + _PHASES.index(state.phase)] = 1.0
    o += _SEG["phase"]

    v[o + _rel(state.dealer, player)] = 1.0
    o += _SEG["dealer_rel"]

    v[o + (0 if state.maker is None else _rel(state.maker, player) + 1)] = 1.0
    o += _SEG["maker_rel"]

    v[o] = 1.0 if state.alone else 0.0
    o += _SEG["alone"]

    for winner, plays in state.completed_tricks:
        for seat, c in plays:
            v[o + _rel(seat, player) * NUM_CARDS + c.id] = 1.0
    o += _SEG["played_by_seat"]

    for seat, c in state.current_trick:
        v[o + _rel(seat, player) * NUM_CARDS + c.id] = 1.0
    o += _SEG["trick_by_seat"]

    my_team = player % 2
    v[o] = state.tricks_won[my_team] / 5.0
    v[o + 1] = state.tricks_won[1 - my_team] / 5.0
    o += _SEG["tricks_won"]

    v[o] = 1.0 if (state.phase == Phase.PLAY and not state.current_trick) else 0.0
    o += _SEG["to_lead"]

    assert o == OBS_SIZE
    return v
