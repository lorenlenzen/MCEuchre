"""Action representation for Euchre.

Actions are lightweight, hashable objects so they can key regret/strategy
tables in CFR. A flat integer encoding (``action_to_index`` /
``index_from_action``) is provided for neural-network policy heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union

from .cards import Card, Suit, DECK, SUITS


class BidKind(Enum):
    PASS = "pass"
    ORDER_UP = "order_up"      # round 1: accept the up-card suit as trump
    CALL = "call"              # round 2: name a suit as trump


@dataclass(frozen=True)
class Pass:
    def __str__(self) -> str:
        return "pass"


@dataclass(frozen=True)
class OrderUp:
    """Round-1 call: accept the turned-up suit. ``alone`` = go it alone."""
    alone: bool = False

    def __str__(self) -> str:
        return "order_up" + (" alone" if self.alone else "")


@dataclass(frozen=True)
class Call:
    """Round-2 call: name a trump suit (not the turned-down suit)."""
    suit: Suit
    alone: bool = False

    def __str__(self) -> str:
        return f"call {self.suit.symbol}" + (" alone" if self.alone else "")


@dataclass(frozen=True)
class Discard:
    """Dealer discards one card after picking up the up-card."""
    card: Card

    def __str__(self) -> str:
        return f"discard {self.card}"


@dataclass(frozen=True)
class Play:
    card: Card

    def __str__(self) -> str:
        return f"play {self.card}"


Action = Union[Pass, OrderUp, Call, Discard, Play]


# --- Flat action index space (for NN policy heads) --------------------------
#
# Layout (total 24 + 24 + 4 + 8 + 1 = 61):
#   [0, 24)   Play card c            -> c.id
#   [24, 48)  Discard card c         -> 24 + c.id
#   [48, 52)  Call suit s (not alone)-> 48 + int(s)
#   [52, 56)  Call suit s alone      -> 52 + int(s)
#   56        OrderUp (not alone)
#   57        OrderUp alone
#   58        Pass
NUM_ACTIONS = 59

_PLAY_BASE = 0
_DISCARD_BASE = 24
_CALL_BASE = 48
_CALL_ALONE_BASE = 52
_ORDER_UP = 56
_ORDER_UP_ALONE = 57
_PASS = 58


def action_to_index(action: Action) -> int:
    if isinstance(action, Play):
        return _PLAY_BASE + action.card.id
    if isinstance(action, Discard):
        return _DISCARD_BASE + action.card.id
    if isinstance(action, Call):
        base = _CALL_ALONE_BASE if action.alone else _CALL_BASE
        return base + int(action.suit)
    if isinstance(action, OrderUp):
        return _ORDER_UP_ALONE if action.alone else _ORDER_UP
    if isinstance(action, Pass):
        return _PASS
    raise ValueError(f"Unknown action: {action!r}")


def index_to_action(index: int) -> Action:
    if _PLAY_BASE <= index < _DISCARD_BASE:
        return Play(Card.from_id(index - _PLAY_BASE))
    if _DISCARD_BASE <= index < _CALL_BASE:
        return Discard(Card.from_id(index - _DISCARD_BASE))
    if _CALL_BASE <= index < _CALL_ALONE_BASE:
        return Call(SUITS[index - _CALL_BASE], alone=False)
    if _CALL_ALONE_BASE <= index < _ORDER_UP:
        return Call(SUITS[index - _CALL_ALONE_BASE], alone=True)
    if index == _ORDER_UP:
        return OrderUp(alone=False)
    if index == _ORDER_UP_ALONE:
        return OrderUp(alone=True)
    if index == _PASS:
        return Pass()
    raise ValueError(f"Action index out of range: {index}")
