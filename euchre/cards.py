"""Card model for Euchre.

The 24-card Euchre deck: 9, 10, J, Q, K, A in each of four suits.

The subtle rule that every Euchre engine must get right is the *left bower*:
when a suit is trump, the Jack of the same color counts as a trump card
(the second-highest trump) and, crucially, no longer belongs to its printed
suit for the purposes of following suit. All of that logic lives here so the
rest of the engine can treat "effective suit" and "trick strength" as simple
lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    @property
    def symbol(self) -> str:
        return {"CLUBS": "♣", "DIAMONDS": "♦",
                "HEARTS": "♥", "SPADES": "♠"}[self.name]


class Rank(IntEnum):
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def symbol(self) -> str:
        return {9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}[self.value]


# Ordered list of ranks, low to high, used to build the deck and index cards.
RANKS: List[Rank] = [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
SUITS: List[Suit] = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]

# Precomputed rank -> index, to keep Card.id off the O(n) list.index path.
_RANK_INDEX = {rank: i for i, rank in enumerate(RANKS)}

# Same-color partner suit. Left bower = Jack of the trump's same-color suit.
_SAME_COLOR = {
    Suit.CLUBS: Suit.SPADES,
    Suit.SPADES: Suit.CLUBS,
    Suit.DIAMONDS: Suit.HEARTS,
    Suit.HEARTS: Suit.DIAMONDS,
}


def same_color_suit(suit: Suit) -> Suit:
    """Return the other suit of the same color."""
    return _SAME_COLOR[suit]


@dataclass(frozen=True, order=True)
class Card:
    suit: Suit
    rank: Rank

    @property
    def id(self) -> int:
        """Stable index in [0, 24) for one-hot encoding: suit * 6 + rank_index.

        Uses a precomputed rank index rather than ``RANKS.index`` -- this is on
        the hottest path in CFR/solver search (millions of calls).
        """
        return int(self.suit) * 6 + _RANK_INDEX[self.rank]

    def __str__(self) -> str:
        return f"{self.rank.symbol}{self.suit.symbol}"

    def __repr__(self) -> str:
        return f"Card({self.suit.name}, {self.rank.name})"

    @staticmethod
    def from_id(card_id: int) -> "Card":
        suit = SUITS[card_id // 6]
        rank = RANKS[card_id % 6]
        return Card(suit, rank)


def make_deck() -> List[Card]:
    """Return a fresh, ordered 24-card Euchre deck."""
    return [Card(s, r) for s in SUITS for r in RANKS]


DECK: List[Card] = make_deck()
NUM_CARDS = len(DECK)  # 24


# --- Trump-aware helpers ----------------------------------------------------

def is_right_bower(card: Card, trump: Suit) -> bool:
    return card.rank == Rank.JACK and card.suit == trump


def is_left_bower(card: Card, trump: Suit) -> bool:
    return card.rank == Rank.JACK and card.suit == same_color_suit(trump)


def is_trump(card: Card, trump: Suit) -> bool:
    """A card is trump if it is in the trump suit OR it is the left bower."""
    return card.suit == trump or is_left_bower(card, trump)


def effective_suit(card: Card, trump: Optional[Suit]) -> Suit:
    """The suit this card behaves as. Left bower behaves as trump.

    With no trump set (during the deal / before a call) the effective suit is
    just the printed suit.
    """
    if trump is not None and is_left_bower(card, trump):
        return trump
    return card.suit


# Strength of a card *within the trump suit*, high value = stronger.
# Right bower > left bower > A > K > Q > 10 > 9 (no plain Jack: it is a bower).
_TRUMP_RANK_STRENGTH = {
    Rank.ACE: 6,
    Rank.KING: 5,
    Rank.QUEEN: 4,
    Rank.TEN: 3,
    Rank.NINE: 2,
}

# Strength within a plain (non-trump) suit. A > K > Q > J > 10 > 9.
_PLAIN_RANK_STRENGTH = {
    Rank.ACE: 6,
    Rank.KING: 5,
    Rank.QUEEN: 4,
    Rank.JACK: 3,
    Rank.TEN: 2,
    Rank.NINE: 1,
}


def card_strength(card: Card, trump: Suit, led: Suit) -> int:
    """Comparable strength for resolving a trick.

    Trump cards beat all led-suit cards, which beat everything off-suit.
    Only cards of the effective led suit or trump can ever win a trick.
    """
    if is_right_bower(card, trump):
        return 200 + 8
    if is_left_bower(card, trump):
        return 200 + 7
    if card.suit == trump:
        return 200 + _TRUMP_RANK_STRENGTH[card.rank]
    if effective_suit(card, trump) == led:
        return 100 + _PLAIN_RANK_STRENGTH[card.rank]
    return _PLAIN_RANK_STRENGTH[card.rank]  # cannot win; ordering only


def trick_winner(plays: List["tuple[int, Card]"], trump: Suit) -> int:
    """Given ordered (player, card) plays for a trick, return the winning player.

    The first play sets the led suit (its effective suit).
    """
    led = effective_suit(plays[0][1], trump)
    best_player = plays[0][0]
    best_strength = card_strength(plays[0][1], trump, led)
    for player, card in plays[1:]:
        s = card_strength(card, trump, led)
        if s > best_strength:
            best_strength = s
            best_player = player
    return best_player
