"""Euchre game engine: state, legal actions, transitions, and scoring.

Design notes
------------
* One *hand* is one episode: deal -> bidding -> play (5 tricks) -> score.
  The running game-to-10 score is meta-state that a wrapper can track; ReBeL
  operates per hand with the hand's point differential as the value.
* Players are 0..3 seated clockwise. Partnerships are fixed:
  team 0 = {0, 2}, team 1 = {1, 3}.
* The state is treated as immutable from the caller's perspective:
  :meth:`EuchreState.apply` returns a new state via a cheap clone.
* Only the standard variant is implemented (makers may go alone; defenders
  may not). "Stick the dealer" is a constructor flag.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from .actions import Action, Call, Discard, OrderUp, Pass, Play
from .cards import (
    Card,
    Suit,
    DECK,
    effective_suit,
    is_trump,
    same_color_suit,
    trick_winner,
)

CHANCE = -1  # sentinel "player" for the deal (chance node)


class Phase(Enum):
    DEAL = auto()          # chance node: deal the cards
    BID_ROUND_1 = auto()   # order up the turned card, or pass
    BID_ROUND_2 = auto()   # name a suit (not the turned-down one), or pass
    DEALER_DISCARD = auto()  # dealer discards after a pickup
    PLAY = auto()          # trick play
    TERMINAL = auto()


def team_of(player: int) -> int:
    return player % 2


def partner_of(player: int) -> int:
    return (player + 2) % 4


@dataclass
class EuchreState:
    dealer: int
    phase: Phase = Phase.DEAL
    current_player: int = CHANCE

    # Cards
    hands: List[List[Card]] = field(default_factory=lambda: [[], [], [], []])
    up_card: Optional[Card] = None       # turned-up card during bidding
    kitty: List[Card] = field(default_factory=list)  # hidden undealt cards

    # Trump / calling
    trump: Optional[Suit] = None
    maker: Optional[int] = None
    alone: bool = False
    lone_player: Optional[int] = None    # the player going alone (maker)
    sitting: Optional[int] = None        # partner of lone player, sits out
    turned_down: Optional[Suit] = None   # up-card suit rejected in round 1

    # Bidding bookkeeping
    bids_seen: int = 0                   # passes/decisions this round
    round1_pickup_pending: bool = False  # up-card ordered up, awaiting discard

    # Play bookkeeping
    trick_leader: int = 0
    current_trick: List[Tuple[int, Card]] = field(default_factory=list)
    completed_tricks: List[Tuple[int, List[Tuple[int, Card]]]] = field(
        default_factory=list
    )  # (winner, plays)
    tricks_won: List[int] = field(default_factory=lambda: [0, 0])  # per team

    # Config
    stick_the_dealer: bool = False

    # Result (set at terminal)
    reward: Optional[Tuple[int, int]] = None  # points to (team0, team1)

    # -- construction --------------------------------------------------------

    def clone(self) -> "EuchreState":
        """Fast copy for search. ``Card`` is immutable and ``apply`` always
        reassigns whole lists (never mutates them in place), so a one-level
        copy of the mutable containers is sufficient -- and far cheaper than
        ``copy.deepcopy``, which dominates CFR/MCCFR runtime.
        """
        s = EuchreState.__new__(EuchreState)
        s.dealer = self.dealer
        s.phase = self.phase
        s.current_player = self.current_player
        s.hands = [list(h) for h in self.hands]
        s.up_card = self.up_card
        s.kitty = list(self.kitty)
        s.trump = self.trump
        s.maker = self.maker
        s.alone = self.alone
        s.lone_player = self.lone_player
        s.sitting = self.sitting
        s.turned_down = self.turned_down
        s.bids_seen = self.bids_seen
        s.round1_pickup_pending = self.round1_pickup_pending
        s.trick_leader = self.trick_leader
        s.current_trick = list(self.current_trick)
        s.completed_tricks = list(self.completed_tricks)
        s.tricks_won = list(self.tricks_won)
        s.stick_the_dealer = self.stick_the_dealer
        s.reward = self.reward
        return s

    @staticmethod
    def new_hand(dealer: int = 0, stick_the_dealer: bool = False) -> "EuchreState":
        return EuchreState(
            dealer=dealer,
            phase=Phase.DEAL,
            current_player=CHANCE,
            stick_the_dealer=stick_the_dealer,
        )

    # -- dealing (chance) ----------------------------------------------------

    def deal(self, rng: Optional[random.Random] = None) -> "EuchreState":
        """Resolve the chance node: shuffle and deal. Returns a new state."""
        assert self.phase == Phase.DEAL
        rng = rng or random
        deck = list(DECK)
        rng.shuffle(deck)
        s = self.clone()
        s.hands = [deck[i * 5:(i + 1) * 5] for i in range(4)]
        rest = deck[20:]
        s.up_card = rest[0]
        s.kitty = rest[1:]  # 3 hidden cards
        s.phase = Phase.BID_ROUND_1
        s.current_player = (self.dealer + 1) % 4
        s.bids_seen = 0
        return s

    def deal_from(self, hands: List[List[Card]], up_card: Card,
                  kitty: List[Card]) -> "EuchreState":
        """Deterministic deal from explicit cards (for tests / determinization)."""
        assert self.phase == Phase.DEAL
        s = self.clone()
        s.hands = [list(h) for h in hands]
        s.up_card = up_card
        s.kitty = list(kitty)
        s.phase = Phase.BID_ROUND_1
        s.current_player = (self.dealer + 1) % 4
        s.bids_seen = 0
        return s

    # -- legal actions -------------------------------------------------------

    def legal_actions(self) -> List[Action]:
        if self.phase == Phase.BID_ROUND_1:
            return [Pass(), OrderUp(alone=False), OrderUp(alone=True)]
        if self.phase == Phase.BID_ROUND_2:
            actions: List[Action] = [Pass()]
            for suit in Suit:
                if suit == self.turned_down:
                    continue
                actions.append(Call(suit, alone=False))
                actions.append(Call(suit, alone=True))
            # Stick-the-dealer: the dealer, acting last, may not pass.
            if (self.stick_the_dealer
                    and self.current_player == self.dealer
                    and self.bids_seen == 3):
                actions = [a for a in actions if not isinstance(a, Pass)]
            return actions
        if self.phase == Phase.DEALER_DISCARD:
            return [Discard(c) for c in self.hands[self.dealer]]
        if self.phase == Phase.PLAY:
            return [Play(c) for c in self._legal_plays(self.current_player)]
        return []

    def _legal_plays(self, player: int) -> List[Card]:
        hand = self.hands[player]
        if not self.current_trick:
            return list(hand)  # leading: anything
        led = effective_suit(self.current_trick[0][1], self.trump)
        follow = [c for c in hand if effective_suit(c, self.trump) == led]
        return follow if follow else list(hand)

    # -- transitions ---------------------------------------------------------

    def apply(self, action: Action) -> "EuchreState":
        if self.phase == Phase.BID_ROUND_1:
            return self._apply_bid1(action)
        if self.phase == Phase.BID_ROUND_2:
            return self._apply_bid2(action)
        if self.phase == Phase.DEALER_DISCARD:
            return self._apply_discard(action)
        if self.phase == Phase.PLAY:
            return self._apply_play(action)
        raise ValueError(f"No actions from phase {self.phase}")

    def _apply_bid1(self, action: Action) -> "EuchreState":
        s = self.clone()
        if isinstance(action, Pass):
            s.bids_seen += 1
            if s.bids_seen == 4:
                # All passed round 1: turn the up-card down, move to round 2.
                s.turned_down = self.up_card.suit
                s.phase = Phase.BID_ROUND_2
                s.current_player = (self.dealer + 1) % 4
                s.bids_seen = 0
            else:
                s.current_player = (self.current_player + 1) % 4
            return s
        if isinstance(action, OrderUp):
            s.trump = self.up_card.suit
            s.maker = self.current_player
            s._set_alone(action.alone, self.current_player)
            # Dealer picks up the up-card, then must discard.
            s.hands[self.dealer] = list(s.hands[self.dealer]) + [self.up_card]
            s.phase = Phase.DEALER_DISCARD
            s.current_player = self.dealer
            return s
        raise ValueError(f"Illegal round-1 bid: {action}")

    def _apply_bid2(self, action: Action) -> "EuchreState":
        s = self.clone()
        if isinstance(action, Pass):
            s.bids_seen += 1
            # With stick-the-dealer the dealer cannot pass, so 4 passes only
            # happens in the no-stick variant -> hand is thrown in (misdeal).
            if s.bids_seen == 4:
                s.phase = Phase.TERMINAL
                s.reward = (0, 0)
                s.current_player = CHANCE
                return s
            s.current_player = (self.current_player + 1) % 4
            return s
        if isinstance(action, Call):
            if action.suit == self.turned_down:
                raise ValueError("Cannot call the turned-down suit")
            s.trump = action.suit
            s.maker = self.current_player
            s._set_alone(action.alone, self.current_player)
            s._begin_play()
            return s
        raise ValueError(f"Illegal round-2 bid: {action}")

    def _apply_discard(self, action: Action) -> "EuchreState":
        if not isinstance(action, Discard):
            raise ValueError("Expected a discard")
        s = self.clone()
        hand = list(s.hands[self.dealer])
        hand.remove(action.card)
        s.hands[self.dealer] = hand
        s.kitty = list(s.kitty) + [action.card]
        s._begin_play()
        return s

    def _apply_play(self, action: Action) -> "EuchreState":
        if not isinstance(action, Play):
            raise ValueError("Expected a play")
        if action.card not in self._legal_plays(self.current_player):
            raise ValueError(f"Illegal play {action.card} for player "
                             f"{self.current_player}")
        s = self.clone()
        hand = list(s.hands[self.current_player])
        hand.remove(action.card)
        s.hands[self.current_player] = hand
        s.current_trick = list(s.current_trick) + [(self.current_player, action.card)]

        expected = 3 if self.alone else 4
        if len(s.current_trick) == expected:
            winner = trick_winner(s.current_trick, s.trump)
            s.completed_tricks = list(s.completed_tricks) + [
                (winner, s.current_trick)
            ]
            s.tricks_won = list(s.tricks_won)
            s.tricks_won[team_of(winner)] += 1
            s.current_trick = []
            if len(s.completed_tricks) == 5:
                s._finish_hand()
            else:
                s.trick_leader = winner
                s.current_player = winner
        else:
            s.current_player = s._next_player(self.current_player)
        return s

    # -- helpers -------------------------------------------------------------

    def _set_alone(self, alone: bool, maker: int) -> None:
        self.alone = alone
        if alone:
            self.lone_player = maker
            self.sitting = partner_of(maker)
        else:
            self.lone_player = None
            self.sitting = None

    def _begin_play(self) -> None:
        self.phase = Phase.PLAY
        leader = (self.dealer + 1) % 4
        if self.sitting is not None and leader == self.sitting:
            leader = self._next_player(leader)
        self.trick_leader = leader
        self.current_player = leader
        self.current_trick = []

    def _next_player(self, player: int) -> int:
        nxt = (player + 1) % 4
        if self.sitting is not None and nxt == self.sitting:
            nxt = (nxt + 1) % 4
        return nxt

    def _finish_hand(self) -> None:
        self.phase = Phase.TERMINAL
        self.current_player = CHANCE
        maker_team = team_of(self.maker)
        maker_tricks = self.tricks_won[maker_team]
        points = [0, 0]
        if maker_tricks >= 3:
            if maker_tricks == 5:
                points[maker_team] = 4 if self.alone else 2
            else:
                points[maker_team] = 1
        else:  # euchred
            points[1 - maker_team] = 2
        self.reward = (points[0], points[1])

    # -- queries -------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.phase == Phase.TERMINAL

    def is_chance(self) -> bool:
        return self.phase == Phase.DEAL

    def returns(self) -> Tuple[int, int]:
        """Points awarded to (team0, team1). Only valid at terminal."""
        assert self.reward is not None
        return self.reward

    def __str__(self) -> str:
        lines = [f"Phase={self.phase.name} dealer={self.dealer} "
                 f"turn={self.current_player}"]
        if self.trump is not None:
            lines.append(f"trump={self.trump.symbol} maker={self.maker} "
                         f"alone={self.alone}")
        if self.up_card is not None and self.trump is None:
            lines.append(f"up={self.up_card}")
        for p in range(4):
            lines.append(f"  P{p}{'*' if p == self.dealer else ' '}: "
                         + " ".join(str(c) for c in self.hands[p]))
        if self.completed_tricks or self.current_trick:
            lines.append(f"tricks_won={self.tricks_won}")
        return "\n".join(lines)
