"""Agents and a head-to-head evaluation harness.

Strength is only meaningful relative to a baseline, so this module provides a
few reference agents and a function to play many hands between two seat
strategies, reporting the average point differential (with a confidence
interval) for team 0.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Protocol

from euchre.actions import Action
from euchre.game import EuchreState, team_of
from .mccfr import MCCFRTrainer


class Agent(Protocol):
    def act(self, state: EuchreState, rng: random.Random) -> Action: ...


class RandomAgent:
    def act(self, state: EuchreState, rng: random.Random) -> Action:
        return rng.choice(state.legal_actions())


class RuleBasedAgent:
    """A simple heuristic: call when the hand is strong, then play greedily.

    Not expert, but a meaningfully non-trivial baseline to measure against.
    """

    def act(self, state: EuchreState, rng: random.Random) -> Action:
        from euchre.actions import Pass, OrderUp, Call, Discard, Play
        from euchre.cards import is_trump, card_strength, Suit
        from euchre.game import Phase

        legal = state.legal_actions()

        if state.phase == Phase.BID_ROUND_1:
            trump = state.up_card.suit
            strength = sum(1 for c in state.hands[state.current_player]
                           if is_trump(c, trump))
            if strength >= 3:
                return OrderUp(alone=False)
            return Pass()

        if state.phase == Phase.BID_ROUND_2:
            best_suit, best_count = None, 0
            for suit in Suit:
                if suit == state.turned_down:
                    continue
                cnt = sum(1 for c in state.hands[state.current_player]
                          if is_trump(c, suit))
                if cnt > best_count:
                    best_suit, best_count = suit, cnt
            calls = [a for a in legal if isinstance(a, Call)]
            if best_count >= 3 and best_suit is not None:
                for a in calls:
                    if a.suit == best_suit and not a.alone:
                        return a
            passes = [a for a in legal if isinstance(a, Pass)]
            if passes:
                return passes[0]
            return calls[0]  # stick-the-dealer: forced to call

        if state.phase == Phase.DEALER_DISCARD:
            # Discard the weakest non-trump card.
            trump = state.trump
            worst = min(state.hands[state.dealer],
                        key=lambda c: (is_trump(c, trump),
                                       card_strength(c, trump, c.suit)))
            return Discard(worst)

        if state.phase == Phase.PLAY:
            trump = state.trump
            if state.current_trick:
                led = state.current_trick[0][1]
                from euchre.cards import effective_suit
                led_suit = effective_suit(led, trump)
            else:
                led_suit = None
            plays = [a for a in legal]
            # Greedy: play the strongest legal card when leading or trying to
            # win, else the weakest to conserve.
            key = lambda a: card_strength(a.card, trump,
                                          led_suit if led_suit else a.card.suit)
            return max(plays, key=key)

        return rng.choice(legal)


class PointCountAgent:
    """A point-count bidding heuristic, stronger than :class:`RuleBasedAgent`.

    Scores a hand's strength for a candidate trump suit as the sum of
    per-card values -- trump cards weighted by rank (bowers highest),
    off-suit aces/kings worth something, everything else off-suit near
    nothing -- plus a ruffing bonus for suits you're void or singleton in,
    capped by how much spare trump you actually have to ruff with. Calling
    (and going alone) is then a fixed threshold on that score, separately
    for round 1, round 2, and alone. Discard and play are unchanged from
    ``RuleBasedAgent``'s simple greedy logic -- the improvement here is
    specifically a sharper bidding decision.
    """

    ORDER_1_THRESH = 2.2
    ORDER_2_THRESH = 2.4
    ALONE_THRESH = 3.6

    _TRUMP_RANK_POINTS = {
        "ACE": 0.72, "KING": 0.50, "QUEEN": 0.30, "TEN": 0.18, "NINE": 0.12,
    }

    def _trump_score(self, card, trump) -> float:
        from euchre.cards import is_right_bower, is_left_bower
        if is_right_bower(card, trump):
            return 0.95
        if is_left_bower(card, trump):
            return 0.85
        return self._TRUMP_RANK_POINTS[card.rank.name]

    def _offsuit_score(self, card) -> float:
        if card.rank.name == "ACE":
            return 0.50
        if card.rank.name == "KING":
            return 0.16
        return 0.03

    def hand_score(self, hand, trump) -> float:
        from euchre.cards import is_trump, effective_suit, Suit
        total = 0.0
        n_trump = 0
        suit_counts = {s: 0 for s in Suit}
        for c in hand:
            suit_counts[effective_suit(c, trump)] += 1
            if is_trump(c, trump):
                total += self._trump_score(c, trump)
                n_trump += 1
            else:
                total += self._offsuit_score(c)

        bonus = 0.0
        for s in Suit:
            if s == trump:
                continue
            cnt = suit_counts[s]
            if cnt == 0:
                bonus += 0.35
            elif cnt == 1:
                bonus += 0.15
        bonus = min(bonus, max(0.0, (n_trump - 1) * 0.40))
        return total + bonus

    def act(self, state: EuchreState, rng: random.Random) -> Action:
        from euchre.actions import Pass, OrderUp, Call, Discard
        from euchre.cards import is_trump, card_strength, effective_suit, Suit
        from euchre.game import Phase

        legal = state.legal_actions()

        if state.phase == Phase.BID_ROUND_1:
            trump = state.up_card.suit
            score = self.hand_score(state.hands[state.current_player], trump)
            if score >= self.ORDER_1_THRESH:
                return OrderUp(alone=score >= self.ALONE_THRESH)
            return Pass()

        if state.phase == Phase.BID_ROUND_2:
            hand = state.hands[state.current_player]
            best_suit, best_score = None, -1.0
            for suit in Suit:
                if suit == state.turned_down:
                    continue
                s = self.hand_score(hand, suit)
                if s > best_score:
                    best_suit, best_score = suit, s
            calls = [a for a in legal if isinstance(a, Call)]
            if best_suit is not None and best_score >= self.ORDER_2_THRESH:
                alone = best_score >= self.ALONE_THRESH
                for a in calls:
                    if a.suit == best_suit and a.alone == alone:
                        return a
            passes = [a for a in legal if isinstance(a, Pass)]
            if passes:
                return passes[0]
            # Stick-the-dealer: forced to call regardless of threshold.
            if best_suit is not None:
                for a in calls:
                    if a.suit == best_suit:
                        return a
            return calls[0]

        if state.phase == Phase.DEALER_DISCARD:
            trump = state.trump
            worst = min(state.hands[state.dealer],
                        key=lambda c: (is_trump(c, trump),
                                       card_strength(c, trump, c.suit)))
            return Discard(worst)

        if state.phase == Phase.PLAY:
            trump = state.trump
            if state.current_trick:
                led = state.current_trick[0][1]
                led_suit = effective_suit(led, trump)
            else:
                led_suit = None
            key = lambda a: card_strength(a.card, trump,
                                          led_suit if led_suit else a.card.suit)
            return max(legal, key=key)

        return rng.choice(legal)


class MCCFRAgent:
    def __init__(self, trainer: MCCFRTrainer, greedy: bool = False) -> None:
        self.trainer = trainer
        self.greedy = greedy

    def act(self, state: EuchreState, rng: random.Random) -> Action:
        dist = self.trainer.average_policy(state)
        actions = list(dist)
        if self.greedy:
            return max(actions, key=lambda a: dist[a])
        return rng.choices(actions, weights=[dist[a] for a in actions])[0]


def play_hand(agents: List[Agent], dealer: int, rng: random.Random,
              stick_the_dealer: bool = False) -> tuple[int, int]:
    state = EuchreState.new_hand(dealer=dealer,
                                 stick_the_dealer=stick_the_dealer).deal(rng)
    while not state.is_terminal():
        agent = agents[state.current_player]
        state = state.apply(agent.act(state, rng))
    return state.returns()


def evaluate(team0_agent_factory: Callable[[], Agent],
             team1_agent_factory: Callable[[], Agent],
             hands: int = 2000, seed: int = 0,
             stick_the_dealer: bool = False) -> Dict[str, float]:
    """Play ``hands`` hands, alternating the dealer. Returns team-0 stats.

    ``*_factory`` build an agent; seats 0/2 use team0, seats 1/3 use team1.
    """
    rng = random.Random(seed)
    a0, a1 = team0_agent_factory(), team1_agent_factory()
    agents = [a0, a1, a0, a1]
    diffs: List[int] = []
    for h in range(hands):
        r0, r1 = play_hand(agents, dealer=h % 4, rng=rng,
                           stick_the_dealer=stick_the_dealer)
        diffs.append(r0 - r1)
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
    stderr = math.sqrt(var / n)
    return {
        "hands": n,
        "team0_mean_point_diff": mean,
        "ci95": 1.96 * stderr,
        "team0_win_rate": sum(1 for d in diffs if d > 0) / n,
    }
