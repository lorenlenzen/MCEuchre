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
