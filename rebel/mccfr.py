"""External-sampling Monte-Carlo CFR for Euchre.

This is the tabular, ground-truth learner. It converges toward an
equilibrium strategy on the real game (no abstraction), which makes it both a
strong baseline and the target that the ReBeL value/policy networks learn to
approximate. The same regret-matching + self-play logic reappears in ReBeL,
just with a neural net generalizing across information sets instead of a hash
table.

Euchre is a two-team game; each player's utility is their team's point
differential for the hand, so partners have aligned objectives.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .match_equity import MatchEquityModel

from euchre.actions import Action, action_to_index
from euchre.game import EuchreState, team_of
from euchre.infoset import infoset_key


def _utility(state: EuchreState, player: int,
            equity_model: Optional["MatchEquityModel"] = None) -> float:
    r = state.returns()
    t = team_of(player)
    if equity_model is not None:
        # Score is fixed for a hand's whole duration, so the terminal
        # state's own team0_score/team1_score fields ARE the pre-hand score
        # -- no separate threading needed. See SubgameSolver._build for the
        # same conversion and why it belongs here (comparing expectations
        # under CFR's regret matching) and not in solve_value/rollout_value.
        diff = equity_model.equity_delta(
            state.team0_score, state.team1_score, r[0], r[1])
        return diff if t == 0 else -diff
    return float(r[t] - r[1 - t])


@dataclass
class Node:
    """Regret and average-strategy accumulators for one information set."""
    action_indices: List[int]          # stable ordering of legal actions
    regret_sum: np.ndarray
    strategy_sum: np.ndarray

    @staticmethod
    def create(legal: List[Action]) -> "Node":
        idx = sorted(action_to_index(a) for a in legal)
        n = len(idx)
        return Node(idx, np.zeros(n), np.zeros(n))

    def strategy(self) -> np.ndarray:
        """Current strategy via regret matching."""
        pos = np.maximum(self.regret_sum, 0.0)
        s = pos.sum()
        if s > 0:
            return pos / s
        return np.full(len(self.regret_sum), 1.0 / len(self.regret_sum))

    def average_strategy(self) -> np.ndarray:
        s = self.strategy_sum.sum()
        if s > 0:
            return self.strategy_sum / s
        return np.full(len(self.strategy_sum), 1.0 / len(self.strategy_sum))


class MCCFRTrainer:
    def __init__(self, stick_the_dealer: bool = False,
                 equity_model: Optional["MatchEquityModel"] = None,
                 seed: int = 0) -> None:
        self.nodes: Dict[str, Node] = {}
        self.stick_the_dealer = stick_the_dealer
        # None (default) preserves exact prior behavior -- raw point-
        # differential utility, every hand starting 0-0 -- for scripts/
        # ladder.py and any other caller that doesn't opt in.
        self.equity_model = equity_model
        self.rng = random.Random(seed)

    def _node(self, state: EuchreState, player: int) -> Node:
        key = infoset_key(state, player)
        node = self.nodes.get(key)
        if node is None:
            node = Node.create(state.legal_actions())
            self.nodes[key] = node
        return node

    def _traverse(self, state: EuchreState, traverser: int) -> float:
        if state.is_terminal():
            return _utility(state, traverser, self.equity_model)
        if state.is_chance():
            return self._traverse(state.deal(self.rng), traverser)

        player = state.current_player
        legal = state.legal_actions()
        node = self._node(state, player)
        sigma = node.strategy()

        if player == traverser:
            child_values = np.empty(len(legal))
            node_value = 0.0
            for i, a in enumerate(legal):
                v = self._traverse(state.apply(a), traverser)
                child_values[i] = v
                node_value += sigma[i] * v
            node.regret_sum += child_values - node_value
            return node_value

        # Opponent / partner node: accumulate their average strategy, sample.
        node.strategy_sum += sigma
        i = self.rng.choices(range(len(legal)), weights=sigma.tolist())[0]
        return self._traverse(state.apply(legal[i]), traverser)

    def iterate(self, dealer: int | None = None) -> None:
        """One MCCFR iteration: traverse once per player on a fresh deal.

        Dealer (when not fixed by the caller) and deal are already
        independently resampled per traverser below -- each of the 4
        traversals is its own fresh, independent hand, not 4 traversals of
        one shared hand. Score is sampled the same way, per traverser, for
        the same reason.
        """
        for p in range(4):
            d = self.rng.randint(0, 3) if dealer is None else dealer
            team0_score = team1_score = 0
            if self.equity_model is not None:
                team0_score, team1_score = self.equity_model.sample_score(self.rng)
            root = EuchreState.new_hand(dealer=d,
                                        stick_the_dealer=self.stick_the_dealer,
                                        team0_score=team0_score,
                                        team1_score=team1_score)
            self._traverse(root, p)

    def train(self, iterations: int, log_every: int = 0) -> None:
        for it in range(1, iterations + 1):
            self.iterate()
            if log_every and it % log_every == 0:
                print(f"iter {it}: {len(self.nodes)} infosets")

    # -- policy access -------------------------------------------------------

    def average_policy(self, state: EuchreState) -> Dict[Action, float]:
        """Average-strategy distribution over legal actions at ``state``."""
        legal = state.legal_actions()
        node = self.nodes.get(infoset_key(state, state.current_player))
        if node is None:
            return {a: 1.0 / len(legal) for a in legal}
        avg = node.average_strategy()
        by_index = dict(zip(node.action_indices, avg))
        return {a: float(by_index[action_to_index(a)]) for a in legal}
