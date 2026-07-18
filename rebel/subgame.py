"""Depth-limited subgame solving with CFR over a sampled belief.

This is the search core of ReBeL. Given the state a player must act from, we:

1. Build a **belief** as a set of determinizations -- full deals consistent
   with what the acting player knows (their hand fixed, opponents sampled).
   Fixing the actor's hand guarantees the actor's real information set is in
   the support, so the resolved strategy is defined for the move we actually
   face. (A learned belief net will replace the uniform sampler later.)
2. Treat "nature picks one of these worlds" as a chance node and run vanilla
   CFR over the resulting extensive game, sharing regret across worlds through
   information-set keys -- so a player's strategy is tied together exactly
   where they cannot distinguish worlds.
3. Cut the tree at a **depth limit** and read leaf values from a value
   function (the ReBeL value net). With no limit the search runs to terminal,
   which exactly solves the sampled-belief subgame and needs no network -- the
   mode used to *generate* value targets during self-play.

The resolved average strategy at the root is the policy ReBeL plays and the
training target for the policy net.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from euchre.actions import Action, action_to_index
from euchre.game import EuchreState, team_of
from euchre.infoset import infoset_key
from .public_belief_state import sample_determinization

# A value function maps a (fully-specified) state to an estimate of the hand's
# team0 - team1 point differential.
ValueFn = Callable[[EuchreState], float]


class _Info:
    __slots__ = ("actions", "regret", "strategy_sum")

    def __init__(self, actions: List[Action]) -> None:
        self.actions = actions
        n = len(actions)
        self.regret = np.zeros(n)
        self.strategy_sum = np.zeros(n)

    def strategy(self) -> np.ndarray:
        pos = np.maximum(self.regret, 0.0)
        s = pos.sum()
        if s > 0:
            return pos / s
        return np.full(len(self.regret), 1.0 / len(self.regret))

    def average(self) -> np.ndarray:
        s = self.strategy_sum.sum()
        if s > 0:
            return self.strategy_sum / s
        return np.full(len(self.strategy_sum), 1.0 / len(self.strategy_sum))


class SubgameSolver:
    def __init__(self, root: EuchreState, actor: int,
                 num_worlds: int = 20, iterations: int = 40,
                 depth_limit: Optional[int] = None,
                 value_fn: Optional[ValueFn] = None,
                 rng: Optional[random.Random] = None) -> None:
        if root.is_terminal() or root.current_player != actor:
            raise ValueError("Subgame root must be a decision node for actor")
        self.actor = actor
        self.iterations = iterations
        self.depth_limit = depth_limit
        self.value_fn = value_fn
        self.rng = rng or random.Random()
        self.infosets: Dict[str, _Info] = {}
        self.worlds = [sample_determinization(root, actor, self.rng)
                       for _ in range(num_worlds)]
        self.weight = 1.0 / num_worlds
        self.root_key = infoset_key(root, actor)

    def _info(self, state: EuchreState, player: int) -> Tuple[str, _Info]:
        key = infoset_key(state, player)
        info = self.infosets.get(key)
        if info is None:
            info = _Info(state.legal_actions())
            self.infosets[key] = info
        return key, info

    def _leaf_value(self, state: EuchreState) -> List[float]:
        v0 = self.value_fn(state) if self.value_fn is not None else 0.0
        return [v0 if team_of(p) == 0 else -v0 for p in range(4)]

    def _cfr(self, state: EuchreState, reach: List[float],
             chance_reach: float, depth: int) -> List[float]:
        if state.is_terminal():
            r = state.returns()
            diff = r[0] - r[1]
            return [diff if team_of(p) == 0 else -diff for p in range(4)]
        if self.depth_limit is not None and depth >= self.depth_limit:
            return self._leaf_value(state)

        player = state.current_player
        _key, info = self._info(state, player)
        strat = info.strategy()
        actions = info.actions

        child_utils: List[List[float]] = []
        node_util = [0.0, 0.0, 0.0, 0.0]
        for i, a in enumerate(actions):
            new_reach = list(reach)
            new_reach[player] *= strat[i]
            cu = self._cfr(state.apply(a), new_reach, chance_reach, depth + 1)
            child_utils.append(cu)
            for p in range(4):
                node_util[p] += strat[i] * cu[p]

        # Counterfactual reach for `player`: everyone else (and chance).
        cf = chance_reach
        for q in range(4):
            if q != player:
                cf *= reach[q]
        for i in range(len(actions)):
            info.regret[i] += cf * (child_utils[i][player] - node_util[player])
            info.strategy_sum[i] += reach[player] * strat[i]
        return node_util

    def run(self) -> None:
        for _ in range(self.iterations):
            for world in self.worlds:
                self._cfr(world, [1.0, 1.0, 1.0, 1.0], self.weight, 0)

    def root_policy(self) -> Dict[Action, float]:
        """Average strategy at the actor's root information set."""
        info = self.infosets.get(self.root_key)
        if info is None:  # no iterations run yet
            self.run()
            info = self.infosets[self.root_key]
        avg = info.average()
        return {a: float(p) for a, p in zip(info.actions, avg)}

    def root_value(self) -> float:
        """Estimated team0 - team1 value of the root under the solved policy."""
        total = 0.0
        for world in self.worlds:
            total += self._cfr(world, [1.0, 1.0, 1.0, 1.0], self.weight, 0)[0]
        return total  # already weighted by self.weight per world


class CFRSearchAgent:
    """Acts by solving a fresh subgame at each decision (ReBeL-style search).

    Uses full-depth CFR over a sampled belief. This is the decision-time search
    agent; with a trained value net passed as ``value_fn`` and a ``depth_limit``
    it becomes the depth-limited ReBeL player.
    """

    def __init__(self, num_worlds: int = 16, iterations: int = 30,
                 depth_limit: Optional[int] = None,
                 value_fn: Optional[ValueFn] = None,
                 greedy: bool = True, seed: int = 0) -> None:
        self.num_worlds = num_worlds
        self.iterations = iterations
        self.depth_limit = depth_limit
        self.value_fn = value_fn
        self.greedy = greedy
        self._rng = random.Random(seed)

    def act(self, state: EuchreState, rng: random.Random) -> Action:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        solver = SubgameSolver(
            state, state.current_player, num_worlds=self.num_worlds,
            iterations=self.iterations, depth_limit=self.depth_limit,
            value_fn=self.value_fn, rng=self._rng)
        solver.run()
        policy = solver.root_policy()
        actions = list(policy)
        if self.greedy:
            return max(actions, key=lambda a: policy[a])
        return rng.choices(actions, weights=[policy[a] for a in actions])[0]
