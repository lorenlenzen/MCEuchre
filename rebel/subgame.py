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

from euchre.actions import Action
from euchre.game import EuchreState, team_of
from euchre.infoset import infoset_key
from .public_belief_state import sample_determinization

# A value function maps a (fully-specified) state to an estimate of the hand's
# team0 - team1 point differential. A batch value function does the same for a
# list of states in one call (e.g. a single batched network forward pass),
# which is how the ReBeL loop keeps net inference off the critical path.
ValueFn = Callable[[EuchreState], float]
BatchValueFn = Callable[[List[EuchreState]], List[float]]


class _Info:
    """Regret and average-strategy accumulators for one information set.

    Uses plain Python lists rather than NumPy arrays: Euchre infosets have only
    a handful of legal actions, and at that size NumPy's per-call overhead is
    pure cost -- the regret-matching arithmetic is faster in native lists.
    """

    __slots__ = ("actions", "regret", "strategy_sum")

    def __init__(self, actions: List[Action]) -> None:
        self.actions = actions
        n = len(actions)
        self.regret = [0.0] * n
        self.strategy_sum = [0.0] * n

    def strategy(self) -> List[float]:
        pos = [r if r > 0.0 else 0.0 for r in self.regret]
        s = sum(pos)
        if s > 0.0:
            inv = 1.0 / s
            return [p * inv for p in pos]
        u = 1.0 / len(pos)
        return [u] * len(pos)

    def average(self) -> List[float]:
        ss = self.strategy_sum
        s = sum(ss)
        if s > 0.0:
            inv = 1.0 / s
            return [x * inv for x in ss]
        n = len(ss)
        u = 1.0 / n
        return [u] * n


class _TNode:
    """A node in the pre-expanded subgame tree.

    Decision nodes carry ``player``, a shared ``info`` (regret/strategy), and
    ``children``. Terminal/leaf nodes carry only a per-player ``util`` vector
    and ``children is None``. Building the tree once and iterating over it keeps
    ``infoset_key``/``apply``/``clone``/``legal_actions`` -- and the value-net
    leaf calls -- out of the per-iteration loop.
    """

    __slots__ = ("util", "info", "player", "children")

    def __init__(self, util=None, info=None, player=-1, children=None) -> None:
        self.util = util
        self.info = info
        self.player = player
        self.children = children


class SubgameSolver:
    def __init__(self, root: EuchreState, actor: int,
                 num_worlds: int = 20, iterations: int = 40,
                 depth_limit: Optional[int] = None,
                 value_fn: Optional[ValueFn] = None,
                 batch_value_fn: Optional[BatchValueFn] = None,
                 rng: Optional[random.Random] = None) -> None:
        if root.is_terminal() or root.current_player != actor:
            raise ValueError("Subgame root must be a decision node for actor")
        self.actor = actor
        self.iterations = iterations
        self.depth_limit = depth_limit
        self.value_fn = value_fn
        self.batch_value_fn = batch_value_fn
        self.rng = rng or random.Random()
        self.infosets: Dict[str, _Info] = {}
        self.worlds = [sample_determinization(root, actor, self.rng)
                       for _ in range(num_worlds)]
        self.weight = 1.0 / num_worlds
        self.root_key = infoset_key(root, actor)
        self.roots: Optional[List[_TNode]] = None
        self._pending_leaves: List[Tuple[_TNode, EuchreState]] = []

    def _leaf_value(self, state: EuchreState) -> List[float]:
        v0 = self.value_fn(state) if self.value_fn is not None else 0.0
        return [v0 if team_of(p) == 0 else -v0 for p in range(4)]

    def _build(self, state: EuchreState, depth: int) -> _TNode:
        if state.is_terminal():
            r = state.returns()
            diff = r[0] - r[1]
            return _TNode(util=[diff if team_of(p) == 0 else -diff
                                for p in range(4)])
        if self.depth_limit is not None and depth >= self.depth_limit:
            if self.batch_value_fn is not None:
                # Defer: collect the leaf and value it in one batched pass.
                node = _TNode(util=None)
                self._pending_leaves.append((node, state))
                return node
            return _TNode(util=self._leaf_value(state))

        player = state.current_player
        key = infoset_key(state, player)
        info = self.infosets.get(key)
        if info is None:
            info = _Info(state.legal_actions())
            self.infosets[key] = info
        children = [self._build(state.apply(a), depth + 1) for a in info.actions]
        return _TNode(info=info, player=player, children=children)

    def _cfr(self, node: _TNode, reach: List[float],
             chance_reach: float) -> List[float]:
        if node.children is None:  # terminal or depth-limit leaf
            return node.util

        info = node.info
        player = node.player
        strat = info.strategy()
        children = node.children
        n = len(children)

        node_util = [0.0, 0.0, 0.0, 0.0]
        child_utils: List[List[float]] = []
        for i in range(n):
            new_reach = list(reach)
            new_reach[player] *= strat[i]
            cu = self._cfr(children[i], new_reach, chance_reach)
            child_utils.append(cu)
            si = strat[i]
            node_util[0] += si * cu[0]
            node_util[1] += si * cu[1]
            node_util[2] += si * cu[2]
            node_util[3] += si * cu[3]

        # Counterfactual reach for `player`: everyone else (and chance).
        cf = chance_reach
        for q in range(4):
            if q != player:
                cf *= reach[q]
        reg = info.regret
        ss = info.strategy_sum
        rp = reach[player]
        npu = node_util[player]
        for i in range(n):
            reg[i] += cf * (child_utils[i][player] - npu)
            ss[i] += rp * strat[i]
        return node_util

    def _build_trees(self) -> None:
        if self.roots is not None:
            return
        self.roots = [self._build(w, 0) for w in self.worlds]
        if self._pending_leaves:
            # One batched network pass values every depth-limit leaf at once.
            states = [st for _node, st in self._pending_leaves]
            values = self.batch_value_fn(states)  # team0 - team1 per state
            for (node, _st), v0 in zip(self._pending_leaves, values):
                node.util = [v0 if team_of(p) == 0 else -v0 for p in range(4)]
            self._pending_leaves = []

    def run(self) -> None:
        self._build_trees()
        for _ in range(self.iterations):
            for root in self.roots:
                self._cfr(root, [1.0, 1.0, 1.0, 1.0], self.weight)

    def root_policy(self) -> Dict[Action, float]:
        """Average strategy at the actor's root information set."""
        if self.roots is None:  # not solved yet
            self.run()
        info = self.infosets[self.root_key]
        avg = info.average()
        return {a: float(p) for a, p in zip(info.actions, avg)}

    def _expected_value(self, node: _TNode) -> List[float]:
        if node.children is None:
            return node.util
        avg = node.info.average()
        node_util = [0.0, 0.0, 0.0, 0.0]
        for i, child in enumerate(node.children):
            cu = self._expected_value(child)
            a = avg[i]
            node_util[0] += a * cu[0]
            node_util[1] += a * cu[1]
            node_util[2] += a * cu[2]
            node_util[3] += a * cu[3]
        return node_util

    def root_value(self) -> float:
        """Estimated team0 - team1 value of the root under the solved (average)
        policy."""
        self._build_trees()
        total = 0.0
        for root in self.roots:
            total += self.weight * self._expected_value(root)[0]
        return total


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
