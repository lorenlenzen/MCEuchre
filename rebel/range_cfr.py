"""Range-based public-tree CFR (experimental; correct but NOT a speedup here).

This builds the DeepStack/ReBeL-style *public* tree: instead of one game tree
per sampled world, it branches on the public action (the card played) and pools
every deal consistent with a public line at each node. The CFR arithmetic is
written as NumPy operations over the deal axis. It is validated correct (M=1
reproduces the double-dummy optimum; root value matches the scalar
``SubgameSolver`` on an identical belief).

**Empirical finding -- why this is not wired into the pipeline.** With a
*sampled* belief it does not vectorize, and is in fact slower than the scalar
solver. A player's strategy is per information set (their exact hand), so the
deals at a node must be grouped by hand -- and randomly sampled deals almost
never share an exact opponent hand. Measured at depth 4 with 80 deals: 885
nodes, 2927 groups, **average group size 1.2** (only the root is dense). NumPy
over size-1 arrays is pure overhead.

Real vectorization needs the *dense enumerated-range* formulation: carry a
belief vector over **all** possible hands (not a sample) and matrix-multiply
strategies against it, with explicit card-removal for joint consistency. Full
joint enumeration is intractable mid-game (>300k consistent deals), so it
requires per-player marginal ranges plus a 4-player + kitty card-removal
correction -- a substantial research effort documented in
``docs/rebel_design.md``. This module is kept as a validated reference for that
tree structure; the trainer and agents use the scalar ``SubgameSolver``.

Scope: the play phase only.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import numpy as np

from euchre.actions import Play, Action
from euchre.cards import Card
from euchre.game import EuchreState, Phase, team_of
from euchre.infoset import infoset_key
from .public_belief_state import sample_determinization

BatchValueFn = Callable[[List[EuchreState]], List[float]]


class _Info:
    """Shared regret/strategy for one information set (actions are card ids)."""

    __slots__ = ("actions", "regret", "strategy_sum")

    def __init__(self, actions: List[int]) -> None:
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


class _Group:
    """Deals at a node that share the acting player's infoset."""

    __slots__ = ("info", "pos", "actions")

    def __init__(self, info: _Info, pos: np.ndarray, actions: List[int]) -> None:
        self.info = info
        self.pos = pos          # local deal positions (into node's deal block)
        self.actions = actions  # card ids, same order as info.actions


class _RNode:
    __slots__ = ("leaf", "util", "player", "width",
                 "groups", "child_cards", "child_nodes", "child_pos",
                 "ga_child")

    def __init__(self) -> None:
        self.leaf = False
        self.util: Optional[np.ndarray] = None     # (W,) team0-team1 per deal
        self.player = -1
        self.width = 0
        self.groups: List[_Group] = []
        self.child_cards: List[int] = []
        self.child_nodes: List[_RNode] = []
        self.child_pos: List[np.ndarray] = []      # per child: parent positions
        # (group_index, action_index) -> (child_index, start, end) slice into
        # that child's deal ordering.
        self.ga_child: Dict[tuple, tuple] = {}


class RangeCFRSolver:
    def __init__(self, root: EuchreState, actor: int,
                 num_deals: int = 100, iterations: int = 40,
                 depth_limit: Optional[int] = None,
                 batch_value_fn: Optional[BatchValueFn] = None,
                 rng: Optional[random.Random] = None) -> None:
        if root.is_terminal() or root.current_player != actor:
            raise ValueError("Subgame root must be a decision node for actor")
        if root.phase != Phase.PLAY:
            raise ValueError("RangeCFRSolver handles the play phase")
        self.actor = actor
        self.iterations = iterations
        self.depth_limit = depth_limit
        self.batch_value_fn = batch_value_fn
        self.rng = rng or random.Random()
        self.infosets: Dict[str, _Info] = {}
        self.root_key = infoset_key(root, actor)
        deals = [sample_determinization(root, actor, self.rng)
                 for _ in range(num_deals)]
        self.M = num_deals
        self.weight = 1.0 / num_deals
        self._pending: List[tuple] = []   # (node, states) leaves awaiting net
        self.root_node = self._build(deals, 0)
        self._resolve_leaves()

    # -- build ---------------------------------------------------------------

    def _build(self, states: List[EuchreState], depth: int) -> _RNode:
        node = _RNode()
        node.width = len(states)
        s0 = states[0]

        if s0.is_terminal():
            r = s0.returns()
            node.leaf = True
            node.util = np.full(len(states), float(r[0] - r[1]))
            return node
        if self.depth_limit is not None and depth >= self.depth_limit:
            node.leaf = True
            if self.batch_value_fn is not None:
                node.util = None
                self._pending.append((node, states))
            else:
                node.util = np.zeros(len(states))
            return node

        player = s0.current_player
        node.player = player

        # Group deals by the acting player's hand (== their infoset here).
        by_hand: Dict[frozenset, List[int]] = {}
        for i, s in enumerate(states):
            key = frozenset(c.id for c in s.hands[player])
            by_hand.setdefault(key, []).append(i)

        # card id -> list of (group_index, positions) that can play it.
        card_contrib: Dict[int, List[tuple]] = {}
        for g, (_hand, positions) in enumerate(by_hand.items()):
            rep = states[positions[0]]
            actions = sorted(c.id for c in rep._legal_plays(player))
            key = infoset_key(rep, player)
            info = self.infosets.get(key)
            if info is None:
                info = _Info(actions)
                self.infosets[key] = info
            pos = np.array(positions, dtype=np.int64)
            node.groups.append(_Group(info, pos, actions))
            for c in actions:
                card_contrib.setdefault(c, []).append((g, pos))

        # Build merged children (one per public card), recording group slices.
        for ic, (c, contribs) in enumerate(card_contrib.items()):
            blocks = []
            offset = 0
            for g, pos in contribs:
                j = node.groups[g].actions.index(c)
                node.ga_child[(g, j)] = (ic, offset, offset + len(pos))
                blocks.append(pos)
                offset += len(pos)
            childpos = np.concatenate(blocks)
            card = Card.from_id(c)
            child_states = [states[i].apply(Play(card)) for i in childpos]
            node.child_cards.append(c)
            node.child_pos.append(childpos)
            node.child_nodes.append(self._build(child_states, depth + 1))
        return node

    def _resolve_leaves(self) -> None:
        if not self._pending:
            return
        flat: List[EuchreState] = []
        spans = []
        for node, states in self._pending:
            spans.append((node, len(flat), len(flat) + len(states)))
            flat.extend(states)
        values = self.batch_value_fn(flat)  # team0-team1 per state
        arr = np.asarray(values, dtype=np.float64)
        for node, a, b in spans:
            node.util = arr[a:b]
        self._pending = []

    # -- CFR (vectorized over deals) -----------------------------------------

    def _cfr(self, node: _RNode, reach: np.ndarray,
             chance: np.ndarray) -> np.ndarray:
        """Return team0-team1 value per deal (aligned to this node's deals)."""
        if node.leaf:
            return node.util

        player = node.player
        sign = 1.0 if team_of(player) == 0 else -1.0
        W = node.width

        # Strategy per group, then recurse into each child with propagated reach.
        strategies = [g.info.strategy() for g in node.groups]
        child_val = [None] * len(node.child_cards)  # per-card value, (W,) scatter
        for ic in range(len(node.child_cards)):
            cpos = node.child_pos[ic]
            creach = reach[:, cpos].copy()
            cchance = chance[cpos]
            child_val[ic] = (cpos, self._child_value(
                node, ic, creach, cchance, strategies))

        # scatter child values into full (W,) arrays keyed by card index
        vfull = []
        for ic in range(len(node.child_cards)):
            cpos, cval = child_val[ic]
            full = np.zeros(W)
            full[cpos] = cval
            vfull.append(full)

        v = np.zeros(W)
        for g, group in enumerate(node.groups):
            pos = group.pos
            strat = strategies[g]
            vg = np.zeros(len(pos))
            for j in range(len(group.actions)):
                ic, _s, _e = node.ga_child[(g, j)]
                vg += strat[j] * vfull[ic][pos]
            v[pos] = vg

            cf = chance[pos].copy()
            for q in range(4):
                if q != player:
                    cf *= reach[q][pos]
            reach_p_sum = reach[player][pos].sum()
            reg = group.info.regret
            ss = group.info.strategy_sum
            for j in range(len(group.actions)):
                ic, _s, _e = node.ga_child[(g, j)]
                action_v = vfull[ic][pos]
                reg[j] += sign * (cf * (action_v - vg)).sum()
                ss[j] += strat[j] * reach_p_sum
        return v

    def _child_value(self, node: _RNode, ic: int, creach: np.ndarray,
                     cchance: np.ndarray, strategies) -> np.ndarray:
        """Scale the acting player's reach into child ``ic`` and recurse."""
        player = node.player
        # Scale each contributing group's block by that group's action prob.
        for (g, j), (cidx, start, end) in node.ga_child.items():
            if cidx == ic:
                creach[player, start:end] *= strategies[g][j]
        return self._cfr(node.child_nodes[ic], creach, cchance)

    def run(self) -> None:
        reach0 = np.ones((4, self.M))
        chance0 = np.full(self.M, self.weight)
        for _ in range(self.iterations):
            self._cfr(self.root_node, reach0.copy(), chance0.copy())

    # -- outputs -------------------------------------------------------------

    def root_policy(self) -> Dict[Action, float]:
        info = self.infosets[self.root_key]
        avg = info.average()
        return {Play(Card.from_id(c)): float(p)
                for c, p in zip(info.actions, avg)}

    def _expected(self, node: _RNode) -> np.ndarray:
        if node.leaf:
            return node.util
        W = node.width
        vfull = []
        for ic in range(len(node.child_cards)):
            cpos = node.child_pos[ic]
            cval = self._expected(node.child_nodes[ic])
            full = np.zeros(W)
            full[cpos] = cval
            vfull.append(full)
        v = np.zeros(W)
        for g, group in enumerate(node.groups):
            pos = group.pos
            avg = group.info.average()
            vg = np.zeros(len(pos))
            for j in range(len(group.actions)):
                ic, _s, _e = node.ga_child[(g, j)]
                vg += avg[j] * vfull[ic][pos]
            v[pos] = vg
        return v

    def root_value(self) -> float:
        """Estimated team0 - team1 value under the solved (average) policy."""
        v = self._expected(self.root_node)
        return float(self.weight * v.sum())


class RangeCFRAgent:
    """Play-phase agent backed by the vectorized range solver."""

    def __init__(self, num_deals: int = 100, iterations: int = 40,
                 depth_limit: Optional[int] = None,
                 batch_value_fn: Optional[BatchValueFn] = None,
                 greedy: bool = True, seed: int = 0) -> None:
        self.num_deals = num_deals
        self.iterations = iterations
        self.depth_limit = depth_limit
        self.batch_value_fn = batch_value_fn
        self.greedy = greedy
        self._rng = random.Random(seed)

    def act(self, state: EuchreState, rng: random.Random) -> Action:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        solver = RangeCFRSolver(
            state, state.current_player, num_deals=self.num_deals,
            iterations=self.iterations, depth_limit=self.depth_limit,
            batch_value_fn=self.batch_value_fn, rng=self._rng)
        solver.run()
        policy = solver.root_policy()
        actions = list(policy)
        if self.greedy:
            return max(actions, key=lambda a: policy[a])
        return rng.choices(actions, weights=[policy[a] for a in actions])[0]
