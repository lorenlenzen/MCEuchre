"""Depth-limited subgame solving with CFR over a sampled belief.

This is the search core of ReBeL. Given the state a player must act from, we:

1. Build a **belief** as a set of determinizations -- full deals consistent
   with what the acting player knows (their hand fixed, opponents sampled
   uniformly). Fixing the actor's hand guarantees the actor's real
   information set is in the support, so the resolved strategy is defined
   for the move we actually face.
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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from euchre.actions import Action
from euchre.game import EuchreState, Phase, team_of
from euchre.infoset import infoset_key
from .public_belief_state import sample_determinization

if TYPE_CHECKING:
    from .match_equity import MatchEquityModel

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
                 equity_model: Optional["MatchEquityModel"] = None,
                 rng: Optional[random.Random] = None) -> None:
        if root.is_terminal() or root.current_player != actor:
            raise ValueError("Subgame root must be a decision node for actor")
        self.actor = actor
        self.iterations = iterations
        self.depth_limit = depth_limit
        self.value_fn = value_fn
        self.batch_value_fn = batch_value_fn
        # Score can't change mid-hand, so it's fixed for this whole subgame --
        # read once here rather than per-terminal-node. None (the default)
        # preserves exact prior behavior (raw point-differential utility) for
        # any caller that doesn't opt in (CFRSearchAgent, existing tests).
        self.equity_model = equity_model
        self.team0_score = root.team0_score
        self.team1_score = root.team1_score
        # Fixed for the whole subgame too -- the deal doesn't change mid-hand.
        self.dealer_is_team0 = team_of(root.dealer) == 0
        self.rng = rng or random.Random()
        self.infosets: Dict[str, _Info] = {}
        self.worlds = [sample_determinization(root, actor, self.rng)
                       for _ in range(num_worlds)]
        self.weights = [1.0 / num_worlds] * num_worlds
        self.root_phase = root.phase
        self.root_key = infoset_key(root, actor)
        self.roots: Optional[List[_TNode]] = None
        self._pending_leaves: List[Tuple[_TNode, EuchreState]] = []

    def _leaf_value(self, state: EuchreState) -> List[float]:
        v0 = self.value_fn(state) if self.value_fn is not None else 0.0
        return [v0 if team_of(p) == 0 else -v0 for p in range(4)]

    def _build(self, state: EuchreState, depth: int) -> _TNode:
        if state.is_terminal():
            r = state.returns()
            if self.equity_model is not None:
                # CFR compares EXPECTATIONS over mixed strategies/hidden-info
                # uncertainty, which is exactly where a saturating equity
                # function can change which option is better -- unlike
                # solve_value/rollout_value's pure double-dummy minimax
                # (see rebel/match_equity.py's module docstring for why that
                # can stay raw-point-based).
                diff = self.equity_model.equity_delta(
                    self.team0_score, self.team1_score, self.dealer_is_team0,
                    r[0], r[1])
            else:
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
        # The subgame boundary is the PHASE boundary, not a fixed ply count,
        # for bidding/discard nodes (BID_ROUND_1/2, DEALER_DISCARD): they
        # expand FULLY (no depth cutoff at all while still bidding -- the
        # auction is short and bounded regardless, <=4 round-1 + <=4 round-2
        # decisions before either trump is set or a real misdeal terminal is
        # hit, caught by the is_terminal() check above), and the INSTANT a
        # child's phase becomes PLAY, that child is cut immediately as a
        # leaf -- no PLAY recursion happens inside a bidding-rooted solve at
        # all. PLAY decisions get their own separate SubgameSolver later,
        # unaffected by any of this.
        #
        # This fixes a real structural asymmetry without the blowup a
        # naive "just don't count bidding plies" version has (measured:
        # 52.7x slower, since that version kept recursing into real,
        # ~4-5-way-branching card play for every distinct bidding-resolution
        # path instead of stopping once). Under the OLD flat ply-count
        # (still used for a state already inside PLAY, below), OrderUp/Call
        # collapse straight into DEALER_DISCARD then real card play, so they
        # reach genuine searched trick outcomes within a few plies, while
        # Pass hands the decision to the next player -- under a flat shared
        # budget, Pass's subtree gets cut off deep in still-uncertain
        # bidding, leaning almost entirely on the value net's guess, while
        # OrderUp's is backed by real search. That asymmetry lets
        # regret-matching settle on whichever branch currently has the
        # more-trustworthy (search-backed) number -- always OrderUp/Call,
        # independent of whether it's actually better. Making every bidding
        # leaf the SAME kind of estimate (one value-net call at the moment
        # trump is fixed, whether reached via an immediate OrderUp or a long
        # chain of passes) removes that asymmetry at its source, in the
        # estimator type, not just the sample count.
        #
        # DEALER_DISCARD and BID_ROUND_2 are free (cut-at-boundary) ONLY when
        # reached as an INTERNAL node of a bidding-rooted solve -- there, a
        # rough single-leaf estimate of "trump fixed, some discard/call
        # chosen" is enough to judge whether an ANCESTOR decision (e.g.
        # ordering up, or passing round 1 toward round 2) looks good, and
        # keeping it cheap is what avoids the blowup above. When either is
        # itself the solve's ROOT (self_play_hand solving the real decision),
        # it must NOT be free: both have at least one action (Discard; Call)
        # that transitions DIRECTLY into Phase.PLAY, so with every such
        # option immediately cut at the boundary, the tree is just the root
        # plus same-depth leaves -- no real search at all, so regret-matching
        # over them degenerates to comparing unbacked value-net guesses
        # (measured for DEALER_DISCARD: exactly 1 infoset, exactly-uniform
        # policy regardless of net quality). BID_ROUND_2 has the identical
        # structural gap for its Call actions specifically -- unlike round
        # 1's OrderUp, Call skips DEALER_DISCARD entirely
        # (euchre/game.py's _apply_bid2 calls _begin_play() directly), so a
        # round-2-rooted solve's own Call-vs-Call-alone comparison was
        # ALSO just comparing unbacked leaves, letting any value-head bias
        # between alone/not-alone train directly into the policy uncorrected
        # -- this is what surfaced as "every alone option outranks its
        # same-suit non-alone twin" on the quiz. BID_ROUND_1 has no such gap
        # (neither Pass nor OrderUp's child is ever directly Phase.PLAY --
        # OrderUp always passes through DEALER_DISCARD first), so it stays
        # free unconditionally, root or not.
        is_free = (state.phase == Phase.BID_ROUND_1
                  or (state.phase in (Phase.BID_ROUND_2, Phase.DEALER_DISCARD)
                      and self.root_phase != state.phase))
        if self.depth_limit is not None and is_free:
            children = []
            for a in info.actions:
                child = state.apply(a)
                if child.phase == Phase.PLAY and not child.is_terminal():
                    # Just crossed the phase boundary -- an immediate leaf,
                    # not a ply-counted continuation.
                    if self.batch_value_fn is not None:
                        node = _TNode(util=None)
                        self._pending_leaves.append((node, child))
                    else:
                        node = _TNode(util=self._leaf_value(child))
                    children.append(node)
                else:
                    children.append(self._build(child, depth))
            return _TNode(info=info, player=player, children=children)

        child_depth = depth + 1 if not is_free else depth
        children = [self._build(state.apply(a), child_depth) for a in info.actions]
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
            for root, w in zip(self.roots, self.weights):
                self._cfr(root, [1.0, 1.0, 1.0, 1.0], w)

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
        for root, w in zip(self.roots, self.weights):
            total += w * self._expected_value(root)[0]
        return total


class CFRSearchAgent:
    """Acts by solving a fresh subgame at each decision (ReBeL-style search).

    Uses full-depth CFR over a sampled belief. This is the decision-time search
    agent; with a trained value net passed as ``value_fn`` and a ``depth_limit``
    it becomes the depth-limited ReBeL player.

    Prefer ``batch_value_fn_factory`` over ``batch_value_fn`` when plugging in
    a trained net: pass ``ReBeLTrainer._value_fn_for`` (or anything with that
    ``actor -> BatchValueFn`` shape) and each solve scores its leaves from the
    acting player's own information set, matching how training now generates
    targets. A plain ``batch_value_fn`` scores every leaf from whoever happens
    to act *at that leaf*, which for a bidding-rooted solve is the opening
    leader -- a view that omits the searching player's own hand (see
    rebel/train_rebel.py's batch_value_fn_from_net docstring for the
    measurements). It stays supported and is still the default so existing
    callers are unaffected.
    """

    def __init__(self, num_worlds: int = 16, iterations: int = 30,
                 depth_limit: Optional[int] = None,
                 value_fn: Optional[ValueFn] = None,
                 batch_value_fn: Optional[BatchValueFn] = None,
                 batch_value_fn_factory=None,
                 greedy: bool = True, seed: int = 0) -> None:
        self.num_worlds = num_worlds
        self.iterations = iterations
        self.depth_limit = depth_limit
        self.value_fn = value_fn
        self.batch_value_fn = batch_value_fn
        self.batch_value_fn_factory = batch_value_fn_factory
        self.greedy = greedy
        self._rng = random.Random(seed)

    def act(self, state: EuchreState, rng: random.Random) -> Action:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        actor = state.current_player
        bvf = (self.batch_value_fn_factory(actor)
               if self.batch_value_fn_factory is not None
               else self.batch_value_fn)
        solver = SubgameSolver(
            state, actor, num_worlds=self.num_worlds,
            iterations=self.iterations, depth_limit=self.depth_limit,
            value_fn=self.value_fn, batch_value_fn=bvf,
            rng=self._rng)
        solver.run()
        policy = solver.root_policy()
        actions = list(policy)
        if self.greedy:
            return max(actions, key=lambda a: policy[a])
        return rng.choices(actions, weights=[policy[a] for a in actions])[0]
