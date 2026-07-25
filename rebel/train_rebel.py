"""The ReBeL self-play training loop.

This wires the pieces together into ReBeL's learning cycle:

1. **Self-play with search.** Play hands; at every decision node run the
   depth-limited CFR subgame solver (`SubgameSolver`), using the *current*
   network to value the leaves. The solved root strategy is the policy we play
   (sampled), and it -- together with the solved root value -- becomes a
   training target.
2. **Learn.** Train one `PolicyValueNet`: the policy head regresses onto the
   CFR strategies (cross-entropy over legal actions); the value head regresses
   onto the CFR root values (MSE). As the value head improves, the leaf
   estimates that feed the next round of search improve too -- the bootstrap
   that lets ReBeL climb past what a reactive policy can reach.

Everything is honest but small by default: reaching genuinely expert play needs
far more self-play and compute (and, in pure Python, a faster engine). The loop
here is designed to *run and learn*, exposing the moving parts, not to train a
finished agent in one sitting. See ``docs/rebel_design.md``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from euchre.actions import Call, NUM_ACTIONS, OrderUp, Pass, action_to_index
from euchre.cards import Suit
from euchre.game import EuchreState, Phase, team_of
from euchre.infoset import observation_tensor, OBS_SIZE
from .evaluate import PointCountAgent
from .networks import PolicyValueNet
from .pimc import rollout_value
from .subgame import SubgameSolver

if TYPE_CHECKING:
    from .match_equity import MatchEquityModel


def legal_mask(state: EuchreState) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in state.legal_actions():
        mask[action_to_index(a)] = True
    return mask


def batch_value_fn_from_net(net: PolicyValueNet):
    """A batched leaf-value function for a net: many states -> one forward pass,
    each returning the team0 - team1 point-differential estimate. Used to plug a
    trained net into the CFR subgame solver at play time."""
    def fn(states: List[EuchreState]) -> List[float]:
        players = [s.current_player if not s.is_terminal() else 0
                   for s in states]
        obs = np.stack([observation_tensor(s, p)
                        for s, p in zip(states, players)])
        with torch.no_grad():
            _, v = net(torch.from_numpy(obs))
        v = v.numpy()
        return [float(v[i]) if team_of(players[i]) == 0 else -float(v[i])
                for i in range(len(states))]
    return fn


# -- C++ hot-path engine (mceuchre_cpp) -------------------------------------
# Optional, opt-in (ReBeLTrainer(engine="cpp")): the engine/observation/
# solver/CFR-search/network hot path ported to C++ (see cpp/README.md), each
# piece differentially verified bit-for-bit / float-tight against this same
# pure-Python implementation (tests/test_cpp_equivalence.py). Only the
# self_play_hand() hot loop is engine-aware -- round2_seed_frac and
# value_ground_frac stay Python-only (see ReBeLTrainer.__init__) since
# they're low-frequency calibration features, not the hot path the C++ port
# targets, and rely on rollout_value, which isn't ported.
def _cpp_module():
    # mceuchre_cpp.cp314-*.pyd is a loose build artifact in the repo root
    # (torch.utils.cpp_extension's build_ext --inplace output), not an
    # installed package -- unlike euchre/rebel, which are always importable
    # via this venv's editable install regardless of caller location.
    # Running a script by path (e.g. `python scripts/train_parallel.py`)
    # sets sys.path[0] to the SCRIPT's own directory (scripts/), not the
    # repo root, so the bare `import mceuchre_cpp` below would raise
    # ModuleNotFoundError there even though the extension is built -- add
    # the repo root explicitly so this works regardless of what invoked it.
    import os
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        import mceuchre_cpp
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "engine='cpp' requires the mceuchre_cpp extension to be built "
            "(see cpp/README.md: python setup.py build_ext --inplace)") from e
    return mceuchre_cpp


def cpp_legal_mask(state) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in state.legal_actions():
        mask[a.index()] = True
    return mask


def cpp_batch_value_fn_from_net(net):
    """Same as batch_value_fn_from_net, but for mceuchre_cpp.EuchreState
    leaves -- team_of is player % 2 (engine.h), matching euchre.game.team_of
    exactly, so no cpp call is needed for it here."""
    cpp = _cpp_module()

    def fn(states) -> List[float]:
        players = [s.current_player if not s.is_terminal() else 0
                   for s in states]
        obs = np.stack([np.asarray(cpp.observation_tensor(s, p))
                        for s, p in zip(states, players)])
        with torch.no_grad():
            _, v = net(torch.from_numpy(obs))
        v = v.numpy()
        return [float(v[i]) if players[i] % 2 == 0 else -float(v[i])
                for i in range(len(states))]
    return fn


@dataclass
class Sample:
    obs: np.ndarray          # observation from the actor's perspective
    mask: np.ndarray         # legal-action mask
    policy: np.ndarray       # CFR target distribution over NUM_ACTIONS
    value: float             # CFR root value, actor's-team differential
    cluster_key: Any = "unknown"  # groups similar decisions for prioritized
                                   # replay sampling (see ReBeLTrainer)
    supervise_policy: bool = True  # False for value-only grounding samples
                                    # (see _grounded_value_sample) -- their
                                    # `policy` field is a placeholder, never
                                    # trained on


class ReBeLTrainer:
    def __init__(self, net: Optional[PolicyValueNet] = None,
                 depth_limit: int = 4, num_worlds: int = 8,
                 cfr_iterations: int = 20, lr: float = 1e-3,
                 buffer_size: int = 20000, belief_model=None,
                 full_depth_cards: int = 0, seed: int = 0,
                 bid_depth_limit: Optional[int] = None,
                 stick_the_dealer: bool = False,
                 grad_clip_norm: float = 5.0,
                 round2_seed_frac: float = 0.0,
                 value_ground_frac: float = 0.0,
                 equity_model: Optional["MatchEquityModel"] = None,
                 engine: str = "python") -> None:
        if engine not in ("python", "cpp"):
            raise ValueError(f"engine must be 'python' or 'cpp', got {engine!r}")
        if engine == "cpp" and (round2_seed_frac > 0 or value_ground_frac > 0):
            # Both features go through rollout_value (rebel/pimc.py), which
            # is deliberately Python-only (see the C++ port plan) -- rather
            # than silently falling back to slow Python for just these
            # calls, require the caller to pick one explicitly.
            raise ValueError(
                "round2_seed_frac/value_ground_frac require engine='python' "
                "(they depend on rollout_value, which isn't ported to C++)")
        if engine == "cpp" and belief_model is not None:
            # cpp.SubgameSolver's production constructor only supports
            # uniform sample_determinization, not belief_model reweighting
            # (rebel/belief_model.py isn't ported).
            raise ValueError("belief_model requires engine='python'")
        self.engine = engine
        self._cpp = _cpp_module() if engine == "cpp" else None
        # A cpp.MatchEquityModel mirror of `equity_model` (built once, not
        # per-hand): self.equity_model itself stays the Python object always
        # -- it's still used for .sample_score() in _fresh_deal, which is
        # pure Python and engine-independent -- but cpp.SubgameSolver needs
        # its own cpp-side equity model instance for equity-aware CFR.
        self._cpp_equity_model = (
            self._cpp.MatchEquityModel(equity_model.target, equity_model.table.flatten().tolist())
            if engine == "cpp" and equity_model is not None else None)
        self.net = net or PolicyValueNet()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        # Safety net, not a tuning knob: caps the gradient norm of any single
        # train_step so one high-loss batch can't produce an outsized update.
        # Matters more now that prioritized replay deliberately biases
        # sampling toward the highest-loss clusters -- if one of those turns
        # out to be irreducibly noisy rather than genuinely learnable (see
        # cluster_stats), this bounds the damage to the update size, the same
        # way priority_ceiling bounds it to the sampling rate.
        self.grad_clip_norm = grad_clip_norm
        self.depth_limit = depth_limit
        # Bidding decisions (BID_ROUND_1/2, DEALER_DISCARD) are the furthest
        # in the game tree from the only exact solves this trainer ever does
        # (the last full_depth_cards tricks of PLAY), so at the shared
        # depth_limit they lean almost entirely on the value net's guess of
        # how the rest of the hand plays out. Letting bidding search deeper
        # gives it more real CFR-solved lookahead before falling back to the
        # net. None = fall back to depth_limit (unchanged behaviour).
        self.bid_depth_limit = bid_depth_limit
        # Off by default: EuchreState.new_hand() then lets round 2 fully pass
        # out into a misdeal, so self-play never faces a forced call. Turning
        # this on trains that decision instead of leaving it unseen.
        self.stick_the_dealer = stick_the_dealer
        # Round 2 only happens after all four players pass round 1, so
        # natural random dealing reaches it in ~5% of hands (measured) --
        # exactly the decision type most in need of training signal and the
        # slowest one to accumulate it. Biasing a fraction of *deals* so the
        # up-card's suit is weak for all four hands raises the prior
        # probability that round 1 naturally resolves into round 2, without
        # ever skipping round 1's own decisions -- every player's round-1
        # turn still gets a real SubgameSolver call and a real Sample; round
        # 2 is only reached if the *actual current strategy* genuinely
        # passes all four times, same as an unbiased deal, just more often.
        # (An earlier version fast-forwarded straight to round 2 instead --
        # that generated zero round-1 samples for those hands and was
        # replaced with this.) Later diagnosis found round 2's real scarcity
        # (measured ~0.67% single-trajectory rate, not the ~5% initially
        # assumed) is mostly a *symptom* of round-1 over-calling (the net
        # called on ~57% of random hands vs. a sensible ~12% baseline), not
        # a sampling problem this alone fixes -- see value_ground_frac below
        # for the fix aimed at the actual root cause.
        self.round2_seed_frac = round2_seed_frac
        # A continuous, low-weight anchor against the self-referential
        # bootstrap drift diagnosed this session: the value head was found
        # to overestimate post-call outcomes by roughly half a point to a
        # full point on average, because nothing outside the last
        # full_depth_cards tricks ever checks its leaf estimates against
        # reality. Mixes a fraction of _grounded_value_sample() calls
        # (exact rollout_value on a genuine post-call state, no CFR, no
        # value-net dependency) into self-play -- value-only supervision
        # (see Sample.supervise_policy), never policy, since rollout_value's
        # own double-dummy assumption has a real bias of its own (overvalues
        # the defense relative to real imperfect-information opponents) --
        # good for correcting gross miscalibration, not for fine policy
        # tuning. Off by default.
        self.value_ground_frac = value_ground_frac
        # None (default) preserves exact prior behavior throughout this
        # class: every deal starts 0-0, SubgameSolver gets no equity_model
        # (raw point-differential CFR targets, unchanged), and
        # _grounded_value_sample's rollout_value calls stay raw-point too.
        # When set, self-play samples a realistic match-score context per
        # hand (via equity_model.sample_score, weighted by how often that
        # score actually arises) and CFR's own terminal utilities become
        # equity-aware -- see rebel/match_equity.py and docs/rebel_design.md.
        self.equity_model = equity_model
        self.num_worlds = num_worlds
        self.cfr_iterations = cfr_iterations
        self.buffer: List[Sample] = []
        self.buffer_size = buffer_size
        self.belief_model = belief_model
        # Prioritized replay, grouped by cluster key instead of per sample:
        # far less state to track (a few dozen hand-strength buckets, not one
        # priority per buffer entry), and a fresh sample in a weak bucket
        # inherits that bucket's known priority immediately instead of
        # starting cold the way per-sample PER does. `by_key` mirrors
        # `buffer`, partitioned; `cluster_priority` is an EMA of per-sample
        # loss for each key, updated in train_step.
        self.by_key: Dict[Any, List[Sample]] = {}
        self.cluster_priority: Dict[Any, float] = {}
        self.priority_alpha = 0.5    # 0 = ignore priority, 1 = fully proportional
        self.priority_ema = 0.3      # weight on the newest loss observation
        self.priority_floor = 0.05   # keeps a "mastered" cluster from starving
        self.priority_default = 1.0  # neutral: makes weighting reduce to plain
                                      # size-proportional (== uniform-over-buffer)
                                      # sampling until a cluster's loss is known
        # Safety valve: loss can stay persistently high for a cluster that's
        # genuinely still learnable, but also for one that's irreducibly noisy
        # (e.g. targets generated from different value-net snapshots over the
        # run, or a position with a genuinely high-entropy equilibrium) --
        # this can't tell those apart, so cap how far any one cluster can be
        # oversampled relative to a "typical" one rather than trying to.
        self.priority_ceiling_mult = 10.0
        self._point_count = PointCountAgent()
        # "As much depth as feasible per position": when the acting player has
        # <= full_depth_cards cards left, solve the subgame to *terminal* (exact
        # CFR targets, no value net) since the tree is then cheap. Deeper into
        # the hand this yields exact endgame targets that anchor the value net,
        # so the depth-limited early-game leaves it feeds are less noisy.
        self.full_depth_cards = full_depth_cards
        self.rng = random.Random(seed)

    def _depth_for(self, state) -> Optional[int]:
        is_play = (state.phase == self._cpp.Phase.Play if self.engine == "cpp"
                  else state.phase == Phase.PLAY)
        if self.engine == "cpp":
            hand_size = state.hands[state.current_player].bit_count()  # bitmask, not a list
        else:
            hand_size = len(state.hands[state.current_player])
        if (self.full_depth_cards > 0 and is_play
                and hand_size <= self.full_depth_cards):
            return None  # full-depth / exact
        is_bid_or_discard = (
            state.phase in (self._cpp.Phase.BidRound1, self._cpp.Phase.BidRound2,
                            self._cpp.Phase.DealerDiscard) if self.engine == "cpp"
            else state.phase in (Phase.BID_ROUND_1, Phase.BID_ROUND_2,
                                 Phase.DEALER_DISCARD))
        if is_bid_or_discard:
            return (self.bid_depth_limit if self.bid_depth_limit is not None
                    else self.depth_limit)
        return self.depth_limit

    _BUCKET_WIDTH = 0.4  # PointCountAgent's thresholds are 2.2/2.4/3.6, so this
                         # gives ~11 buckets over the practical [0, ~4.5] range

    def _cluster_key(self, state, actor: int) -> Any:
        """Group a decision into a rough hand-strength bucket for prioritized
        replay. Only bidding phases get fine-grained buckets, via the same
        point-count score used by PointCountAgent -- that's the axis the quiz
        scorecard actually showed weakness on (under-calling marginal
        ace-heavy hands, spurious alone calls). Discard/play get one coarse
        bucket each for now; no diagnosed weakness there yet to target.

        Cluster keys are only ever compared within one trainer's lifetime
        (self.engine is fixed at construction), so it's fine that the cpp
        branch's phase-name strings ("BidRound1") differ in spelling from
        the Python branch's ("BID_ROUND_1") -- they never need to match
        across engines, only to group consistently within one."""
        if self.engine == "cpp":
            from euchre.cards import Card as PyCard, Suit as PySuit
            hand = [PyCard.from_id(c) for c in range(24) if (state.hands[actor] >> c) & 1]
            if state.phase == self._cpp.Phase.BidRound1:
                up_suit = PyCard.from_id(state.up_card).suit
                score = self._point_count.hand_score(hand, up_suit)
                return ("bid1", int(score // self._BUCKET_WIDTH))
            if state.phase == self._cpp.Phase.BidRound2:
                turned = PySuit(state.turned_down)
                score = max(self._point_count.hand_score(hand, s) for s in PySuit
                           if s != turned)
                return ("bid2", int(score // self._BUCKET_WIDTH))
            return (state.phase.name,)

        hand = state.hands[actor]
        if state.phase == Phase.BID_ROUND_1:
            score = self._point_count.hand_score(hand, state.up_card.suit)
            return ("bid1", int(score // self._BUCKET_WIDTH))
        if state.phase == Phase.BID_ROUND_2:
            score = max(self._point_count.hand_score(hand, s) for s in Suit
                       if s != state.turned_down)
            return ("bid2", int(score // self._BUCKET_WIDTH))
        return (state.phase.name,)

    # -- leaf value from the current network ---------------------------------

    def value_fn(self, state) -> float:
        """Estimate a subgame leaf's value: team0 - team1 point differential,
        or (when equity_model is set) a team0-signed match win-probability
        delta -- whichever units the net is currently being trained to
        predict, since this just reads its raw output."""
        return self.batch_value_fn([state])[0]

    def batch_value_fn(self, states) -> List[float]:
        """Value many leaves in a single network forward pass (see
        ``batch_value_fn_from_net`` / ``cpp_batch_value_fn_from_net``)."""
        if self.engine == "cpp":
            return cpp_batch_value_fn_from_net(self.net)(states)
        return batch_value_fn_from_net(self.net)(states)

    # -- self-play -----------------------------------------------------------

    def _fresh_deal(self):
        dealer = self.rng.randint(0, 3)
        team0_score = team1_score = 0
        if self.equity_model is not None:
            # sample_score draws (dealing team's score, other team's score)
            # -- map onto team0/team1 using whichever team `dealer` is.
            dealer_score, other_score = self.equity_model.sample_score(self.rng)
            if team_of(dealer) == 0:
                team0_score, team1_score = dealer_score, other_score
            else:
                team0_score, team1_score = other_score, dealer_score
        if self.engine == "cpp":
            deck = list(range(24))
            self.rng.shuffle(deck)
            return self._cpp.EuchreState.new_hand(
                dealer=dealer,
                stick_the_dealer=self.stick_the_dealer,
                team0_score=team0_score, team1_score=team1_score).deal_from_deck(deck)
        return EuchreState.new_hand(
            dealer=dealer,
            stick_the_dealer=self.stick_the_dealer,
            team0_score=team0_score, team1_score=team1_score).deal(self.rng)

    # ORDER_1_THRESH (2.2) is where PointCountAgent itself calls -- 86% of
    # random first-actor hands already score under it (measured), so
    # filtering there barely biases anything and sits right at the
    # ambiguous boundary where a reasonably-calibrated bid1 policy has real
    # reason to mix rather than confidently pass. This needs hands that are
    # unambiguously weak, not merely "under the calling line" -- close to
    # the scoring floor (5 off-suit junk cards score 0.15, the minimum
    # possible), not the decision boundary.
    _ROUND2_BIAS_MAX_SCORE = 0.3  # ~4.8% of hands qualify (measured)

    def _biased_deal(self, max_tries: int = 200) -> EuchreState:
        """Deal, rejecting and redealing until the *first-to-act* hand is
        unambiguously weak for the up-card's suit -- raises the odds round 1
        genuinely resolves toward round 2, without touching how round 1
        itself gets decided or recorded, and without needing all four hands
        to qualify (only checking the first actor is both simpler and a much
        higher acceptance rate). Falls back to a normal deal if no
        qualifying deal turns up within the try budget."""
        for _ in range(max_tries):
            state = self._fresh_deal()
            suit = state.up_card.suit
            first = state.current_player
            if (self._point_count.hand_score(state.hands[first], suit)
                    <= self._ROUND2_BIAS_MAX_SCORE):
                return state
        return self._fresh_deal()

    # Matches recalibrate_value.py's default -- round 2 is the specific spot
    # this session's diagnosis traced the bias to, so it stays deliberately
    # over-represented relative to its natural (~1%) frequency here too.
    _VALUE_GROUND_ROUND2_FRAC = 0.3

    def _grounded_value_sample(self) -> Optional[Sample]:
        """One exact rollout_value-grounded post-call sample, built the same
        way recalibrate_value.py's build_samples() does -- a real deal, a
        real call (round 1 directly, or round 2 via four genuine passes),
        then the exact double-dummy value of the resulting state. No CFR, no
        dependence on the value net currently being trained, so it can't
        inherit that net's own bias. Returns None on the (rare, defensive)
        case a round-2 walk doesn't land on a callable state -- callers
        should just skip storing anything that turn rather than retry, to
        avoid a hidden retry loop on a state space we already know is thin."""
        state = self._fresh_deal()
        if self.rng.random() < self._VALUE_GROUND_ROUND2_FRAC:
            for _ in range(4):
                state = state.apply(Pass())
            if state.phase != Phase.BID_ROUND_2:
                return None
            calls = [a for a in state.legal_actions()
                     if isinstance(a, Call) and not a.alone]
            if not calls:
                return None
            nxt = state.apply(self.rng.choice(calls))
        else:
            nxt = state.apply(OrderUp(alone=False))

        # exact -- all 4 hands already known; nxt already carries whatever
        # score _fresh_deal sampled (apply()/clone() preserve it), so no
        # separate sampling call is needed here.
        v0 = rollout_value(nxt, team0_score=nxt.team0_score,
                           team1_score=nxt.team1_score,
                           equity_model=self.equity_model)
        leaf_player = nxt.current_player
        target = v0 if team_of(leaf_player) == 0 else -v0
        return Sample(
            obs=observation_tensor(nxt, leaf_player),
            mask=legal_mask(nxt),
            policy=np.zeros(NUM_ACTIONS, dtype=np.float32),  # unused, see
                                                              # supervise_policy
            value=target,
            cluster_key=("value_ground",),
            supervise_policy=False)

    def self_play_hand(self) -> Tuple[int, int]:
        if self.value_ground_frac > 0 and self.rng.random() < self.value_ground_frac:
            gs = self._grounded_value_sample()
            if gs is not None:
                self._store(gs)
        if self.round2_seed_frac > 0 and self.rng.random() < self.round2_seed_frac:
            state = self._biased_deal()
        else:
            state = self._fresh_deal()
        while not state.is_terminal():
            legal = state.legal_actions()
            if len(legal) == 1:
                state = state.apply(legal[0])
                continue
            actor = state.current_player
            if self.engine == "cpp":
                depth = self._depth_for(state)
                solver = self._cpp.SubgameSolver(
                    state, actor, self.num_worlds, self.cfr_iterations,
                    -1 if depth is None else depth, self.batch_value_fn,
                    self._cpp_equity_model, self.rng.getrandbits(63))
            else:
                solver = SubgameSolver(
                    state, actor, num_worlds=self.num_worlds,
                    iterations=self.cfr_iterations, depth_limit=self._depth_for(state),
                    batch_value_fn=self.batch_value_fn,
                    belief_model=self.belief_model, equity_model=self.equity_model,
                    rng=self.rng)
            solver.run()
            policy = solver.root_policy()
            # team0 - team1 raw points, or (equity_model set) a team0-signed
            # match win-probability delta -- either way, root_val's units
            # match whatever Sample.value trains the value head to predict.
            root_val = solver.root_value()

            target = np.zeros(NUM_ACTIONS, dtype=np.float32)
            if self.engine == "cpp":
                for idx, p in policy.items():
                    target[idx] = p
                obs = np.asarray(self._cpp.observation_tensor(state, actor))
                mask = cpp_legal_mask(state)
            else:
                for a, p in policy.items():
                    target[action_to_index(a)] = p
                obs = observation_tensor(state, actor)
                mask = legal_mask(state)
            actor_val = root_val if team_of(actor) == 0 else -root_val
            self._store(Sample(
                obs=obs,
                mask=mask,
                policy=target,
                value=actor_val,
                cluster_key=self._cluster_key(state, actor)))

            actions = list(policy)
            chosen = self.rng.choices(
                actions, weights=[policy[a] for a in actions])[0]
            if self.engine == "cpp":
                chosen = self._cpp.Action.from_index(chosen)
            state = state.apply(chosen)
        return state.returns()

    def _store(self, sample: Sample) -> None:
        self.buffer.append(sample)
        self.by_key.setdefault(sample.cluster_key, []).append(sample)
        if len(self.buffer) > self.buffer_size:
            old = self.buffer.pop(0)
            # `buffer` and each `by_key[k]` list are both append-only, so the
            # globally oldest sample is always at index 0 of its own key's
            # list too -- no scan needed to find it.
            lst = self.by_key.get(old.cluster_key)
            if lst:
                (lst.pop(0) if lst[0] is old else lst.remove(old))
                if not lst:
                    del self.by_key[old.cluster_key]

    # -- learning ------------------------------------------------------------

    def _cluster_weights(self) -> Tuple[List[Any], List[float]]:
        """Sampling weight per known cluster key: size * min(priority, ceiling)^alpha.

        Shared by `_sample_batch` (actual sampling) and `cluster_stats`
        (diagnostics) so the reported sample share always matches what's
        really drawn -- one source of truth for the formula.

        The ceiling is relative, not a fixed number: a multiple of the
        median *measured* priority, so it adapts as the overall loss level
        drops over training instead of needing manual retuning. Guards
        against a cluster whose loss stays high for irreducible reasons
        (moving bootstrap targets, a genuinely high-entropy equilibrium)
        rather than genuinely-still-learnable ones -- this can't tell those
        apart, so it just bounds the worst case instead.
        """
        keys = list(self.by_key.keys())
        if not keys:
            return [], []
        measured = list(self.cluster_priority.values())
        typical = (sorted(measured)[len(measured) // 2] if measured
                  else self.priority_default)
        ceiling = max(typical * self.priority_ceiling_mult, self.priority_default)
        weights = []
        for k in keys:
            p = max(self.cluster_priority.get(k, self.priority_default),
                    self.priority_floor)
            p = min(p, ceiling)
            weights.append(len(self.by_key[k]) * p ** self.priority_alpha)
        return keys, weights

    def _sample_batch(self, n: int):
        """Draw `n` samples, weighted by cluster priority.

        Two-stage: pick a cluster key (weighted), then pick uniformly within
        it -- O(num_keys + n) instead of scanning the whole buffer to build a
        per-sample weight array (num_keys is a few dozen; the buffer can be
        up to buffer_size). Weighting each key by `len(by_key[k])` alongside
        its priority means a neutral/unmeasured priority (the default) makes
        this mathematically equivalent to uniform sampling over the flat
        buffer -- any skew comes only from the learned priority signal, not
        from cluster granularity itself.
        """
        keys, weights = self._cluster_weights()
        drawn_keys = self.rng.choices(keys, weights=weights, k=n)
        return [self.rng.choice(self.by_key[k]) for k in drawn_keys], drawn_keys

    def cluster_stats(self, top_n: int = 5) -> List[dict]:
        """Diagnostic snapshot: the top clusters by *effective sample share*
        (post-ceiling, matching what's actually drawn) -- not just raw
        priority, so it's directly visible whether the ceiling is doing
        anything and whether any cluster is dominating training."""
        keys, weights = self._cluster_weights()
        if not keys:
            return []
        total = sum(weights) or 1.0
        rows = [
            {"key": list(k) if isinstance(k, tuple) else k,
             "priority": round(self.cluster_priority.get(k, self.priority_default), 3),
             "count": len(self.by_key[k]),
             "sample_share": round(w / total, 4)}
            for k, w in zip(keys, weights)
        ]
        rows.sort(key=lambda r: -r["sample_share"])
        return rows[:top_n]

    def train_step(self, batch_size: int = 128) -> dict:
        if not self.buffer:
            return {"policy_loss": 0.0, "value_loss": 0.0, "grad_norm": 0.0}
        n = min(batch_size, len(self.buffer))
        batch, drawn_keys = self._sample_batch(n)
        obs = torch.from_numpy(np.stack([s.obs for s in batch]))
        mask = torch.from_numpy(np.stack([s.mask for s in batch]))
        target_p = torch.from_numpy(np.stack([s.policy for s in batch]))
        target_v = torch.tensor([s.value for s in batch], dtype=torch.float32)
        supervise_p = torch.tensor([1.0 if s.supervise_policy else 0.0
                                    for s in batch], dtype=torch.float32)

        logits, value = self.net(obs)
        logits = logits.masked_fill(~mask, float("-inf"))
        logp = F.log_softmax(logits, dim=-1)
        # Cross-entropy against the CFR target distribution (legal-only). Zero
        # out illegal entries so the target's 0 * (-inf) does not become NaN.
        logp = torch.where(mask, logp, torch.zeros_like(logp))
        # Value-only grounding samples (supervise_policy=False) contribute
        # nothing to policy_loss -- their `policy` field is a placeholder,
        # never a real target -- and get excluded from the averaging
        # denominator too, not just zeroed in the numerator.
        per_policy_loss = -(target_p * logp).sum(dim=-1) * supervise_p
        n_policy = supervise_p.sum().clamp(min=1.0)
        per_value_loss = F.mse_loss(value, target_v, reduction="none")
        policy_loss = per_policy_loss.sum() / n_policy
        value_loss = per_value_loss.mean()
        loss = policy_loss + value_loss

        self.opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                                    self.grad_clip_norm)
        self.opt.step()

        # Feed this batch's per-sample loss back into each drawn cluster's
        # priority (EMA), so future batches lean toward clusters the net is
        # currently getting wrong -- without tracking priority per sample.
        per_sample_loss = (per_policy_loss + per_value_loss).detach().numpy()
        for k, l in zip(drawn_keys, per_sample_loss):
            old = self.cluster_priority.get(k, self.priority_default)
            self.cluster_priority[k] = ((1 - self.priority_ema) * old
                                        + self.priority_ema * float(l))

        return {"policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "grad_norm": float(grad_norm)}

    def train(self, generations: int, hands_per_gen: int = 4,
              train_steps: int = 8, batch_size: int = 128,
              log: bool = False) -> List[dict]:
        history = []
        for g in range(1, generations + 1):
            for _ in range(hands_per_gen):
                self.self_play_hand()
            stats = {}
            for _ in range(train_steps):
                stats = self.train_step(batch_size)
            stats = {"gen": g, "buffer": len(self.buffer), **stats}
            history.append(stats)
            if log:
                print(f"gen {g}: buffer={stats['buffer']} "
                      f"policy_loss={stats['policy_loss']:.4f} "
                      f"value_loss={stats['value_loss']:.4f}")
        return history


class ReBeLNetAgent:
    """Fast inference agent: acts from the trained policy head, no search.

    This is what you deploy once the net has learned; decision-time search
    (`CFRSearchAgent` with ``value_fn=trainer.value_fn``) can be layered back on
    top for extra strength.
    """

    def __init__(self, net: PolicyValueNet, greedy: bool = True,
                 temperature: float = 1.0) -> None:
        self.net = net
        self.greedy = greedy
        self.temperature = temperature

    def act(self, state: EuchreState, rng: random.Random):
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        obs = torch.from_numpy(
            observation_tensor(state, state.current_player)).unsqueeze(0)
        mask = torch.from_numpy(legal_mask(state)).unsqueeze(0)
        with torch.no_grad():
            dist = self.net.policy(obs, mask).squeeze(0).numpy()
        idxs = [action_to_index(a) for a in legal]
        probs = np.array([dist[i] for i in idxs], dtype=np.float64)
        if probs.sum() <= 0:
            probs = np.ones(len(legal)) / len(legal)
        else:
            probs = probs / probs.sum()
        if self.greedy:
            return legal[int(np.argmax(probs))]
        # Optimal play in an imperfect-information game is a *mixed* strategy;
        # temperature keeps the agent from collapsing to a deterministic (and
        # thus exploitable) policy.
        if self.temperature != 1.0:
            probs = probs ** (1.0 / self.temperature)
            probs = probs / probs.sum()
        return legal[rng.choices(range(len(legal)), weights=probs.tolist())[0]]
