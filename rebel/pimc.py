"""Perfect-Information Monte-Carlo (PIMC) agent.

The classic strong-baseline method for trick-taking card games: sample many
full deals ("worlds") consistent with what the player can observe, solve each
world with the perfect-information solver, and pick the action with the best
average value. It gives genuine decision-time reasoning about hidden cards --
the thing a reactive policy net cannot do -- and serves as both a strong
opponent and a benchmark for the ReBeL agent.

Strength scales with the number of determinizations, but so does cost, and the
per-world double-dummy solve is far more expensive early in the hand (5 cards)
than late (nearly free). So the strongest *practical* configuration uses a
per-decision **time budget** (``play_budget`` / ``call_budget``): keep sampling
and solving worlds until the budget runs out. That spends the search where it
is cheap -- hundreds of worlds late, a few dozen on the first trick -- for the
best play at a fixed latency. ``strong_pimc()`` is the ready-made preset
(belief refinement on, budgeted search). Set a fixed ``worlds`` count instead
for reproducible behaviour (used in tests).

Known limitation (documented, not hidden): PIMC assumes perfect information
*within* each world, so it cannot reason about its own future information
gain or hide information from opponents ("strategy fusion"). ISMCTS / the
ReBeL subgame solver address this; PIMC remains an excellent baseline.
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional

from euchre.actions import (
    Action, Pass, OrderUp, Call, Discard, Play,
)
from euchre.game import EuchreState, Phase, team_of
from .solver import solve_value
from .public_belief_state import sample_determinization


def _team_sign(value_team0: int, team: int) -> int:
    """Convert a team0-team1 value into the given team's point differential."""
    return value_team0 if team == 0 else -value_team0


def rollout_value(state: EuchreState, memo: Optional[dict] = None) -> int:
    """Value (team0 - team1) of a fully-known world under optimal play.

    Resolves a pending dealer discard by choosing the discard that is best for
    the dealer's team, then solves the play phase.
    """
    if state.is_terminal():
        r = state.returns()
        return r[0] - r[1]
    if state.phase == Phase.PLAY:
        return solve_value(state, memo)
    if state.phase == Phase.DEALER_DISCARD:
        dealer_team = team_of(state.dealer)
        best: Optional[int] = None
        shared: dict = {}
        for a in state.legal_actions():
            v = rollout_value(state.apply(a), shared)
            if best is None or (v > best if dealer_team == 0 else v < best):
                best = v
        return best
    raise ValueError(f"rollout_value cannot start from phase {state.phase}")


class PIMCAgent:
    """A PIMC player.

    Args:
        worlds: determinizations sampled per decision (more = stronger, slower).
        call_worlds: worlds used for the (cheaper-per-world) bidding search.
        call_threshold: minimum expected point differential to make trump; the
            neutral value of passing is approximated as 0.
        seed: RNG seed.
    """

    def __init__(self, worlds: int = 20, call_worlds: int = 10,
                 call_threshold: float = 0.4, belief_model=None,
                 play_budget: Optional[float] = None,
                 call_budget: Optional[float] = None,
                 min_worlds: int = 8, max_worlds: int = 400,
                 seed: int = 0) -> None:
        self.worlds = worlds
        self.call_worlds = call_worlds
        self.call_threshold = call_threshold
        self.belief_model = belief_model
        # When a *_budget (seconds) is set, keep sampling and solving worlds
        # until the budget runs out (bounded by [min_worlds, max_worlds]). This
        # spends the search where it is cheap -- many worlds late in the hand,
        # fewer on the expensive first trick -- for the strongest play at a
        # fixed per-decision latency. When None, use the fixed ``worlds`` count.
        self.play_budget = play_budget
        self.call_budget = call_budget
        self.min_worlds = min_worlds
        self.max_worlds = max_worlds
        self._rng = random.Random(seed)

    # -- public API ----------------------------------------------------------

    def act(self, state: EuchreState, rng: random.Random) -> Action:
        phase = state.phase
        if phase == Phase.PLAY:
            return self._act_play(state)
        if phase == Phase.DEALER_DISCARD:
            return self._act_discard(state)
        if phase in (Phase.BID_ROUND_1, Phase.BID_ROUND_2):
            return self._act_bid(state)
        return rng.choice(state.legal_actions())

    # -- play / discard via the solver --------------------------------------

    def _weighted_totals(self, state: EuchreState, actor: int,
                         actions: List[Action], team: int) -> Dict[Action, float]:
        """Sample worlds (fixed count or within ``play_budget``), solve each
        action in each world, and return the belief-weighted total value per
        action. Belief weights condition on the bidding when a model is set."""
        model = self.belief_model
        bids = None
        if model is not None:
            from .belief_model import reconstruct_bids
            bids = reconstruct_bids(state)

        collected: List[tuple] = []  # (log_weight, {action: value})
        budget = self.play_budget
        deadline = time.time() + budget if budget is not None else None
        n = 0
        while True:
            world = sample_determinization(state, actor, self._rng)
            memo: dict = {}  # shared across sibling actions in this world
            vals = {a: _team_sign(solve_value(world.apply(a), memo), team)
                    for a in actions}
            log_w = 0.0
            if model is not None and bids:
                from .belief_model import (
                    reconstruct_original_hands, deal_log_weight)
                log_w = deal_log_weight(reconstruct_original_hands(world), bids,
                                        model, state.alone, state.dealer)
            collected.append((log_w, vals))
            n += 1
            if deadline is not None:
                if n >= self.min_worlds and (time.time() >= deadline
                                             or n >= self.max_worlds):
                    break
            elif n >= self.worlds:
                break

        m = max(lw for lw, _ in collected)
        totals: Dict[Action, float] = {a: 0.0 for a in actions}
        for log_w, vals in collected:
            w = math.exp(log_w - m)
            for a in actions:
                totals[a] += w * vals[a]
        return totals

    def _act_play(self, state: EuchreState) -> Action:
        actor = state.current_player
        legal = [Play(c) for c in state._legal_plays(actor)]
        if len(legal) == 1:
            return legal[0]
        totals = self._weighted_totals(state, actor, legal, team_of(actor))
        return max(totals, key=lambda a: totals[a])

    def _act_discard(self, state: EuchreState) -> Action:
        actor = state.dealer
        legal = state.legal_actions()
        totals = self._weighted_totals(state, actor, legal, team_of(actor))
        return max(totals, key=lambda a: totals[a])

    # -- bidding via a determinized value search ----------------------------

    def _act_bid(self, state: EuchreState) -> Action:
        actor = state.current_player
        team = team_of(actor)
        legal = state.legal_actions()
        options = [a for a in legal if not isinstance(a, Pass)]
        # No trump is set yet, so there is no bidding to condition on; the
        # belief is uniform. Estimate each option's value by rolling the
        # determinized hand forward and solving it.
        sums = {a: 0.0 for a in options}
        deadline = (time.time() + self.call_budget
                    if self.call_budget is not None else None)
        n = 0
        while True:
            world = sample_determinization(state, actor, self._rng)
            for a in options:
                sums[a] += _team_sign(rollout_value(world.apply(a)), team)
            n += 1
            if deadline is not None:
                if n >= self.min_worlds and (time.time() >= deadline
                                             or n >= self.max_worlds):
                    break
            elif n >= self.call_worlds:
                break

        # Value each non-pass option; passing is approximated as neutral (0).
        best_action: Optional[Action] = None
        best_value = self.call_threshold
        for a in options:
            val = sums[a] / n
            if best_action is None or val > best_value:
                best_action, best_value = a, val

        passes = [a for a in legal if isinstance(a, Pass)]
        if best_action is not None and (not passes or best_value >=
                                        self.call_threshold):
            return best_action
        if passes:
            return passes[0]
        # Stick-the-dealer: forced to call -> take the best option regardless.
        non_pass = [a for a in legal if not isinstance(a, Pass)]
        return best_action or non_pass[0]


def strong_pimc(play_budget: float = 5.0, call_budget: float = 4.0,
                seed: int = 0) -> PIMCAgent:
    """The strongest reasonable PIMC preset: bidding-conditioned belief on, and
    time-budgeted search that uses as many determinizations as each decision
    affords (many late in the hand where solves are cheap, fewer on the
    expensive first trick). Tune the budgets to trade latency for strength.
    """
    from .belief_model import BiddingBeliefModel
    return PIMCAgent(
        belief_model=BiddingBeliefModel(),
        play_budget=play_budget, call_budget=call_budget,
        min_worlds=12, max_worlds=400, call_threshold=0.4, seed=seed)
