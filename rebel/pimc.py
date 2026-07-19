"""Perfect-Information Monte-Carlo (PIMC) agent.

The classic strong-baseline method for trick-taking card games: sample many
full deals ("worlds") consistent with what the player can observe, solve each
world with the perfect-information solver, and pick the action with the best
average value. It gives genuine decision-time reasoning about hidden cards --
the thing a reactive policy net cannot do -- and serves as both a strong
opponent and a benchmark for the ReBeL agent.

Known limitation (documented, not hidden): PIMC assumes perfect information
*within* each world, so it cannot reason about its own future information
gain or hide information from opponents ("strategy fusion"). ISMCTS / the
ReBeL subgame solver address this; PIMC remains an excellent baseline.
"""

from __future__ import annotations

import random
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
                 seed: int = 0) -> None:
        self.worlds = worlds
        self.call_worlds = call_worlds
        self.call_threshold = call_threshold
        self.belief_model = belief_model
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

    def _sample_belief(self, state: EuchreState, actor: int, n: int
                       ) -> "tuple[List[EuchreState], List[float]]":
        """Return (worlds, weights). Weights condition on the bidding when a
        belief model is set, else uniform."""
        if self.belief_model is not None:
            from .belief_model import sample_weighted_belief
            return sample_weighted_belief(state, actor, n, self.belief_model,
                                          self._rng)
        worlds = [sample_determinization(state, actor, self._rng)
                  for _ in range(n)]
        return worlds, [1.0 / n] * n

    def _act_play(self, state: EuchreState) -> Action:
        actor = state.current_player
        team = team_of(actor)
        legal = [Play(c) for c in state._legal_plays(actor)]
        if len(legal) == 1:
            return legal[0]
        totals: Dict[Action, float] = {a: 0.0 for a in legal}
        worlds, weights = self._sample_belief(state, actor, self.worlds)
        for world, w in zip(worlds, weights):
            memo: dict = {}  # shared across sibling actions in this world
            for a in legal:
                v = solve_value(world.apply(a), memo)
                totals[a] += w * _team_sign(v, team)
        return max(totals, key=lambda a: totals[a])

    def _act_discard(self, state: EuchreState) -> Action:
        actor = state.dealer
        team = team_of(actor)
        legal = state.legal_actions()
        totals: Dict[Action, float] = {a: 0.0 for a in legal}
        worlds, weights = self._sample_belief(state, actor, self.worlds)
        for world, w in zip(worlds, weights):
            memo: dict = {}
            for a in legal:
                v = solve_value(world.apply(a), memo)
                totals[a] += w * _team_sign(v, team)
        return max(totals, key=lambda a: totals[a])

    # -- bidding via a determinized value search ----------------------------

    def _option_value(self, state: EuchreState, action: Action,
                      actor: int, worlds: List[EuchreState]) -> float:
        team = team_of(actor)
        total = 0.0
        for world in worlds:
            total += _team_sign(rollout_value(world.apply(action)), team)
        return total / len(worlds)

    def _act_bid(self, state: EuchreState) -> Action:
        actor = state.current_player
        legal = state.legal_actions()
        # No trump is set yet, so there is no bidding to condition on; the
        # belief is uniform here regardless of model.
        worlds, _weights = self._sample_belief(state, actor, self.call_worlds)

        # Value each non-pass option; passing is approximated as neutral (0).
        best_action: Optional[Action] = None
        best_value = self.call_threshold
        for a in legal:
            if isinstance(a, Pass):
                continue
            val = self._option_value(state, a, actor, worlds)
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
