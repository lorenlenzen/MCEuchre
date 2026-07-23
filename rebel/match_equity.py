"""Match equity: converting a hand's raw point differential into a
race-to-target win-probability delta, conditioned on the current match score.

Real Euchre strategy is score-dependent (analogous to backgammon match equity
or poker tournament ICM) -- a team one point from winning should strongly
prefer a safe sure-thing over a high-variance loner attempt with equal or
better *raw* expected points. Nothing in this codebase's search or training
loop knew the match score existed until this module.

Two pieces:

* :func:`fit_hand_outcome_distribution` / :func:`build_equity_table` --
  empirically measure how often each of Euchre's 7 discrete single-hand
  outcomes occurs (fit against a heuristic agent, not the eventual trained
  net -- an approximation, cheap to refit later against a stronger agent),
  then value-iterate a table E(a, b) = team0's probability of winning the
  match from score (a, b), to a fixed point over the resulting absorbing
  Markov chain.
* :class:`MatchEquityModel` -- wraps the table with the two operations the
  rest of the pipeline needs: converting a hand's raw (p0, p1) point outcome
  into a win-probability delta, and sampling a realistic starting score for
  self-play.

Design note (see docs/rebel_design.md "Milestone 3.6" and the plan this
session): Euchre's scoring has a structural property -- exactly one team
scores per non-misdeal hand -- that guarantees the *ordinal* ranking of the 7
discrete outcomes under equity always matches their ranking under raw point
differential, for any monotonic equity table and any fixed starting score.
That is why `solve_value`/`rollout_value`'s internal double-dummy search can
stay completely raw-point-based (no transposition-table re-keying needed):
the conversion here only needs to happen at the boundary where CFR compares
*expectations* over mixed/uncertain outcomes (SubgameSolver, MCCFRTrainer),
where a saturating equity function genuinely can change which option is
better -- the whole reason match equity matters.
"""

from __future__ import annotations

import json
import random
from typing import Callable, Dict, List, Tuple

import numpy as np

from .evaluate import RuleBasedAgent, play_hand

Outcome = Tuple[int, int]  # (team0_points, team1_points) from one hand


def fit_hand_outcome_distribution(
        hands: int = 20000, seed: int = 0,
        agent_factory: Callable[[], object] = RuleBasedAgent,
        ) -> Dict[Outcome, float]:
    """Empirical frequency of each single-hand (team0_points, team1_points)
    outcome, via the same `play_hand` harness `evaluate()` uses -- already
    dealer-alternating, so naturally team-symmetric. Approximate by
    construction (fit against a heuristic agent, not the eventual trained
    net); refitting later against a stronger agent is just rerunning this,
    not required now.
    """
    rng = random.Random(seed)
    a0, a1 = agent_factory(), agent_factory()
    agents = [a0, a1, a0, a1]
    counts: Dict[Outcome, int] = {}
    for h in range(hands):
        r0, r1 = play_hand(agents, dealer=h % 4, rng=rng)
        key = (r0, r1)
        counts[key] = counts.get(key, 0) + 1

    # The underlying process IS exactly team-symmetric (both seats run the
    # same agent class, dealer alternates evenly across hands via h % 4), but
    # a finite empirical sample won't measure it as exactly symmetric --
    # e.g. one run measured (2,0) at 0.2554 vs (0,2) at 0.2477, pure sampling
    # noise, not a real asymmetry. Symmetrizing here (fold each outcome
    # together with its mirror) turns that into an exact invariant rather
    # than an approximate one: it makes win_prob(0,0)==0.5 and
    # win_prob(a,b)+win_prob(b,a)==1 provably exact (see MatchEquityModel),
    # not just approximately true up to sampling noise.
    sym_counts: Dict[Outcome, int] = {}
    for (p0, p1), c in counts.items():
        sym_counts[(p0, p1)] = sym_counts.get((p0, p1), 0) + c
        sym_counts[(p1, p0)] = sym_counts.get((p1, p0), 0) + c
    total = float(sum(sym_counts.values()))
    return {k: v / total for k, v in sym_counts.items()}


def build_equity_table(outcome_dist: Dict[Outcome, float], target: int = 10,
                       max_iterations: int = 500, tol: float = 1e-12,
                       ) -> np.ndarray:
    """Value-iterate E[a, b] = team0's match win probability from score
    (a, b), a, b in [0, target), to a fixed point.

    E is defined by the Bellman equation E[a,b] = sum_outcome
    p(outcome) * E'[a+p0, b+p1], where E' is 1/0 at the (a,b)-exceeds-target
    absorbing boundaries and E otherwise. A misdeal ((0,0), genuinely possible
    without stick-the-dealer) is a real self-loop in this chain; plain
    Jacobi-style value iteration handles it without needing to eliminate it
    analytically -- convergence is geometric at a rate bounded by the
    misdeal probability, which is well under 1, so this converges in well
    under `max_iterations` in practice.
    """
    outcomes = list(outcome_dist.items())
    E = np.full((target, target), 0.5, dtype=np.float64)
    for _ in range(max_iterations):
        newE = np.empty_like(E)
        for a in range(target):
            for b in range(target):
                total = 0.0
                for (p0, p1), prob in outcomes:
                    na, nb = a + p0, b + p1
                    if na >= target:
                        val = 1.0
                    elif nb >= target:
                        val = 0.0
                    else:
                        val = E[na, nb]
                    total += prob * val
                newE[a, b] = total
        delta = float(np.abs(newE - E).max())
        E = newE
        if delta < tol:
            break
    return E


class MatchEquityModel:
    """Wraps an equity table with the operations the training/search pipeline
    needs: converting a hand's raw outcome into a win-probability delta, and
    sampling a realistic starting score for self-play."""

    def __init__(self, table: np.ndarray, outcome_dist: Dict[Outcome, float]):
        assert table.shape[0] == table.shape[1]
        self.table = table
        self.target = table.shape[0]
        self.outcome_dist = dict(outcome_dist)
        self._visit_weights: List[Tuple[Tuple[int, int], float]] | None = None

    # -- core lookups ---------------------------------------------------

    def win_prob(self, team0_score: int, team1_score: int) -> float:
        if team0_score >= self.target:
            return 1.0
        if team1_score >= self.target:
            return 0.0
        return float(self.table[team0_score, team1_score])

    def equity_delta(self, team0_score: int, team1_score: int,
                     p0: int, p1: int) -> float:
        """Team0-signed win-probability delta from one hand's (p0, p1)
        outcome. Team1's own delta is exactly its negation (win
        probabilities are complementary), so this plugs directly into the
        existing `util = [diff if team_of(p) == 0 else -diff ...]` pattern
        used throughout the CFR code, unchanged."""
        before = self.win_prob(team0_score, team1_score)
        after = self.win_prob(team0_score + p0, team1_score + p1)
        return after - before

    # -- realistic score sampling ----------------------------------------

    def _build_visit_weights(self) -> None:
        """Forward visitation mass: V[a,b] = probability that a random match
        (starting 0-0, hand outcomes drawn from outcome_dist) ever has a hand
        that STARTS at score (a,b). Computed exactly via forward propagation
        in increasing a+b order, with the (0,0) misdeal self-loop eliminated
        analytically (dividing by 1 - p_misdeal) rather than iterated, since
        forward order only works for strictly-increasing transitions.
        Cheap: O(target^2 * num_outcomes)."""
        target = self.target
        p_misdeal = self.outcome_dist.get((0, 0), 0.0)
        denom = 1.0 - p_misdeal
        non_misdeal = [(o, p / denom) for o, p in self.outcome_dist.items()
                       if o != (0, 0)]
        V = np.zeros((target, target), dtype=np.float64)
        V[0, 0] = 1.0
        for total in range(2 * target):
            for a in range(max(0, total - target + 1), min(total, target - 1) + 1):
                b = total - a
                if b < 0 or b >= target:
                    continue
                mass = V[a, b]
                if mass <= 0.0:
                    continue
                for (p0, p1), prob in non_misdeal:
                    na, nb = a + p0, b + p1
                    if na >= target or nb >= target:
                        continue  # match ends; no further hand starts here
                    V[na, nb] += mass * prob
        weights = [((a, b), float(V[a, b]))
                  for a in range(target) for b in range(target)
                  if V[a, b] > 0.0]
        self._visit_weights = weights

    def sample_score(self, rng: random.Random) -> Tuple[int, int]:
        """Draw a (team0_score, team1_score) starting context, weighted by
        how often that score actually arises across real matches under this
        model -- so self-play spends training proportionally to how often a
        score situation really occurs, not uniformly over the whole grid."""
        if self._visit_weights is None:
            self._build_visit_weights()
        scores, weights = zip(*self._visit_weights)
        return rng.choices(scores, weights=weights, k=1)[0]

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "target": self.target,
            "outcome_dist": {f"{p0},{p1}": prob
                             for (p0, p1), prob in self.outcome_dist.items()},
            "table": self.table.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> "MatchEquityModel":
        with open(path) as f:
            data = json.load(f)
        outcome_dist = {}
        for k, v in data["outcome_dist"].items():
            p0, p1 = k.split(",")
            outcome_dist[(int(p0), int(p1))] = v
        table = np.array(data["table"], dtype=np.float64)
        return MatchEquityModel(table, outcome_dist)
