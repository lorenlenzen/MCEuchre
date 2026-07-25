"""Match equity: converting a hand's raw point differential into a
race-to-target win-probability delta, conditioned on the current match score
AND on which team deals the next hand.

Real Euchre strategy is score-dependent (analogous to backgammon match equity
or poker tournament ICM) -- a team one point from winning should strongly
prefer a safe sure-thing over a high-variance loner attempt with equal or
better *raw* expected points. It is also dealer-dependent: the dealer sees
the up-card and decides whether to pick it up with strictly more information
than anyone else at the table, a real, measurable edge, and the deal passes
to the other team every hand (teams sit in alternating seats), so knowing
who deals next is part of knowing how good a given score really is.

Two pieces:

* :func:`fit_hand_outcome_distribution` / :func:`build_equity_table` --
  empirically measure how often each single-hand outcome occurs, keyed by
  (dealing team's points, other team's points) rather than by an arbitrary
  team label (fit against a heuristic agent, not the eventual trained net --
  an approximation, cheap to refit later against a stronger agent), then
  value-iterate a table Ed(a, b) = the probability that the team about to
  deal wins the match, given their own score a and the opponent's score b.
* :class:`MatchEquityModel` -- wraps the table with the operations the rest
  of the pipeline needs: converting a hand's raw (p0, p1) point outcome into
  a win-probability delta (correctly handling the deal passing to the other
  team between hands), and sampling a realistic starting score for
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

Dealer-relative table, not team0/team1-relative (see the session's design
discussion): only ONE (target x target) table is stored -- Ed(a,b), the
DEALING team's win probability. The non-dealing team's win probability at
the same (a,b) is not a separate quantity needing its own table; it is the
exact identity Eo(a,b) == 1 - Ed(b,a) (swap whose score is whose, negate,
since exactly one team wins). MatchEquityModel.win_prob applies that
identity directly rather than storing a second, redundant array. The one
place this asymmetry needs real care is `equity_delta`: the deal passes to
the OTHER team for the next hand, so its `before`/`after` win_prob calls
deliberately use opposite dealer orientations, not the same one.
"""

from __future__ import annotations

import json
import random
from typing import Callable, Dict, List, Tuple

import numpy as np

from .evaluate import RuleBasedAgent, play_hand

Outcome = Tuple[int, int]  # (dealing_team_points, other_team_points) from one hand


def fit_hand_outcome_distribution(
        hands: int = 20000, seed: int = 0,
        agent_factory: Callable[[], object] = RuleBasedAgent,
        ) -> Dict[Outcome, float]:
    """Empirical frequency of each single-hand (dealing team's points, other
    team's points) outcome, via the same `play_hand` harness `evaluate()`
    uses. Deliberately NOT symmetrized across dealer/non-dealer -- unlike the
    old team0/team1 label (nothing structurally distinguished those, so
    symmetrizing across them just removed sampling noise), dealer vs
    non-dealer is a real asymmetry this distribution exists to capture.
    Approximate by construction (fit against a heuristic agent, not the
    eventual trained net); refitting later against a stronger agent is just
    rerunning this, not required now.
    """
    from euchre.game import team_of

    rng = random.Random(seed)
    a0, a1 = agent_factory(), agent_factory()
    agents = [a0, a1, a0, a1]
    counts: Dict[Outcome, int] = {}
    for h in range(hands):
        dealer = h % 4
        r0, r1 = play_hand(agents, dealer=dealer, rng=rng)
        key = (r0, r1) if team_of(dealer) == 0 else (r1, r0)
        counts[key] = counts.get(key, 0) + 1
    total = float(sum(counts.values()))
    return {k: v / total for k, v in counts.items()}


def build_equity_table(outcome_dist: Dict[Outcome, float], target: int = 10,
                       max_iterations: int = 500, tol: float = 1e-12,
                       ) -> np.ndarray:
    """Value-iterate Ed[a, b] = the probability that the team about to deal
    wins the match, given their own score a and the opponent's score b, to a
    fixed point.

    The deal passes to the other team every hand (real Euchre rule -- teams
    sit in alternating seats), so this is a coupled recursion with Eo[a,b]
    (the NON-dealing team's win probability at the same (a,b)): after I
    deal, I'm non-dealing next hand (look up Eo); after the opponent deals,
    I deal next hand (look up Ed). Both arrays are iterated together
    (Jacobi-style) to a shared fixed point; only Ed is returned. Eo is not
    persisted -- it is exactly recoverable as Eo[a,b] == 1 - Ed[b,a], an
    identity of the underlying symmetric process (see module docstring), not
    an approximation, so storing it separately would be redundant.

    outcome_dist is (dealing team's points, other team's points) -- see
    fit_hand_outcome_distribution. A misdeal ((0,0), genuinely possible
    without stick-the-dealer) is a real self-loop in this chain; plain
    Jacobi-style value iteration handles it without needing to eliminate it
    analytically.
    """
    outcomes = list(outcome_dist.items())
    Ed = np.full((target, target), 0.5, dtype=np.float64)
    Eo = np.full((target, target), 0.5, dtype=np.float64)
    for _ in range(max_iterations):
        newEd = np.empty_like(Ed)
        newEo = np.empty_like(Eo)
        for a in range(target):
            for b in range(target):
                d_total = 0.0
                o_total = 0.0
                for (pd, po), prob in outcomes:
                    # Ed[a,b]: my team deals this hand -- my points pd,
                    # opponent's points po. Afterward I'm non-dealing.
                    na, nb = a + pd, b + po
                    if na >= target:
                        d_val = 1.0
                    elif nb >= target:
                        d_val = 0.0
                    else:
                        d_val = Eo[na, nb]
                    d_total += prob * d_val

                    # Eo[a,b]: the OPPONENT deals this hand, so the
                    # distribution's "dealer" slot is their points (added to
                    # b, not a) and "other" is mine (added to a). Afterward
                    # I'm dealing.
                    na2, nb2 = a + po, b + pd
                    if na2 >= target:
                        o_val = 1.0
                    elif nb2 >= target:
                        o_val = 0.0
                    else:
                        o_val = Ed[na2, nb2]
                    o_total += prob * o_val
                newEd[a, b] = d_total
                newEo[a, b] = o_total
        delta = max(float(np.abs(newEd - Ed).max()), float(np.abs(newEo - Eo).max()))
        Ed, Eo = newEd, newEo
        if delta < tol:
            break
    return Ed


class MatchEquityModel:
    """Wraps a dealer-relative equity table with the operations the
    training/search pipeline needs: converting a hand's raw outcome into a
    win-probability delta (correctly handling the deal passing to the other
    team between hands), and sampling a realistic starting score for
    self-play."""

    def __init__(self, table: np.ndarray, outcome_dist: Dict[Outcome, float]):
        assert table.shape[0] == table.shape[1]
        self.table = table  # Ed[a, b]: the DEALING team's win probability
        self.target = table.shape[0]
        self.outcome_dist = dict(outcome_dist)
        self._visit_weights: List[Tuple[Tuple[int, int], float]] | None = None

    # -- core lookups ---------------------------------------------------

    def win_prob(self, my_score: int, opp_score: int, am_i_dealer: bool) -> float:
        """My team's win probability, given my score, the opponent's score,
        and whether my team deals the upcoming hand. The non-dealing case is
        the exact identity 1 - Ed[opp_score, my_score] (see module
        docstring), not a separately stored table."""
        if am_i_dealer:
            if my_score >= self.target:
                return 1.0
            if opp_score >= self.target:
                return 0.0
            return float(self.table[my_score, opp_score])
        if opp_score >= self.target:
            return 0.0
        if my_score >= self.target:
            return 1.0
        return 1.0 - float(self.table[opp_score, my_score])

    def equity_delta(self, team0_score: int, team1_score: int,
                     dealer_is_team0: bool, p0: int, p1: int) -> float:
        """Team0-signed win-probability delta from one hand's (p0, p1)
        outcome. The deal passes to the OTHER team for the next hand (real
        Euchre rule), so `before` and `after` deliberately query opposite
        dealer orientations -- getting this backwards is the one place this
        model is easy to get subtly wrong (see tests/test_match_equity.py).
        Team1's own delta is exactly this value's negation (win
        probabilities are complementary), so this plugs directly into the
        existing `util = [diff if team_of(p) == 0 else -diff ...]` pattern
        used throughout the CFR code, unchanged."""
        before = self.win_prob(team0_score, team1_score, dealer_is_team0)
        after = self.win_prob(team0_score + p0, team1_score + p1, not dealer_is_team0)
        return after - before

    # -- realistic score sampling ----------------------------------------

    def _build_visit_weights(self) -> None:
        """Forward visitation mass: V[a,b] = probability that a random match
        (starting 0-0, hand outcomes drawn from outcome_dist) ever has a hand
        that STARTS at score (a,b), viewed from whichever team is about to
        deal that hand -- outcome_dist is already dealer-relative, so this
        stays a single (target x target) table exactly as before; it does
        not need its own dealer axis. Computed exactly via forward
        propagation in increasing a+b order, with the (0,0) misdeal
        self-loop eliminated analytically (dividing by 1 - p_misdeal) rather
        than iterated, since forward order only works for strictly-
        increasing transitions. Cheap: O(target^2 * num_outcomes)."""
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
        """Draw a (dealing team's score, other team's score) starting
        context, weighted by how often that score actually arises across
        real matches under this model. Callers assign the drawn pair to
        team0/team1 using whichever team they already know deals the
        sampled hand (e.g. ReBeLTrainer._fresh_deal already picks a dealer
        seat) -- this does not need to sample a dealer itself, only a score,
        exactly as before."""
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
