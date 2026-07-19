"""Tests for the TMECor (correlated team play) solver.

The coordination game has a known answer and is the crisp demonstration that
the solver captures the value of correlation; the Euchre endgame checks the
always-true inequality TMECor >= Nash on the real game.
"""

import random

import numpy as np
import pytest

from euchre.game import EuchreState, Phase
from rebel.tmecor import (
    tmecor_value, independent_nash_value, solve_zero_sum, collect_infosets,
)
from rebel.team_games import CoordinationGame, EuchreEndgame, sample_endgame_worlds


def test_solve_zero_sum_matching_pennies():
    M = np.array([[1.0, -1.0], [-1.0, 1.0]])
    value, a, b = solve_zero_sum(M, iters=5000)
    assert abs(value) < 0.02
    assert abs(a[0] - 0.5) < 0.05 and abs(b[0] - 0.5) < 0.05


def test_coordination_game_correlation_beats_independent():
    """TMECor must reach the correlated optimum (0) and strictly beat the
    independent-CFR equilibrium (a coordination trap at -1). The analytic best
    *independent* value (TME) is -0.5, so correlation is worth +0.5 over the
    best product strategy and more over what CFR reaches."""
    g = CoordinationGame()
    tv, pures, dist = tmecor_value(g, iters=8000)
    nv = independent_nash_value(g, iters=8000)
    assert abs(tv - 0.0) < 0.05
    assert tv - nv > 0.4          # correlation strictly helps
    assert abs(sum(dist) - 1.0) < 1e-6
    assert len(pures) == len(dist)


def _reach_endgame(seed, max_hand=2):
    rng = random.Random(seed)
    s = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
    while not s.is_terminal() and s.phase != Phase.PLAY:
        s = s.apply(rng.choice(s.legal_actions()))
    while not s.is_terminal() and len(s.hands[s.current_player]) > max_hand:
        s = s.apply(rng.choice(s.legal_actions()))
    return None if (s.is_terminal() or s.phase != Phase.PLAY) else s


@pytest.mark.slow
def test_euchre_endgame_tmecor_at_least_nash():
    """On real Euchre endgames, correlated team play is never worse than the
    independent equilibrium (TMECor >= Nash)."""
    checked = 0
    for seed in range(12):
        s = _reach_endgame(seed, max_hand=2)
        if s is None:
            continue
        worlds = sample_endgame_worlds(s, 2, random.Random(seed))
        g = EuchreEndgame(worlds)
        try:
            tv, _p, _d = tmecor_value(g, cap=20000, iters=1500)
        except ValueError:
            continue  # enumeration too large for this position
        nv = independent_nash_value(g, iters=800)
        assert tv >= nv - 0.05, f"seed {seed}: TMECor {tv} < Nash {nv}"
        checked += 1
        if checked >= 4:
            break
    assert checked >= 2


def test_coordination_infosets_are_unlinked():
    """Each player has exactly one information set (nobody observes others)."""
    info = collect_infosets(CoordinationGame())
    assert {p: len(v) for p, v in info.items()} == {0: 1, 1: 1, 2: 1}
