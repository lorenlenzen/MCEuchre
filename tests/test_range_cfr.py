"""Correctness tests for the experimental range-based public-tree CFR.

These lock in that the solver is *correct* (even though it is not a speedup for
sampled beliefs -- see the module docstring and docs/rebel_design.md).
"""

import random

import pytest

import rebel.range_cfr as R
from euchre.game import EuchreState, Phase
from rebel.range_cfr import RangeCFRSolver
from rebel.subgame import SubgameSolver
from rebel.solver import solve_value


def _reach_play(seed, max_hand=3):
    rng = random.Random(seed)
    s = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
    while not s.is_terminal() and s.phase != Phase.PLAY:
        s = s.apply(rng.choice(s.legal_actions()))
    while not s.is_terminal() and len(s.hands[s.current_player]) > max_hand:
        s = s.apply(rng.choice(s.legal_actions()))
    return s


def test_range_cfr_single_deal_matches_double_dummy(monkeypatch):
    """With one deal (the true world) it must pick a double-dummy optimum."""
    optimal = 0
    trials = 0
    for seed in range(16):
        s = _reach_play(seed, max_hand=2)
        if s.phase != Phase.PLAY:
            continue
        trials += 1
        monkeypatch.setattr(R, "sample_determinization",
                            lambda root, actor, rng, _s=s: _s.clone())
        solver = RangeCFRSolver(s, s.current_player, num_deals=1,
                                iterations=100, rng=random.Random(0))
        solver.run()
        chosen = max(solver.root_policy(), key=lambda a: solver.root_policy()[a])
        if solve_value(s.apply(chosen)) == solve_value(s):
            optimal += 1
    assert trials > 10
    assert optimal >= 0.95 * trials


@pytest.mark.slow
def test_range_cfr_root_value_matches_scalar_on_same_belief():
    """Same belief, same game: the two solvers agree on the root value."""
    worst = 0.0
    for seed in range(3):
        s = _reach_play(seed, max_hand=2)
        if s.phase != Phase.PLAY:
            continue
        M, N = 12, 100
        sc = SubgameSolver(s, s.current_player, num_worlds=M, iterations=N,
                           rng=random.Random(7))
        sc.run()
        rg = RangeCFRSolver(s, s.current_player, num_deals=M, iterations=N,
                            rng=random.Random(7))
        rg.run()
        worst = max(worst, abs(sc.root_value() - rg.root_value()))
    assert worst < 0.15, f"root-value mismatch {worst}"


def test_range_cfr_policy_is_valid_distribution():
    s = _reach_play(2, max_hand=3)
    solver = RangeCFRSolver(s, s.current_player, num_deals=15, iterations=20,
                            depth_limit=4,
                            batch_value_fn=lambda states: [0.0] * len(states),
                            rng=random.Random(0))
    solver.run()
    pol = solver.root_policy()
    assert abs(sum(pol.values()) - 1.0) < 1e-6
    assert all(p >= 0 for p in pol.values())
    assert set(pol) == set(s.legal_actions())
