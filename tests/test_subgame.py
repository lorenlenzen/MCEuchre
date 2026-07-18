"""Tests for the depth-limited CFR subgame solver."""

import random

import pytest

from euchre.game import EuchreState, Phase, team_of
from rebel.subgame import SubgameSolver, CFRSearchAgent
from rebel.solver import solve_value


def _reach_play(seed, max_hand=5):
    rng = random.Random(seed)
    s = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
    g = 0
    while not s.is_terminal() and s.phase != Phase.PLAY:
        s = s.apply(rng.choice(s.legal_actions()))
        g += 1
        if g > 20:
            break
    if s.is_terminal() or s.phase != Phase.PLAY:
        return None
    while not s.is_terminal() and len(s.hands[s.current_player]) > max_hand:
        s = s.apply(rng.choice(s.legal_actions()))
    return None if s.is_terminal() else s


def test_shallow_solve_returns_valid_distribution():
    s = _reach_play(1, max_hand=5)
    solver = SubgameSolver(s, s.current_player, num_worlds=4, iterations=10,
                           depth_limit=4, value_fn=lambda st: 0.0,
                           rng=random.Random(0))
    solver.run()
    pol = solver.root_policy()
    assert abs(sum(pol.values()) - 1.0) < 1e-6
    assert all(p >= 0 for p in pol.values())
    assert set(pol) == set(s.legal_actions())


def test_single_world_cfr_matches_double_dummy():
    """With the true world as the only belief, full-depth CFR should pick a
    double-dummy *optimal* action (there may be several equally-good ones, so
    we check the achieved value, not a specific card)."""
    optimal = 0
    trials = 0
    for seed in range(60):
        s = _reach_play(seed, max_hand=2)  # tiny position: full CFR is cheap
        if s is None:
            continue
        trials += 1
        solver = SubgameSolver(s, s.current_player, num_worlds=1,
                               iterations=300, rng=random.Random(0))
        solver.worlds = [s]  # belief = the true world only
        solver.run()
        pol = solver.root_policy()
        chosen = max(pol, key=lambda a: pol[a])
        # An action is optimal iff taking it preserves the game value.
        if solve_value(s.apply(chosen)) == solve_value(s):
            optimal += 1
    assert trials > 10
    assert optimal >= 0.9 * trials


def test_cfr_search_agent_plays_legally():
    agent = CFRSearchAgent(num_worlds=4, iterations=8, depth_limit=4,
                           value_fn=lambda st: 0.0, seed=0)
    s = _reach_play(3, max_hand=4)
    rng = random.Random(0)
    for _ in range(3):
        if s.is_terminal():
            break
        a = agent.act(s, rng)
        assert a in s.legal_actions()
        s = s.apply(a)
