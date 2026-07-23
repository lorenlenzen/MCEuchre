"""Tests for match-equity awareness: the equity table itself, and its
integration points in infoset_key, SubgameSolver, MCCFRTrainer, and
rollout_value.

Uses a small, self-contained synthetic outcome distribution (not the
generated rebel/match_equity_table.json) so these tests are fast,
deterministic, and don't depend on a precompute step having been run.
"""

import random

import numpy as np
import pytest

from euchre.game import EuchreState, Phase
from euchre.infoset import infoset_key
from rebel.match_equity import MatchEquityModel, build_equity_table
from rebel.mccfr import MCCFRTrainer, _utility
from rebel.pimc import _rollout_value_raw, rollout_value
from rebel.subgame import SubgameSolver

# Already exactly team-symmetric by construction (no fitting/sampling noise
# to correct), unlike the real fit_hand_outcome_distribution() output.
SYNTH_DIST = {(2, 0): 0.25, (0, 2): 0.25, (1, 0): 0.20, (0, 1): 0.20,
              (0, 0): 0.10}
TARGET = 10


def _model():
    table = build_equity_table(SYNTH_DIST, target=TARGET)
    return MatchEquityModel(table, SYNTH_DIST)


# --- equity table sanity ----------------------------------------------------

def test_equity_zero_zero_is_exactly_half():
    m = _model()
    assert m.win_prob(0, 0) == pytest.approx(0.5, abs=1e-9)


def test_equity_boundaries():
    m = _model()
    for b in range(TARGET):
        assert m.win_prob(TARGET, b) == 1.0
        assert m.win_prob(0, TARGET) == 0.0  # symmetric check via b=TARGET


def test_equity_complementary():
    m = _model()
    for a in range(TARGET):
        for b in range(TARGET):
            assert m.win_prob(a, b) + m.win_prob(b, a) == pytest.approx(1.0, abs=1e-9)


def test_equity_monotonic():
    m = _model()
    for a in range(TARGET - 1):
        for b in range(TARGET):
            assert m.table[a + 1, b] >= m.table[a, b] - 1e-12
    for a in range(TARGET):
        for b in range(TARGET - 1):
            assert m.table[a, b + 1] <= m.table[a, b] + 1e-12


def test_sample_score_covers_near_terminal_states():
    m = _model()
    rng = random.Random(0)
    draws = {m.sample_score(rng) for _ in range(5000)}
    assert any(a == TARGET - 1 or b == TARGET - 1 for a, b in draws)


def test_save_load_roundtrip(tmp_path):
    m = _model()
    path = str(tmp_path / "table.json")
    m.save(path)
    m2 = MatchEquityModel.load(path)
    assert np.allclose(m.table, m2.table)
    assert m2.outcome_dist == m.outcome_dist


# --- the behavioral point: risk-shaping actually has teeth ------------------

def test_equity_prefers_safe_option_at_saturating_boundary():
    """The core claim this feature exists for: at a score one point from
    winning, a certain +1 must beat a 50/50 gamble between +2 and +0 --
    RAW point EV ties (1.0 both), but equity strictly prefers the safe
    option, because the gamble's +2 branch overshoots the win condition and
    that extra point is provably wasted (win probability saturates at 1.0
    the instant a team reaches target). This holds for ANY equity table
    that is merely monotonic -- not a property of this specific table."""
    m = _model()
    a, b = TARGET - 1, 0  # one point from winning

    raw_safe_ev = 1.0
    raw_risky_ev = 0.5 * 2 + 0.5 * 0
    assert raw_safe_ev == pytest.approx(raw_risky_ev)  # tied under raw points

    safe_equity = m.equity_delta(a, b, 1, 0)
    risky_equity = 0.5 * m.equity_delta(a, b, 2, 0) + 0.5 * m.equity_delta(a, b, 0, 0)
    assert safe_equity > risky_equity + 1e-9


# --- infoset_key: score must distinguish, team-relative -------------------

def _dealt(seed=1):
    return EuchreState.new_hand(dealer=0).deal(random.Random(seed))


def test_infoset_key_distinguishes_score():
    s = _dealt()
    s_scored = s.clone()
    s_scored.team0_score, s_scored.team1_score = 3, 1
    p = s.current_player
    assert infoset_key(s, p) != infoset_key(s_scored, p)


def test_infoset_key_score_is_team_relative():
    s = _dealt()
    s.team0_score, s.team1_score = 5, 3
    # player 0/2 are team0 (mine=5,their=3); player 1/3 are team1 (mine=3,their=5)
    assert "sc5,3" in infoset_key(s, 0)
    assert "sc3,5" in infoset_key(s, 1)


# --- backward compatibility: equity_model=None is byte-identical ----------

def test_mccfr_utility_unchanged_without_equity_model():
    s = _dealt()
    s.phase = Phase.TERMINAL
    s.reward = (2, 0)
    assert _utility(s, 0, None) == 2.0
    assert _utility(s, 1, None) == -2.0


def test_mccfr_utility_uses_equity_when_set():
    s = _dealt()
    s.team0_score, s.team1_score = 9, 0
    s.phase = Phase.TERMINAL
    s.reward = (1, 0)
    m = _model()
    expected = m.equity_delta(9, 0, 1, 0)
    assert _utility(s, 0, m) == pytest.approx(expected)
    assert _utility(s, 1, m) == pytest.approx(-expected)


def test_mccfr_iterate_samples_score_when_equity_model_set():
    """End-to-end: iterate() should deal hands at a variety of sampled
    scores (not always the 0-0 default) when equity_model is set, and stay
    at 0-0 when it isn't -- confirmed indirectly via infoset_key's score
    component appearing in the node table."""
    t_none = MCCFRTrainer(seed=0)
    t_none.iterate()
    assert all("sc0,0" in k for k in t_none.nodes)

    m = _model()
    t_eq = MCCFRTrainer(seed=0, equity_model=m)
    for _ in range(5):  # several draws so "all landed on 0-0" is negligible
        t_eq.iterate()
    assert any("sc0,0" not in k for k in t_eq.nodes)


def test_subgame_solver_terminal_branch_unchanged_without_equity_model():
    root = _dealt()
    solver = SubgameSolver(root, root.current_player, num_worlds=1, iterations=1,
                           batch_value_fn=lambda states: [0.0] * len(states),
                           rng=random.Random(0))
    term = root.clone()
    term.phase = Phase.TERMINAL
    term.reward = (2, 0)
    node = solver._build(term, 0)
    assert node.util == [2.0, -2.0, 2.0, -2.0]


def test_subgame_solver_terminal_branch_uses_equity_when_set():
    root = _dealt()
    root.team0_score, root.team1_score = 9, 0
    m = _model()
    solver = SubgameSolver(root, root.current_player, num_worlds=1, iterations=1,
                           batch_value_fn=lambda states: [0.0] * len(states),
                           equity_model=m, rng=random.Random(0))
    term = root.clone()
    term.phase = Phase.TERMINAL
    term.reward = (1, 0)
    node = solver._build(term, 0)
    expected = m.equity_delta(9, 0, 1, 0)
    assert node.util[0] == pytest.approx(expected)
    assert node.util[1] == pytest.approx(-expected)
    assert node.util[2] == pytest.approx(expected)
    assert node.util[3] == pytest.approx(-expected)


def _post_orderup_state(seed):
    """A DEALER_DISCARD state -- always reachable deterministically (round 1's
    OrderUp is legal for the first-to-act on any fresh deal), unlike walking
    toward round 2 which can require several genuine passes."""
    from euchre.actions import OrderUp
    return _dealt(seed).apply(OrderUp(alone=False))


def test_rollout_value_unchanged_without_equity_model():
    """Same result whether or not team0_score/team1_score are passed, as
    long as equity_model isn't -- the default path is untouched."""
    state = _post_orderup_state(3)
    raw = _rollout_value_raw(state)
    assert rollout_value(state) == raw
    assert rollout_value(state, team0_score=3, team1_score=1) == raw


def test_rollout_value_converts_with_equity_model():
    state = _post_orderup_state(5)
    raw = _rollout_value_raw(state)
    m = _model()
    got = rollout_value(state, team0_score=9, team1_score=0, equity_model=m)
    p0, p1 = (raw, 0) if raw >= 0 else (0, -raw)
    expected = m.equity_delta(9, 0, p0, p1)
    assert got == pytest.approx(expected)
