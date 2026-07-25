"""Tests for match-equity awareness: the dealer-relative equity table
itself, and its integration points in infoset_key, SubgameSolver,
MCCFRTrainer, and rollout_value.

Uses small, self-contained synthetic outcome distributions (not the
generated rebel/match_equity_table.json) so these tests are fast,
deterministic, and don't depend on a precompute step having been run.
"""

import random

import numpy as np
import pytest

from euchre.game import EuchreState, Phase, team_of
from euchre.infoset import infoset_key
from rebel.match_equity import MatchEquityModel, build_equity_table
from rebel.mccfr import MCCFRTrainer, _utility
from rebel.pimc import _rollout_value_raw, rollout_value
from rebel.subgame import SubgameSolver

# A "neutral" distribution: the dealing team has no real edge (dealer/other
# outcome mass is symmetric), so Ed and Eo should come out ~equal -- a good
# check that nothing is spuriously asymmetric when the underlying process
# genuinely isn't.
SYNTH_DIST = {(2, 0): 0.25, (0, 2): 0.25, (1, 0): 0.20, (0, 1): 0.20,
              (0, 0): 0.10}

# A distribution where the dealing team has a real, sizeable edge -- for
# tests that specifically exercise the dealer-vs-non-dealer machinery, which
# a neutral distribution can't distinguish (Ed ~= Eo there by coincidence,
# not because the lookup logic is right).
ASYMMETRIC_DIST = {(1, 0): 0.45, (0, 1): 0.25, (2, 0): 0.15, (0, 2): 0.05,
                   (4, 0): 0.02, (0, 0): 0.08}
TARGET = 10


def _model(dist=SYNTH_DIST):
    table = build_equity_table(dist, target=TARGET)
    return MatchEquityModel(table, dist)


# --- equity table sanity ----------------------------------------------------

def test_equity_zero_zero_neutral_dist_is_close_to_half():
    """With no real dealer edge in the outcome distribution, Ed(0,0) and
    Eo(0,0) should both land close to 0.5 -- not hardcoded (this model no
    longer symmetrizes team labels the way the old one did), just an
    emergent property of a genuinely-symmetric outcome distribution."""
    m = _model(SYNTH_DIST)
    assert m.win_prob(0, 0, True) == pytest.approx(0.5, abs=0.02)
    assert m.win_prob(0, 0, False) == pytest.approx(0.5, abs=0.02)


def test_equity_zero_zero_reflects_a_real_dealer_edge():
    """With a real edge in the outcome distribution, an even score is NOT a
    coin flip once you know who deals next -- the whole reason this model
    tracks dealer identity at all."""
    m = _model(ASYMMETRIC_DIST)
    assert m.win_prob(0, 0, True) > 0.5
    assert m.win_prob(0, 0, False) < 0.5


def test_equity_boundaries():
    m = _model()
    for b in range(TARGET):
        assert m.win_prob(TARGET, b, True) == 1.0
        assert m.win_prob(TARGET, b, False) == 1.0
        assert m.win_prob(0, TARGET, True) == 0.0
        assert m.win_prob(0, TARGET, False) == 0.0


def test_win_prob_dealer_nondealer_identity():
    """win_prob(a, b, True) and win_prob(b, a, False) describe the same real
    situation from the two teams' perspectives (whoever holds score a is
    dealing), so they must sum to exactly 1 -- this is closer to a check on
    the lookup implementation itself (Eo(a,b) == 1 - Ed(b,a) is applied
    directly, not approximated) than a deep numerical property."""
    m = _model(ASYMMETRIC_DIST)
    for a in range(TARGET):
        for b in range(TARGET):
            assert m.win_prob(a, b, True) + m.win_prob(b, a, False) == pytest.approx(1.0, abs=1e-9)
            assert m.win_prob(a, b, False) + m.win_prob(b, a, True) == pytest.approx(1.0, abs=1e-9)


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


# --- equity_delta: the before/after dealer flip ----------------------------

def test_equity_delta_flips_dealer_orientation_between_hands():
    """The deal passes to the OTHER team for the next hand (real Euchre
    rule), so equity_delta's `after` must query the OPPOSITE dealer
    orientation from `before`, not the same one -- constructed with a real
    dealer edge (ASYMMETRIC_DIST) so the two choices provably give different
    numbers, not coincidentally-equal ones."""
    m = _model(ASYMMETRIC_DIST)
    a, b = 3, 2
    before = m.win_prob(a, b, True)
    correct_after = m.win_prob(a + 1, b, False)   # I dealt, opponent deals next
    wrong_after = m.win_prob(a + 1, b, True)       # bug: same orientation both times
    assert correct_after != pytest.approx(wrong_after), (
        "test can't distinguish correct vs. buggy orientation at this (a,b) -- pick another")
    got = m.equity_delta(a, b, True, 1, 0)
    assert got == pytest.approx(correct_after - before)
    assert got != pytest.approx(wrong_after - before)


# --- the behavioral point: risk-shaping actually has teeth ------------------

def test_equity_prefers_safe_option_at_saturating_boundary():
    """The core claim this feature exists for: at a score one point from
    winning, a certain +1 must beat a 50/50 gamble between +2 and +0 --
    RAW point EV ties (1.0 both), but equity strictly prefers the safe
    option, because the gamble's +2 branch overshoots the win condition and
    that extra point is provably wasted (win probability saturates at 1.0
    the instant a team reaches target). This holds for ANY equity table
    that is merely monotonic -- not a property of this specific table --
    and for either dealer orientation, since it's about MY team's own raw
    outcome, not the dealer edge itself."""
    m = _model()
    a, b = TARGET - 1, 0  # one point from winning

    raw_safe_ev = 1.0
    raw_risky_ev = 0.5 * 2 + 0.5 * 0
    assert raw_safe_ev == pytest.approx(raw_risky_ev)  # tied under raw points

    for dealer_is_team0 in (True, False):
        safe_equity = m.equity_delta(a, b, dealer_is_team0, 1, 0)
        risky_equity = (0.5 * m.equity_delta(a, b, dealer_is_team0, 2, 0)
                        + 0.5 * m.equity_delta(a, b, dealer_is_team0, 0, 0))
        assert safe_equity > risky_equity + 1e-9


# --- infoset_key: score must distinguish, team-relative -------------------

def _dealt(seed=1, dealer=0):
    return EuchreState.new_hand(dealer=dealer).deal(random.Random(seed))


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
    s = _dealt(dealer=0)  # team_of(0) == 0 -> dealer_is_team0 True
    s.team0_score, s.team1_score = 9, 0
    s.phase = Phase.TERMINAL
    s.reward = (1, 0)
    m = _model()
    expected = m.equity_delta(9, 0, True, 1, 0)
    assert _utility(s, 0, m) == pytest.approx(expected)
    assert _utility(s, 1, m) == pytest.approx(-expected)


def test_mccfr_utility_uses_equity_with_team1_dealer():
    """Same as above but with team1 dealing -- exercises the branch the
    default dealer=0 fixture never reaches, since _utility derives
    dealer_is_team0 straight from state.dealer, not from a caller-supplied
    flag."""
    s = _dealt(dealer=1)  # team_of(1) == 1 -> dealer_is_team0 False
    s.team0_score, s.team1_score = 9, 0
    s.phase = Phase.TERMINAL
    s.reward = (1, 0)
    m = _model(ASYMMETRIC_DIST)
    expected = m.equity_delta(9, 0, False, 1, 0)
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
    root = _dealt(dealer=0)  # dealer_is_team0 True
    root.team0_score, root.team1_score = 9, 0
    m = _model()
    solver = SubgameSolver(root, root.current_player, num_worlds=1, iterations=1,
                           batch_value_fn=lambda states: [0.0] * len(states),
                           equity_model=m, rng=random.Random(0))
    assert solver.dealer_is_team0 is True
    term = root.clone()
    term.phase = Phase.TERMINAL
    term.reward = (1, 0)
    node = solver._build(term, 0)
    expected = m.equity_delta(9, 0, True, 1, 0)
    assert node.util[0] == pytest.approx(expected)
    assert node.util[1] == pytest.approx(-expected)
    assert node.util[2] == pytest.approx(expected)
    assert node.util[3] == pytest.approx(-expected)


def test_subgame_solver_reads_dealer_from_root():
    """A root dealt by team1 should flip dealer_is_team0 to False, without
    any separate flag -- SubgameSolver derives it straight from root.dealer."""
    root = _dealt(seed=2, dealer=1)  # team_of(1) == 1
    m = _model(ASYMMETRIC_DIST)
    solver = SubgameSolver(root, root.current_player, num_worlds=1, iterations=1,
                           batch_value_fn=lambda states: [0.0] * len(states),
                           equity_model=m, rng=random.Random(0))
    assert solver.dealer_is_team0 is False


def _post_orderup_state(seed, dealer=0):
    """A DEALER_DISCARD state -- always reachable deterministically (round 1's
    OrderUp is legal for the first-to-act on any fresh deal), unlike walking
    toward round 2 which can require several genuine passes."""
    from euchre.actions import OrderUp
    return _dealt(seed, dealer=dealer).apply(OrderUp(alone=False))


def test_rollout_value_unchanged_without_equity_model():
    """Same result whether or not team0_score/team1_score are passed, as
    long as equity_model isn't -- the default path is untouched."""
    state = _post_orderup_state(3)
    raw = _rollout_value_raw(state)
    assert rollout_value(state) == raw
    assert rollout_value(state, team0_score=3, team1_score=1) == raw


def test_rollout_value_converts_with_equity_model():
    state = _post_orderup_state(5, dealer=0)  # dealer_is_team0 True
    raw = _rollout_value_raw(state)
    m = _model()
    got = rollout_value(state, team0_score=9, team1_score=0, equity_model=m)
    p0, p1 = (raw, 0) if raw >= 0 else (0, -raw)
    expected = m.equity_delta(9, 0, True, p0, p1)
    assert got == pytest.approx(expected)


def test_rollout_value_converts_with_team1_dealer():
    """rollout_value derives dealer_is_team0 from state.dealer itself, not a
    caller-supplied flag -- exercise the team1-deals branch explicitly."""
    state = _post_orderup_state(7, dealer=1)  # team_of(1) == 1
    raw = _rollout_value_raw(state)
    m = _model(ASYMMETRIC_DIST)
    got = rollout_value(state, team0_score=9, team1_score=0, equity_model=m)
    p0, p1 = (raw, 0) if raw >= 0 else (0, -raw)
    expected = m.equity_delta(9, 0, False, p0, p1)
    assert got == pytest.approx(expected)
