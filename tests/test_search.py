"""Tests for the double-dummy solver and the PIMC agent."""

import random

import pytest

from euchre.game import EuchreState, Phase, team_of
from euchre.actions import Play, OrderUp, Discard
from rebel.solver import solve_value, best_play, action_values, _ordered_plays
from rebel.pimc import PIMCAgent, rollout_value
from rebel.evaluate import evaluate, RandomAgent


def _brute(state):
    if state.is_terminal():
        r = state.returns()
        return r[0] - r[1]
    p = state.current_player
    maxi = team_of(p) == 0
    vals = [_brute(state.apply(Play(c))) for c in state._legal_plays(p)]
    return max(vals) if maxi else min(vals)


def _reach_play(seed, max_hand=3):
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


def test_solver_matches_brute_force():
    checked = 0
    for seed in range(120):
        s = _reach_play(seed)
        if s is None:
            continue
        checked += 1
        assert solve_value(s) == _brute(s), f"seed {seed}"
    assert checked > 20


def test_move_reduction_keeps_a_winning_card():
    """Regression: a card in the current trick must break equivalence runs."""
    # Build the exact situation that exposed the bug: following hearts with
    # A and J while the K sits in the trick winning it.
    from euchre.cards import Card, Suit, Rank
    s = EuchreState.new_hand(dealer=3)
    hands = [
        [Card(Suit.HEARTS, Rank.ACE), Card(Suit.HEARTS, Rank.JACK),
         Card(Suit.SPADES, Rank.ACE)],
        [Card(Suit.HEARTS, Rank.TEN), Card(Suit.DIAMONDS, Rank.NINE),
         Card(Suit.CLUBS, Rank.NINE)],
        [Card(Suit.DIAMONDS, Rank.ACE), Card(Suit.CLUBS, Rank.QUEEN),
         Card(Suit.HEARTS, Rank.QUEEN)],
        [Card(Suit.SPADES, Rank.TEN), Card(Suit.DIAMONDS, Rank.TEN),
         Card(Suit.SPADES, Rank.NINE)],
    ]
    s = s.deal_from(hands, Card(Suit.CLUBS, Rank.ACE), [])
    s = s.apply(OrderUp(alone=False))          # trump clubs, maker P0
    s = s.apply(Discard(Card(Suit.CLUBS, Rank.ACE)))
    # Reach the leading state at P1 and lead into the trick.
    # Simpler: just assert both A and J are kept when K is in the trick.
    from euchre.cards import Card as C
    # Construct a mid-trick state directly.
    s2 = s.clone()
    s2.current_trick = [(1, C(Suit.HEARTS, Rank.KING))]
    s2.current_player = 0
    reduced = {c for c in _ordered_plays(s2, 0)}
    assert C(Suit.HEARTS, Rank.ACE) in reduced
    assert C(Suit.HEARTS, Rank.JACK) in reduced


def test_best_play_is_optimal():
    for seed in range(40):
        s = _reach_play(seed, max_hand=3)
        if s is None:
            continue
        bp = best_play(s)
        vals = action_values(s)
        maxi = team_of(s.current_player) == 0
        target = (max if maxi else min)(vals.values())
        assert vals[bp] == target


def test_rollout_value_handles_discard():
    rng = random.Random(4)
    s = EuchreState.new_hand(dealer=0).deal(rng)
    s = s.apply(OrderUp(alone=False))
    assert s.phase == Phase.DEALER_DISCARD
    v = rollout_value(s)
    assert isinstance(v, int)


def test_pimc_plays_legally_and_finishes():
    agent = PIMCAgent(worlds=4, call_worlds=3, seed=1)
    rng = random.Random(2)
    s = EuchreState.new_hand(dealer=0).deal(rng)
    guard = 0
    while not s.is_terminal():
        a = agent.act(s, rng)
        assert a in s.legal_actions()
        s = s.apply(a)
        guard += 1
        assert guard < 60
    assert sum(s.tricks_won) in (0, 5)


def test_strong_pimc_time_budget_plays_legally():
    from rebel.pimc import strong_pimc
    agent = strong_pimc(play_budget=0.2, call_budget=0.2, seed=0)
    s = EuchreState.new_hand(dealer=0).deal(random.Random(3))
    rng = random.Random(0)
    guard = 0
    while not s.is_terminal():
        a = agent.act(s, rng)
        assert a in s.legal_actions()
        s = s.apply(a)
        guard += 1
        assert guard < 60
    assert sum(s.tricks_won) in (0, 5)


def test_pimc_budget_samples_at_least_min_worlds(monkeypatch):
    """A time budget keeps sampling worlds (>= min_worlds), unlike the fixed
    small-count mode."""
    import rebel.pimc as P
    from rebel.pimc import PIMCAgent
    calls = {"n": 0}
    orig = P.sample_determinization

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    monkeypatch.setattr(P, "sample_determinization", counting)

    agent = PIMCAgent(play_budget=0.15, min_worlds=10, max_worlds=400, seed=0)
    s = None
    for seed in range(10):  # skip the occasional all-pass misdeal
        s = EuchreState.new_hand(dealer=0).deal(random.Random(seed))
        while s.phase != Phase.PLAY and not s.is_terminal():
            s = s.apply(agent.act(s, random.Random(0)))
        if s.phase == Phase.PLAY:
            break
    assert s.phase == Phase.PLAY
    calls["n"] = 0
    a = agent.act(s, random.Random(0))
    assert a in s.legal_actions()
    assert calls["n"] >= 10


@pytest.mark.slow
def test_pimc_beats_random():
    stats = evaluate(lambda: PIMCAgent(worlds=6, call_worlds=4),
                     RandomAgent, hands=20, seed=3)
    assert stats["team0_mean_point_diff"] > 0
