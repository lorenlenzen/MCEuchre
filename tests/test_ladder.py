"""Tests for the Elo evaluation ladder."""

import random

from rebel.ladder import (
    AgentSpec, MatchResult, play_match, round_robin, bradley_terry, _elo,
    evaluate_ladder, format_leaderboard,
)
from rebel.evaluate import RandomAgent, RuleBasedAgent


def test_bradley_terry_recovers_ordering():
    # A beats B beats C in a round robin; strengths must be ordered.
    games = [[0, 10, 10], [10, 0, 10], [10, 10, 0]]
    wins = [0.5 + 8 + 8, 0.5 + 2 + 8, 0.5 + 2 + 2]  # A 8/10, C 2/10
    elos = _elo(bradley_terry(3, wins, games))
    assert elos[0] > elos[1] > elos[2]
    # geometric-mean normalization keeps the average near the 1500 anchor
    assert abs(sum(elos) / 3 - 1500) < 1.0


def test_bradley_terry_equal_strength_equal_elo():
    games = [[0, 10], [10, 0]]
    wins = [0.5 + 5, 0.5 + 5]
    elos = _elo(bradley_terry(2, wins, games))
    assert abs(elos[0] - elos[1]) < 1e-6


def test_self_play_is_fair():
    """An agent against itself must be ~even in margin and Elo (no seat bias)."""
    res = play_match(RandomAgent, RandomAgent, 400, seed=3)
    mean_margin = sum(res.margins) / res.hands
    assert abs(mean_margin) < 0.2
    win_bias = (res.a_wins - res.b_wins) / res.hands
    assert abs(win_bias) < 0.15


def test_match_result_accounting():
    res = play_match(RuleBasedAgent, RandomAgent, 100, seed=1)
    assert res.hands == 100
    assert res.a_wins + res.b_wins + res.draws == 100
    assert len(res.margins) == 100


def test_ladder_ranks_stronger_agent_higher():
    specs = [AgentSpec("random", RandomAgent),
             AgentSpec("rule_based", RuleBasedAgent)]
    standings = evaluate_ladder(specs, hands_per_pair=300, seed=0)
    assert standings[0].name == "rule_based"
    assert standings[1].name == "random"
    assert standings[0].elo > standings[1].elo
    assert standings[0].win_rate > 0.5 > standings[1].win_rate


def test_bootstrap_cis_bracket_elo():
    specs = [AgentSpec("random", RandomAgent),
             AgentSpec("rule_based", RuleBasedAgent)]
    standings = evaluate_ladder(specs, hands_per_pair=200, seed=0,
                                bootstrap=100)
    for st in standings:
        assert st.elo_lo is not None and st.elo_hi is not None
        assert st.elo_lo <= st.elo <= st.elo_hi


def test_format_leaderboard_lists_all_agents():
    specs = [AgentSpec("random", RandomAgent),
             AgentSpec("rule_based", RuleBasedAgent)]
    standings = evaluate_ladder(specs, hands_per_pair=120, seed=0)
    text = format_leaderboard(standings)
    assert "random" in text and "rule_based" in text
    assert "Elo" in text
    assert len(text.splitlines()) == len(specs) + 2  # header + rule + rows
