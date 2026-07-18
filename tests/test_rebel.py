"""Tests for encoding, networks, and the MCCFR trainer."""

import random

import numpy as np
import pytest

from euchre.game import EuchreState, Phase
from euchre.infoset import infoset_key, observation_tensor, OBS_SIZE
from euchre.actions import Pass, OrderUp

from rebel.mccfr import MCCFRTrainer, Node
from rebel.evaluate import (
    RandomAgent, RuleBasedAgent, MCCFRAgent, evaluate, play_hand,
)


def _dealt(seed=1):
    return EuchreState.new_hand(dealer=0).deal(random.Random(seed))


def test_observation_tensor_shape_and_range():
    s = _dealt()
    obs = observation_tensor(s, s.current_player)
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_infoset_key_hides_opponent_hands():
    """Two deals identical to player 0 but differing in others share a key."""
    p0 = _dealt(1).hands[0]
    # Build two states where player 0 holds the same cards but opponents differ.
    from euchre.cards import DECK
    rng = random.Random(5)
    rest = [c for c in DECK if c not in p0]
    rng.shuffle(rest)
    hands_a = [list(p0), rest[0:5], rest[5:10], rest[10:15]]
    up = rest[15]; kitty_a = rest[16:19]
    sA = EuchreState.new_hand(dealer=0).deal_from(hands_a, up, kitty_a)

    rng.shuffle(rest)  # reshuffle opponents, keep p0 and up-card
    rest2 = [c for c in DECK if c not in p0 and c != up]
    rng.shuffle(rest2)
    hands_b = [list(p0), rest2[0:5], rest2[5:10], rest2[10:15]]
    kitty_b = rest2[15:18]
    sB = EuchreState.new_hand(dealer=0).deal_from(hands_b, up, kitty_b)

    assert infoset_key(sA, 0) == infoset_key(sB, 0)


def test_infoset_key_distinguishes_different_hands():
    sA = _dealt(1)
    sB = _dealt(2)
    # Different deals almost surely give player 0 different hands -> different key
    assert infoset_key(sA, 0) != infoset_key(sB, 0)


def test_node_regret_matching_uniform_when_no_regret():
    node = Node.create([Pass(), OrderUp(False), OrderUp(True)])
    strat = node.strategy()
    assert np.allclose(strat, 1 / 3)


def test_node_regret_matching_prefers_positive_regret():
    node = Node.create([Pass(), OrderUp(False), OrderUp(True)])
    node.regret_sum[1] = 5.0
    node.regret_sum[0] = -2.0
    strat = node.strategy()
    assert strat[1] == pytest.approx(1.0)
    assert strat[0] == pytest.approx(0.0)


def test_mccfr_runs_and_builds_infosets():
    trainer = MCCFRTrainer(seed=0)
    trainer.train(iterations=25)
    assert len(trainer.nodes) > 0
    # Average policy is a valid distribution.
    s = _dealt()
    dist = trainer.average_policy(s)
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in dist.values())


def test_network_forward_and_masked_policy():
    import torch
    from rebel.networks import PolicyValueNet
    from euchre.actions import NUM_ACTIONS

    net = PolicyValueNet()
    s = _dealt()
    obs = torch.tensor(observation_tensor(s, s.current_player)).unsqueeze(0)
    logits, value = net(obs)
    assert logits.shape == (1, NUM_ACTIONS)
    assert value.shape == (1,)

    mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
    from euchre.actions import action_to_index
    for a in s.legal_actions():
        mask[0, action_to_index(a)] = True
    dist = net.policy(obs, mask)
    assert dist.sum().item() == pytest.approx(1.0, abs=1e-5)
    # Illegal actions get zero probability.
    assert dist[~mask].sum().item() == pytest.approx(0.0, abs=1e-6)


def test_evaluate_rulebased_beats_random():
    stats = evaluate(RuleBasedAgent, RandomAgent, hands=300, seed=7)
    assert stats["hands"] == 300
    # A heuristic that calls on strong hands should beat random on average.
    assert stats["team0_mean_point_diff"] > 0
