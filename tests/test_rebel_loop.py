"""Tests for the ReBeL self-play training loop."""

import random

import numpy as np
import pytest
import torch

from euchre.game import EuchreState
from euchre.actions import NUM_ACTIONS
from rebel.train_rebel import (
    ReBeLTrainer, ReBeLNetAgent, legal_mask, Sample,
)
from rebel.networks import PolicyValueNet


def test_legal_mask_matches_legal_actions():
    from euchre.actions import action_to_index
    s = EuchreState.new_hand(dealer=0).deal(random.Random(0))
    mask = legal_mask(s)
    assert mask.dtype == bool and mask.shape == (NUM_ACTIONS,)
    assert mask.sum() == len(s.legal_actions())
    for a in s.legal_actions():
        assert mask[action_to_index(a)]


def test_value_fn_returns_finite_scalar():
    trainer = ReBeLTrainer(num_worlds=2, cfr_iterations=2, depth_limit=2)
    s = EuchreState.new_hand(dealer=0).deal(random.Random(1))
    v = trainer.value_fn(s)
    assert isinstance(v, float) and np.isfinite(v)


@pytest.mark.slow
def test_self_play_hand_collects_samples_and_finishes():
    trainer = ReBeLTrainer(num_worlds=2, cfr_iterations=2, depth_limit=2, seed=0)
    result = trainer.self_play_hand()
    assert sum(result) in (0, 1, 2, 4)
    assert len(trainer.buffer) > 0
    s = trainer.buffer[0]
    assert s.obs.shape[0] > 0
    assert abs(s.policy.sum() - 1.0) < 1e-5 or s.policy.sum() == 0.0
    assert bool(s.mask.any())


def test_train_step_reduces_loss_on_a_fixed_batch():
    """Overfit a tiny synthetic batch: the loss must go down."""
    torch.manual_seed(0)
    trainer = ReBeLTrainer()
    # Build a small fixed set of samples with a definite best action.
    from euchre.infoset import OBS_SIZE
    rng = np.random.default_rng(0)
    for _ in range(16):
        obs = rng.random(OBS_SIZE).astype(np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        legal = rng.choice(NUM_ACTIONS, size=3, replace=False)
        mask[legal] = True
        pol = np.zeros(NUM_ACTIONS, dtype=np.float32)
        pol[legal[0]] = 1.0  # deterministic target
        trainer.buffer.append(Sample(obs, mask, pol, float(rng.uniform(-2, 2))))
    first = trainer.train_step(batch_size=16)
    for _ in range(60):
        last = trainer.train_step(batch_size=16)
    assert last["policy_loss"] < first["policy_loss"]
    assert last["value_loss"] < first["value_loss"]


@pytest.mark.slow
def test_train_loop_runs_and_updates_weights():
    trainer = ReBeLTrainer(num_worlds=2, cfr_iterations=2, depth_limit=2, seed=0)
    before = [p.detach().clone() for p in trainer.net.parameters()]
    history = trainer.train(generations=1, hands_per_gen=1, train_steps=3,
                            batch_size=32)
    assert len(history) == 1
    after = list(trainer.net.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_net_agent_plays_legally():
    net = PolicyValueNet()
    agent = ReBeLNetAgent(net)
    s = EuchreState.new_hand(dealer=0).deal(random.Random(2))
    rng = random.Random(0)
    steps = 0
    while not s.is_terminal():
        a = agent.act(s, rng)
        assert a in s.legal_actions()
        s = s.apply(a)
        steps += 1
        assert steps < 60
