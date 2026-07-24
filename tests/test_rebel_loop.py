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
        trainer._store(Sample(obs, mask, pol, float(rng.uniform(-2, 2))))
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


# --- engine="cpp" selector (Task 54): not a differential test against the
# Python path (world sampling isn't required to match, see cpp/belief.cpp's
# docstring) -- these confirm the C++ self-play hot path actually runs
# end-to-end and produces well-formed samples/training updates, and that the
# documented incompatibilities (belief_model, round2_seed_frac,
# value_ground_frac all depend on un-ported Python-only features) are
# rejected up front rather than failing deep inside self-play. -------------

def test_cpp_engine_rejects_unsupported_options():
    with pytest.raises(ValueError):
        ReBeLTrainer(engine="nonsense")
    with pytest.raises(ValueError):
        ReBeLTrainer(engine="cpp", round2_seed_frac=0.1)
    with pytest.raises(ValueError):
        ReBeLTrainer(engine="cpp", value_ground_frac=0.1)
    with pytest.raises(ValueError):
        ReBeLTrainer(engine="cpp", belief_model=object())


def test_cpp_engine_value_fn_returns_finite_scalar():
    import mceuchre_cpp as cpp

    trainer = ReBeLTrainer(num_worlds=2, cfr_iterations=2, depth_limit=2, engine="cpp")
    deck = list(range(24))
    random.Random(1).shuffle(deck)
    s = cpp.EuchreState.new_hand(dealer=0).deal_from_deck(deck)
    v = trainer.value_fn(s)
    assert isinstance(v, float) and np.isfinite(v)


@pytest.mark.slow
def test_cpp_engine_self_play_hand_collects_samples_and_finishes():
    trainer = ReBeLTrainer(num_worlds=2, cfr_iterations=2, depth_limit=2,
                           seed=0, engine="cpp")
    result = trainer.self_play_hand()
    assert sum(result) in (0, 1, 2, 4)
    assert len(trainer.buffer) > 0
    s = trainer.buffer[0]
    assert s.obs.shape[0] > 0
    assert abs(s.policy.sum() - 1.0) < 1e-5 or s.policy.sum() == 0.0
    assert bool(s.mask.any())


@pytest.mark.slow
def test_cpp_engine_train_loop_runs_and_updates_weights():
    trainer = ReBeLTrainer(num_worlds=2, cfr_iterations=2, depth_limit=2,
                           seed=0, engine="cpp")
    before = [p.detach().clone() for p in trainer.net.parameters()]
    history = trainer.train(generations=1, hands_per_gen=1, train_steps=3,
                            batch_size=32)
    assert len(history) == 1
    after = list(trainer.net.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


@pytest.mark.slow
def test_cpp_engine_equity_aware_self_play_runs():
    from rebel.match_equity import MatchEquityModel, build_equity_table

    dist = {(2, 0): 0.25, (0, 2): 0.25, (1, 0): 0.20, (0, 1): 0.20, (0, 0): 0.10}
    table = build_equity_table(dist, target=10)
    equity_model = MatchEquityModel(table, dist)
    trainer = ReBeLTrainer(num_worlds=2, cfr_iterations=2, depth_limit=2,
                           seed=0, engine="cpp", equity_model=equity_model)
    assert trainer._cpp_equity_model is not None
    result = trainer.self_play_hand()
    assert sum(result) in (0, 1, 2, 4)
    assert len(trainer.buffer) > 0


@pytest.mark.slow
def test_cpp_engine_full_depth_cards_matches_bitmask_hand_size():
    """Regression test for a real bug: _depth_for used to call
    len(state.hands[player]), which works for Python's list-of-cards hands
    but raised TypeError on cpp's bitmask-int hands (only exercised when
    full_depth_cards > 0, which none of the other cpp-engine tests set).
    Caught by scripts/bench measurement, not by CI -- this closes that gap."""
    trainer = ReBeLTrainer(num_worlds=2, cfr_iterations=2, depth_limit=2,
                           full_depth_cards=2, seed=0, engine="cpp")
    result = trainer.self_play_hand()
    assert sum(result) in (0, 1, 2, 4)
    assert len(trainer.buffer) > 0


@pytest.mark.slow
def test_cpp_net_trains_via_rebel_trainer_and_checkpoint_interops():
    """Task 55's checkpoint-interop requirement: a checkpoint produced by
    training the C++ PolicyValueNet (real LibTorch autograd, driven by the
    ordinary ReBeLTrainer.train_step -- no separate C++ training loop needed,
    since cpp.PolicyValueNet's parameters are genuine leaf tensors in the
    same autograd graph torch.optim already knows how to update) loads
    correctly into the existing pure-Python PolicyValueNet, matching
    forward() output exactly. This is the direction that matters for
    existing tooling (quiz_eval.py, warm_start_value.py, ...): they only
    ever need to torch.load a plain state_dict and hand it to a Python net,
    which state_dict_() (cpp/network.h) produces directly."""
    import mceuchre_cpp as cpp

    cpp_net = cpp.PolicyValueNet()
    trainer = ReBeLTrainer(net=cpp_net, num_worlds=2, cfr_iterations=2,
                           depth_limit=2, seed=0, engine="cpp")
    before = [p.detach().clone() for p in trainer.net.parameters()]
    trainer.self_play_hand()
    trainer.train_step(batch_size=8)
    after = list(trainer.net.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        "cpp.PolicyValueNet weights did not change after train_step")

    py_net = PolicyValueNet()
    py_net.load_state_dict(trainer.net.state_dict_())
    py_net.eval()
    trainer.net.eval()
    from euchre.infoset import OBS_SIZE
    obs = torch.randn(4, OBS_SIZE)
    with torch.no_grad():
        cpp_logits, cpp_value = trainer.net(obs)
        py_logits, py_value = py_net(obs)
    assert torch.equal(cpp_logits, py_logits)
    assert torch.equal(cpp_value, py_value)
