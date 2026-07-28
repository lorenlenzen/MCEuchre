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
# one remaining documented incompatibility (belief_model, which depends on
# rebel/belief_model.py, itself un-ported) is rejected up front rather than
# failing deep inside self-play. round2_seed_frac and value_ground_frac are
# NOT incompatibilities: _biased_deal never calls rollout_value (only needed
# engine-aware hand/up_card conversion, same as _cluster_key -- see
# test_cpp_engine_biased_deal_works below), and value_ground_frac's cpp path
# uses cpp_rollout_value (a Python-level mirror built on the already-bound
# mceuchre_cpp.solve_value, see rebel/train_rebel.py) instead of
# rebel.pimc.rollout_value -- see test_cpp_engine_value_ground_frac_self_play_runs
# below and the cpp_rollout_value differential tests in
# test_cpp_equivalence.py. -------------

def test_cpp_engine_rejects_unsupported_options():
    with pytest.raises(ValueError):
        ReBeLTrainer(engine="nonsense")
    with pytest.raises(ValueError):
        ReBeLTrainer(engine="cpp", belief_model=object())


def test_cpp_engine_biased_deal_works():
    trainer = ReBeLTrainer(num_worlds=2, cfr_iterations=2, depth_limit=2,
                           engine="cpp", round2_seed_frac=1.0, seed=0)
    state = trainer._biased_deal()
    assert state.phase == trainer._cpp.Phase.BidRound1


def test_biased_deal_weakens_the_hands_versus_a_natural_deal():
    """Regression test for real, measured bugs this session found in
    _biased_deal, in order:

    1. An early version rejected/redealt the WHOLE deck until seat 1 alone
       was weak. By card conservation on a fixed 24-card deck, that doesn't
       just weaken seat 1 -- it systematically concentrates the up-card
       suit's strength onto seats 2-4 instead, which were never supposed to
       be biased. Measured: round2_seed_frac made round 2 LESS reachable,
       worse as the fraction increased (9.8% -> 9.5% -> 8.8% -> 5.5%).
    2. A fix that swapped only seat 1's hand (leaving 2-4 untouched)
       resolved that regression but gave a much weaker, noisier effect,
       since only one of four seats was ever biased.
    3. A generalization to all 4 hands (_weaken_all_hands_for_suit,
       thresholding their SUM) was first written assuming sum <=
       _ROUND2_BIAS_SUM_THRESHOLD (2.0) was a *provable* guarantee (since
       hand_score is non-negative). That claim was falsified by this very
       test: a real draw only reached sum=4.67 against the 2.0 threshold.
       Root cause, found by measurement (not guesswork): the swap loop
       only ever swapped against kitty[0], silently leaving 2 of the
       kitty's 3 cards untouched (fixed -- now all kitty slots are
       candidates), and the up-card itself (never part of any hand's
       score) wasn't being used as a free extra sink for the single
       highest-value same-suit card (fixed -- see
       _weaken_all_hands_for_suit's docstring). Even with both fixes,
       there are up to 7 suit-relevant card values and only 4 total sink
       slots (up-card + 3 kitty), so 2.0 is still not always reachable --
       measured achievable range is roughly mean 3.17, max ~3.9 over 500
       natural deals with max-effort swapping (threshold=0). This is
       therefore a best-effort minimization, not a hard guarantee, and the
       test below checks the properties that actually hold: the swap
       process never makes the sum worse than the natural deal, and stays
       within the measured achievable range (with slack)."""
    trainer = ReBeLTrainer(seed=0, round2_seed_frac=1.0)
    for i in range(20):
        natural = trainer._fresh_deal()
        suit = natural.up_card.suit
        natural_hands = [list(natural.hands[seat]) for seat in range(4)]
        natural_sum = sum(trainer._point_count.hand_score(h, suit) for h in natural_hands)

        new_hands, new_up, new_kitty = trainer._weaken_all_hands_for_suit(
            natural_hands, suit, natural.up_card, natural.kitty,
            trainer._ROUND2_BIAS_SUM_THRESHOLD)
        scores = [trainer._point_count.hand_score(h, suit) for h in new_hands]
        biased_sum = sum(scores)

        assert biased_sum <= natural_sum + 1e-9, (
            f"draw {i}: biased sum {biased_sum} exceeds the SAME deal's natural "
            f"sum {natural_sum} -- swapping should never make things worse")
        assert biased_sum <= 4.5, (
            f"draw {i}: biased sum {biased_sum} far exceeds the measured "
            f"achievable range (mean ~3.2, max ~3.9), scores={scores}")


def test_cpp_engine_biased_deal_matches_python_achievable_range():
    trainer = ReBeLTrainer(engine="cpp", seed=0, round2_seed_frac=1.0)
    from euchre.cards import Card as PyCard
    for i in range(20):
        state = trainer._biased_deal()
        suit = PyCard.from_id(state.up_card).suit
        scores = []
        for seat in range(4):
            hand = [PyCard.from_id(c) for c in range(24) if (state.hands[seat] >> c) & 1]
            scores.append(trainer._point_count.hand_score(hand, suit))
        assert sum(scores) <= 4.5, (
            f"draw {i}: sum {sum(scores)} far exceeds the measured achievable "
            f"range (mean ~3.2, max ~3.9), scores={scores}")


@pytest.mark.slow
def test_cpp_engine_round2_seed_frac_self_play_runs():
    trainer = ReBeLTrainer(num_worlds=2, cfr_iterations=2, depth_limit=2,
                           engine="cpp", round2_seed_frac=1.0, seed=0)
    result = trainer.self_play_hand()
    assert sum(result) in (0, 1, 2, 4)
    assert len(trainer.buffer) > 0


def test_cpp_engine_grounded_value_sample_well_formed():
    from euchre.infoset import OBS_SIZE
    trainer = ReBeLTrainer(engine="cpp", value_ground_frac=1.0, seed=3)
    n_ok = 0
    for _ in range(30):
        s = trainer._grounded_value_sample()
        if s is None:
            continue
        n_ok += 1
        assert s.obs.shape == (OBS_SIZE,)
        assert s.mask.dtype == bool
        assert s.supervise_policy is False
        assert s.cluster_key[0] in ("bid1_ground", "bid2_ground"), s.cluster_key
        assert isinstance(s.cluster_key[1], int)
        assert np.isfinite(s.value)
    assert n_ok > 0, "expected at least one non-None grounded sample in 30 tries"


def test_resolve_dealer_discard_lands_on_fresh_play_state():
    """The retargeting fix: resolve_dealer_discard must return a state at
    the same leaf type SubgameSolver's bidding-rooted solves actually use
    (Phase.PLAY, zero cards played) -- not the pre-discard DEALER_DISCARD
    state -- and its returned value must be self-consistent (an independent
    solve_value on the returned state matches exactly)."""
    from euchre.game import EuchreState, Phase
    from euchre.actions import OrderUp
    from rebel.pimc import resolve_dealer_discard
    from rebel.solver import solve_value

    for seed in range(15):
        rng = random.Random(seed)
        st = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
        dd = st.apply(OrderUp(alone=False))
        assert dd.phase == Phase.DEALER_DISCARD
        nxt, value = resolve_dealer_discard(dd)
        assert nxt.phase == Phase.PLAY
        assert nxt.completed_tricks == []
        assert nxt.current_trick == []
        assert len(nxt.hands[dd.dealer]) == 5
        assert value == solve_value(nxt)


def test_cpp_resolve_dealer_discard_lands_on_fresh_play_state():
    import mceuchre_cpp as cpp
    from rebel.train_rebel import cpp_resolve_dealer_discard

    for seed in range(15):
        rng = random.Random(seed)
        deck = list(range(24))
        rng.shuffle(deck)
        st = cpp.EuchreState.new_hand(dealer=rng.randint(0, 3)).deal_from_deck(deck)
        dd = st.apply(cpp.Action.order_up(False))
        assert dd.phase == cpp.Phase.DealerDiscard
        nxt, value = cpp_resolve_dealer_discard(dd)
        assert nxt.phase == cpp.Phase.Play
        assert list(nxt.completed_tricks) == []
        assert list(nxt.current_trick) == []
        assert bin(nxt.hands[dd.dealer]).count("1") == 5
        assert value == cpp.solve_value(nxt)


def test_cpp_engine_grounded_value_sample_equity_aware():
    """value_ground_frac's cpp path also needs to honor equity_model when
    one is set (mirrors test_cpp_engine_equity_aware_self_play_runs below,
    which covers the CFR-search path but not this separate one)."""
    from rebel.match_equity import MatchEquityModel
    eq = MatchEquityModel.load("rebel/match_equity_table.json")
    trainer = ReBeLTrainer(engine="cpp", value_ground_frac=1.0,
                           equity_model=eq, seed=4)
    assert trainer._cpp_equity_model is not None
    n_ok = 0
    for _ in range(30):
        s = trainer._grounded_value_sample()
        if s is None:
            continue
        n_ok += 1
        # equity units are a win-probability delta, bounded in [-1, 1]
        assert -1.0 <= s.value <= 1.0
    assert n_ok > 0


@pytest.mark.slow
def test_cpp_engine_value_ground_frac_self_play_runs():
    trainer = ReBeLTrainer(num_worlds=2, cfr_iterations=2, depth_limit=2,
                           engine="cpp", value_ground_frac=1.0, seed=5)
    result = trainer.self_play_hand()
    assert sum(result) in (0, 1, 2, 4)
    assert len(trainer.buffer) > 0


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
