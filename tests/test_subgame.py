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


# --- subgame boundary = phase boundary (bidding-rooted solves) -------------
# A bidding-rooted solve (BID_ROUND_1/2, DEALER_DISCARD) must expand the
# WHOLE auction regardless of depth_limit (it's short/bounded on its own --
# see _build's comment), and cut the instant a child transitions into PLAY,
# never recursing into real card play inside the same solve. These are the
# correctness gate for that behavior; the throughput side was measured
# separately (see session notes) against the pre-fix version, which instead
# kept recursing into PLAY for every distinct bidding-resolution path and
# was ~53x slower at production settings.

def _fresh_bid1(seed):
    rng = random.Random(seed)
    s = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
    assert s.phase == Phase.BID_ROUND_1
    return s


@pytest.mark.parametrize("seed", range(10))
def test_bidding_subgame_leaves_are_exactly_at_phase_boundary(seed):
    """Every deferred (batched) leaf produced while solving a bidding-rooted
    decision must be a state whose phase just became PLAY -- not deeper into
    a trick, and not still bidding (those aren't leaves, they're expanded)."""
    s = _fresh_bid1(seed)
    seen_states = []

    def batch_fn(states):
        seen_states.extend(states)
        return [0.0] * len(states)

    solver = SubgameSolver(s, s.current_player, num_worlds=3, iterations=2,
                           depth_limit=1, batch_value_fn=batch_fn,
                           rng=random.Random(0))
    solver.run()
    assert len(seen_states) > 0, "expected at least one phase-boundary leaf"
    for st in seen_states:
        assert st.phase == Phase.PLAY
        # A freshly-entered PLAY state has all four hands at their original
        # size minus whatever the dealer discarded down to -- specifically,
        # nothing has been played yet: no completed tricks, empty current
        # trick.
        assert st.completed_tricks == []
        assert st.current_trick == []


@pytest.mark.parametrize("seed", range(6))
def test_bidding_subgame_expands_full_auction_regardless_of_depth_limit(seed):
    """depth_limit=1 used to cut a bidding-rooted solve down to almost
    nothing (old flat ply-count). Now the whole auction should still be
    explored -- multiple players' bidding decisions show up as distinct
    infosets, not just the root's."""
    s = _fresh_bid1(seed)
    solver = SubgameSolver(s, s.current_player, num_worlds=4, iterations=2,
                           depth_limit=1, value_fn=lambda st: 0.0,
                           rng=random.Random(0))
    solver.run()
    # More than just the root's own infoset -- at least the next player's
    # decision (reached via Pass) must have been expanded too.
    assert len(solver.infosets) > 1


def test_play_rooted_solve_identical_to_before_the_change():
    """A solve rooted already inside PLAY never touches the new bidding-leaf
    logic (is_free is False from the start) -- confirm its root
    value/policy are deterministic and sane, matching ordinary depth-limited
    behavior, not accidentally affected by the bidding-rooted code path."""
    s = _reach_play(2, max_hand=4)
    solver = SubgameSolver(s, s.current_player, num_worlds=4, iterations=10,
                           depth_limit=3, value_fn=lambda st: 0.0,
                           rng=random.Random(0))
    solver.run()
    pol = solver.root_policy()
    assert abs(sum(pol.values()) - 1.0) < 1e-6
    assert set(pol) == set(s.legal_actions())
    # No bidding phases should ever be visited from a PLAY root.
    for st in [w for w in solver.worlds]:
        assert st.phase == Phase.PLAY


# --- DEALER_DISCARD: free only as an internal node, not as the solve root --
# Regression tests for a real bug the phase-boundary fix introduced: treating
# DEALER_DISCARD as always-free (cut immediately at the PLAY boundary) meant
# a discard-rooted solve was nothing but the root plus 6 same-depth leaves --
# no real search -- so regret-matching over them degenerated to comparing
# unbacked value-net guesses and produced an exactly-uniform policy
# regardless of net quality (measured: exactly 1 infoset, confirmed against a
# fully-trained checkpoint too, not just a fresh one).

def _fresh_dealer_discard(seed):
    from euchre.actions import OrderUp
    rng = random.Random(seed)
    s = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
    s = s.apply(OrderUp(alone=False))
    assert s.phase == Phase.DEALER_DISCARD
    return s


@pytest.mark.parametrize("seed", range(8))
def test_dealer_discard_rooted_solve_gets_real_search_depth(seed):
    s = _fresh_dealer_discard(seed)
    solver = SubgameSolver(s, s.current_player, num_worlds=4, iterations=10,
                           depth_limit=6, value_fn=lambda st: 0.0,
                           rng=random.Random(0))
    solver.run()
    # Pre-fix this was exactly 1 (root only, 6 same-depth unsearched leaves).
    assert len(solver.infosets) > 1, (
        f"seed={seed}: discard-rooted solve built only "
        f"{len(solver.infosets)} infoset(s) -- no real search depth")
    pol = solver.root_policy()
    assert abs(sum(pol.values()) - 1.0) < 1e-6
    assert set(pol) == set(s.legal_actions())


def test_dealer_discard_root_policy_is_differentiated_with_trained_net():
    """The structural check above (infoset count) doesn't prove the policy
    is actually useful -- confirm a trained net's discard root_policy isn't
    (numerically indistinguishable from) uniform, the symptom that surfaced
    this bug via the quiz."""
    import os
    net_path = "checkpoints/rebel_sa.pt"
    if not os.path.exists(net_path):
        pytest.skip(f"{net_path} not present in this checkout")
    import torch
    from rebel.networks import PolicyValueNet
    from rebel.train_rebel import batch_value_fn_from_net

    net = PolicyValueNet()
    net.load_state_dict(torch.load(net_path, map_location="cpu"))
    value_fn = batch_value_fn_from_net(net)

    s = _fresh_dealer_discard(0)
    solver = SubgameSolver(s, s.current_player, num_worlds=8, iterations=15,
                           depth_limit=6, batch_value_fn=value_fn,
                           rng=random.Random(0))
    solver.run()
    pol = solver.root_policy()
    n = len(pol)
    uniform = 1.0 / n
    # A collapsed/degenerate solve gives every action within float noise of
    # uniform; a real search should show a meaningfully preferred action.
    assert max(pol.values()) > uniform + 0.05, (
        f"root_policy looks uniform (max={max(pol.values()):.4f}, "
        f"uniform={uniform:.4f}) -- discard search may have collapsed again")


def test_dealer_discard_internal_to_bidding_solve_stays_free():
    """The fix must not reintroduce the blowup the phase-boundary change
    itself fixed: DEALER_DISCARD reached INSIDE a BID_ROUND_1-rooted solve
    (exploring the hypothetical "what if I order up" branch) must stay a
    cheap single-leaf estimate, not ply-counted real search -- confirmed by
    the same small-infoset-count signature test_subgame.py's other bidding
    tests already rely on."""
    rng = random.Random(0)
    s = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
    assert s.phase == Phase.BID_ROUND_1
    solver = SubgameSolver(s, s.current_player, num_worlds=4, iterations=5,
                           depth_limit=1, value_fn=lambda st: 0.0,
                           rng=random.Random(0))
    solver.run()
    # Small and bounded, matching the ~10-30 infoset range measured for a
    # free-auction solve -- nowhere near the thousands a real card-play
    # expansion through DEALER_DISCARD would produce.
    assert len(solver.infosets) < 60, (
        f"bidding-rooted solve built {len(solver.infosets)} infosets -- "
        f"DEALER_DISCARD may no longer be free as an internal node")


# --- BID_ROUND_2: same structural gap as DEALER_DISCARD, for Call --------
# Round 2's Call skips DEALER_DISCARD entirely and goes straight to PLAY
# (euchre/game.py's _apply_bid2 calls _begin_play() directly) -- unlike
# round 1's OrderUp, which always passes through a discard sub-tree first.
# So a BID_ROUND_2-rooted solve's own Call/Call-alone/suit comparison had
# the identical "root's own action goes straight to an unbacked leaf" gap
# DEALER_DISCARD did, just less visible (the tree isn't literally 1 infoset,
# since Pass still expands the rest of the auction) -- any value-head bias
# between alone/not-alone trained straight into the policy uncorrected,
# surfacing as "every alone option outranks its same-suit non-alone twin"
# on the quiz.

def _fresh_bid_round2(seed):
    from euchre.actions import Pass
    rng = random.Random(seed)
    s = EuchreState.new_hand(dealer=rng.randint(0, 3), stick_the_dealer=True).deal(rng)
    for _ in range(4):
        s = s.apply(Pass())
    assert s.phase == Phase.BID_ROUND_2
    return s


@pytest.mark.parametrize("seed", range(8))
def test_bid_round2_rooted_solve_gets_real_search_depth(seed):
    s = _fresh_bid_round2(seed)
    solver = SubgameSolver(s, s.current_player, num_worlds=4, iterations=10,
                           depth_limit=6, value_fn=lambda st: 0.0,
                           rng=random.Random(0))
    solver.run()
    # A degenerate pre-fix solve here wasn't necessarily exactly 1 (Pass's
    # own subtree still expands), but the Call options themselves had zero
    # depth; a healthy solve should be well beyond that.
    assert len(solver.infosets) > 5, (
        f"seed={seed}: round-2-rooted solve built only "
        f"{len(solver.infosets)} infoset(s) -- Call options may still lack "
        f"real search depth")
    pol = solver.root_policy()
    assert abs(sum(pol.values()) - 1.0) < 1e-6
    assert set(pol) == set(s.legal_actions())


def test_bid_round2_root_policy_is_differentiated_with_trained_net():
    import os
    net_path = "checkpoints/rebel_sa.pt"
    if not os.path.exists(net_path):
        pytest.skip(f"{net_path} not present in this checkout")
    import torch
    from rebel.networks import PolicyValueNet
    from rebel.train_rebel import batch_value_fn_from_net

    net = PolicyValueNet()
    net.load_state_dict(torch.load(net_path, map_location="cpu"))
    value_fn = batch_value_fn_from_net(net)

    s = _fresh_bid_round2(0)
    solver = SubgameSolver(s, s.current_player, num_worlds=8, iterations=15,
                           depth_limit=6, batch_value_fn=value_fn,
                           rng=random.Random(0))
    solver.run()
    pol = solver.root_policy()
    n = len(pol)
    uniform = 1.0 / n
    assert max(pol.values()) > uniform + 0.05, (
        f"root_policy looks uniform (max={max(pol.values()):.4f}, "
        f"uniform={uniform:.4f}) -- round-2 search may have collapsed again")


def test_bid_round2_internal_to_bid_round1_solve_stays_free():
    """BID_ROUND_2 reached INSIDE a BID_ROUND_1-rooted solve (the "everyone
    passes" branch) must stay a cheap free expansion, not ply-counted --
    confirms the fix didn't reintroduce the original blowup."""
    rng = random.Random(0)
    s = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
    assert s.phase == Phase.BID_ROUND_1
    solver = SubgameSolver(s, s.current_player, num_worlds=4, iterations=5,
                           depth_limit=1, value_fn=lambda st: 0.0,
                           rng=random.Random(0))
    solver.run()
    assert len(solver.infosets) < 60, (
        f"bidding-rooted solve built {len(solver.infosets)} infosets -- "
        f"BID_ROUND_2 may no longer be free as an internal node")
