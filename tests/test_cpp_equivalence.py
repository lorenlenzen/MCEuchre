"""Differential/golden testing: the C++ engine (mceuchre_cpp) must produce
results IDENTICAL to the already-trusted Python engine (euchre.game) for the
same inputs. This is the primary correctness gate for the C++ port -- see
the plan's "Risk management" section: rather than re-deriving correctness
from scratch in a new test framework, we prove equivalence to the reference
implementation.

Both engines are driven through the SAME deal (an explicit card-id list, so
there's no cross-language RNG-algorithm concern) and the SAME sequence of
chosen actions (picked by shared action *index*, since both engines use the
identical NUM_ACTIONS=59 flat index space from euchre/actions.py), with
full state compared after every single step, not just at the end.
"""

import math
import random

import pytest

import numpy as np
import torch

import mceuchre_cpp as cpp
from euchre.actions import Call, Discard, OrderUp, Pass, Play, action_to_index, index_to_action
from euchre.cards import Card
from euchre.game import EuchreState, Phase
from euchre.infoset import observation_tensor as py_observation_tensor
from euchre.infoset import infoset_key as py_infoset_key
from euchre.infoset import OBS_SIZE as PY_OBS_SIZE

# Python Phase -> expected cpp.Phase name (naming conventions differ deliberately).
_PHASE_MAP = {
    Phase.DEAL: "Deal",
    Phase.BID_ROUND_1: "BidRound1",
    Phase.BID_ROUND_2: "BidRound2",
    Phase.DEALER_DISCARD: "DealerDiscard",
    Phase.PLAY: "Play",
    Phase.TERMINAL: "Terminal",
}


def _deal_both(seed, dealer=0, stick_the_dealer=False, team0_score=0, team1_score=0):
    """Deal identical hands into both engines from the same explicit shuffle."""
    deck = list(range(24))
    random.Random(seed).shuffle(deck)

    py_st = EuchreState.new_hand(dealer=dealer, stick_the_dealer=stick_the_dealer,
                                 team0_score=team0_score, team1_score=team1_score)
    py_hands = [[Card.from_id(c) for c in deck[i * 5:(i + 1) * 5]] for i in range(4)]
    py_up = Card.from_id(deck[20])
    py_kitty = [Card.from_id(c) for c in deck[21:24]]
    py_st = py_st.deal_from(py_hands, py_up, py_kitty)

    cpp_st = cpp.EuchreState.new_hand(dealer=dealer, stick_the_dealer=stick_the_dealer,
                                      team0_score=team0_score, team1_score=team1_score)
    cpp_st = cpp_st.deal_from_deck(deck)

    return py_st, cpp_st


def _hand_ids(py_hand):
    return frozenset(c.id for c in py_hand)


def _cpp_hand_ids(bitmask):
    return frozenset(i for i in range(24) if (bitmask >> i) & 1)


def _assert_states_equal(py_st, cpp_st, step_desc=""):
    ctx = f" (after {step_desc})" if step_desc else ""
    assert py_st.dealer == cpp_st.dealer, f"dealer mismatch{ctx}"
    assert _PHASE_MAP[py_st.phase] == cpp_st.phase.name, (
        f"phase mismatch{ctx}: {py_st.phase} vs {cpp_st.phase.name}")
    assert py_st.current_player == cpp_st.current_player, f"current_player mismatch{ctx}"

    for p in range(4):
        assert _hand_ids(py_st.hands[p]) == _cpp_hand_ids(cpp_st.hands[p]), \
            f"hand[{p}] mismatch{ctx}"

    py_up = py_st.up_card.id if py_st.up_card is not None else None
    assert py_up == cpp_st.up_card, f"up_card mismatch{ctx}: {py_up} vs {cpp_st.up_card}"

    py_kitty = [c.id for c in py_st.kitty]
    assert py_kitty == list(cpp_st.kitty), f"kitty mismatch{ctx}: {py_kitty} vs {list(cpp_st.kitty)}"

    py_trump = int(py_st.trump) if py_st.trump is not None else None
    assert py_trump == cpp_st.trump, f"trump mismatch{ctx}: {py_trump} vs {cpp_st.trump}"
    assert py_st.maker == cpp_st.maker, f"maker mismatch{ctx}"
    assert py_st.alone == cpp_st.alone, f"alone mismatch{ctx}"
    assert py_st.lone_player == cpp_st.lone_player, f"lone_player mismatch{ctx}"
    assert py_st.sitting == cpp_st.sitting, f"sitting mismatch{ctx}"
    py_td = int(py_st.turned_down) if py_st.turned_down is not None else None
    assert py_td == cpp_st.turned_down, f"turned_down mismatch{ctx}"
    assert py_st.bids_seen == cpp_st.bids_seen, f"bids_seen mismatch{ctx}"
    assert py_st.trick_leader == cpp_st.trick_leader, f"trick_leader mismatch{ctx}"
    assert list(py_st.tricks_won) == list(cpp_st.tricks_won), f"tricks_won mismatch{ctx}"

    py_cur_trick = [(p, c.id) for p, c in py_st.current_trick]
    cpp_cur_trick = [(t.player, t.card) for t in cpp_st.current_trick]
    assert py_cur_trick == cpp_cur_trick, f"current_trick mismatch{ctx}"

    py_completed = [(w, [(p, c.id) for p, c in plays]) for w, plays in py_st.completed_tricks]
    cpp_completed = [(t.winner, [(p.player, p.card) for p in t.plays])
                     for t in cpp_st.completed_tricks]
    assert py_completed == cpp_completed, f"completed_tricks mismatch{ctx}"

    assert py_st.is_terminal() == cpp_st.is_terminal(), f"is_terminal mismatch{ctx}"
    if py_st.is_terminal():
        assert py_st.returns() == cpp_st.returns(), f"returns mismatch{ctx}"


def _legal_action_indices(py_st):
    return frozenset(action_to_index(a) for a in py_st.legal_actions())


def _cpp_legal_action_indices(cpp_st):
    return frozenset(a.index() for a in cpp_st.legal_actions())


def _apply_index(py_st, cpp_st, idx):
    py_st2 = py_st.apply(index_to_action(idx))
    cpp_st2 = cpp_st.apply(cpp.Action.from_index(idx))
    return py_st2, cpp_st2


def _walk_random(seed, dealer=0, stick_the_dealer=False, max_steps=100,
                 team0_score=0, team1_score=0):
    py_st, cpp_st = _deal_both(seed, dealer, stick_the_dealer, team0_score, team1_score)
    _assert_states_equal(py_st, cpp_st, "deal")
    rng = random.Random(seed + 100000)
    steps = 0
    while not py_st.is_terminal():
        py_legal = _legal_action_indices(py_st)
        cpp_legal = _cpp_legal_action_indices(cpp_st)
        assert py_legal == cpp_legal, (
            f"legal actions mismatch at step {steps} (seed {seed}): "
            f"py={sorted(py_legal)} cpp={sorted(cpp_legal)}")
        idx = rng.choice(sorted(py_legal))
        py_st, cpp_st = _apply_index(py_st, cpp_st, idx)
        _assert_states_equal(py_st, cpp_st, f"step {steps} (action {idx})")
        steps += 1
        assert steps <= max_steps, f"didn't terminate within {max_steps} steps (seed {seed})"
    return py_st, cpp_st


# --- bulk random coverage ---------------------------------------------------

@pytest.mark.parametrize("seed", range(300))
def test_random_hands_match(seed):
    _walk_random(seed, dealer=seed % 4)


@pytest.mark.parametrize("seed", range(100))
def test_random_hands_with_stick_the_dealer_match(seed):
    _walk_random(seed, dealer=seed % 4, stick_the_dealer=True)


@pytest.mark.parametrize("seed", range(50))
def test_random_hands_with_score_match(seed):
    rng = random.Random(seed + 999)
    _walk_random(seed, dealer=seed % 4, team0_score=rng.randint(0, 9),
                team1_score=rng.randint(0, 9))


# --- targeted scenarios (force paths bulk random coverage might rarely hit) --

def test_misdeal_matches():
    """All-pass round 1 then all-pass round 2 -> misdeal, reward (0,0). Force
    it explicitly (natural random walks hit this ~1/150 hands) rather than
    relying on the bulk sweep to stumble into it."""
    py_st, cpp_st = _deal_both(seed=7)
    for _ in range(8):  # 4 round-1 passes + 4 round-2 passes
        py_idx = action_to_index(Pass())
        assert py_idx in _legal_action_indices(py_st)
        py_st, cpp_st = _apply_index(py_st, cpp_st, py_idx)
        _assert_states_equal(py_st, cpp_st, "forced pass")
    assert py_st.is_terminal()
    assert py_st.returns() == (0, 0)
    assert cpp_st.returns() == (0, 0)


def test_stick_the_dealer_forces_call():
    """With stick_the_dealer, the dealer's round-2 turn must not offer Pass."""
    py_st, cpp_st = _deal_both(seed=7, stick_the_dealer=True)
    for _ in range(4):  # round-1 all pass -> round 2
        idx = action_to_index(Pass())
        py_st, cpp_st = _apply_index(py_st, cpp_st, idx)
    for _ in range(3):  # first 3 round-2 seats pass
        idx = action_to_index(Pass())
        py_st, cpp_st = _apply_index(py_st, cpp_st, idx)
    _assert_states_equal(py_st, cpp_st, "round2 3 passes")
    assert py_st.current_player == py_st.dealer
    py_legal = _legal_action_indices(py_st)
    cpp_legal = _cpp_legal_action_indices(cpp_st)
    assert py_legal == cpp_legal
    assert action_to_index(Pass()) not in py_legal, "dealer must not be able to pass"


@pytest.mark.parametrize("seed", range(30))
def test_alone_calls_match(seed):
    """Force an alone OrderUp/Call so the sitting-out player-skip logic
    (next_player/begin_play) gets exercised and compared."""
    py_st, cpp_st = _deal_both(seed=seed)
    idx = action_to_index(OrderUp(alone=True))
    if idx not in _legal_action_indices(py_st):
        pytest.skip("not first-to-act's turn for a round-1 alone order-up in this seed")
    py_st, cpp_st = _apply_index(py_st, cpp_st, idx)
    _assert_states_equal(py_st, cpp_st, "alone order-up")
    assert py_st.sitting is not None
    assert cpp_st.sitting is not None
    # play the rest out randomly (dealer discard + 3-handed tricks)
    rng = random.Random(seed + 5000)
    steps = 0
    while not py_st.is_terminal():
        legal = sorted(_legal_action_indices(py_st))
        a_idx = rng.choice(legal)
        py_st, cpp_st = _apply_index(py_st, cpp_st, a_idx)
        _assert_states_equal(py_st, cpp_st, f"alone-hand step {steps}")
        steps += 1
        assert steps <= 100


# --- action index round-trip (both directions, both languages) -------------

def test_action_index_roundtrip_matches_python():
    """Every C++ Action.from_index/.index() must agree with the already
    -verified Python action_to_index/index_to_action (tests/test_engine.py)
    for the full NUM_ACTIONS space."""
    assert cpp.NUM_ACTIONS == 59
    for idx in range(cpp.NUM_ACTIONS):
        py_action = index_to_action(idx)
        py_idx_back = action_to_index(py_action)
        assert py_idx_back == idx  # sanity on the Python side itself

        cpp_action = cpp.Action.from_index(idx)
        assert cpp_action.index() == idx

        # cross-check the *kind* and payload agree
        if isinstance(py_action, Pass):
            assert cpp_action.kind == cpp.ActionKind.Pass
        elif isinstance(py_action, OrderUp):
            assert cpp_action.kind == cpp.ActionKind.OrderUp
            assert cpp_action.alone == py_action.alone
        elif isinstance(py_action, Call):
            assert cpp_action.kind == cpp.ActionKind.Call
            assert cpp_action.alone == py_action.alone
            assert cpp_action.suit == int(py_action.suit)
        elif isinstance(py_action, Discard):
            assert cpp_action.kind == cpp.ActionKind.Discard
            assert cpp_action.card == py_action.card.id
        elif isinstance(py_action, Play):
            assert cpp_action.kind == cpp.ActionKind.Play
            assert cpp_action.card == py_action.card.id


# --- legal_mask vs rebel.train_rebel.legal_mask -----------------------------

def test_legal_mask_matches_python():
    """cpp.legal_mask must set exactly the flat indices legal_actions()
    reports, at every step of a random hand -- the C++ port added so
    net.policy(obs, legal_mask) can be called without a Python round-trip
    for legal-mask construction (see docs/rebel_design.md's planned
    net-native self-play section)."""
    from rebel.train_rebel import legal_mask as py_legal_mask
    for seed in range(30):
        py_st, cpp_st = _deal_both(seed, dealer=seed % 4)
        rng = random.Random(seed + 200000)
        steps = 0
        while not py_st.is_terminal():
            py_mask = py_legal_mask(py_st)
            cpp_mask = np.asarray(cpp.legal_mask(cpp_st))
            assert py_mask.shape == cpp_mask.shape == (cpp.NUM_ACTIONS,)
            assert np.array_equal(py_mask, cpp_mask), (
                f"legal_mask mismatch at step {steps} (seed {seed}): "
                f"py={np.nonzero(py_mask)[0].tolist()} "
                f"cpp={np.nonzero(cpp_mask)[0].tolist()}")
            idx = rng.choice(sorted(_legal_action_indices(py_st)))
            py_st, cpp_st = _apply_index(py_st, cpp_st, idx)
            steps += 1
            assert steps <= 100, f"didn't terminate within 100 steps (seed {seed})"


# --- observation_tensor / infoset_key vs euchre.infoset --------------------

def _assert_obs_and_key_match(py_st, cpp_st, player, ctx=""):
    py_obs = py_observation_tensor(py_st, player)
    cpp_obs = np.asarray(cpp.observation_tensor(cpp_st, player))
    assert py_obs.shape == cpp_obs.shape == (PY_OBS_SIZE,), (
        f"obs shape mismatch{ctx}: py={py_obs.shape} cpp={cpp_obs.shape}")
    assert np.array_equal(py_obs, cpp_obs), (
        f"observation_tensor mismatch{ctx} for player {player}: "
        f"{np.nonzero(py_obs != cpp_obs)[0].tolist()} differ")

    py_key = py_infoset_key(py_st, player)
    cpp_key = cpp.infoset_key(cpp_st, player)
    assert py_key == cpp_key, f"infoset_key mismatch{ctx}: {py_key!r} vs {cpp_key!r}"


@pytest.mark.parametrize("seed", range(150))
def test_observation_and_key_match_current_player(seed):
    """At every step of a random hand, for the player about to act (the
    realistic case -- this is how both functions are actually called
    throughout the codebase), obs/key must match exactly."""
    py_st, cpp_st = _deal_both(seed, dealer=seed % 4)
    rng = random.Random(seed + 200000)
    steps = 0
    while not py_st.is_terminal():
        p = py_st.current_player
        _assert_obs_and_key_match(py_st, cpp_st, p, f" step {steps}")
        legal = sorted(_legal_action_indices(py_st))
        idx = rng.choice(legal)
        py_st, cpp_st = _apply_index(py_st, cpp_st, idx)
        steps += 1
        assert steps <= 100


@pytest.mark.parametrize("seed", range(40))
def test_observation_and_key_match_all_players(seed):
    """Same, but also check the 3 non-acting players' hypothetical views at
    each step -- observation_tensor/infoset_key both take an explicit
    `player` arg and are used for any seat, not just the actor."""
    py_st, cpp_st = _deal_both(seed, dealer=seed % 4)
    rng = random.Random(seed + 300000)
    steps = 0
    while not py_st.is_terminal():
        for p in range(4):
            _assert_obs_and_key_match(py_st, cpp_st, p, f" step {steps} player {p}")
        legal = sorted(_legal_action_indices(py_st))
        idx = rng.choice(legal)
        py_st, cpp_st = _apply_index(py_st, cpp_st, idx)
        steps += 1
        assert steps <= 100


@pytest.mark.parametrize("seed", range(50))
def test_observation_and_key_match_with_score(seed):
    rng = random.Random(seed + 400000)
    py_st, cpp_st = _deal_both(seed, dealer=seed % 4,
                               team0_score=rng.randint(0, 9), team1_score=rng.randint(0, 9))
    walk_rng = random.Random(seed + 500000)
    steps = 0
    while not py_st.is_terminal():
        _assert_obs_and_key_match(py_st, cpp_st, py_st.current_player, f" step {steps}")
        legal = sorted(_legal_action_indices(py_st))
        idx = walk_rng.choice(legal)
        py_st, cpp_st = _apply_index(py_st, cpp_st, idx)
        steps += 1
        assert steps <= 100


def test_obs_size_matches():
    assert cpp.OBS_SIZE == PY_OBS_SIZE


# --- solve_value (exact double-dummy) vs rebel.solver -----------------------

def _walk_to_play(seed, dealer=0, min_cards_left=None, max_tries=50):
    """Walk both engines to a random PLAY-phase decision (deterministically
    identical between them), optionally continuing further until at most
    `min_cards_left` cards remain in the acting player's hand (keeps exact
    solves fast for the bulk of the sweep)."""
    for attempt in range(max_tries):
        py_st, cpp_st = _deal_both(seed * 1000 + attempt, dealer)
        rng = random.Random(seed * 1000 + attempt + 700000)
        steps = 0
        while not py_st.is_terminal() and py_st.phase != Phase.PLAY:
            legal = sorted(_legal_action_indices(py_st))
            idx = rng.choice(legal)
            py_st, cpp_st = _apply_index(py_st, cpp_st, idx)
            steps += 1
            if steps > 30:
                break
        if py_st.phase != Phase.PLAY:
            continue  # misdealt or didn't reach play in time; redeal
        if min_cards_left is not None:
            while (not py_st.is_terminal()
                  and len(py_st.hands[py_st.current_player]) > min_cards_left):
                legal = sorted(_legal_action_indices(py_st))
                idx = rng.choice(legal)
                py_st, cpp_st = _apply_index(py_st, cpp_st, idx)
            if py_st.is_terminal():
                continue
        return py_st, cpp_st
    pytest.skip(f"couldn't reach a PLAY state in {max_tries} tries for seed {seed}")


@pytest.mark.parametrize("seed", range(40))
def test_solve_value_matches_late_game(seed):
    """Exact solves near the end of the hand (<=3 cards left) -- fast, and
    this is the realistic regime (full_depth_cards) the live trainer uses."""
    from rebel.solver import solve_value as py_solve_value
    py_st, cpp_st = _walk_to_play(seed, dealer=seed % 4, min_cards_left=3)
    py_val = py_solve_value(py_st)
    cpp_val = cpp.solve_value(cpp_st)
    assert py_val == cpp_val, f"solve_value mismatch seed={seed}: py={py_val} cpp={cpp_val}"


@pytest.mark.parametrize("seed", range(8))
def test_solve_value_matches_full_hand(seed):
    """A handful of full (unrestricted) exact solves from the start of PLAY
    -- slower, kept to a small count."""
    from rebel.solver import solve_value as py_solve_value
    py_st, cpp_st = _walk_to_play(seed, dealer=seed % 4)
    py_val = py_solve_value(py_st)
    cpp_val = cpp.solve_value(cpp_st)
    assert py_val == cpp_val, f"solve_value mismatch seed={seed}: py={py_val} cpp={cpp_val}"


# --- rollout_value (rebel.pimc) vs cpp_rollout_value (rebel.train_rebel) ---
# cpp_rollout_value is a thin Python-level mirror (not a new C++ port) built
# on the already-bound mceuchre_cpp.solve_value -- see rebel/train_rebel.py's
# module docstring for why. These tests are the correctness gate for it, the
# same role test_solve_value_matches_* plays for the lower-level solve.

def _order_up_index(py_st, alone=False):
    for a in py_st.legal_actions():
        if isinstance(a, OrderUp) and a.alone == alone:
            return action_to_index(a)
    raise AssertionError("no matching OrderUp action legal in this state")


def _call_index(py_st, alone=False):
    for a in py_st.legal_actions():
        if isinstance(a, Call) and a.alone == alone:
            return action_to_index(a)
    return None


@pytest.mark.parametrize("seed", range(20))
def test_rollout_value_matches_python_dealer_discard(seed):
    """cpp_rollout_value must resolve a pending DEALER_DISCARD -- trying
    every discard, keeping whichever is best for the dealer's team --
    identically to rebel.pimc.rollout_value's pure-Python version. Reached
    via a round-1 OrderUp(not alone), which always transitions straight to
    DEALER_DISCARD."""
    from rebel.pimc import rollout_value as py_rollout_value
    from rebel.train_rebel import cpp_rollout_value
    py_st, cpp_st = _deal_both(seed, dealer=seed % 4)
    idx = _order_up_index(py_st, alone=False)
    py_next, cpp_next = _apply_index(py_st, cpp_st, idx)
    py_val = py_rollout_value(py_next)
    cpp_val = cpp_rollout_value(cpp_next)
    assert py_val == cpp_val, f"rollout_value mismatch seed={seed}: py={py_val} cpp={cpp_val}"


@pytest.mark.parametrize("seed", range(20))
def test_resolve_dealer_discard_value_matches_python(seed):
    """resolve_dealer_discard/cpp_resolve_dealer_discard must agree on the
    optimal discard's VALUE. The specific discarded CARD is allowed to
    differ when multiple discards tie for that value (Python's and C++'s
    legal_actions() enumeration orders differ, so ties break differently --
    harmless, since each engine's own (state, value) pair stays internally
    self-consistent regardless of which tied-optimal discard it picked)."""
    from rebel.pimc import resolve_dealer_discard as py_resolve
    from rebel.train_rebel import cpp_resolve_dealer_discard as cpp_resolve
    from rebel.solver import solve_value as py_solve_value
    py_st, cpp_st = _deal_both(seed, dealer=seed % 4)
    idx = _order_up_index(py_st, alone=False)
    py_dd, cpp_dd = _apply_index(py_st, cpp_st, idx)

    py_next, py_val = py_resolve(py_dd)
    cpp_next, cpp_val = cpp_resolve(cpp_dd)
    assert py_val == cpp_val, (
        f"resolve_dealer_discard value mismatch seed={seed}: py={py_val} cpp={cpp_val}")
    # Each engine's returned state must independently justify its own value.
    assert py_solve_value(py_next) == py_val
    assert cpp.solve_value(cpp_next) == cpp_val


@pytest.mark.parametrize("seed", range(20))
def test_rollout_value_matches_python_round2_call(seed):
    """Same, but reached via four real passes into BID_ROUND_2 then a
    non-alone Call -- covers rollout_value's other entry point (straight
    into PLAY, no DEALER_DISCARD involved)."""
    from rebel.pimc import rollout_value as py_rollout_value
    from rebel.train_rebel import cpp_rollout_value
    for attempt in range(10):
        py_st, cpp_st = _deal_both(seed * 100 + attempt, dealer=seed % 4)
        pass_idx = action_to_index(Pass())
        for _ in range(4):
            py_st, cpp_st = _apply_index(py_st, cpp_st, pass_idx)
        if py_st.phase != Phase.BID_ROUND_2:
            continue
        idx = _call_index(py_st, alone=False)
        if idx is None:
            continue
        py_next, cpp_next = _apply_index(py_st, cpp_st, idx)
        py_val = py_rollout_value(py_next)
        cpp_val = cpp_rollout_value(cpp_next)
        assert py_val == cpp_val, (
            f"rollout_value mismatch seed={seed}: py={py_val} cpp={cpp_val}")
        return
    pytest.skip(f"couldn't reach a callable BID_ROUND_2 state for seed {seed}")


# --- DEALER_DISCARD: free only as an internal node, not as the solve root -
# Regression coverage for a real bug the phase-boundary fix introduced (see
# rebel/subgame.py's build() and cpp/subgame.cpp's build()): treating
# DEALER_DISCARD as always-free meant a discard-rooted solve was nothing but
# the root plus 6 same-depth leaves -- no real search -- producing an
# exactly-uniform policy regardless of net quality. Ported to both engines;
# this is the differential half of that fix's verification.

@pytest.mark.parametrize("seed", range(10))
def test_dealer_discard_rooted_solve_matches_python(seed):
    """With a zero value function, pure double-dummy CFR determines the
    result -- if both engines now explore real depth into the resulting
    hand instead of stopping at the root, they should still agree exactly,
    the same correctness gate test_subgame_solver_depth_limited_matches
    already applies to bidding-rooted solves."""
    py_st, cpp_st = _deal_both(seed, dealer=seed % 4)
    idx = _order_up_index(py_st, alone=False)
    py_dd, cpp_dd = _apply_index(py_st, cpp_st, idx)
    assert py_dd.phase == Phase.DEALER_DISCARD
    assert cpp_dd.phase == cpp.Phase.DealerDiscard

    py_solver = _py_solver_with_worlds(py_dd, py_dd.current_player, [py_dd], [1.0],
                                       iterations=5, depth_limit=3,
                                       batch_value_fn=_zero_batch_value_fn, equity_model=None)
    cpp_solver = cpp.SubgameSolver(cpp_dd, cpp_dd.current_player, [cpp_dd], [1.0],
                                   5, 3, _zero_batch_value_fn, None)

    py_policy = {action_to_index(a): p for a, p in py_solver.root_policy().items()}
    cpp_policy = cpp_solver.root_policy()
    assert set(py_policy) == set(cpp_policy)
    for idx2 in py_policy:
        assert py_policy[idx2] == pytest.approx(cpp_policy[idx2], abs=1e-9), (
            f"discard root_policy[{idx2}] mismatch seed={seed}: "
            f"py={py_policy[idx2]} cpp={cpp_policy[idx2]}")


def test_cpp_dealer_discard_root_policy_is_differentiated_with_trained_net():
    """Same collapse check as the Python side
    (test_subgame.py::test_dealer_discard_root_policy_is_differentiated_with_trained_net),
    against the cpp engine -- this is what actually surfaced the bug (the
    quiz runs the trained net, not a zero value function)."""
    import os
    net_path = "checkpoints/rebel_sa.pt"
    if not os.path.exists(net_path):
        pytest.skip(f"{net_path} not present in this checkout")
    import torch
    from rebel.networks import PolicyValueNet
    from rebel.train_rebel import cpp_batch_value_fn_from_net

    net = PolicyValueNet()
    net.load_state_dict(torch.load(net_path, map_location="cpu"))
    value_fn = cpp_batch_value_fn_from_net(net)

    import random as _random
    rng = _random.Random(0)
    deck = list(range(24))
    rng.shuffle(deck)
    st = cpp.EuchreState.new_hand(dealer=rng.randint(0, 3)).deal_from_deck(deck)
    st = st.apply(cpp.Action.order_up(False))
    assert st.phase == cpp.Phase.DealerDiscard

    solver = cpp.SubgameSolver(st, st.current_player, 8, 15, 6, value_fn, None, 0)
    solver.run()
    pol = solver.root_policy()
    n = len(pol)
    uniform = 1.0 / n
    assert max(pol.values()) > uniform + 0.05, (
        f"cpp root_policy looks uniform (max={max(pol.values()):.4f}, "
        f"uniform={uniform:.4f}) -- discard search may have collapsed again")


# --- BID_ROUND_2: same structural gap as DEALER_DISCARD, for Call ---------
# Round 2's Call skips DealerDiscard entirely and goes straight to Play, so
# a round-2-rooted solve's own Call/Call-alone/suit comparison had the same
# "root's own action goes straight to an unbacked leaf" gap DealerDiscard
# did -- surfaced as "every alone option outranks its same-suit non-alone
# twin" on the quiz.

@pytest.mark.parametrize("seed", range(10))
def test_bid_round2_rooted_solve_matches_python(seed):
    py_st, cpp_st = _deal_both(seed, dealer=seed % 4, stick_the_dealer=True)
    pass_idx = action_to_index(Pass())
    for _ in range(4):
        py_st, cpp_st = _apply_index(py_st, cpp_st, pass_idx)
    assert py_st.phase == Phase.BID_ROUND_2
    assert cpp_st.phase == cpp.Phase.BidRound2

    py_solver = _py_solver_with_worlds(py_st, py_st.current_player, [py_st], [1.0],
                                       iterations=5, depth_limit=3,
                                       batch_value_fn=_zero_batch_value_fn, equity_model=None)
    cpp_solver = cpp.SubgameSolver(cpp_st, cpp_st.current_player, [cpp_st], [1.0],
                                   5, 3, _zero_batch_value_fn, None)

    py_policy = {action_to_index(a): p for a, p in py_solver.root_policy().items()}
    cpp_policy = cpp_solver.root_policy()
    assert set(py_policy) == set(cpp_policy)
    for idx2 in py_policy:
        assert py_policy[idx2] == pytest.approx(cpp_policy[idx2], abs=1e-9), (
            f"round-2 root_policy[{idx2}] mismatch seed={seed}: "
            f"py={py_policy[idx2]} cpp={cpp_policy[idx2]}")


def test_cpp_bid_round2_root_policy_is_differentiated_with_trained_net():
    import os
    net_path = "checkpoints/rebel_sa.pt"
    if not os.path.exists(net_path):
        pytest.skip(f"{net_path} not present in this checkout")
    import torch
    from rebel.networks import PolicyValueNet
    from rebel.train_rebel import cpp_batch_value_fn_from_net

    net = PolicyValueNet()
    net.load_state_dict(torch.load(net_path, map_location="cpu"))
    value_fn = cpp_batch_value_fn_from_net(net)

    import random as _random
    rng = _random.Random(0)
    deck = list(range(24))
    rng.shuffle(deck)
    st = cpp.EuchreState.new_hand(dealer=rng.randint(0, 3), stick_the_dealer=True).deal_from_deck(deck)
    for _ in range(4):
        st = st.apply(cpp.Action.pass_())
    assert st.phase == cpp.Phase.BidRound2

    solver = cpp.SubgameSolver(st, st.current_player, 8, 15, 6, value_fn, None, 0)
    solver.run()
    pol = solver.root_policy()
    n = len(pol)
    uniform = 1.0 / n
    assert max(pol.values()) > uniform + 0.05, (
        f"cpp round-2 root_policy looks uniform (max={max(pol.values()):.4f}, "
        f"uniform={uniform:.4f}) -- search may have collapsed again")


def test_rollout_value_equity_mode_matches_python():
    """The equity-delta conversion boundary (team0_score/team1_score +
    equity_model) must match too, not just the raw double-dummy path --
    cpp_rollout_value takes a mceuchre_cpp.MatchEquityModel here, a
    different type than rebel.match_equity.MatchEquityModel but built from
    the same table (see ReBeLTrainer._cpp_equity_model)."""
    from rebel.match_equity import MatchEquityModel
    from rebel.pimc import rollout_value as py_rollout_value
    from rebel.train_rebel import cpp_rollout_value
    py_eq = MatchEquityModel.load("rebel/match_equity_table.json")
    cpp_eq = cpp.MatchEquityModel(py_eq.target, py_eq.table.flatten().tolist())
    rng = random.Random(0)
    for trial in range(15):
        t0, t1 = rng.randint(0, 9), rng.randint(0, 9)
        py_st, cpp_st = _deal_both(trial, dealer=trial % 4,
                                   team0_score=t0, team1_score=t1)
        idx = _order_up_index(py_st, alone=False)
        py_next, cpp_next = _apply_index(py_st, cpp_st, idx)
        py_val = py_rollout_value(py_next, team0_score=py_next.team0_score,
                                  team1_score=py_next.team1_score,
                                  equity_model=py_eq)
        cpp_val = cpp_rollout_value(cpp_next, team0_score=cpp_next.team0_score,
                                    team1_score=cpp_next.team1_score,
                                    equity_model=cpp_eq)
        assert abs(py_val - cpp_val) < 1e-9, (
            f"rollout_value equity mismatch trial={trial}: py={py_val} cpp={cpp_val}")


# --- MatchEquityModel vs rebel.match_equity ---------------------------------

def test_match_equity_model_matches_python():
    """Dealer-relative model (see rebel/match_equity.py's module docstring):
    a single target x target table (Ed[a,b], the dealing team's win prob),
    win_prob(my_score, opp_score, am_i_dealer) applies the non-dealing
    identity 1 - Ed[opp,my] directly rather than storing a second table.
    Exhaustive over all target*target*2 = 200 (a, b, am_i_dealer)
    combinations -- cheap enough not to sample."""
    from rebel.match_equity import MatchEquityModel as PyModel, build_equity_table

    # Deliberately asymmetric (a real dealer edge) so this test would catch a
    # bug that only shows up when Ed != Eo, not just a coincidentally-neutral
    # distribution where the dealer flag wouldn't matter either way.
    dist = {(1, 0): 0.45, (0, 1): 0.25, (2, 0): 0.15, (0, 2): 0.05,
            (4, 0): 0.02, (0, 0): 0.08}
    target = 10
    table = build_equity_table(dist, target=target)
    py_model = PyModel(table, dist)
    cpp_model = cpp.MatchEquityModel(target, table.flatten().tolist())

    for a in range(target):
        for b in range(target):
            for am_i_dealer in (True, False):
                py_wp = py_model.win_prob(a, b, am_i_dealer)
                cpp_wp = cpp_model.win_prob(a, b, am_i_dealer)
                assert abs(py_wp - cpp_wp) < 1e-12, (
                    f"win_prob({a},{b},{am_i_dealer}) mismatch: {py_wp} vs {cpp_wp}")

    for (p0, p1) in [(1, 0), (2, 0), (0, 1), (0, 2), (4, 0), (0, 4), (0, 0)]:
        for a in (0, 5, 9):
            for b in (0, 5, 9):
                for dealer_is_team0 in (True, False):
                    py_d = py_model.equity_delta(a, b, dealer_is_team0, p0, p1)
                    cpp_d = cpp_model.equity_delta(a, b, dealer_is_team0, p0, p1)
                    assert abs(py_d - cpp_d) < 1e-12, (
                        f"equity_delta({a},{b},{dealer_is_team0},{p0},{p1}) "
                        f"mismatch: {py_d} vs {cpp_d}")

    assert cpp_model.win_prob(target, 3, True) == 1.0
    assert cpp_model.win_prob(3, target, True) == 0.0
    assert cpp_model.win_prob(target, 3, False) == 1.0
    assert cpp_model.win_prob(3, target, False) == 0.0


# --- trump-table helpers vs euchre.cards -----------------------------------

def test_trump_helpers_match_python():
    from euchre.cards import DECK, SUITS
    from euchre.cards import is_right_bower as py_rb, is_left_bower as py_lb
    from euchre.cards import is_trump as py_trump, effective_suit as py_eff

    for c in DECK:
        for t in SUITS:
            assert cpp.is_right_bower(c.id, int(t)) == py_rb(c, t)
            assert cpp.is_left_bower(c.id, int(t)) == py_lb(c, t)
            assert cpp.is_trump(c.id, int(t)) == py_trump(c, t)
            assert cpp.effective_suit(c.id, int(t)) == int(py_eff(c, t))
        assert cpp.effective_suit(c.id, -1) == int(c.suit)  # trump=None case


# --- sample_determinization: correctness properties, NOT cross-language RNG
# equality (deliberate design decision -- see module docstring / plan: the
# sampler only needs to be an unbiased, CONSISTENT sampler, not bit-identical
# to Python's for the same seed). -----------------------------------------

def _cpp_hand_size(bitmask):
    return bin(bitmask).count("1")


@pytest.mark.parametrize("seed", range(20))
def test_sample_determinization_consistent(seed):
    from rebel.public_belief_state import known_voids as py_known_voids

    py_st, cpp_st = _walk_to_play(seed, dealer=seed % 4, min_cards_left=2)
    player = cpp_st.current_player
    voids = py_known_voids(py_st)
    trump = cpp_st.trump

    for s in range(5):
        world = cpp.sample_determinization(cpp_st, player, seed=seed * 100 + s)

        # Actor's own hand is preserved exactly.
        assert _cpp_hand_ids(world.hands[player]) == _cpp_hand_ids(cpp_st.hands[player])

        # Hand sizes and kitty size unchanged.
        for p in range(4):
            assert _cpp_hand_size(world.hands[p]) == _cpp_hand_size(cpp_st.hands[p])
        assert len(world.kitty) == len(cpp_st.kitty)

        # No card assigned twice across the 4 hands + kitty.
        all_cards = []
        for p in range(4):
            all_cards.extend(_cpp_hand_ids(world.hands[p]))
        all_cards.extend(world.kitty)
        assert len(all_cards) == len(set(all_cards)), f"card reused (seed {seed}, sample {s})"

        # Opponents never hold a card in a suit they're already known void in.
        # Exception: the up-card itself, when its location is still genuinely
        # ambiguous between the dealer's hand and the kitty (post-pickup,
        # post-discard, queried by a non-dealer) -- both engines special-case
        # its destination as {dealer, kitty} without consulting voids there,
        # matching rebel/public_belief_state.py's own (pre-existing, shared)
        # simplification, not a language discrepancy.
        up_card = cpp_st.up_card
        for p in range(4):
            if p == player:
                continue
            void_suits = {int(vs) for vs in voids[p]}
            if not void_suits:
                continue
            for c in _cpp_hand_ids(world.hands[p]):
                if c == up_card:
                    continue
                assert cpp.effective_suit(c, trump) not in void_suits, (
                    f"seat {p} assigned a card in a known-void suit (seed {seed}, sample {s})")


# --- SubgameSolver: CFR math itself, testing constructor with explicit,
# identically-constructed worlds (bypasses sample_determinization on both
# sides, per the plan's design -- CFR's regret matching is fully
# deterministic given fixed worlds/weights, so it CAN be compared exactly). -

def _build_state_pair(dealer, hands_ids, up_id, kitty_ids, stick_the_dealer=False,
                      team0_score=0, team1_score=0):
    py_st = EuchreState.new_hand(dealer=dealer, stick_the_dealer=stick_the_dealer,
                                 team0_score=team0_score, team1_score=team1_score)
    py_hands = [[Card.from_id(c) for c in h] for h in hands_ids]
    py_st = py_st.deal_from(py_hands, Card.from_id(up_id), [Card.from_id(c) for c in kitty_ids])

    cpp_st = cpp.EuchreState.new_hand(dealer=dealer, stick_the_dealer=stick_the_dealer,
                                      team0_score=team0_score, team1_score=team1_score)
    cpp_hands = [sum(1 << c for c in h) for h in hands_ids]
    cpp_st = cpp_st.deal_from(cpp_hands, up_id, list(kitty_ids))
    return py_st, cpp_st


def _bidding_worlds(seed, num_worlds=3, dealer=0, actor=1):
    """Several BID_ROUND_1 (hands, kitty) assignments sharing the actor's own
    hand and the public up-card -- so every world shares the actor's exact
    root infoset, differing only in opponents' hidden hands/kitty."""
    deck = list(range(24))
    random.Random(seed).shuffle(deck)
    actor_hand_ids = deck[:5]
    up_id = deck[5]
    pool = deck[6:]

    rng = random.Random(seed + 42)
    seats = [p for p in range(4) if p != actor]
    worlds = []
    for _ in range(num_worlds):
        shuffled = list(pool)
        rng.shuffle(shuffled)
        hands_ids = [None] * 4
        hands_ids[actor] = list(actor_hand_ids)
        idx = 0
        for s in seats:
            hands_ids[s] = shuffled[idx:idx + 5]
            idx += 5
        kitty_ids = shuffled[idx:idx + 3]
        worlds.append((hands_ids, kitty_ids))
    return dealer, actor, up_id, worlds


def _py_solver_with_worlds(root, actor, worlds, weights, iterations, depth_limit,
                           batch_value_fn, equity_model):
    from rebel.subgame import SubgameSolver as PySubgameSolver
    from euchre.game import team_of
    from euchre.infoset import infoset_key as py_infoset_key

    solver = PySubgameSolver.__new__(PySubgameSolver)
    solver.actor = actor
    solver.iterations = iterations
    solver.depth_limit = depth_limit
    solver.value_fn = None
    solver.batch_value_fn = batch_value_fn
    solver.equity_model = equity_model
    solver.team0_score = root.team0_score
    solver.team1_score = root.team1_score
    solver.dealer_is_team0 = team_of(root.dealer) == 0
    solver.rng = random.Random(0)
    solver.infosets = {}
    solver.worlds = worlds
    solver.weights = weights
    solver.root_phase = root.phase
    solver.root_key = py_infoset_key(root, actor)
    solver.roots = None
    solver._pending_leaves = []
    return solver


def _zero_batch_value_fn(states):
    return [0.0] * len(states)


@pytest.mark.parametrize("seed", range(10))
def test_subgame_solver_depth_limited_matches(seed):
    """Depth-limited, batched-leaf CFR from a BID_ROUND_1 root, several
    explicit shared-infoset worlds -- exercises tree building, batched leaf
    deferral, and the CFR regret-matching recursion itself."""
    dealer, actor, up_id, world_specs = _bidding_worlds(seed, num_worlds=3)
    py_root, cpp_root = _build_state_pair(dealer, world_specs[0][0], up_id, world_specs[0][1])

    py_worlds, cpp_worlds = [], []
    for hands_ids, kitty_ids in world_specs:
        p, c = _build_state_pair(dealer, hands_ids, up_id, kitty_ids)
        py_worlds.append(p)
        cpp_worlds.append(c)
    weights = [0.5, 0.3, 0.2]

    py_solver = _py_solver_with_worlds(py_root, actor, py_worlds, list(weights),
                                       iterations=5, depth_limit=2,
                                       batch_value_fn=_zero_batch_value_fn, equity_model=None)
    cpp_solver = cpp.SubgameSolver(cpp_root, actor, cpp_worlds, list(weights),
                                   5, 2, _zero_batch_value_fn, None)

    py_policy = {action_to_index(a): p for a, p in py_solver.root_policy().items()}
    cpp_policy = cpp_solver.root_policy()
    assert set(py_policy) == set(cpp_policy)
    for idx in py_policy:
        assert py_policy[idx] == pytest.approx(cpp_policy[idx], abs=1e-9), (
            f"root_policy[{idx}] mismatch seed={seed}: py={py_policy[idx]} cpp={cpp_policy[idx]}")

    py_value = py_solver.root_value()
    cpp_value = cpp_solver.root_value()
    assert py_value == pytest.approx(cpp_value, abs=1e-9), (
        f"root_value mismatch seed={seed}: py={py_value} cpp={cpp_value}")


@pytest.mark.parametrize("seed", range(6))
def test_subgame_solver_full_solve_matches(seed):
    """Full (unlimited depth) solve to terminal from a late-PLAY root, single
    world (the actual concrete deal), with a real MatchEquityModel -- exercises
    the equity-aware terminal-utility path end to end."""
    from rebel.match_equity import MatchEquityModel as PyModel, build_equity_table

    py_st, cpp_st = _walk_to_play(seed, dealer=seed % 4, min_cards_left=2)
    actor = py_st.current_player

    dist = {(2, 0): 0.25, (0, 2): 0.25, (1, 0): 0.20, (0, 1): 0.20, (0, 0): 0.10}
    target = 10
    table = build_equity_table(dist, target=target)
    py_model = PyModel(table, dist)
    cpp_model = cpp.MatchEquityModel(target, table.flatten().tolist())

    py_solver = _py_solver_with_worlds(py_st, actor, [py_st], [1.0],
                                       iterations=5, depth_limit=None,
                                       batch_value_fn=None, equity_model=py_model)
    cpp_solver = cpp.SubgameSolver(cpp_st, actor, [cpp_st], [1.0],
                                   5, -1, None, cpp_model)

    py_policy = {action_to_index(a): p for a, p in py_solver.root_policy().items()}
    cpp_policy = cpp_solver.root_policy()
    assert set(py_policy) == set(cpp_policy)
    for idx in py_policy:
        assert py_policy[idx] == pytest.approx(cpp_policy[idx], abs=1e-9), (
            f"root_policy[{idx}] mismatch seed={seed}: py={py_policy[idx]} cpp={cpp_policy[idx]}")

    py_value = py_solver.root_value()
    cpp_value = cpp_solver.root_value()
    assert py_value == pytest.approx(cpp_value, abs=1e-9), (
        f"root_value mismatch seed={seed}: py={py_value} cpp={cpp_value}")


# --- SubgameSolver: net-native leaf eval vs the Python-callback BatchValueFn
# path, on IDENTICAL worlds/weights -- proves build_trees_net_native() (Part
# B of the planned net-native self-play work; see docs/rebel_design.md)
# computes byte-identical leaf values to cpp_batch_value_fn_from_net, the
# per-solve Python callback it replaces on the self-play hot path. Isolates
# "did porting leaf eval into C++ change the math" from world sampling,
# which this doesn't touch. -------------------------------------------------

@pytest.mark.parametrize("perspective", [None, 0, 1])
@pytest.mark.parametrize("seed", range(10))
def test_subgame_solver_net_native_matches_callback(seed, perspective):
    from rebel.train_rebel import cpp_batch_value_fn_from_net

    torch.manual_seed(seed)
    net = cpp.PolicyValueNet()
    net.eval()

    dealer, actor, up_id, world_specs = _bidding_worlds(seed, num_worlds=4)
    _, cpp_root = _build_state_pair(dealer, world_specs[0][0], up_id, world_specs[0][1])
    cpp_worlds = []
    for hands_ids, kitty_ids in world_specs:
        _, c = _build_state_pair(dealer, hands_ids, up_id, kitty_ids)
        cpp_worlds.append(c)
    weights = [0.4, 0.3, 0.2, 0.1]

    callback_solver = cpp.SubgameSolver(
        cpp_root, actor, list(cpp_worlds), list(weights), 5, 2,
        cpp_batch_value_fn_from_net(net, perspective=perspective), None)
    native_solver = cpp.SubgameSolver(
        cpp_root, actor, list(cpp_worlds), list(weights), 5, 2,
        net.cpp_module, perspective, None)

    cb_policy = callback_solver.root_policy()
    nn_policy = native_solver.root_policy()
    assert set(cb_policy) == set(nn_policy)
    for idx in cb_policy:
        assert cb_policy[idx] == pytest.approx(nn_policy[idx], abs=1e-6), (
            f"root_policy[{idx}] mismatch seed={seed} perspective={perspective}: "
            f"callback={cb_policy[idx]} native={nn_policy[idx]}")

    assert callback_solver.root_value() == pytest.approx(
        native_solver.root_value(), abs=1e-6), (
        f"root_value mismatch seed={seed} perspective={perspective}")


# --- sample_weighted_worlds: belief-weighted determinization (Part A of the
# planned net-native self-play work; see docs/rebel_design.md). No Python
# reference implementation exists (rebel/belief_model.py, the heuristic this
# replaces, was deleted on purpose this session) -- so instead of a
# cross-language differential test, this checks (1) properties that must
# hold regardless of what the net has learned, and (2) that the returned
# weights match an INDEPENDENT reimplementation of the same algorithm
# (replay the known pass-prefix, score it with net.policy()) applied to the
# SAME worlds the C++ side returned -- proving the weighting math itself is
# right without needing to control sample_determinization's RNG (which,
# per belief.h, is deliberately not cross-language-identical). -------------

def _reach_bidding_decision(seed, want_phase, want_bids_seen_gt_0=False, max_deals=50):
    """Deal hands and take PASS at every bidding decision until reaching
    `want_phase` (BidRound1 or BidRound2) with len(legal_actions()) > 1 --
    redealing (not retrying mid-hand) on a misdeal or a forced call, so the
    returned root is always a genuine decision node for its current_player.
    """
    rng = random.Random(seed)
    for _ in range(max_deals):
        s = cpp.EuchreState.new_hand(dealer=rng.randint(0, 3))
        deck = list(range(24))
        rng.shuffle(deck)
        s = s.deal_from_deck(deck)
        while True:
            if s.is_terminal():
                break
            if s.phase == want_phase and len(s.legal_actions()) > 1:
                if not want_bids_seen_gt_0 or s.bids_seen > 0:
                    return s
            pass_action = next((a for a in s.legal_actions()
                               if a.kind == cpp.ActionKind.Pass), None)
            if pass_action is None:
                break  # stick-the-dealer forced call; redeal
            s = s.apply(pass_action)
    raise RuntimeError(f"couldn't reach phase={want_phase} in {max_deals} deals")


def _independent_weights(root, worlds, net, weight_floor):
    """Python-side reimplementation of belief.cpp's sample_weighted_worlds
    weighting step (NOT its world sampling -- takes worlds as given), used
    to verify the C++ math independently."""
    pass_idx = cpp.Action.pass_().index()
    n_round1 = 4 if root.phase == cpp.Phase.BidRound2 else root.bids_seen
    n_round2 = root.bids_seen if root.phase == cpp.Phase.BidRound2 else 0
    steps = n_round1 + n_round2

    log_w = []
    for w in worlds:
        replay = cpp.EuchreState.new_hand(
            dealer=root.dealer, stick_the_dealer=root.stick_the_dealer,
            team0_score=root.team0_score, team1_score=root.team1_score
        ).deal_from(list(w.hands), w.up_card, list(w.kitty))
        lw = 0.0
        for _ in range(steps):
            obs = torch.from_numpy(np.asarray(
                cpp.observation_tensor(replay, replay.current_player))).unsqueeze(0)
            mask = torch.from_numpy(np.asarray(cpp.legal_mask(replay))).unsqueeze(0)
            with torch.no_grad():
                probs = net.policy(obs, mask)
            p = max(float(probs[0, pass_idx]), 1e-6)
            lw += math.log(p)
            replay = replay.apply(cpp.Action.pass_())
        log_w.append(lw)

    m = max(log_w)
    raw = [math.exp(l - m) for l in log_w]
    total = sum(raw)
    n = len(worlds)
    uniform = 1.0 / n
    floored = [max(x / total, weight_floor * uniform) for x in raw]
    total2 = sum(floored)
    return [x / total2 for x in floored]


@pytest.mark.parametrize("phase,bids_seen_gt_0", [
    (cpp.Phase.BidRound1, False),
    (cpp.Phase.BidRound1, True),
    (cpp.Phase.BidRound2, False),
    (cpp.Phase.BidRound2, True),
])
@pytest.mark.parametrize("seed", range(5))
def test_sample_weighted_worlds_matches_independent_reimplementation(seed, phase, bids_seen_gt_0):
    torch.manual_seed(seed + 1000)
    net = cpp.PolicyValueNet()
    net.eval()

    root = _reach_bidding_decision(seed, phase, bids_seen_gt_0)
    actor = root.current_player

    worlds, weights = cpp.sample_weighted_worlds(
        root, actor, 6, net.cpp_module, seed=seed + 2000, weight_floor=0.05)

    assert len(worlds) == len(weights) == 6
    assert sum(weights) == pytest.approx(1.0, abs=1e-9)

    expected = _independent_weights(root, worlds, net, weight_floor=0.05)
    for got, exp in zip(weights, expected):
        assert got == pytest.approx(exp, abs=1e-5), (
            f"weight mismatch seed={seed} phase={phase}: {weights} vs {expected}")


@pytest.mark.parametrize("seed", range(20))
def test_sample_weighted_worlds_respects_floor(seed):
    """No world's weight can fall below weight_floor * uniform, however
    confidently the net disagrees with it."""
    torch.manual_seed(seed)
    net = cpp.PolicyValueNet()
    net.eval()

    root = _reach_bidding_decision(seed, cpp.Phase.BidRound2, want_bids_seen_gt_0=True)
    actor = root.current_player
    n = 10
    floor = 0.1
    _, weights = cpp.sample_weighted_worlds(
        root, actor, n, net.cpp_module, seed=seed, weight_floor=floor)
    uniform = 1.0 / n
    for w in weights:
        assert w >= floor * uniform - 1e-9


def test_sample_weighted_worlds_uniform_when_nothing_observed():
    """First-to-act in round 1 (bids_seen == 0, not yet BidRound2): no bids
    to condition on, so weights must be exactly uniform."""
    torch.manual_seed(0)
    net = cpp.PolicyValueNet()
    net.eval()

    root = _reach_bidding_decision(0, cpp.Phase.BidRound1, want_bids_seen_gt_0=False)
    assert root.bids_seen == 0
    n = 7
    worlds, weights = cpp.sample_weighted_worlds(root, root.current_player, n, net.cpp_module, seed=0)
    assert len(worlds) == n
    assert weights == pytest.approx([1.0 / n] * n)


def test_sample_weighted_worlds_rejects_non_bidding_root():
    torch.manual_seed(0)
    net = cpp.PolicyValueNet()
    net.eval()

    py_st, cpp_st = _walk_to_play(0, dealer=0, min_cards_left=3)
    with pytest.raises(Exception):
        cpp.sample_weighted_worlds(cpp_st, cpp_st.current_player, 4, net.cpp_module, seed=0)


@pytest.mark.parametrize("seed", range(10))
def test_subgame_solver_belief_weighted_matches_uniform_when_untrained_ish(seed):
    """belief_weighted=True must still produce a complete, legal solve (not
    just a plausible one) -- root_policy covers exactly the actor's legal
    actions and sums to 1, for both a BidRound1 and BidRound2 root."""
    torch.manual_seed(seed)
    net = cpp.PolicyValueNet()
    net.eval()

    phase = cpp.Phase.BidRound1 if seed % 2 == 0 else cpp.Phase.BidRound2
    root = _reach_bidding_decision(seed, phase, want_bids_seen_gt_0=True)
    actor = root.current_player

    solver = cpp.SubgameSolver(root, actor, 6, 10, 2, net.cpp_module, actor, True, None, seed)
    solver.run()
    policy = solver.root_policy()
    legal_idxs = {a.index() for a in root.legal_actions()}
    assert set(policy) == legal_idxs
    assert sum(policy.values()) == pytest.approx(1.0, abs=1e-6)


# --- PolicyValueNet: torch::nn::Module port vs rebel.networks -------------
# torch::python::bind_module only gives the TOP-level module full Python
# nn.Module machinery, so state_dict_()/load_state_dict_() (cpp/network.h)
# walk named_parameters()/named_buffers() directly rather than going through
# Python's generic (child-recursing) load_state_dict -- see that file's
# docstring. Verified here: state_dict interop in both directions, and
# forward()/policy() bit-parity given identical weights.

def _real_observations(seed, n):
    """Real observation_tensor rows from actual random game states (not pure
    Gaussian noise) -- exercises the network on its actual input distribution
    (one-hot blocks, mostly-zero features)."""
    from euchre.infoset import observation_tensor as py_obs

    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        st = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(random.Random(rng.random()))
        steps = rng.randint(0, 20)
        for _ in range(steps):
            if st.is_terminal():
                break
            st = st.apply(rng.choice(st.legal_actions()))
        if st.is_terminal():
            continue
        rows.append(py_obs(st, st.current_player))
    arr = np.stack(rows[:n]).astype(np.float32)
    return torch.from_numpy(arr)


def test_policy_value_net_forward_matches_python():
    from rebel.networks import PolicyValueNet as PyNet

    torch.manual_seed(0)
    py_net = PyNet()
    py_net.eval()
    cpp_net = cpp.PolicyValueNet()
    cpp_net.eval()
    cpp_net.load_state_dict_(py_net.state_dict())

    obs = _real_observations(seed=1, n=17)
    with torch.no_grad():
        py_logits, py_value = py_net(obs)
        cpp_logits, cpp_value = cpp_net.forward(obs)

    assert py_logits.shape == cpp_logits.shape == (17, cpp.NUM_ACTIONS)
    assert torch.allclose(py_logits, cpp_logits, atol=1e-5), (
        f"logits max diff: {(py_logits - cpp_logits).abs().max().item()}")
    assert torch.allclose(py_value, cpp_value, atol=1e-5), (
        f"value max diff: {(py_value - cpp_value).abs().max().item()}")


def test_policy_value_net_policy_matches_python():
    from rebel.networks import PolicyValueNet as PyNet

    torch.manual_seed(2)
    py_net = PyNet()
    py_net.eval()
    cpp_net = cpp.PolicyValueNet()
    cpp_net.eval()
    cpp_net.load_state_dict_(py_net.state_dict())

    obs = _real_observations(seed=3, n=11)
    rng = random.Random(4)
    legal_mask = torch.zeros(11, cpp.NUM_ACTIONS, dtype=torch.bool)
    for i in range(11):
        idxs = rng.sample(range(cpp.NUM_ACTIONS), k=rng.randint(1, 5))
        legal_mask[i, idxs] = True

    with torch.no_grad():
        py_policy = py_net.policy(obs, legal_mask)
        cpp_policy = cpp_net.policy(obs, legal_mask)

    assert torch.allclose(py_policy, cpp_policy, atol=1e-6), (
        f"policy max diff: {(py_policy - cpp_policy).abs().max().item()}")


def test_policy_value_net_state_dict_roundtrip():
    """A checkpoint saved from the C++ module's own weights (state_dict_())
    loads correctly into a plain Python PolicyValueNet via its normal
    load_state_dict -- the checkpoint-interop direction that matters for
    Task 55 (a C++-trained checkpoint must work with existing Python
    tooling)."""
    from rebel.networks import PolicyValueNet as PyNet

    cpp_net = cpp.PolicyValueNet()  # cpp's own default init, not loaded from python
    cpp_net.eval()
    cpp_sd = cpp_net.state_dict_()

    py_net = PyNet()
    py_net.load_state_dict(cpp_sd)
    py_net.eval()

    obs = _real_observations(seed=5, n=9)
    with torch.no_grad():
        cpp_logits, cpp_value = cpp_net.forward(obs)
        py_logits, py_value = py_net(obs)
    assert torch.equal(py_logits, cpp_logits)
    assert torch.equal(py_value, cpp_value)


def test_policy_value_net_equivariant_under_relabeling():
    """Re-verify the suit-agnostic architecture's core guarantee (see
    tests/test_suit_symmetry.py) transferred exactly to the C++ module: exact
    equivariance under the 8-element color-preserving suit-relabeling group,
    using the C++ module's OWN native weight initialization (not weights
    copied from Python) -- a genuine structural check, not just a consequence
    of forward() being bit-identical for arbitrary tensors."""
    from test_suit_symmetry import (
        color_preserving_group, _relabel_state, _permute_logits, _sample_states,
    )
    from euchre.infoset import observation_tensor as py_obs

    torch.manual_seed(7)
    net = cpp.PolicyValueNet()
    net.eval()
    perms = color_preserving_group()
    for st in _sample_states(15, 42):
        p = st.current_player
        obs = torch.from_numpy(py_obs(st, p)).unsqueeze(0)
        with torch.no_grad():
            lg, val = net.forward(obs)
        lg = lg.squeeze(0).numpy()
        val = float(val)
        for pi in perms:
            robs = torch.from_numpy(py_obs(_relabel_state(st, pi), p)).unsqueeze(0)
            with torch.no_grad():
                rlg, rval = net.forward(robs)
            rlg = rlg.squeeze(0).numpy()
            assert np.allclose(rlg, _permute_logits(lg, pi), atol=1e-5)
            assert abs(float(rval) - val) < 1e-5


# --- actor-conditioned leaf values: both engines pick the same perspective -
# The leaf value used to come from the leaf's own acting player (the opening
# leader), which averages away the searching actor's hand -- see
# rebel/train_rebel.py's batch_value_fn_from_net docstring. The perspective is
# bound into the Python callable handed to the solver at construction, so
# nothing under cpp/ changed; these confirm both engines therefore agree.

@pytest.mark.parametrize("seed", range(8))
def test_leaf_value_perspective_matches_python(seed):
    """Applies one EXPLICIT shared discard rather than resolve_dealer_discard,
    so both engines are provably at the same state -- tied-optimal discards
    can break differently between engines (see
    test_resolve_dealer_discard_value_matches_python), which would show up
    here as a spurious mismatch whenever the perspective IS the dealer."""
    from rebel.train_rebel import (batch_value_fn_from_net,
                                   cpp_batch_value_fn_from_net)
    from rebel.networks import PolicyValueNet
    torch.manual_seed(seed)
    net = PolicyValueNet()
    py_st, cpp_st = _deal_both(seed, dealer=seed % 4)
    py_dd, cpp_dd = _apply_index(py_st, cpp_st,
                                 _order_up_index(py_st, alone=False))
    assert py_dd.phase == Phase.DEALER_DISCARD
    py_play, cpp_play = _apply_index(
        py_dd, cpp_dd, action_to_index(py_dd.legal_actions()[0]))
    assert py_play.phase == Phase.PLAY

    for p in range(4):
        py_v = batch_value_fn_from_net(net, perspective=p)([py_play])[0]
        cpp_v = cpp_batch_value_fn_from_net(net, perspective=p)([cpp_play])[0]
        assert py_v == pytest.approx(cpp_v, abs=1e-6), (
            f"perspective={p} seed={seed}: py={py_v} cpp={cpp_v}")


@pytest.mark.parametrize("seed", range(6))
def test_grounded_value_sample_matches_across_engines(seed):
    """_grounded_value_sample now captures the BIDDER's observation and
    varies which seat bids; both engine branches must stay in step.

    Compares cluster_key and value, not the raw observation: the two engines
    may pick different tied-optimal discards, and when the bidder IS the
    dealer that changes the observation while leaving the value identical (a
    tie means equal value by definition)."""
    from rebel.train_rebel import ReBeLTrainer
    py_t = ReBeLTrainer(engine="python", value_ground_frac=1.0, seed=seed)
    cpp_t = ReBeLTrainer(engine="cpp", value_ground_frac=1.0, seed=seed)
    for _ in range(12):
        a, b = py_t._grounded_value_sample(), cpp_t._grounded_value_sample()
        assert (a is None) == (b is None)
        if a is None:
            continue
        assert a.cluster_key == b.cluster_key
        assert a.value == pytest.approx(b.value, abs=1e-6)
        assert a.obs.shape == b.obs.shape == (PY_OBS_SIZE,)
