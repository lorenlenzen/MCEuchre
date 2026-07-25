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
