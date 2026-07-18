"""Consistency tests for determinization / belief sampling."""

import random

from euchre.cards import DECK, effective_suit
from euchre.game import EuchreState, Phase
from euchre.actions import OrderUp, Discard, Pass, Call
from rebel.public_belief_state import sample_determinization, known_voids


def _play_to_midhand(seed: int) -> EuchreState:
    """Deal, order up, and play a few cards to reach a mid-play state."""
    rng = random.Random(seed)
    s = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
    # Reach a trump by any means: everyone in a simple loop.
    from rebel.evaluate import RuleBasedAgent
    agent = RuleBasedAgent()
    steps = 0
    while (s.phase != Phase.PLAY and not s.is_terminal()) or (
            s.phase == Phase.PLAY and len(s.completed_tricks) < 2):
        if s.is_terminal():
            break
        s = s.apply(agent.act(s, rng))
        steps += 1
        if steps > 60:
            break
    return s


def _assert_consistent(state: EuchreState, sample: EuchreState, player: int):
    # Player's own hand preserved.
    assert sorted(c.id for c in sample.hands[player]) == \
        sorted(c.id for c in state.hands[player])
    # Hand sizes preserved for everyone.
    for p in range(4):
        assert len(sample.hands[p]) == len(state.hands[p])
    assert len(sample.kitty) == len(state.kitty)
    # Already-played cards were removed from hands (not re-dealt).
    played = set()
    for _w, plays in state.completed_tricks:
        played.update(c for _, c in plays)
    played.update(c for _, c in state.current_trick)
    for p in range(4):
        assert not (set(sample.hands[p]) & played)
    # Every card is accounted for exactly once. Before a pickup the up-card
    # sits in its own public slot (not in any hand or the kitty).
    from rebel.public_belief_state import _pickup_happened
    all_cards = ([c for p in range(4) for c in sample.hands[p]]
                 + list(sample.kitty) + list(played))
    if sample.up_card is not None and not _pickup_happened(sample):
        all_cards.append(sample.up_card)
    assert len(all_cards) == 24
    assert set(all_cards) == set(DECK)


def test_determinization_consistent_many_states():
    for seed in range(60):
        state = _play_to_midhand(seed)
        if state.is_terminal():
            continue
        player = state.current_player if state.current_player >= 0 else 0
        rng = random.Random(1000 + seed)
        for _ in range(5):
            sample = sample_determinization(state, player, rng)
            _assert_consistent(state, sample, player)


def test_determinization_respects_voids():
    """A sampled opponent must not hold a suit they've shown void of."""
    for seed in range(40):
        state = _play_to_midhand(seed)
        if state.is_terminal() or state.trump is None:
            continue
        player = state.current_player
        voids = known_voids(state)
        rng = random.Random(seed)
        sample = sample_determinization(state, player, rng)
        for p in range(4):
            if p == player:
                continue
            for c in sample.hands[p]:
                assert effective_suit(c, state.trump) not in voids[p], (
                    f"seat {p} void {voids[p]} but holds {c}")


def test_determinization_at_start_of_play():
    rng = random.Random(3)
    s = EuchreState.new_hand(dealer=0).deal(rng)
    s = s.apply(OrderUp(alone=False))
    # dealer discards
    s = s.apply(Discard(s.hands[s.dealer][0]))
    assert s.phase == Phase.PLAY
    player = s.current_player
    sample = sample_determinization(s, player, random.Random(9))
    _assert_consistent(s, sample, player)
