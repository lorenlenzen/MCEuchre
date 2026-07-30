"""Tests for scripts/train_pattern.py's --require constrained dealer.

The point of --require is reaching structural patterns that cluster selection
cannot express and rejection sampling cannot find: all four jacks spreads
across strength buckets 5-8 depending on the up-card suit, and occurs in
0.047% of hands per seat (~640,000 draws for 300 samples). So the cards are
placed rather than waited for -- which only helps if the constructed deal is
still a legal, properly varied deal, which is what these check.
"""

import collections
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from train_pattern import (apply_passes, choose_required,  # noqa: E402
                           constrained_deal, parse_require)

from euchre.cards import Card, Rank, Suit  # noqa: E402
from rebel.train_rebel import ReBeLTrainer  # noqa: E402


def _hand(state, seat, engine):
    if engine == "cpp":
        return [Card.from_id(c) for c in range(24) if (state.hands[seat] >> c) & 1]
    return list(state.hands[seat])


def test_parse_require_wildcards():
    assert [len(c) for _t, c in parse_require("JS")] == [1]
    assert [len(c) for _t, c in parse_require("J*")] == [4]
    assert [len(c) for _t, c in parse_require("*S")] == [6]
    assert [len(c) for _t, c in parse_require("**")] == [24]
    assert [len(c) for _t, c in parse_require("JS,J*,*S")] == [1, 4, 6]
    # case-insensitive and whitespace-tolerant
    assert [t for t, _c in parse_require(" js , j* ")] == ["JS", "J*"]


@pytest.mark.parametrize("bad", ["XX", "J", "JZ", "ZS", "JSS"])
def test_parse_require_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_require(bad)


def test_parse_require_rejects_more_patterns_than_a_hand_holds():
    with pytest.raises(ValueError):
        parse_require(",".join(["J*"] * 6))


def test_choose_required_picks_distinct_cards():
    rng = random.Random(0)
    got = choose_required(parse_require("J*,J*,J*"), rng)
    assert len(got) == 3 and len(set(got)) == 3
    assert all(c.rank == Rank.JACK for c in got)


def test_choose_required_detects_the_impossible():
    """Only four jacks exist, so five J* patterns can never be satisfied --
    this must fail fast rather than look like bad luck at --max-tries."""
    assert choose_required(parse_require(",".join(["J*"] * 5)),
                           random.Random(0)) is None


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_constrained_deal_is_a_legal_deal(engine):
    trainer = ReBeLTrainer(engine=engine, seed=0)
    rng = random.Random(7)
    patterns = parse_require("JS,JH,JD,JC")
    seen = 0
    for _ in range(40):
        drawn = constrained_deal(trainer, patterns, "bid1", rng)
        assert drawn is not None
        state = apply_passes(trainer, *drawn)
        if state is None:
            continue
        seen += 1
        actor = state.current_player
        hand = _hand(state, actor, engine)
        assert len(hand) == 5
        assert sum(c.rank == Rank.JACK for c in hand) == 4

        dealt = [c for seat in range(4) for c in _hand(state, seat, engine)]
        assert len(dealt) == 20, "four hands of five"
        assert len(set(dealt)) == 20, "a card was dealt twice"
        up = Card.from_id(state.up_card) if engine == "cpp" else state.up_card
        assert up not in dealt, "the up-card is also in someone's hand"
    assert seen > 0


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_constrained_deal_varies_everything_it_should(engine):
    """Only the required cards are pinned. If the seat, up-card suit or score
    were fixed too, this would be one position dressed up as many -- exactly
    the memorization the script exists to avoid."""
    trainer = ReBeLTrainer(engine=engine, seed=0)
    rng = random.Random(7)
    patterns = parse_require("JS,JH,JD,JC")
    seats, up_suits, scores = collections.Counter(), set(), set()
    for _ in range(120):
        drawn = constrained_deal(trainer, patterns, "bid1", rng)
        state = apply_passes(trainer, *drawn)
        if state is None:
            continue
        seats[(state.current_player - state.dealer) % 4] += 1
        up = Card.from_id(state.up_card) if engine == "cpp" else state.up_card
        up_suits.add(up.suit)
        scores.add((state.team0_score, state.team1_score))
    assert set(seats) == {0, 1, 2, 3}, f"actor seat not varied: {dict(seats)}"
    assert len(up_suits) == 4, f"up-card suit not varied: {up_suits}"
    # equity_model is None here, so scores stay 0-0; just assert it's coherent
    assert scores == {(0, 0)}


def test_constrained_deal_reaches_round_two():
    trainer = ReBeLTrainer(engine="python", seed=0)
    rng = random.Random(3)
    patterns = parse_require("A*,K*")
    from euchre.game import Phase
    reached = 0
    for _ in range(40):
        drawn = constrained_deal(trainer, patterns, "bid2", rng)
        state = apply_passes(trainer, *drawn)
        if state is None:
            continue
        assert state.phase == Phase.BID_ROUND_2
        hand = state.hands[state.current_player]
        assert any(c.rank == Rank.ACE for c in hand)
        assert any(c.rank == Rank.KING for c in hand)
        reached += 1
    assert reached > 0
