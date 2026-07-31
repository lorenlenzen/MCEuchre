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


# --- --void / --score constrained deals -------------------------------------

def test_parse_voids_and_score():
    from train_pattern import parse_score, parse_voids
    assert parse_voids("up") == ["up"]
    assert parse_voids("trump") == ["up"]        # alias
    assert parse_voids("up,H") == ["up", "H"]
    assert parse_score("9,6") == (9, 6)
    for bad in ("X", "up,X", ""):
        if bad == "":
            assert parse_voids(bad) == []
            continue
        with pytest.raises(ValueError):
            parse_voids(bad)
    for bad in ("9", "9,6,1", "a,b"):
        with pytest.raises(ValueError):
            parse_score(bad)


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_void_up_means_zero_effective_trump(engine):
    """The whole point of `--void up` is a TRUMP void, so it has to exclude
    the left bower as well -- a hand holding the off-suit jack of the
    up-card's colour is not trump-void, and training it as though it were
    would teach the donation on hands that can actually take a trick."""
    from euchre.cards import effective_suit
    from train_pattern import (apply_passes, constrained_deal, parse_require,
                               parse_voids)
    trainer = ReBeLTrainer(engine=engine, seed=0)
    rng = random.Random(5)
    seen = 0
    for _ in range(60):
        drawn = constrained_deal(trainer, parse_require("**"), "bid1", rng,
                                 voids=parse_voids("up"))
        assert drawn is not None
        state = apply_passes(trainer, *drawn)
        if state is None:
            continue
        seen += 1
        actor = state.current_player
        hand = _hand(state, actor, engine)
        up = Card.from_id(state.up_card) if engine == "cpp" else state.up_card
        trump = [c for c in hand if effective_suit(c, up.suit) == up.suit]
        assert trump == [], [str(c) for c in trump]
    assert seen > 0


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_score_is_pinned_from_the_actors_side(engine):
    """--score is MINE,THEIRS for whoever acts -- so it must land on team0 or
    team1 depending on the actor's team, not always team0."""
    from euchre.game import team_of
    from train_pattern import (apply_passes, constrained_deal, parse_require,
                               parse_score)
    trainer = ReBeLTrainer(engine=engine, seed=0)
    rng = random.Random(5)
    teams = set()
    for _ in range(60):
        drawn = constrained_deal(trainer, parse_require("**"), "bid1", rng,
                                 score=parse_score("9,6"))
        state = apply_passes(trainer, *drawn)
        if state is None:
            continue
        actor = state.current_player
        mine, theirs = ((state.team0_score, state.team1_score)
                        if team_of(actor) == 0
                        else (state.team1_score, state.team0_score))
        assert (mine, theirs) == (9, 6)
        teams.add(team_of(actor))
    assert teams == {0, 1}, "actor should land on both teams across draws"


def test_void_and_require_can_be_mutually_unsatisfiable():
    """Requiring a club while demanding a club void must fail the draw rather
    than silently dropping one of the two constraints."""
    from train_pattern import constrained_deal, parse_require, parse_voids
    trainer = ReBeLTrainer(seed=0)
    rng = random.Random(0)
    # every up-card that makes JC/JS trump-relevant still leaves suits where
    # 'C*' is legal, so this asks for the strictly impossible: all four clubs
    # required AND void in clubs.
    pats = parse_require("*C,*C,*C,*C")
    fails = sum(constrained_deal(trainer, pats, "bid1", rng,
                                 voids=parse_voids("C")) is None
                for _ in range(20))
    assert fails == 20, "a contradictory require/void pair must never deal"
