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
                           constrained_deal, parse_require, relsuit_map,
                           resolve_patterns)

from euchre.cards import Card, Rank, Suit  # noqa: E402
from rebel.train_rebel import ReBeLTrainer  # noqa: E402


def _hand(state, seat, engine):
    if engine == "cpp":
        return [Card.from_id(c) for c in range(24) if (state.hands[seat] >> c) & 1]
    return list(state.hands[seat])


def _resolved(text, up_suit=Suit.CLUBS, seed=0):
    """--require text -> [(token, candidate_cards), ...] for a fixed up-card
    suit -- the shape choose_required/tests need, without going through a
    full constrained_deal. Any up_suit works for counting candidates: each
    U/N/G/g letter always names exactly one physical suit (6 cards)
    regardless of which one is picked."""
    return resolve_patterns(parse_require(text),
                            relsuit_map(up_suit, random.Random(seed)))


def test_parse_require_wildcards():
    assert [len(c) for _t, c in _resolved("JU")] == [1]
    assert [len(c) for _t, c in _resolved("J*")] == [4]
    assert [len(c) for _t, c in _resolved("*U")] == [6]
    assert [len(c) for _t, c in _resolved("**")] == [24]
    assert [len(c) for _t, c in _resolved("JU,J*,*U")] == [1, 4, 6]
    # case-insensitive on rank and U/N, whitespace-tolerant; G/g stay
    # case-distinct (the two off-color suits are otherwise indistinguishable)
    assert [t for t, _r, _s in parse_require(" ju , j* , jg ")] == \
        ["JU", "J*", "Jg"]


@pytest.mark.parametrize("bad", ["XX", "J", "JZ", "ZU", "JUU", "JS", "JC"])
def test_parse_require_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_require(bad)


def test_parse_require_rejects_more_patterns_than_a_hand_holds():
    with pytest.raises(ValueError):
        parse_require(",".join(["J*"] * 6))


def test_choose_required_picks_distinct_cards():
    rng = random.Random(0)
    got = choose_required(_resolved("J*,J*,J*"), rng)
    assert len(got) == 3 and len(set(got)) == 3
    assert all(c.rank == Rank.JACK for c in got)


def test_choose_required_detects_the_impossible():
    """Only four jacks exist, so five J* patterns can never be satisfied --
    this must fail fast rather than look like bad luck at --max-tries."""
    assert choose_required(_resolved(",".join(["J*"] * 5)),
                           random.Random(0)) is None


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_constrained_deal_is_a_legal_deal(engine):
    trainer = ReBeLTrainer(engine=engine, seed=0)
    rng = random.Random(7)
    patterns = parse_require("JU,JN,JG,Jg")
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
    patterns = parse_require("JU,JN,JG,Jg")
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


# --- --void-up / --score constrained deals -----------------------------------

def test_parse_score():
    from train_pattern import parse_score
    assert parse_score("9,6") == (9, 6)
    for bad in ("9", "9,6,1", "a,b"):
        with pytest.raises(ValueError):
            parse_score(bad)


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_void_up_means_zero_effective_trump(engine):
    """The whole point of `--void-up` is a TRUMP void, so it has to exclude
    the left bower as well -- a hand holding the off-suit jack of the
    up-card's colour is not trump-void, and training it as though it were
    would teach the donation on hands that can actually take a trick."""
    from euchre.cards import effective_suit
    from train_pattern import apply_passes, constrained_deal, parse_require
    trainer = ReBeLTrainer(engine=engine, seed=0)
    rng = random.Random(5)
    seen = 0
    for _ in range(60):
        drawn = constrained_deal(trainer, parse_require("**"), "bid1", rng,
                                 void_up=True)
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
    """Requiring cards of the up-card's own suit while demanding the hand be
    void-in-up must fail the draw rather than silently dropping one of the
    two constraints."""
    from train_pattern import constrained_deal, parse_require
    trainer = ReBeLTrainer(seed=0)
    rng = random.Random(0)
    # *U resolves to the up-card's own suit -- requiring four of its cards
    # while also demanding the hand hold none of that (effective) suit is
    # strictly impossible.
    pats = parse_require("*U,*U,*U,*U")
    fails = sum(constrained_deal(trainer, pats, "bid1", rng,
                                 void_up=True) is None
                for _ in range(20))
    assert fails == 20, "a contradictory require/void pair must never deal"
