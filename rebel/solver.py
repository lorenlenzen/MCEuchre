"""Perfect-information ("double-dummy") solver for the Euchre play phase.

Given a fully-specified state (all four hands known), compute the value under
optimal play by both teams, where value is ``team0_points - team1_points`` for
the hand. This is exact game-tree search with alpha-beta pruning, a
transposition table, and move ordering, which makes five-card double-dummy
solves fast enough to sit inside PIMC's per-decision world loop.

It is a *team* game rather than a strictly alternating one (partners share an
objective), but it is still one scalar that team 0 maximizes and team 1
minimizes, so alpha-beta applies.

The solver is the engine behind PIMC search (`rebel/pimc.py`): sample many
worlds consistent with what a player knows, solve each, and aggregate.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from euchre.actions import Play, Action
from euchre.cards import Card, card_strength, effective_suit
from euchre.game import EuchreState, Phase, team_of

_INF = 10 ** 9
_LOWER, _EXACT, _UPPER = 0, 1, 2  # transposition-table bound flags


def _key(state: EuchreState) -> Tuple:
    hands = tuple(tuple(sorted(c.id for c in state.hands[p])) for p in range(4))
    trick = tuple((seat, c.id) for seat, c in state.current_trick)
    return (state.current_player, trick, hands,
            state.tricks_won[0], state.tricks_won[1])


def _ordered_plays(state: EuchreState, player: int) -> List[Card]:
    """Legal plays, de-duplicated by equivalence and ordered strongest-first.

    Two of a player's cards in the same effective suit are strategically
    equivalent when they form a consecutive run among the still-unplayed cards
    (no opponent card ranks between them): playing either yields the same
    double-dummy value, so we keep a single representative. This is the classic
    double-dummy move reduction and it slashes the branching of the first
    tricks. The remaining moves are then ordered strongest-first for pruning.
    """
    plays = state._legal_plays(player)
    if len(plays) <= 1:
        return plays
    trump = state.trump
    myset = set(state.hands[player])
    playset = set(plays)

    # Bucket cards by effective suit, ordered strongest-first. Cards already
    # played to the *current* trick must be included: they are not playable but
    # they still separate ranks (a card sitting in the trick can rank between
    # two of my cards, so playing the higher wins the trick while the lower
    # does not -- those two are NOT equivalent).
    buckets: Dict[object, List[Card]] = defaultdict(list)
    for p in range(4):
        for c in state.hands[p]:
            buckets[effective_suit(c, trump)].append(c)
    for _seat, c in state.current_trick:
        buckets[effective_suit(c, trump)].append(c)

    keep: List[Card] = []
    for suit, cards in buckets.items():
        cards.sort(key=lambda c: card_strength(c, trump, suit), reverse=True)
        i, n = 0, len(cards)
        while i < n:
            if cards[i] in myset:
                run = []
                while i < n and cards[i] in myset:
                    run.append(cards[i])
                    i += 1
                legal_in_run = [c for c in run if c in playset]
                if legal_in_run:
                    keep.append(legal_in_run[-1])  # lowest legal representative
            else:
                i += 1

    reduced = keep if keep else plays
    if state.current_trick:
        led = effective_suit(state.current_trick[0][1], trump)
    else:
        led = None
    return sorted(
        reduced,
        key=lambda c: card_strength(c, trump, led if led is not None else c.suit),
        reverse=True,
    )


def _ab(state: EuchreState, alpha: int, beta: int,
        memo: Dict[Tuple, Tuple[int, int]]) -> int:
    if state.is_terminal():
        r = state.returns()
        return r[0] - r[1]

    a0, b0 = alpha, beta
    key = _key(state)
    entry = memo.get(key)
    if entry is not None:
        val, flag = entry
        if flag == _EXACT:
            return val
        if flag == _LOWER:
            alpha = max(alpha, val)
        else:  # _UPPER
            beta = min(beta, val)
        if alpha >= beta:
            return val

    player = state.current_player
    maximizing = team_of(player) == 0
    plays = _ordered_plays(state, player)

    if maximizing:
        value = -_INF
        for c in plays:
            value = max(value, _ab(state.apply(Play(c)), alpha, beta, memo))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = _INF
        for c in plays:
            value = min(value, _ab(state.apply(Play(c)), alpha, beta, memo))
            beta = min(beta, value)
            if alpha >= beta:
                break

    if value <= a0:
        flag = _UPPER
    elif value >= b0:
        flag = _LOWER
    else:
        flag = _EXACT
    memo[key] = (value, flag)
    return value


def solve_value(state: EuchreState,
                memo: Optional[Dict[Tuple, Tuple[int, int]]] = None) -> int:
    """Optimal ``team0 - team1`` point differential from ``state`` (PLAY phase).

    ``memo`` may be shared across sibling calls to reuse transpositions.
    """
    if state.is_terminal():
        r = state.returns()
        return r[0] - r[1]
    assert state.phase == Phase.PLAY, "solver runs on the play phase"
    if memo is None:
        memo = {}
    return _ab(state, -_INF, _INF, memo)


def action_values(state: EuchreState,
                  memo: Optional[Dict[Tuple, Tuple[int, int]]] = None
                  ) -> Dict[Action, int]:
    """Optimal continuation value (team0 - team1) for each legal play.

    Each child is solved with a full window so the returned values are exact;
    a shared transposition table keeps sibling solves cheap.
    """
    assert state.phase == Phase.PLAY
    if memo is None:
        memo = {}
    out: Dict[Action, int] = {}
    for c in _ordered_plays(state, state.current_player):
        out[Play(c)] = _ab(state.apply(Play(c)), -_INF, _INF, memo)
    return out


def best_play(state: EuchreState,
              memo: Optional[Dict[Tuple, Tuple[int, int]]] = None) -> Action:
    """The optimal play for the side on move (from its own team's view)."""
    maximizing = team_of(state.current_player) == 0
    values = action_values(state, memo)
    return (max if maximizing else min)(values, key=lambda a: values[a])
