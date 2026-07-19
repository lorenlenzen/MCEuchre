"""Correlated team play (TMECor) for small imperfect-information team games.

Independent per-player CFR (what the main pipeline uses) finds a Nash
equilibrium, but for a *team* -- players who share a reward yet cannot see each
other's cards -- Nash is generally not optimal. The team can do strictly better
with a **Team-Maxmin Equilibrium with Correlation (TMECor)**: before play, the
partners agree on a *correlated* joint plan (a shared random signal), then each
plays their part from their own information alone. That shared randomness lets
them coordinate in ways independent strategies cannot, which is the formal basis
of signalling conventions.

Computation (Celli & Gatti; von Stengel & Koller): a team playing with
correlation is a single **coordinator** whose pure strategies are the team's
*joint* deterministic plans (a "prescription" -- one action per team infoset).
TMECor is then a Nash equilibrium of the two-player zero-sum game
coordinator-A vs coordinator-B, whose payoff matrix we solve with regret
matching (no LP dependency). See ``docs/rebel_design.md`` for the scaling story.

Scope, honestly: enumerating joint pure strategies is exponential in the number
of team infosets, so this is exact only for *small* subgames (an endgame, or a
toy). It is a validated reference for the technique -- like ``range_cfr`` -- not
a full-game solver. It is wired into nothing; it is here to be correct and to
demonstrate the gap between correlated and independent team play.

The game is supplied through a tiny duck-typed interface (see ``TeamGame``),
so the same solver runs on a canonical signalling game (known answer) and on
Euchre endgames.
"""

from __future__ import annotations

from itertools import product
from math import prod
from typing import Dict, Hashable, List, Tuple

import numpy as np


class TeamGame:
    """Interface a game must provide. ``team`` is the set of player ids on the
    team we solve for; everyone else is the adversary. Payoff is from the
    team's perspective (team maximizes, adversary minimizes)."""

    team: frozenset

    def worlds(self) -> List[Tuple[object, float]]:
        """Chance outcomes: (root_state, probability)."""
        raise NotImplementedError

    def is_terminal(self, s) -> bool: raise NotImplementedError
    def current_player(self, s) -> int: raise NotImplementedError
    def infoset(self, s, player: int) -> Hashable: raise NotImplementedError
    def legal_actions(self, s) -> list: raise NotImplementedError
    def apply(self, s, a): raise NotImplementedError
    def payoff(self, s) -> float: raise NotImplementedError


# -- infoset collection ------------------------------------------------------

def collect_infosets(game: TeamGame) -> Dict[int, Dict[Hashable, list]]:
    """Walk every world's tree (branching all actions) to enumerate, per
    player, each information set and its legal actions."""
    info: Dict[int, Dict[Hashable, list]] = {}

    def rec(s):
        if game.is_terminal(s):
            return
        p = game.current_player(s)
        key = game.infoset(s, p)
        acts = game.legal_actions(s)
        info.setdefault(p, {})
        if key not in info[p]:
            info[p][key] = list(acts)
        for a in acts:
            rec(game.apply(s, a))

    for root, _w in game.worlds():
        rec(root)
    return info


def _enumerate_pures(info: Dict[int, Dict[Hashable, list]],
                     players: List[int], cap: int) -> List[Dict]:
    """All deterministic prescriptions for ``players`` (one action per
    infoset). A prescription is a dict {(player, infoset): action}."""
    keys = [(p, k) for p in sorted(players) for k in info.get(p, {})]
    action_lists = [info[p][k] for (p, k) in keys]
    sizes = [len(a) for a in action_lists]
    total = prod(sizes) if sizes else 1
    if total > cap:
        raise ValueError(f"team pure-strategy space too large: {total} > {cap}")
    pures = []
    for combo in product(*[range(s) for s in sizes]):
        strat = {keys[i]: action_lists[i][combo[i]] for i in range(len(keys))}
        pures.append(strat)
    return pures


# -- payoff matrix and zero-sum solve ----------------------------------------

def _playout(game: TeamGame, root, a_strat: Dict, b_strat: Dict,
             team: frozenset) -> float:
    s = root
    while not game.is_terminal(s):
        p = game.current_player(s)
        key = (p, game.infoset(s, p))
        a = (a_strat if p in team else b_strat)[key]
        s = game.apply(s, a)
    return game.payoff(s)


def _payoff_matrix(game: TeamGame, A: List[Dict], B: List[Dict],
                   team: frozenset) -> np.ndarray:
    W = game.worlds()
    M = np.zeros((len(A), len(B)))
    for i, ap in enumerate(A):
        for j, bp in enumerate(B):
            M[i, j] = sum(w * _playout(game, root, ap, bp, team)
                          for root, w in W)
    return M


def _regret_match(r: np.ndarray) -> np.ndarray:
    pos = np.maximum(r, 0.0)
    s = pos.sum()
    if s > 0:
        return pos / s
    return np.full(len(r), 1.0 / len(r))


def solve_zero_sum(M: np.ndarray, iters: int = 8000
                   ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Value and average strategies for the zero-sum game where the row player
    maximizes ``M`` and the column player minimizes it (regret-matching
    self-play converges to the minimax)."""
    nA, nB = M.shape
    rA = np.zeros(nA)
    rB = np.zeros(nB)
    SA = np.zeros(nA)
    SB = np.zeros(nB)
    for _ in range(iters):
        a = _regret_match(rA)
        b = _regret_match(rB)
        SA += a
        SB += b
        uA = M @ b               # row utilities (maximize)
        rA += uA - a @ uA
        uB = -(a @ M)            # column utilities (maximize -M)
        rB += uB - b @ uB
    a = SA / SA.sum()
    b = SB / SB.sum()
    return float(a @ M @ b), a, b


# -- top-level ---------------------------------------------------------------

def tmecor_value(game: TeamGame, cap: int = 50000, iters: int = 8000
                 ) -> Tuple[float, List[Dict], np.ndarray]:
    """TMECor value for ``game.team`` and the team's correlated strategy.

    Returns (value, team_pure_strategies, distribution_over_them).
    """
    info = collect_infosets(game)
    team = frozenset(game.team)
    adversary = [p for p in info if p not in team]
    A = _enumerate_pures(info, list(team), cap)
    B = _enumerate_pures(info, adversary, cap)
    value, a_dist, _b = solve_zero_sum(_payoff_matrix(game, A, B, team), iters)
    return value, A, a_dist


def independent_nash_value(game: TeamGame, iters: int = 4000) -> float:
    """Team value under an independent (per-player) Nash equilibrium, via
    vanilla CFR. Compared against ``tmecor_value`` this exposes the gap that
    correlation buys."""
    team = frozenset(game.team)
    regret: Dict = {}
    strat_sum: Dict = {}

    def node(p, key, acts):
        k = (p, key)
        if k not in regret:
            regret[k] = np.zeros(len(acts))
            strat_sum[k] = np.zeros(len(acts))
        return regret[k], strat_sum[k]

    def cfr(s, reaches: Dict) -> float:
        if game.is_terminal(s):
            return game.payoff(s)
        p = game.current_player(s)
        key = game.infoset(s, p)
        acts = game.legal_actions(s)
        r, ss = node(p, key, acts)
        strat = _regret_match(r)
        child = np.zeros(len(acts))
        util = 0.0
        for i, a in enumerate(acts):
            nr = dict(reaches)
            nr[p] = reaches[p] * strat[i]
            child[i] = cfr(game.apply(s, a), nr)
            util += strat[i] * child[i]
        sign = 1.0 if p in team else -1.0
        cf = 1.0
        for q, rp in reaches.items():
            if q != p:
                cf *= rp
        r += sign * cf * (child - util)
        ss += reaches[p] * strat
        return util

    players = list(collect_infosets(game).keys())
    W = game.worlds()
    for _ in range(iters):
        for root, w in W:
            reaches = {pl: 1.0 for pl in players}
            reaches["chance"] = w
            cfr(root, reaches)

    avg = {k: (ss / ss.sum() if ss.sum() > 0
               else np.full(len(ss), 1.0 / len(ss)))
           for k, ss in strat_sum.items()}

    def ev(s) -> float:
        if game.is_terminal(s):
            return game.payoff(s)
        p = game.current_player(s)
        key = game.infoset(s, p)
        acts = game.legal_actions(s)
        a = avg.get((p, key))
        if a is None:
            a = np.full(len(acts), 1.0 / len(acts))
        return sum(a[i] * ev(game.apply(s, acts[i])) for i in range(len(acts)))

    return sum(w * ev(root) for root, w in W)
