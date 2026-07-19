"""Evaluation ladder: a round-robin tournament with Elo ratings.

Strength only means something relative to a field, so this runs every agent
against every other, fits **Bradley-Terry** strengths from the pairwise results
(the maximum-likelihood model behind Elo), and reports a leaderboard with Elo
ratings, win rates, average margins, and bootstrap confidence intervals.

Seat bias is cancelled by alternating orientation within each match (each agent
plays team 0 on half the hands and team 1 on the other half), and the dealer
rotates hand to hand, so every seat deals equally.

On measuring "expert": exact exploitability (a full best response) is
intractable here, so the ladder measures *relative* strength. The practical
proxies are Elo separation from a strong searcher (PIMC in the pool) and the
average point margin against the field; a local-best-response lower bound is
the natural future extension (see docs/rebel_design.md).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from euchre.game import EuchreState
from .evaluate import Agent, play_hand

AgentFactory = Callable[[], Agent]


@dataclass
class AgentSpec:
    name: str
    factory: AgentFactory


@dataclass
class MatchResult:
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    margins: List[int] = field(default_factory=list)  # per-hand (a_pts - b_pts)

    @property
    def hands(self) -> int:
        return self.a_wins + self.b_wins + self.draws


def play_match(a_factory: AgentFactory, b_factory: AgentFactory,
               hands: int, seed: int = 0,
               stick_the_dealer: bool = False) -> MatchResult:
    """Play ``hands`` hands between two agents, alternating which team each
    plays to cancel seat bias. Results are from A's perspective."""
    rng = random.Random(seed)
    a, b = a_factory(), b_factory()
    res = MatchResult()
    for h in range(hands):
        # Decouple orientation from the dealer seat: flip which team A plays
        # every 4 hands while the dealer rotates every hand, so each orientation
        # sees every dealer equally and neither agent is always the dealing team
        # (the dealer gets the up-card, a real edge).
        a_team0 = (h // 4) % 2 == 0
        agents = [a, b, a, b] if a_team0 else [b, a, b, a]
        r0, r1 = play_hand(agents, dealer=h % 4, rng=rng,
                           stick_the_dealer=stick_the_dealer)
        a_pts, b_pts = (r0, r1) if a_team0 else (r1, r0)
        res.margins.append(a_pts - b_pts)
        if a_pts > b_pts:
            res.a_wins += 1
        elif b_pts > a_pts:
            res.b_wins += 1
        else:
            res.draws += 1
    return res


def round_robin(specs: List[AgentSpec], hands_per_pair: int = 200,
                seed: int = 0, stick_the_dealer: bool = False
                ) -> Dict[Tuple[int, int], MatchResult]:
    """Every agent vs every other. Keyed by (i, j) with i < j."""
    results: Dict[Tuple[int, int], MatchResult] = {}
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            results[(i, j)] = play_match(
                specs[i].factory, specs[j].factory, hands_per_pair,
                seed=seed + 1000 * i + j, stick_the_dealer=stick_the_dealer)
    return results


# -- Bradley-Terry / Elo -----------------------------------------------------

def bradley_terry(n: int, wins: List[float], games: List[List[float]],
                  iters: int = 1000, tol: float = 1e-9) -> List[float]:
    """MM (minorization-maximization) fit of Bradley-Terry strengths.

    ``wins[i]`` counts i's wins (draws as half); ``games[i][j]`` is the number
    of games between i and j. Returns strengths normalized to geometric mean 1.
    """
    gamma = [1.0] * n
    for _ in range(iters):
        new = [0.0] * n
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if j == i or games[i][j] == 0:
                    continue
                denom += games[i][j] / (gamma[i] + gamma[j])
            new[i] = wins[i] / denom if denom > 0 else gamma[i]
        # normalize to geometric mean 1 for identifiability
        logmean = sum(math.log(max(g, 1e-12)) for g in new) / n
        scale = math.exp(logmean)
        new = [g / scale for g in new]
        if max(abs(new[i] - gamma[i]) for i in range(n)) < tol:
            gamma = new
            break
        gamma = new
    return gamma


def _aggregate(n: int, results: Dict[Tuple[int, int], MatchResult]
               ) -> Tuple[List[float], List[List[float]]]:
    """Wins (draws as half, with light smoothing) and game counts. The
    smoothing (one virtual drawn game per pair) keeps a winless or perfect
    agent from mapping to +/- infinite Elo."""
    wins = [0.5] * n  # half a virtual win each, paired below
    games = [[0.0] * n for _ in range(n)]
    for (i, j), r in results.items():
        wins[i] += r.a_wins + 0.5 * r.draws + 0.5   # +0.5 virtual draw
        wins[j] += r.b_wins + 0.5 * r.draws + 0.5
        g = r.hands + 1                               # +1 virtual game
        games[i][j] += g
        games[j][i] += g
    return wins, games


def _elo(gamma: List[float], anchor: float = 1500.0) -> List[float]:
    # geometric mean of gamma is 1, so mean Elo == anchor.
    return [anchor + 400.0 * math.log10(max(g, 1e-12)) for g in gamma]


@dataclass
class Standing:
    name: str
    elo: float
    elo_lo: Optional[float]
    elo_hi: Optional[float]
    win_rate: float
    avg_margin: float
    games: int


def _standings(specs, results, elos) -> List[Standing]:
    n = len(specs)
    wins = [0.0] * n
    margin = [0.0] * n
    games = [0] * n
    for (i, j), r in results.items():
        wins[i] += r.a_wins + 0.5 * r.draws
        wins[j] += r.b_wins + 0.5 * r.draws
        margin[i] += sum(r.margins)
        margin[j] -= sum(r.margins)
        games[i] += r.hands
        games[j] += r.hands
    out = []
    for i, spec in enumerate(specs):
        g = max(games[i], 1)
        out.append(Standing(
            name=spec.name, elo=elos[i], elo_lo=None, elo_hi=None,
            win_rate=wins[i] / g, avg_margin=margin[i] / g, games=games[i]))
    return out


def evaluate_ladder(specs: List[AgentSpec], hands_per_pair: int = 200,
                    seed: int = 0, bootstrap: int = 0,
                    stick_the_dealer: bool = False) -> List[Standing]:
    """Run the tournament and return standings sorted by Elo (descending).

    ``bootstrap`` > 0 adds Elo confidence intervals by resampling each pair's
    hands that many times (2.5/97.5 percentiles).
    """
    n = len(specs)
    results = round_robin(specs, hands_per_pair, seed, stick_the_dealer)
    wins, games = _aggregate(n, results)
    elos = _elo(bradley_terry(n, wins, games))
    standings = _standings(specs, results, elos)

    if bootstrap > 0:
        samples = [[] for _ in range(n)]
        rng = random.Random(seed + 7)
        keys = list(results)
        for _ in range(bootstrap):
            bw, bg = [0.5] * n, [[0.0] * n for _ in range(n)]
            for (i, j) in keys:
                margins = results[(i, j)].margins
                m = len(margins)
                a_w = d = 0
                for _k in range(m):
                    x = margins[rng.randrange(m)]
                    if x > 0:
                        a_w += 1
                    elif x == 0:
                        d += 1
                b_w = m - a_w - d
                bw[i] += a_w + 0.5 * d + 0.5
                bw[j] += b_w + 0.5 * d + 0.5
                bg[i][j] += m + 1
                bg[j][i] += m + 1
            be = _elo(bradley_terry(n, bw, bg, iters=300))
            for i in range(n):
                samples[i].append(be[i])
        for i in range(n):
            s = sorted(samples[i])
            lo = s[int(0.025 * len(s))]
            hi = s[min(int(0.975 * len(s)), len(s) - 1)]
            standings[i].elo_lo = lo
            standings[i].elo_hi = hi

    standings.sort(key=lambda st: st.elo, reverse=True)
    return standings


def format_leaderboard(standings: List[Standing]) -> str:
    lines = []
    header = f"{'#':>2}  {'agent':<18}{'Elo':>7}{'95% CI':>18}" \
             f"{'win%':>8}{'margin':>9}{'games':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for rank, st in enumerate(standings, 1):
        ci = (f"[{st.elo_lo:.0f}, {st.elo_hi:.0f}]"
              if st.elo_lo is not None else "-")
        lines.append(
            f"{rank:>2}  {st.name:<18}{st.elo:>7.0f}{ci:>18}"
            f"{100 * st.win_rate:>7.1f}%{st.avg_margin:>+9.3f}{st.games:>8}")
    return "\n".join(lines)
