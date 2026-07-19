"""TeamGame adapters for the TMECor solver: a canonical coordination game with
a known answer, and Euchre endgames."""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from euchre.cards import effective_suit
from euchre.game import EuchreState
from euchre.infoset import infoset_key
from .public_belief_state import known_voids
from .tmecor import TeamGame


class CoordinationGame(TeamGame):
    """A minimal team game where correlation strictly helps.

    Team {0, 1} each pick A or B without seeing the other; adversary {2} then
    picks, also blind. The team scores +1 when they match on a value the
    adversary did not pick, -1 when the adversary catches their common value,
    and -1 when they fail to match. Independent (Nash) play can guarantee only
    -0.5 (they cannot both randomize *and* stay matched); with a shared coin
    they always match yet stay unpredictable, for value 0. So TMECor (0) beats
    Nash (-0.5) -- exactly the correlation gap this solver must capture.
    """

    team = frozenset({0, 1})

    def worlds(self):
        return [((), 1.0)]

    def is_terminal(self, s):
        return len(s) == 3

    def current_player(self, s):
        return len(s)

    def infoset(self, s, player):
        # Nobody observes anyone else's move: one infoset per player.
        return f"p{player}"

    def legal_actions(self, s):
        return ["A", "B"]

    def apply(self, s, a):
        return s + (a,)

    def payoff(self, s):
        x, y, z = s
        if x != y:
            return -1.0
        return 1.0 if z != x else -1.0


def sample_endgame_worlds(state: EuchreState, num_worlds: int,
                          rng: Optional[random.Random] = None
                          ) -> List[Tuple[EuchreState, float]]:
    """Belief for a team (seats 0 and 2) at a play state: keep both team hands
    fixed and redistribute the adversary's cards (seats 1 and 3), respecting
    known voids. Uniform weights."""
    rng = rng or random.Random()
    voids = known_voids(state)
    adv_cards = list(state.hands[1]) + list(state.hands[3])
    n1, n3 = len(state.hands[1]), len(state.hands[3])
    trump = state.trump
    worlds: List[Tuple[EuchreState, float]] = []
    seen = set()
    tries = 0
    while len(worlds) < num_worlds and tries < num_worlds * 50:
        tries += 1
        pool = list(adv_cards)
        rng.shuffle(pool)
        h1, h3 = pool[:n1], pool[n1:n1 + n3]
        ok = all(effective_suit(c, trump) not in voids[1] for c in h1) and \
            all(effective_suit(c, trump) not in voids[3] for c in h3)
        if not ok:
            continue
        key = (tuple(sorted(c.id for c in h1)), tuple(sorted(c.id for c in h3)))
        if key in seen:
            continue
        seen.add(key)
        w = state.clone()
        w.hands = [list(state.hands[0]), h1, list(state.hands[2]), h3]
        worlds.append((w, 0.0))
    if not worlds:  # fall back to the true world
        worlds = [(state.clone(), 1.0)]
    weight = 1.0 / len(worlds)
    return [(w, weight) for w, _ in worlds]


class EuchreEndgame(TeamGame):
    """Team {0, 2} vs {1, 3} over an explicit set of belief worlds (deals)."""

    team = frozenset({0, 2})

    def __init__(self, worlds: List[Tuple[EuchreState, float]]) -> None:
        self._worlds = worlds

    def worlds(self):
        return self._worlds

    def is_terminal(self, s):
        return s.is_terminal()

    def current_player(self, s):
        return s.current_player

    def infoset(self, s, player):
        return infoset_key(s, player)

    def legal_actions(self, s):
        return list(s.legal_actions())

    def apply(self, s, a):
        return s.apply(a)

    def payoff(self, s):
        r = s.returns()
        return float(r[0] - r[1])
