"""Per-seat bidding-behaviour counters.

`match_equity_table.json`'s outcome_dist records only what hands were WORTH
(1 / 2 / 4 points), which cannot answer behavioural questions: a failed
loner that still took three tricks scores 1 and is indistinguishable from an
ordinary make, and a euchred loner just hands the defenders 2. So "how often
does this agent call alone" is not recoverable from the equity table at all
-- it has to be counted at the moment the bid is made, which is what this
does.

Bids are bucketed by SEAT (bidding-order position: first/second/third/dealer,
quiz_eval.py's SEAT_ORDER convention) because position is the dominant
strategic axis in the auction -- first seat and the dealer face genuinely
different decisions on the same cards, and an aggression problem in one seat
would be invisible in a pooled total.

Round-2 calls are split by suit ROLE, not suit identity:

* **next** -- the same-colour suit as the turned-down up-card. It holds the
  left bower of the turned-down suit, so calling next is a materially
  different proposition from calling green.
* **green** -- either off-colour suit.

That's the same U/N/G/g vocabulary train_pattern.py's --require uses, minus
U: the up-card's own suit is the turned-down one and is illegal to call in
round 2, so next + green covers every legal round-2 call exactly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

SEAT_NAMES = {1: "first", 2: "second", 3: "third", 4: "dealer"}

# Ordered for reporting; the two round groups have separate denominators.
ROUND1_CATEGORIES = ("r1_orderup", "r1_orderup_alone", "r1_pass")
ROUND2_CATEGORIES = ("r2_next", "r2_next_alone", "r2_green", "r2_green_alone",
                     "r2_pass")
CATEGORIES = ROUND1_CATEGORIES + ROUND2_CATEGORIES


def _same_color(suit: int) -> int:
    """Partner suit of the same colour. Delegates to euchre/cards.py rather
    than restating the pairing, so this can't drift from the engine's own
    left-bower logic. Suit is an IntEnum, so a plain int (what the cpp
    engine hands back) looks up correctly."""
    from euchre.cards import same_color_suit
    return int(same_color_suit(suit))


def classify(state, action, engine: str = "python") -> Optional[Tuple[int, str]]:
    """(seat_position, category) for a bidding action, or None if `state`
    isn't a bidding decision.

    `state` must be the state BEFORE `action` is applied -- the classifier
    reads the phase and turned-down suit from it.
    """
    if engine == "cpp":
        import mceuchre_cpp as cpp
        phase, kind = state.phase, action.kind
        is_r1 = phase == cpp.Phase.BidRound1
        is_r2 = phase == cpp.Phase.BidRound2
        is_pass = kind == cpp.ActionKind.Pass
        is_orderup = kind == cpp.ActionKind.OrderUp
        is_call = kind == cpp.ActionKind.Call
        alone = bool(action.alone)
        called_suit = int(action.suit) if is_call else None
        turned_down = (int(state.turned_down)
                       if state.turned_down is not None else None)
    else:
        from euchre.actions import Call, OrderUp, Pass
        from euchre.game import Phase
        phase = state.phase
        is_r1 = phase == Phase.BID_ROUND_1
        is_r2 = phase == Phase.BID_ROUND_2
        is_pass = isinstance(action, Pass)
        is_orderup = isinstance(action, OrderUp)
        is_call = isinstance(action, Call)
        alone = bool(getattr(action, "alone", False))
        called_suit = int(action.suit) if is_call else None
        turned_down = (int(state.turned_down)
                       if state.turned_down is not None else None)

    if not (is_r1 or is_r2):
        return None

    # Bidding-order position: 1 = first (left of dealer) .. 4 = dealer.
    seat = (int(state.current_player) - int(state.dealer) - 1) % 4 + 1

    if is_r1:
        if is_pass:
            return seat, "r1_pass"
        if is_orderup:
            return seat, "r1_orderup_alone" if alone else "r1_orderup"
        return None

    if is_pass:
        return seat, "r2_pass"
    if not is_call or called_suit is None or turned_down is None:
        return None
    role = "next" if called_suit == _same_color(turned_down) else "green"
    return seat, f"r2_{role}_alone" if alone else f"r2_{role}"


class BidCounter:
    """Tallies classified bids per seat, and reports rates within each round.

    Round 1 and round 2 are normalized separately: a seat only reaches round
    2 when everyone passed round 1, so the two rounds have different (and
    very unequal) denominators. Pooling them would make round-2 rates look
    vanishingly small for reasons that have nothing to do with behaviour.
    """

    def __init__(self) -> None:
        self.counts: Dict[Tuple[int, str], int] = defaultdict(int)

    def record(self, state, action, engine: str = "python") -> bool:
        hit = classify(state, action, engine)
        if hit is None:
            return False
        self.counts[hit] += 1
        return True

    def seat_total(self, seat: int, categories) -> int:
        return sum(self.counts[(seat, c)] for c in categories)

    def as_rows(self) -> List[Dict[str, Any]]:
        """One row per seat: raw counts plus within-round rates."""
        rows = []
        for seat in (1, 2, 3, 4):
            r1 = self.seat_total(seat, ROUND1_CATEGORIES)
            r2 = self.seat_total(seat, ROUND2_CATEGORIES)
            row: Dict[str, Any] = {"seat": SEAT_NAMES[seat],
                                   "r1_decisions": r1, "r2_decisions": r2}
            for c in CATEGORIES:
                n = self.counts[(seat, c)]
                denom = r1 if c in ROUND1_CATEGORIES else r2
                row[c] = n
                row[f"{c}_rate"] = (n / denom) if denom else 0.0
            rows.append(row)
        return rows

    def format_table(self) -> str:
        rows = self.as_rows()
        out = []
        for title, cats, dec_key in (
                ("ROUND 1", ROUND1_CATEGORIES, "r1_decisions"),
                ("ROUND 2", ROUND2_CATEGORIES, "r2_decisions")):
            head = f"{'seat':>7} {'decisions':>10} " + " ".join(
                f"{c[3:]:>16}" for c in cats)
            out.append(f"\n{title}")
            out.append(head)
            out.append("-" * len(head))
            for r in rows:
                cells = " ".join(f"{r[c]:>7} {r[c + '_rate']:>7.1%}" for c in cats)
                out.append(f"{r['seat']:>7} {r[dec_key]:>10} {cells}")
        return "\n".join(out)
