"""End-to-end smoke demo: train tabular MCCFR briefly and measure strength.

Run:  python scripts/demo.py

This is a sanity/orientation script, not a full training run. It shows the
moving parts working together: the engine, the MCCFR learner, and the
evaluation harness. Real strength needs far more iterations (and, ultimately,
the ReBeL neural pipeline described in docs/rebel_design.md).
"""

import random
import time

from rebel.mccfr import MCCFRTrainer
from rebel.evaluate import (
    evaluate, RandomAgent, RuleBasedAgent, MCCFRAgent,
)
from rebel.public_belief_state import sample_determinization
from euchre.game import EuchreState
from euchre.infoset import infoset_key


def main() -> None:
    print("== Baselines ==")
    stats = evaluate(RuleBasedAgent, RandomAgent, hands=400, seed=1)
    print(f"RuleBased vs Random: mean pt diff "
          f"{stats['team0_mean_point_diff']:+.3f} "
          f"± {stats['ci95']:.3f}, win rate {stats['team0_win_rate']:.3f}")

    print("\n== Training MCCFR (tiny illustrative run) ==")
    print("(Full Euchre has a huge infoset space, so a short tabular run stays")
    print(" near-random -- this is exactly the motivation for ReBeL's neural")
    print(" generalization; see docs/rebel_design.md.)")
    trainer = MCCFRTrainer(seed=0)
    t0 = time.time()
    trainer.train(iterations=100, log_every=50)
    print(f"trained in {time.time() - t0:.1f}s, "
          f"{len(trainer.nodes)} infosets visited")

    print("\n== Determinization sample (decision-time belief) ==")
    s = EuchreState.new_hand(dealer=0).deal(random.Random(7))
    p = s.current_player
    print(f"Player {p} sees their hand; sampling a consistent world:")
    sample = sample_determinization(s, p, random.Random(0))
    for q in range(4):
        tag = "  (me)" if q == p else ""
        print(f"  P{q}: " + " ".join(str(c) for c in sample.hands[q]) + tag)


if __name__ == "__main__":
    main()
