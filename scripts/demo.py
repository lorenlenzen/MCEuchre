"""End-to-end tour of the MCEuchre stack.

Run:  python scripts/demo.py

Shows the pieces working together: the engine, the double-dummy solver, PIMC
search, the CFR subgame solver, and one generation of the ReBeL self-play loop.
Counts are deliberately tiny so it finishes in a couple of minutes -- real
strength needs far more search and self-play (see docs/rebel_design.md).
"""

import random
import time

from euchre.game import EuchreState, Phase
from rebel.evaluate import evaluate, RandomAgent, RuleBasedAgent
from rebel.solver import solve_value, best_play
from rebel.pimc import PIMCAgent
from rebel.subgame import SubgameSolver
from rebel.train_rebel import ReBeLTrainer


def _reach_play(seed):
    rng = random.Random(seed)
    s = EuchreState.new_hand(dealer=rng.randint(0, 3)).deal(rng)
    while not s.is_terminal() and s.phase != Phase.PLAY:
        s = s.apply(rng.choice(s.legal_actions()))
    while not s.is_terminal() and len(s.hands[s.current_player]) > 3:
        s = s.apply(rng.choice(s.legal_actions()))
    return s


def main() -> None:
    print("== 1. Baseline: heuristic vs random ==")
    stats = evaluate(RuleBasedAgent, RandomAgent, hands=400, seed=1)
    print(f"   RuleBased vs Random: {stats['team0_mean_point_diff']:+.3f} "
          f"pt/hand (win rate {stats['team0_win_rate']:.2f})")

    print("\n== 2. Double-dummy solver ==")
    s = _reach_play(2)
    print(f"   Perfect-info value of a mid-play position (team0-team1): "
          f"{solve_value(s)}; optimal play: {best_play(s)}")

    print("\n== 3. PIMC search picks a move under hidden information ==")
    agent = PIMCAgent(worlds=10, seed=0)
    move = agent.act(s, random.Random(0))
    print(f"   PIMC (10 worlds) chooses: {move}")

    print("\n== 4. CFR subgame solver (ReBeL's search core) ==")
    t0 = time.time()
    solver = SubgameSolver(s, s.current_player, num_worlds=10, iterations=40,
                           rng=random.Random(0))
    solver.run()
    pol = solver.root_policy()
    print(f"   Solved a play subgame in {time.time() - t0:.1f}s. Root policy:")
    for a, p in sorted(pol.items(), key=lambda kv: -kv[1])[:3]:
        print(f"     {p:.3f}  {a}")

    print("\n== 5. ReBeL self-play loop ==")
    print("   (Solves a depth-limited subgame at every decision, batch-values")
    print("    the leaves with the net, then trains the net on the results.)")
    trainer = ReBeLTrainer(num_worlds=6, cfr_iterations=15, depth_limit=4,
                           seed=0)
    t0 = time.time()
    hands = 8
    hist = trainer.train(generations=2, hands_per_gen=hands // 2, train_steps=8,
                         batch_size=64)
    dt = time.time() - t0
    st = hist[-1]
    print(f"   {hands} self-play hands + training in {dt:.0f}s "
          f"({dt / hands:.2f}s/hand); buffer={st['buffer']} samples")
    print(f"   policy_loss={st['policy_loss']:.4f}  "
          f"value_loss={st['value_loss']:.4f}")
    print("\nDone. See docs/rebel_design.md for scaling this to expert play.")


if __name__ == "__main__":
    main()
