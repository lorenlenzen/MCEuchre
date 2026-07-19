"""The ReBeL self-play training loop.

This wires the pieces together into ReBeL's learning cycle:

1. **Self-play with search.** Play hands; at every decision node run the
   depth-limited CFR subgame solver (`SubgameSolver`), using the *current*
   network to value the leaves. The solved root strategy is the policy we play
   (sampled), and it -- together with the solved root value -- becomes a
   training target.
2. **Learn.** Train one `PolicyValueNet`: the policy head regresses onto the
   CFR strategies (cross-entropy over legal actions); the value head regresses
   onto the CFR root values (MSE). As the value head improves, the leaf
   estimates that feed the next round of search improve too -- the bootstrap
   that lets ReBeL climb past what a reactive policy can reach.

Everything is honest but small by default: reaching genuinely expert play needs
far more self-play and compute (and, in pure Python, a faster engine). The loop
here is designed to *run and learn*, exposing the moving parts, not to train a
finished agent in one sitting. See ``docs/rebel_design.md``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from euchre.actions import NUM_ACTIONS, action_to_index
from euchre.game import EuchreState, Phase, team_of
from euchre.infoset import observation_tensor, OBS_SIZE
from .networks import PolicyValueNet
from .subgame import SubgameSolver


def legal_mask(state: EuchreState) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in state.legal_actions():
        mask[action_to_index(a)] = True
    return mask


def batch_value_fn_from_net(net: PolicyValueNet):
    """A batched leaf-value function for a net: many states -> one forward pass,
    each returning the team0 - team1 point-differential estimate. Used to plug a
    trained net into the CFR subgame solver at play time."""
    def fn(states: List[EuchreState]) -> List[float]:
        players = [s.current_player if not s.is_terminal() else 0
                   for s in states]
        obs = np.stack([observation_tensor(s, p)
                        for s, p in zip(states, players)])
        with torch.no_grad():
            _, v = net(torch.from_numpy(obs))
        v = v.numpy()
        return [float(v[i]) if team_of(players[i]) == 0 else -float(v[i])
                for i in range(len(states))]
    return fn


@dataclass
class Sample:
    obs: np.ndarray          # observation from the actor's perspective
    mask: np.ndarray         # legal-action mask
    policy: np.ndarray       # CFR target distribution over NUM_ACTIONS
    value: float             # CFR root value, actor's-team differential


class ReBeLTrainer:
    def __init__(self, net: Optional[PolicyValueNet] = None,
                 depth_limit: int = 4, num_worlds: int = 8,
                 cfr_iterations: int = 20, lr: float = 1e-3,
                 buffer_size: int = 20000, belief_model=None,
                 full_depth_cards: int = 0, seed: int = 0) -> None:
        self.net = net or PolicyValueNet()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.depth_limit = depth_limit
        self.num_worlds = num_worlds
        self.cfr_iterations = cfr_iterations
        self.buffer: List[Sample] = []
        self.buffer_size = buffer_size
        self.belief_model = belief_model
        # "As much depth as feasible per position": when the acting player has
        # <= full_depth_cards cards left, solve the subgame to *terminal* (exact
        # CFR targets, no value net) since the tree is then cheap. Deeper into
        # the hand this yields exact endgame targets that anchor the value net,
        # so the depth-limited early-game leaves it feeds are less noisy.
        self.full_depth_cards = full_depth_cards
        self.rng = random.Random(seed)

    def _depth_for(self, state: EuchreState) -> Optional[int]:
        if (self.full_depth_cards > 0 and state.phase == Phase.PLAY
                and len(state.hands[state.current_player])
                <= self.full_depth_cards):
            return None  # full-depth / exact
        return self.depth_limit

    # -- leaf value from the current network ---------------------------------

    def value_fn(self, state: EuchreState) -> float:
        """Estimate team0 - team1 point differential at a subgame leaf."""
        return self.batch_value_fn([state])[0]

    def batch_value_fn(self, states: List[EuchreState]) -> List[float]:
        """Value many leaves in a single network forward pass (see
        ``batch_value_fn_from_net``)."""
        return batch_value_fn_from_net(self.net)(states)

    # -- self-play -----------------------------------------------------------

    def self_play_hand(self) -> Tuple[int, int]:
        state = EuchreState.new_hand(
            dealer=self.rng.randint(0, 3)).deal(self.rng)
        while not state.is_terminal():
            legal = state.legal_actions()
            if len(legal) == 1:
                state = state.apply(legal[0])
                continue
            actor = state.current_player
            solver = SubgameSolver(
                state, actor, num_worlds=self.num_worlds,
                iterations=self.cfr_iterations, depth_limit=self._depth_for(state),
                batch_value_fn=self.batch_value_fn,
                belief_model=self.belief_model, rng=self.rng)
            solver.run()
            policy = solver.root_policy()
            root_val = solver.root_value()  # team0 - team1

            target = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for a, p in policy.items():
                target[action_to_index(a)] = p
            actor_val = root_val if team_of(actor) == 0 else -root_val
            self._store(Sample(
                obs=observation_tensor(state, actor),
                mask=legal_mask(state),
                policy=target,
                value=actor_val))

            actions = list(policy)
            chosen = self.rng.choices(
                actions, weights=[policy[a] for a in actions])[0]
            state = state.apply(chosen)
        return state.returns()

    def _store(self, sample: Sample) -> None:
        self.buffer.append(sample)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    # -- learning ------------------------------------------------------------

    def train_step(self, batch_size: int = 128) -> dict:
        if not self.buffer:
            return {"policy_loss": 0.0, "value_loss": 0.0}
        batch = self.rng.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs = torch.from_numpy(np.stack([s.obs for s in batch]))
        mask = torch.from_numpy(np.stack([s.mask for s in batch]))
        target_p = torch.from_numpy(np.stack([s.policy for s in batch]))
        target_v = torch.tensor([s.value for s in batch], dtype=torch.float32)

        logits, value = self.net(obs)
        logits = logits.masked_fill(~mask, float("-inf"))
        logp = F.log_softmax(logits, dim=-1)
        # Cross-entropy against the CFR target distribution (legal-only). Zero
        # out illegal entries so the target's 0 * (-inf) does not become NaN.
        logp = torch.where(mask, logp, torch.zeros_like(logp))
        policy_loss = -(target_p * logp).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, target_v)
        loss = policy_loss + value_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item())}

    def train(self, generations: int, hands_per_gen: int = 4,
              train_steps: int = 8, batch_size: int = 128,
              log: bool = False) -> List[dict]:
        history = []
        for g in range(1, generations + 1):
            for _ in range(hands_per_gen):
                self.self_play_hand()
            stats = {}
            for _ in range(train_steps):
                stats = self.train_step(batch_size)
            stats = {"gen": g, "buffer": len(self.buffer), **stats}
            history.append(stats)
            if log:
                print(f"gen {g}: buffer={stats['buffer']} "
                      f"policy_loss={stats['policy_loss']:.4f} "
                      f"value_loss={stats['value_loss']:.4f}")
        return history


class ReBeLNetAgent:
    """Fast inference agent: acts from the trained policy head, no search.

    This is what you deploy once the net has learned; decision-time search
    (`CFRSearchAgent` with ``value_fn=trainer.value_fn``) can be layered back on
    top for extra strength.
    """

    def __init__(self, net: PolicyValueNet, greedy: bool = True,
                 temperature: float = 1.0) -> None:
        self.net = net
        self.greedy = greedy
        self.temperature = temperature

    def act(self, state: EuchreState, rng: random.Random):
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        obs = torch.from_numpy(
            observation_tensor(state, state.current_player)).unsqueeze(0)
        mask = torch.from_numpy(legal_mask(state)).unsqueeze(0)
        with torch.no_grad():
            dist = self.net.policy(obs, mask).squeeze(0).numpy()
        idxs = [action_to_index(a) for a in legal]
        probs = np.array([dist[i] for i in idxs], dtype=np.float64)
        if probs.sum() <= 0:
            probs = np.ones(len(legal)) / len(legal)
        else:
            probs = probs / probs.sum()
        if self.greedy:
            return legal[int(np.argmax(probs))]
        # Optimal play in an imperfect-information game is a *mixed* strategy;
        # temperature keeps the agent from collapsing to a deterministic (and
        # thus exploitable) policy.
        if self.temperature != 1.0:
            probs = probs ** (1.0 / self.temperature)
            probs = probs / probs.sum()
        return legal[rng.choices(range(len(legal)), weights=probs.tolist())[0]]
