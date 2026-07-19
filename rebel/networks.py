"""Neural networks for the ReBeL pipeline.

* :class:`PolicyValueNet` maps an observation to (policy logits, value). It is
  the deep function approximator that will replace the tabular strategy tables
  once the CFR core is validated.
* :class:`PBSValueNet` maps a *public belief state* feature vector to a value
  for each infostate in that PBS — the leaf evaluator ReBeL calls during
  depth-limited subgame solving.

Both are deliberately simple MLPs; the interfaces matter more than the depth,
and they can be swapped for an LSTM/Transformer over the play history later.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from euchre.infoset import OBS_SIZE
from euchre.actions import NUM_ACTIONS


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 depth: int = 3) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


# The flat action space splits cleanly into play (card plays) and bidding
# (discard / call / order-up / pass). Giving each its own output head lets the
# two very different decision types specialise instead of sharing one linear
# layer. Going alone is already first-class (OrderUp(alone), Call(alone)).
_NUM_PLAY_ACTIONS = 24  # indices [0, 24)


class PolicyValueNet(nn.Module):
    """Shared trunk with separate play/bidding policy heads and a value head."""

    def __init__(self, obs_size: int = OBS_SIZE, num_actions: int = NUM_ACTIONS,
                 hidden: int = 256, depth: int = 3) -> None:
        super().__init__()
        layers = [nn.Linear(obs_size, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.play_head = nn.Linear(hidden, _NUM_PLAY_ACTIONS)
        self.bid_head = nn.Linear(hidden, num_actions - _NUM_PLAY_ACTIONS)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        logits = torch.cat([self.play_head(h), self.bid_head(h)], dim=-1)
        return logits, self.value_head(h).squeeze(-1)

    def policy(self, obs: torch.Tensor,
               legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return a legal, normalized action distribution."""
        logits, _ = self.forward(obs)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        return F.softmax(logits, dim=-1)


class PBSValueNet(nn.Module):
    """Value network over public belief states (the ReBeL leaf evaluator).

    Input: a PBS feature vector (public state features + the belief, i.e. a
    probability distribution over each player's possible hands). Output: the
    expected value for the acting team. The per-infostate value formulation
    from the ReBeL paper can be recovered by querying with the belief basis
    vectors; this scalar-output version is the minimal starting point.
    """

    def __init__(self, pbs_size: int, hidden: int = 512, depth: int = 4) -> None:
        super().__init__()
        self.net = MLP(pbs_size, hidden, 1, depth=depth)

    def forward(self, pbs_features: torch.Tensor) -> torch.Tensor:
        return self.net(pbs_features).squeeze(-1)
