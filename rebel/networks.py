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

from euchre.infoset import (
    OBS_SIZE, GLOBAL_DIM, SUIT_OFF, SUIT_BLOCK_DIM, CARD_OFF, CARD_FEAT_DIM,
    NUM_SUITS,
)
from euchre.cards import NUM_CARDS
from euchre.actions import NUM_ACTIONS

_CARDS_PER_SUIT = NUM_CARDS // NUM_SUITS  # 6


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


class PolicyValueNet(nn.Module):
    """Suit-agnostic relational policy/value network.

    Every suit-bearing output is produced by a *shared* tower applied to each
    suit's (or card's) role-relative feature block, so the two off-color
    ("green") suits are handled by identical weights and no per-absolute-suit
    preference can form -- the failure mode diagnosed and fixed this session.
    Absolute suit identity never enters as a parameter; only role (encoded in
    the observation relative to the trump/up-card suit) does. See
    euchre/infoset.py for the matching encoding and docs/rebel_design.md.

    The external interface is unchanged: forward(obs) -> (logits[59], value),
    with logits laid out exactly as euchre/actions.py's flat index space, so
    the CFR solver, training loop, and agents are untouched.
    """

    def __init__(self, obs_size: int = OBS_SIZE, num_actions: int = NUM_ACTIONS,
                 suit_emb: int = 64, hidden: int = 128, context: int = 128
                 ) -> None:
        super().__init__()
        assert obs_size == OBS_SIZE and num_actions == NUM_ACTIONS
        self.suit_emb = suit_emb
        # Shared suit encoder: one per-suit role-relative block -> embedding.
        self.suit_encoder = MLP(SUIT_BLOCK_DIM, hidden, suit_emb, depth=2)
        # Context trunk: symmetric pool over the 4 suit embeddings (permutation
        # invariant -> green-symmetric) concatenated with the global block.
        self.context = MLP(suit_emb + GLOBAL_DIM, hidden, context, depth=2)
        # Shared "value of making suit s trump" scorer -> (not-alone, alone).
        # Used for BOTH OrderUp (the reference suit, round 1) and Call (round
        # 2), so round 1's abundant order-up gradient trains the very weights
        # that score round-2 suit calls.
        self.make_trump = MLP(suit_emb + context, hidden, 2, depth=2)
        # Shared per-card scorers (play and discard), on role-relative card
        # features + that card's suit embedding + context.
        self.play_scorer = MLP(suit_emb + CARD_FEAT_DIM + context, hidden, 1,
                               depth=2)
        self.discard_scorer = MLP(suit_emb + CARD_FEAT_DIM + context, hidden, 1,
                                  depth=2)
        self.pass_head = nn.Linear(context, 1)
        self.value_head = MLP(context, hidden, 1, depth=2)

    def forward(self, obs: torch.Tensor):
        b = obs.shape[0]
        glob = obs[:, :GLOBAL_DIM]
        suit_blocks = obs[:, SUIT_OFF:SUIT_OFF + NUM_SUITS * SUIT_BLOCK_DIM]
        suit_blocks = suit_blocks.reshape(b, NUM_SUITS, SUIT_BLOCK_DIM)
        card_feats = obs[:, CARD_OFF:CARD_OFF + NUM_CARDS * CARD_FEAT_DIM]
        card_feats = card_feats.reshape(b, NUM_CARDS, CARD_FEAT_DIM)

        # role one-hot is the first N_ROLES dims of each suit block; index 0 is
        # the reference role (the up-card / trump suit), used to route the
        # make-trump score to OrderUp.
        ref_ind = suit_blocks[:, :, 0]  # (b, 4) -- exactly one 1 when defined

        suit_e = self.suit_encoder(suit_blocks)              # (b, 4, E)
        pooled = suit_e.mean(dim=1)                          # (b, E) symmetric
        ctx = self.context(torch.cat([pooled, glob], dim=-1))  # (b, C)

        ctx_suit = ctx.unsqueeze(1).expand(b, NUM_SUITS, -1)
        make = self.make_trump(torch.cat([suit_e, ctx_suit], dim=-1))  # (b,4,2)
        call_na = make[:, :, 0]                              # (b, 4)
        call_al = make[:, :, 1]                              # (b, 4)
        orderup_na = (ref_ind * call_na).sum(dim=1, keepdim=True)   # (b, 1)
        orderup_al = (ref_ind * call_al).sum(dim=1, keepdim=True)   # (b, 1)

        # each card's suit embedding: cards are id-ordered (suit*6 + rank), so
        # repeat each suit embedding for its 6 ranks.
        card_suit_e = suit_e.repeat_interleave(_CARDS_PER_SUIT, dim=1)  # (b,24,E)
        ctx_card = ctx.unsqueeze(1).expand(b, NUM_CARDS, -1)
        card_in = torch.cat([card_suit_e, card_feats, ctx_card], dim=-1)
        play = self.play_scorer(card_in).squeeze(-1)         # (b, 24)
        discard = self.discard_scorer(card_in).squeeze(-1)   # (b, 24)
        pass_l = self.pass_head(ctx)                         # (b, 1)

        # Assemble in the exact order of euchre/actions.py's flat index space:
        # play[0:24], discard[24:48], call[48:52], call_alone[52:56],
        # orderup[56], orderup_alone[57], pass[58].
        logits = torch.cat([play, discard, call_na, call_al,
                            orderup_na, orderup_al, pass_l], dim=-1)
        return logits, self.value_head(ctx).squeeze(-1)

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
