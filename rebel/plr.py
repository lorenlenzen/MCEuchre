"""Prioritized Level Replay (PLR) over DEALS.

Self-play deals hands uniformly at random, which is the right *evaluation*
distribution but a wasteful *training* one: the overwhelming majority of
hands are ones the net already plays correctly, and the compute spent
solving them teaches nothing. PLR (Jiang et al. 2021, and the regret-based
environment-design line it belongs to -- see docs/rebel_design.md) keeps a
rolling buffer of the highest-learning-potential instances found by random
search and replays them preferentially.

Adapted here, a "level" is a DEAL -- the four hands, up-card, kitty, dealer
and match score -- stored as plain card ids so it is engine-neutral,
picklable, and hashable (so re-encountering a deal updates its score rather
than duplicating it).

Why replay a deal rather than a stored training sample: replaying re-runs
the real CFR solves, producing FRESH targets from the current net, and it
exercises every seat and phase of that hand again. Re-training on a stored
sample would just re-fit a stale target. This is also why the granularity
is the deal and not the individual decision: the decision-level equivalent
is ordinary prioritized replay, which `ReBeLTrainer.cluster_priority`
already does (at cluster granularity).

Scoring uses the same per-sample loss (policy cross-entropy + value MSE)
that `ReBeLTrainer.train_step` feeds into `cluster_priority`, so a deal's
score is exactly the per-deal generalization of the existing cluster-level
signal -- finer-grained, but the same quantity, not a competing one.

Two deliberate departures from the published algorithm, both documented at
their call sites:

* **Vanilla PLR, not Robust PLR (PLR^perp).** Robust PLR trains ONLY on
  replayed levels, scoring new ones without a gradient step, which is what
  earns its minimax-regret guarantee. Here a "new level" costs a full
  self-play hand of CFR solves -- by far the dominant cost -- so discarding
  those samples would roughly halve throughput. We train on both and accept
  the weaker guarantee.
* **On-distribution by construction.** Deals enter the buffer only by being
  dealt naturally, never by mutation/editing (the ACCEL extension). Euchre's
  deal distribution is FIXED (a uniform shuffle), unlike the procedurally
  generated environments UED targets, so inventing high-regret deals would
  optimize for hands that never occur. `replay_prob` is the dial on how far
  the training distribution is allowed to drift from the natural one.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class DealSpec(NamedTuple):
    """A dealt hand, as engine-neutral card ids. Hashable, so it doubles as
    the buffer's dedupe key."""
    dealer: int
    hands: Tuple[Tuple[int, ...], ...]   # 4 seats, sorted card ids
    up_card: int
    kitty: Tuple[int, ...]
    team0_score: int
    team1_score: int
    stick_the_dealer: bool


def deal_to_spec(state, engine: str) -> DealSpec:
    """Freeze a freshly-dealt state into a replayable DealSpec.

    Must be called BEFORE any action is applied: it captures the deal, not
    the position. (After a pickup the dealer's hand has 6 cards and the
    kitty 4, which `deal_from` would reject on replay.)
    """
    if engine == "cpp":
        hands = tuple(tuple(c for c in range(24) if (state.hands[p] >> c) & 1)
                      for p in range(4))
        up = int(state.up_card)
        kitty = tuple(int(c) for c in state.kitty)
    else:
        hands = tuple(tuple(sorted(c.id for c in state.hands[p]))
                      for p in range(4))
        up = state.up_card.id
        kitty = tuple(c.id for c in state.kitty)
    return DealSpec(dealer=int(state.dealer), hands=hands, up_card=up,
                    kitty=kitty, team0_score=int(state.team0_score),
                    team1_score=int(state.team1_score),
                    stick_the_dealer=bool(state.stick_the_dealer))


def spec_to_state(spec: DealSpec, trainer):
    """Rebuild a dealt state from a DealSpec, in `trainer`'s own engine."""
    if trainer.engine == "cpp":
        bitmasks = [sum(1 << c for c in h) for h in spec.hands]
        return trainer._cpp.EuchreState.new_hand(
            dealer=spec.dealer, stick_the_dealer=spec.stick_the_dealer,
            team0_score=spec.team0_score, team1_score=spec.team1_score
        ).deal_from(bitmasks, spec.up_card, list(spec.kitty))
    from euchre.cards import Card
    from euchre.game import EuchreState
    hands = [[Card.from_id(c) for c in h] for h in spec.hands]
    return EuchreState.new_hand(
        dealer=spec.dealer, stick_the_dealer=spec.stick_the_dealer,
        team0_score=spec.team0_score, team1_score=spec.team1_score
    ).deal_from(hands, Card.from_id(spec.up_card),
                [Card.from_id(c) for c in spec.kitty])


def score_samples(net, samples: Sequence) -> float:
    """A deal's learning potential: mean per-sample loss over the samples it
    produced, under the CURRENT net.

    Deliberately the same quantity ReBeLTrainer.train_step feeds into
    cluster_priority (policy cross-entropy + value MSE, per sample), so this
    is a finer-grained view of the existing priority signal rather than a
    second, differently-scaled one. Value-only grounding samples
    (supervise_policy=False) contribute their value term only, matching
    train_step's own handling.
    """
    if not samples:
        return 0.0
    obs = torch.from_numpy(np.stack([s.obs for s in samples]))
    mask = torch.from_numpy(np.stack([s.mask for s in samples]))
    target_p = torch.from_numpy(np.stack([s.policy for s in samples]))
    target_v = torch.tensor([s.value for s in samples], dtype=torch.float32)
    supervise_p = torch.tensor([1.0 if s.supervise_policy else 0.0
                                for s in samples], dtype=torch.float32)
    with torch.no_grad():
        logits, value = net(obs)
        logits = logits.masked_fill(~mask, float("-inf"))
        logp = F.log_softmax(logits, dim=-1)
        logp = torch.where(mask, logp, torch.zeros_like(logp))
        per_policy = -(target_p * logp).sum(dim=-1) * supervise_p
        per_value = F.mse_loss(value, target_v, reduction="none")
        return float((per_policy + per_value).mean())


class _Entry:
    __slots__ = ("spec", "score", "last_used")

    def __init__(self, spec: DealSpec, score: float, last_used: int) -> None:
        self.spec = spec
        self.score = score
        self.last_used = last_used


class PLRBuffer:
    """Rolling buffer of high-learning-potential deals, sampled by rank.

    Rank-based rather than score-proportional prioritization (as in the
    paper): proportional sampling is dominated by whatever the current loss
    scale happens to be, which drifts a lot over training as the net
    improves, whereas ranks are invariant to that drift.

    `staleness_coef` mixes in a term favouring deals not replayed recently.
    Without it the buffer collapses onto whichever handful of deals scored
    highest early and never revisits the rest, so their scores -- measured
    against a long-obsolete net -- never get refreshed.

    Entries have no TTL: a deal leaves only by being out-scored, and its
    score only refreshes when it is sampled. Measured refresh coverage,
    sampling `capacity` times against a full buffer: 38% of entries at
    staleness_coef=0.1, 58% at 0.5. Hence the 0.3 default (the published
    algorithm uses ~0.1, but over far smaller level sets) and, more
    importantly, hence storing scores relative to `_typical` -- with
    normalization an unrefreshed entry is merely OLD rather than
    systematically inflated, so it no longer blocks fresher, genuinely
    harder deals from displacing it.
    """

    def __init__(self, capacity: int = 1000, replay_prob: float = 0.5,
                 temperature: float = 1.0, staleness_coef: float = 0.3,
                 typical_ema: float = 0.01,
                 rng: Optional[random.Random] = None) -> None:
        if not 0.0 <= replay_prob <= 1.0:
            raise ValueError(f"replay_prob must be in [0, 1], got {replay_prob}")
        if not 0.0 <= staleness_coef <= 1.0:
            raise ValueError(f"staleness_coef must be in [0, 1], "
                             f"got {staleness_coef}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not 0.0 < typical_ema <= 1.0:
            raise ValueError(f"typical_ema must be in (0, 1], got {typical_ema}")
        self.capacity = capacity
        self.replay_prob = replay_prob
        self.temperature = temperature
        self.staleness_coef = staleness_coef
        self.typical_ema = typical_ema
        # Running "what does a hand cost right now" level, as an EMA of every
        # raw score observed. Stored scores are kept RELATIVE to this (see
        # update), because absolute losses drift a lot over training: a deal
        # scored 2.5 against an early weak net would otherwise sit in the
        # buffer forever out-ranking genuinely-harder deals scored 0.9
        # against a much stronger one, purely because the whole loss level
        # moved. This mirrors how cluster_priority's ceiling is computed
        # from the *median measured* priority rather than an absolute
        # constant (rebel/train_rebel.py) -- same drift, same remedy.
        self._typical: Optional[float] = None
        self.rng = rng or random.Random()
        self._entries: Dict[DealSpec, _Entry] = {}
        self._step = 0
        # Counters for reporting -- how much of training is actually replay,
        # and how often a fresh deal was good enough to displace a stored one.
        self.n_replayed = 0
        self.n_inserted = 0
        self.n_evicted = 0

    def __len__(self) -> int:
        return len(self._entries)

    def should_replay(self) -> bool:
        """True if this hand should replay a stored deal rather than deal a
        fresh one. Always False while the buffer is empty."""
        if not self._entries:
            return False
        return self.rng.random() < self.replay_prob

    def _weights(self) -> Tuple[List[_Entry], np.ndarray]:
        entries = list(self._entries.values())
        scores = np.array([e.score for e in entries], dtype=np.float64)
        # Rank 1 = highest score; weight ~ (1/rank)^(1/temperature).
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(entries), dtype=np.float64)
        ranks[order] = np.arange(1, len(entries) + 1)
        w_score = (1.0 / ranks) ** (1.0 / self.temperature)
        w_score /= w_score.sum()

        if self.staleness_coef > 0.0:
            staleness = np.array([self._step - e.last_used for e in entries],
                                 dtype=np.float64)
            total = staleness.sum()
            w_stale = (staleness / total if total > 0
                       else np.full(len(entries), 1.0 / len(entries)))
            w = (1.0 - self.staleness_coef) * w_score + self.staleness_coef * w_stale
        else:
            w = w_score
        return entries, w

    def sample(self) -> DealSpec:
        """Draw a stored deal, marking it as just-used for staleness."""
        if not self._entries:
            raise IndexError("PLRBuffer.sample() on an empty buffer")
        entries, w = self._weights()
        idx = self.rng.choices(range(len(entries)), weights=w.tolist())[0]
        entry = entries[idx]
        self._step += 1
        entry.last_used = self._step
        self.n_replayed += 1
        return entry.spec

    def update(self, spec: DealSpec, score: float) -> bool:
        """Record `score` for `spec`, inserting it if it earns a slot.

        `score` is the raw loss; what gets STORED is score / typical, its
        ratio to the current running loss level (see _typical). That keeps
        entries measured generations apart comparable, so eviction ranks
        deals by "harder than its contemporaries" rather than by "measured
        back when everything was hard".

        Returns True if the deal is in the buffer afterwards. An already-
        stored deal always has its score refreshed (that's the point of
        replaying it -- the old score was measured against an older net); a
        new deal is admitted if there's room, or if it outscores the current
        weakest entry, which it then evicts.
        """
        # Normalize against the typical level as it stood BEFORE this
        # observation, then fold this one in. Order matters: dividing by a
        # typical that already contains `score` pulls every ratio toward 1.0
        # (exactly 1.0 at typical_ema=1.0), which would silently disable the
        # normalization entirely -- caught by
        # test_stale_high_scores_do_not_block_fresher_harder_deals.
        baseline = self._typical if self._typical is not None else score
        self._typical = (score if self._typical is None
                         else (1.0 - self.typical_ema) * self._typical
                              + self.typical_ema * score)
        score = score / max(baseline, 1e-9)
        self._step += 1
        existing = self._entries.get(spec)
        if existing is not None:
            existing.score = score
            existing.last_used = self._step
            return True
        if len(self._entries) < self.capacity:
            self._entries[spec] = _Entry(spec, score, self._step)
            self.n_inserted += 1
            return True
        weakest = min(self._entries.values(), key=lambda e: e.score)
        if score > weakest.score:
            del self._entries[weakest.spec]
            self._entries[spec] = _Entry(spec, score, self._step)
            self.n_inserted += 1
            self.n_evicted += 1
            return True
        return False

    def stats(self) -> Dict[str, Any]:
        scores = [e.score for e in self._entries.values()]
        return {"size": len(self._entries),
                "replayed": self.n_replayed,
                "inserted": self.n_inserted,
                "evicted": self.n_evicted,
                # Scores below are RATIOS to `typical`, not raw losses: 1.0
                # means "an average hand for the net as it stands now".
                "typical": float(self._typical) if self._typical is not None else 0.0,
                "score_mean": float(np.mean(scores)) if scores else 0.0,
                "score_max": float(np.max(scores)) if scores else 0.0,
                "score_min": float(np.min(scores)) if scores else 0.0}
