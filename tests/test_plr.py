"""Tests for rebel/plr.py -- Prioritized Level Replay over deals.

The properties that matter are behavioural, not numeric: the buffer must
actually favour high-loss deals, must not collapse onto a stale handful,
must round-trip a deal through both engines so a replay is the SAME deal,
and must degrade to ordinary uniform dealing when switched off.
"""

import random

import pytest
import torch

from rebel.plr import (DealSpec, PLRBuffer, deal_to_spec, score_samples,
                       spec_to_state)
from rebel.train_rebel import ReBeLTrainer


def _spec(seed):
    """A distinct (not necessarily legal) DealSpec, for buffer-mechanics
    tests that never deal it -- only the hashing/ordering matters here."""
    rng = random.Random(seed)
    deck = list(range(24))
    rng.shuffle(deck)
    return DealSpec(dealer=seed % 4,
                    hands=tuple(tuple(sorted(deck[i * 5:(i + 1) * 5]))
                                for i in range(4)),
                    up_card=deck[20], kitty=tuple(deck[21:24]),
                    team0_score=0, team1_score=0, stick_the_dealer=False)


# --- deal round-trip: a replay must be the identical deal -----------------

@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_deal_spec_round_trips(engine):
    """spec -> state -> spec must be identity, or a "replay" would silently
    train on a different hand than the one that scored highly."""
    trainer = ReBeLTrainer(engine=engine, num_worlds=2, cfr_iterations=2,
                           depth_limit=2, seed=0)
    for _ in range(5):
        state = trainer._default_deal()
        spec = deal_to_spec(state, engine)
        assert sum(len(h) for h in spec.hands) == 20, "four hands of five"
        assert len(spec.kitty) == 3
        assert spec.up_card not in {c for h in spec.hands for c in h}
        again = deal_to_spec(spec_to_state(spec, trainer), engine)
        assert spec == again


def test_deal_spec_is_engine_neutral():
    """The same 24 cards dealt into either engine must produce an EQUAL
    spec -- the buffer is engine-agnostic, and a spec stored by one engine
    has to be replayable by the other."""
    py = ReBeLTrainer(engine="python", seed=0)
    cpp = ReBeLTrainer(engine="cpp", seed=0)
    base = _spec(7)
    py_spec = deal_to_spec(spec_to_state(base, py), "python")
    cpp_spec = deal_to_spec(spec_to_state(base, cpp), "cpp")
    assert py_spec == cpp_spec == base


# --- buffer mechanics -----------------------------------------------------

def test_empty_buffer_never_replays():
    """With nothing stored there is nothing to replay -- the caller must
    fall through to ordinary dealing rather than raise."""
    buf = PLRBuffer(replay_prob=1.0, rng=random.Random(0))
    assert not buf.should_replay()
    with pytest.raises(IndexError):
        buf.sample()


def test_replay_prob_zero_never_replays():
    buf = PLRBuffer(replay_prob=0.0, rng=random.Random(0))
    for i in range(5):
        buf.update(_spec(i), score=1.0)
    assert len(buf) == 5
    assert not any(buf.should_replay() for _ in range(200))


def test_update_refreshes_rather_than_duplicates():
    """Re-scoring a replayed deal must overwrite its score, not insert a
    second copy -- otherwise the buffer fills with duplicates of whatever
    gets replayed most. (Stored scores are ratios to the running typical
    level, so assert on the drop, not on the raw number. The gate is off
    here so the re-score exercises refresh rather than graduation, which
    test_learned_deals_graduate_out_of_the_buffer covers.)"""
    buf = PLRBuffer(capacity=10, typical_ema=0.01, min_score_ratio=0.0,
                    rng=random.Random(0))
    s = _spec(1)
    buf.update(s, 5.0)
    high = buf.stats()["score_max"]
    buf.update(s, 0.1)
    assert len(buf) == 1
    assert buf.stats()["score_max"] < high


def test_capacity_evicts_the_weakest_only_when_beaten():
    """Eviction ranks by score relative to the contemporaneous typical
    level; typical_ema is tiny here so that level barely moves and the
    ordering follows the raw numbers."""
    buf = PLRBuffer(capacity=3, typical_ema=1e-6, rng=random.Random(0))
    for i, sc in enumerate([1.0, 2.0, 3.0]):
        buf.update(_spec(i), sc)
    assert len(buf) == 3
    lowest = buf.stats()["score_min"]
    # Weaker than the weakest stored -> rejected, buffer unchanged.
    assert buf.update(_spec(99), 0.5) is False
    assert len(buf) == 3 and buf.stats()["score_min"] == pytest.approx(lowest)
    # Stronger -> displaces the weakest entry.
    assert buf.update(_spec(100), 9.0) is True
    assert len(buf) == 3 and buf.stats()["score_min"] > lowest
    assert buf.stats()["evicted"] == 1


def test_sampling_favours_high_score_deals():
    """The whole point: high-loss deals must be drawn more often. Staleness
    and the admission gate are both off here so this measures sampling
    prioritization alone."""
    buf = PLRBuffer(capacity=10, staleness_coef=0.0, min_score_ratio=0.0,
                    rng=random.Random(0))
    best = _spec(0)
    buf.update(best, 100.0)
    for i in range(1, 10):
        buf.update(_spec(i), 1.0)
    drawn = [buf.sample() for _ in range(400)]
    share = drawn.count(best) / len(drawn)
    assert share > 0.25, f"top-scoring deal drawn only {share:.0%} of the time"


def test_staleness_prevents_collapse_onto_one_deal():
    """Without a staleness term the buffer fixates on early high scorers
    whose scores were measured against a long-obsolete net. With it, every
    stored deal should eventually come up."""
    buf = PLRBuffer(capacity=5, staleness_coef=0.5, min_score_ratio=0.0,
                    rng=random.Random(0))
    specs = [_spec(i) for i in range(5)]
    buf.update(specs[0], 100.0)
    for s in specs[1:]:
        buf.update(s, 1.0)
    seen = {buf.sample() for _ in range(300)}
    assert seen == set(specs), f"only {len(seen)}/5 deals ever sampled"


@pytest.mark.parametrize("kwargs", [
    {"replay_prob": -0.1}, {"replay_prob": 1.5},
    {"staleness_coef": -0.1}, {"staleness_coef": 2.0},
    {"temperature": 0.0}, {"temperature": -1.0},
    {"typical_ema": 0.0}, {"typical_ema": 1.5},
])
def test_rejects_out_of_range_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        PLRBuffer(**kwargs)


# --- score normalization: entries must not ossify -------------------------

def test_stale_high_scores_do_not_block_fresher_harder_deals():
    """Regression test for buffer ossification. Absolute losses fall as the
    net improves, so scores measured generations apart aren't comparable.
    Without normalization, deals scored 2.5 against an early weak net would
    permanently out-rank deals scored 0.9 against a strong one -- even when
    the latter are far harder RELATIVE to what the net can now do -- and the
    buffer would freeze around its oldest entries.

    Here: fill at an early loss level, then present a deal that is much
    harder for the *current* net but numerically smaller. It must still get
    in."""
    buf = PLRBuffer(capacity=5, typical_ema=1.0, rng=random.Random(0))
    for i in range(5):                      # early era: losses around 2.5
        buf.update(_spec(i), 2.5)
    assert len(buf) == 5

    for i in range(100, 140):               # net improves; typical drops to ~0.5
        buf.update(_spec(i), 0.5)

    # 0.9 is far below the stored 2.5s in absolute terms, but ~1.8x the
    # current typical -- a genuinely hard hand for the net as it stands.
    assert buf.update(_spec(999), 0.9) is True, (
        "a deal 1.8x harder than current typical was rejected because old "
        "entries kept inflated absolute scores -- buffer has ossified")


def test_typical_tracks_the_current_loss_level():
    buf = PLRBuffer(capacity=50, typical_ema=1.0, rng=random.Random(0))
    buf.update(_spec(0), 2.0)
    assert buf.stats()["typical"] == pytest.approx(2.0)
    for i in range(1, 20):
        buf.update(_spec(i), 0.4)
    assert buf.stats()["typical"] == pytest.approx(0.4)


def test_scores_are_stored_relative_to_typical():
    """A deal exactly at the typical level scores ~1.0 whatever the era."""
    buf = PLRBuffer(capacity=10, typical_ema=1.0, rng=random.Random(0))
    buf.update(_spec(0), 3.0)
    assert buf.stats()["score_max"] == pytest.approx(1.0)
    buf.update(_spec(1), 0.2)
    assert buf.stats()["score_max"] == pytest.approx(1.0)


# --- scoring --------------------------------------------------------------

@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_score_samples_matches_train_step_loss(engine):
    """A deal's score must be the same per-sample loss train_step feeds into
    cluster_priority -- a differently-scaled signal would silently compete
    with the existing prioritization instead of refining it."""
    import numpy as np
    import torch.nn.functional as F

    trainer = ReBeLTrainer(engine=engine, num_worlds=2, cfr_iterations=3,
                           depth_limit=2, seed=0)
    trainer.self_play_hand()
    samples = trainer.last_hand_samples
    assert samples, "self_play_hand must expose the samples it generated"

    got = score_samples(trainer.net, samples)

    # Recompute train_step's own per-sample loss independently.
    obs = torch.from_numpy(np.stack([s.obs for s in samples]))
    mask = torch.from_numpy(np.stack([s.mask for s in samples]))
    tp = torch.from_numpy(np.stack([s.policy for s in samples]))
    tv = torch.tensor([s.value for s in samples], dtype=torch.float32)
    sp = torch.tensor([1.0 if s.supervise_policy else 0.0 for s in samples])
    with torch.no_grad():
        logits, value = trainer.net(obs)
        logits = logits.masked_fill(~mask, float("-inf"))
        logp = F.log_softmax(logits, dim=-1)
        logp = torch.where(mask, logp, torch.zeros_like(logp))
        want = float(((-(tp * logp).sum(dim=-1) * sp)
                      + F.mse_loss(value, tv, reduction="none")).mean())
    assert got == pytest.approx(want, abs=1e-6)


def test_score_samples_handles_empty():
    trainer = ReBeLTrainer(seed=0)
    assert score_samples(trainer.net, []) == 0.0


def test_typical_is_not_anchored_to_the_first_hand():
    """Bias correction: a plain 1% EMA would leave `typical` dominated by
    hand 1 for ~100 hands, so the admission gate would spend that stretch
    calibrated to one arbitrary deal."""
    buf = PLRBuffer(min_score_ratio=0.0, typical_ema=0.01, rng=random.Random(0))
    buf.update(_spec(0), 5.0)           # an unrepresentative first hand
    for i in range(1, 30):
        buf.update(_spec(i), 1.0)
    assert buf.stats()["typical"] < 1.5, (
        f"typical={buf.stats()['typical']:.2f} is still anchored to hand 1")


# --- admission gate: a struggle-finder, not a recency cache ---------------

def test_easy_hands_are_refused_even_when_there_is_room():
    """The gate is what makes this PLR rather than "replay recent hands".
    Textbook PLR admits unconditionally until full, which is fine when the
    buffer is small relative to levels seen -- but a per-actor buffer here
    is huge relative to hands per actor (a 404-hand run over 14 actors
    leaves each buffer ~3% full), so without a gate NOTHING is ever
    selected on difficulty."""
    buf = PLRBuffer(capacity=100, typical_ema=1e-6, min_score_ratio=1.0,
                    rng=random.Random(0))
    buf.update(_spec(0), 1.0)                 # sets typical ~= 1.0
    assert buf.update(_spec(1), 0.4) is False, "easy hand took a free slot"
    assert buf.update(_spec(2), 1.6) is True, "hard hand refused"
    assert len(buf) == 2
    assert buf.stats()["rejected"] == 1


def test_learned_deals_graduate_out_of_the_buffer():
    """A stored deal that stops clearing the admission bar has been learned
    and must leave. Previously a refresh kept it unconditionally, so a
    mastered hand held its slot forever: eviction only fires when the buffer
    is FULL and something better arrives, so in an unfilled buffer nothing
    ever left, and the staleness term kept re-drawing it."""
    buf = PLRBuffer(capacity=50, min_score_ratio=1.0, typical_ema=1e-6,
                    rng=random.Random(0))
    buf.update(_spec(0), 1.0)                       # sets typical ~= 1.0
    hard = _spec(1)
    assert buf.update(hard, 1.8) is True            # admitted: 1.8x typical
    assert buf.update(hard, 1.5) is True            # still hard: retained
    assert hard in buf._entries
    assert buf.update(hard, 0.4) is False           # learned: graduated out
    assert hard not in buf._entries
    assert buf.stats()["graduated"] == 1


def test_buffer_may_drain_when_everything_is_learned():
    """Draining is a legitimate outcome, not a failure -- it just means
    fresh dealing resumes until hard hands turn up again."""
    buf = PLRBuffer(capacity=50, min_score_ratio=1.0, typical_ema=1e-6,
                    rng=random.Random(0))
    buf.update(_spec(0), 1.0)
    specs = [_spec(i) for i in range(1, 11)]
    for s in specs:
        buf.update(s, 2.0)
    assert len(buf) == 11
    for s in specs:                                  # the net learns them all
        buf.update(s, 0.2)
    assert len(buf) == 1                             # only the seed remains
    assert buf.stats()["graduated"] == 10


def test_gate_zero_restores_unconditional_admission():
    buf = PLRBuffer(capacity=100, typical_ema=1e-6, min_score_ratio=0.0,
                    rng=random.Random(0))
    buf.update(_spec(0), 1.0)
    assert buf.update(_spec(1), 0.01) is True
    assert buf.stats()["rejected"] == 0


def test_gate_keeps_only_above_typical_hands_over_a_run():
    """End to end: with scores drawn around a typical of 1.0, a gated buffer
    should admit roughly the harder half and reject the rest."""
    buf = PLRBuffer(capacity=500, replay_prob=0.0, min_score_ratio=1.0,
                    rng=random.Random(0))
    rng = random.Random(1)
    for i in range(400):
        buf.update(_spec(i), max(0.05, rng.gauss(1.0, 0.3)))
    s = buf.stats()
    assert s["rejected"] > 100, "gate admitted nearly everything"
    assert s["size"] > 50, "gate admitted almost nothing"
    assert s["size"] + s["rejected"] == 400


# --- buffer inspection: which hands need more training --------------------

def test_replay_prob_one_starves_the_buffer():
    """Documents a real footgun rather than pretending it away: new deals
    only reach the buffer on NON-replay hands, so replay_prob=1.0 stores the
    first hand and then replays it forever. Both scripts warn about this.

    Both training scripts warn when --plr-replay-prob is >= 0.9."""
    from rebel.plr import PLRBuffer as B
    for prob, expect in ((1.0, 1), (0.5, None)):
        buf = B(capacity=50, replay_prob=prob, min_score_ratio=0.0,
                rng=random.Random(0))
        for i in range(200):
            spec = buf.sample() if buf.should_replay() else _spec(i)
            buf.update(spec, score=1.0)
        if expect is not None:
            assert len(buf) == expect, (
                f"replay_prob={prob} should freeze the buffer at {expect}")
        else:
            assert len(buf) > 10, "a moderate replay_prob must still grow"


def test_describe_spec_is_readable_and_role_labelled():
    from rebel.plr import describe_spec
    from euchre.cards import Card, Rank, Suit, same_color_suit

    up = Card(Suit.DIAMONDS, Rank.QUEEN)
    spec = _spec(4)._replace(dealer=0, up_card=up.id)
    d = describe_spec(spec)

    assert d["up_card"].endswith("(U)"), d["up_card"]
    assert d["up_card"].startswith("QD")
    # Every seat is named by POSITION, not absolute index.
    assert set(d["hands_by_position"]) == {"first", "second", "third", "dealer"}
    # Every card carries exactly one role tag, and all four roles are used
    # across the deck.
    tags = set()
    for hand in d["hands_by_position"].values():
        for card in hand.split():
            assert card[-1] == ")" and card[-3] == "("
            tags.add(card[-2])
    assert tags <= {"U", "N", "G", "g"}


def test_describe_spec_next_role_follows_up_card_colour():
    from rebel.plr import describe_spec
    from euchre.cards import Card, Rank, Suit, same_color_suit

    for up_suit in Suit:
        spec = _spec(9)._replace(dealer=0, up_card=Card(up_suit, Rank.NINE).id)
        d = describe_spec(spec)
        nxt = same_color_suit(up_suit)
        joined = " ".join(d["hands_by_position"].values())
        for card in joined.split():
            suit_ch = card[1]
            role = card[-2]
            suit = "CDHS".index(suit_ch)
            if suit == int(up_suit):
                assert role == "U", card
            elif suit == int(nxt):
                assert role == "N", card
            else:
                assert role in ("G", "g"), card


def test_top_is_ordered_hardest_first():
    buf = PLRBuffer(capacity=10, typical_ema=1e-6, rng=random.Random(0))
    for i, sc in enumerate([1.0, 5.0, 3.0]):
        buf.update(_spec(i), sc)
    scores = [e["score"] for e in buf.top(3)]
    assert scores == sorted(scores, reverse=True)
    assert len(buf.top(2)) == 2


def test_dump_writes_stats_and_hardest(tmp_path):
    import json
    buf = PLRBuffer(capacity=10, rng=random.Random(0))
    for i in range(4):
        buf.update(_spec(i), 1.0 + i)
    p = tmp_path / "b.json"
    buf.dump(str(p), n=2)
    d = json.loads(p.read_text())
    assert d["stats"]["size"] == 4
    assert len(d["hardest"]) == 2
    assert "hands_by_position" in d["hardest"][0]


# --- integration: replay actually re-plays the stored deal ----------------

@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_deal_fn_replay_reproduces_the_stored_deal(engine):
    """End-to-end shape used by train_scale.py: a deal_fn that replays from
    the buffer must hand self_play_hand the identical deal, and the hand
    must still play out normally from it."""
    trainer = ReBeLTrainer(engine=engine, num_worlds=2, cfr_iterations=2,
                           depth_limit=2, seed=0)
    buf = PLRBuffer(replay_prob=1.0, rng=random.Random(0))

    trainer.self_play_hand()
    stored = deal_to_spec(trainer.last_hand_state, engine)
    buf.update(stored, 42.0)

    trainer.deal_fn = lambda: spec_to_state(buf.sample(), trainer)
    trainer.self_play_hand()
    assert deal_to_spec(trainer.last_hand_state, engine) == stored
    assert trainer.last_hand_samples, "replayed hand still produced samples"
