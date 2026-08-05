// Determinization sampling, ported from rebel/public_belief_state.py.
// Deliberately does NOT aim for byte-identical RNG sequences with the Python
// version (a randomized backtracking sampler's *correctness* criterion is
// "produces a valid, unbiased completion consistent with the player's
// knowledge", not "same literal draws as Python for the same seed") -- see
// tests/test_cpp_equivalence.py's belief tests, which check consistency
// properties rather than cross-language equality.
#pragma once

#include <array>
#include <random>
#include <utility>
#include <vector>

#include "engine.h"
#include "network.h"

namespace mceuchre {

// Per-seat suits a player is known to be void in, inferred from failing to
// follow suit so far (mirrors known_voids()).
std::array<uint8_t, 4> known_voids(const EuchreState& state);  // bitmask of suits, bit s set = void in suit s

// Sample a full-information EuchreState consistent with `player`'s
// information: their own hand and all public information preserved exactly;
// hidden hands/kitty are a random consistent completion. Throws
// std::runtime_error if no consistent completion is found within max_tries
// (mirrors the Python RuntimeError).
EuchreState sample_determinization(const EuchreState& state, int player,
                                   std::mt19937_64& rng, int max_tries = 400);

// Sample `num_worlds` determinizations for a BIDDING-phase root (BidRound1
// or BidRound2) and importance-weight them by how well each hypothesized
// hand explains the PASS sequence actually observed getting to `root`,
// under `net`'s own policy -- replaces the deleted rebel/belief_model.py
// heuristic with the net's own belief instead of a hand-tuned formula (see
// docs/rebel_design.md's "Planned: net-native C++ self-play" section for
// the full design rationale, including why this is always self-referential
// belief, not opponent modeling).
//
// No rejection sampling: every world from sample_determinization is kept,
// just reweighted, so this costs exactly one extra batched net.policy()
// call per solve (covering every world's every prefix step at once), not
// one call per world or a "redeal until N passes line up" retry loop.
//
// Every state on the path to a bidding-phase root is reachable by ONLY
// Pass actions from the top of the auction: turned_down is set (round 1
// was 4 passes) iff root.phase is BidRound2, and bids_seen counts passes
// within the current round (the first non-pass ends it) -- so the prefix
// is fully determined by (root.phase, root.bids_seen), never inferred from
// hands. Each world's own (already-fixed) hand is replayed through that
// same deterministic prefix to see what the net would have made of it.
//
// `weight_floor` is a fraction of the uniform weight (1/num_worlds) below
// which no world's final weight can fall -- keeps an early, noisy policy
// from fully starving a world of support. Throws std::invalid_argument if
// root isn't a BidRound1/BidRound2 decision.
std::pair<std::vector<EuchreState>, std::vector<double>>
sample_weighted_worlds(const EuchreState& root, int actor, int num_worlds,
                       PolicyValueNetImpl& net, std::mt19937_64& rng,
                       double weight_floor = 0.05);

}  // namespace mceuchre
