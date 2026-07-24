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

#include "engine.h"

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

}  // namespace mceuchre
