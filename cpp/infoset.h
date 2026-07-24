// Observation encoding + infoset key, ported from euchre/infoset.py. Must
// match that file bit-for-bit (observation_tensor) / character-for-character
// (infoset_key) -- verified by tests/test_cpp_equivalence.py.
#pragma once

#include <string>
#include <vector>

#include "engine.h"

namespace mceuchre {

// Race-to-target match score, must agree with rebel/match_equity.py /
// euchre/infoset.py's MATCH_TARGET.
constexpr int MATCH_TARGET = 10;

// Roles of a suit relative to the reference suit R.
constexpr int ROLE_REF = 0, ROLE_NEXT = 1, ROLE_GREEN = 2;
constexpr int N_ROLES = 3;

// Segment dims -- must equal euchre/infoset.py's _GLOBAL/_SUIT_BLOCK/_CARD_FEAT.
constexpr int GLOBAL_DIM = 32;
constexpr int SUIT_BLOCK_DIM = 25;
constexpr int CARD_FEAT_DIM = 11;
constexpr int GLOBAL_OFF = 0;
constexpr int SUIT_OFF = GLOBAL_DIM;
constexpr int CARD_OFF = GLOBAL_DIM + NUM_SUITS * SUIT_BLOCK_DIM;
constexpr int OBS_SIZE = GLOBAL_DIM + NUM_SUITS * SUIT_BLOCK_DIM + NUM_CARDS * CARD_FEAT_DIM;  // 396

// Fills a caller-provided buffer of size OBS_SIZE (zero-initialized by this
// function). Kept as a raw-buffer API (not returning a container) so the
// pybind11 binding can write straight into a numpy array with no copy.
void observation_tensor(const EuchreState& state, int player, float* out);

std::string infoset_key(const EuchreState& state, int player);

}  // namespace mceuchre
