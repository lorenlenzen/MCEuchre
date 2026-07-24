// Exact double-dummy alpha-beta solver for the PLAY phase, ported from
// rebel/solver.py. Internal minimax stays raw-point (team0 - team1)
// throughout, deliberately -- see rebel/match_equity.py's module docstring
// for why that's exact, not an approximation, even under match equity.
#pragma once

#include <unordered_map>

#include "engine.h"

namespace mceuchre {

// team0 - team1 optimal point differential from `state` (must be PLAY phase
// or terminal). `memo` may be shared across sibling calls to reuse
// transpositions; pass nullptr for a fresh one-off solve.
int solve_value(const EuchreState& state, std::unordered_map<std::string, std::pair<int, int>>* memo = nullptr);

}  // namespace mceuchre
