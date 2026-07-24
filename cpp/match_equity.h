// C++ side of rebel/match_equity.py's MatchEquityModel -- read-only lookups
// (win_prob/equity_delta) needed by the CFR search's terminal-utility
// computation. Deliberately does NOT parse the JSON table file itself:
// Python's existing, already-tested MatchEquityModel.load() does that, and
// just hands the resulting flat table + target size to this constructor --
// avoids reimplementing JSON parsing (and its bug surface) in C++ for a
// format Python already loads correctly. Table *fitting* and score
// *sampling* also stay Python-only (self-play orchestration, never called
// from the C++ search itself).
#pragma once

#include <vector>

namespace mceuchre {

class MatchEquityModel {
public:
    // `table` is target*target values, row-major (table[a*target+b] ==
    // win_prob(a,b)) -- exactly numpy's default C-contiguous layout, so
    // pybind11 can hand over a flattened array with no reshaping needed.
    MatchEquityModel(int target, std::vector<double> table)
        : target_(target), table_(std::move(table)) {}

    double win_prob(int team0_score, int team1_score) const;
    // Team0-signed win-probability delta from one hand's (p0, p1) outcome.
    double equity_delta(int team0_score, int team1_score, int p0, int p1) const;

    int target() const { return target_; }

private:
    int target_ = 0;
    std::vector<double> table_;  // target_ x target_, row-major

    double at(int a, int b) const { return table_[a * target_ + b]; }
};

}  // namespace mceuchre
