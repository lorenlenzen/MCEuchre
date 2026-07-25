// C++ side of rebel/match_equity.py's MatchEquityModel -- read-only lookups
// (win_prob/equity_delta) needed by the CFR search's terminal-utility
// computation. Deliberately does NOT parse the JSON table file itself:
// Python's existing, already-tested MatchEquityModel.load() does that, and
// just hands the resulting flat table + target size to this constructor --
// avoids reimplementing JSON parsing (and its bug surface) in C++ for a
// format Python already loads correctly. Table *fitting* and score
// *sampling* also stay Python-only (self-play orchestration, never called
// from the C++ search itself).
//
// Dealer-relative, not team0/team1-relative (matches rebel/match_equity.py
// exactly -- see that file's module docstring): `table` holds Ed[a,b], the
// win probability of the team about to deal, given their own score a and
// the opponent's score b. The non-dealing case is the exact identity
// 1 - Ed[b,a] (an identity of the underlying symmetric process, not an
// approximation), applied directly in win_prob rather than stored
// separately -- there is only ever one target*target table.
#pragma once

#include <vector>

namespace mceuchre {

class MatchEquityModel {
public:
    // `table` is target*target values, row-major (table[a*target+b] ==
    // Ed(a,b), the DEALING team's win prob) -- exactly numpy's default
    // C-contiguous layout, so pybind11 can hand over a flattened array with
    // no reshaping needed.
    MatchEquityModel(int target, std::vector<double> table)
        : target_(target), table_(std::move(table)) {}

    // My team's win probability, given my score, the opponent's score, and
    // whether my team deals the upcoming hand.
    double win_prob(int my_score, int opp_score, bool am_i_dealer) const;
    // Team0-signed win-probability delta from one hand's (p0, p1) outcome.
    // The deal passes to the OTHER team for the next hand (real Euchre
    // rule), so before/after deliberately query opposite dealer
    // orientations -- see rebel/match_equity.py's equity_delta docstring
    // for why getting this backwards is the one easy mistake here.
    double equity_delta(int team0_score, int team1_score, bool dealer_is_team0,
                        int p0, int p1) const;

    int target() const { return target_; }

private:
    int target_ = 0;
    std::vector<double> table_;  // target_ x target_, row-major, Ed[a,b]

    double at(int a, int b) const { return table_[a * target_ + b]; }
};

}  // namespace mceuchre
