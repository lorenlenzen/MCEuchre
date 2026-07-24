#include "match_equity.h"

namespace mceuchre {

double MatchEquityModel::win_prob(int team0_score, int team1_score) const {
    if (team0_score >= target_) return 1.0;
    if (team1_score >= target_) return 0.0;
    return at(team0_score, team1_score);
}

double MatchEquityModel::equity_delta(int team0_score, int team1_score, int p0, int p1) const {
    double before = win_prob(team0_score, team1_score);
    double after = win_prob(team0_score + p0, team1_score + p1);
    return after - before;
}

}  // namespace mceuchre
