#include "match_equity.h"

namespace mceuchre {

double MatchEquityModel::win_prob(int my_score, int opp_score, bool am_i_dealer) const {
    if (am_i_dealer) {
        if (my_score >= target_) return 1.0;
        if (opp_score >= target_) return 0.0;
        return at(my_score, opp_score);
    }
    if (opp_score >= target_) return 0.0;
    if (my_score >= target_) return 1.0;
    return 1.0 - at(opp_score, my_score);
}

double MatchEquityModel::equity_delta(int team0_score, int team1_score, bool dealer_is_team0,
                                      int p0, int p1) const {
    double before = win_prob(team0_score, team1_score, dealer_is_team0);
    double after = win_prob(team0_score + p0, team1_score + p1, !dealer_is_team0);
    return after - before;
}

}  // namespace mceuchre
