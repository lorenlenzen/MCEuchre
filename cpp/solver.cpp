#include "solver.h"

#include <algorithm>
#include <sstream>

namespace mceuchre {

namespace {

constexpr int INF = 1'000'000'000;
enum Bound { LOWER = 0, EXACT = 1, UPPER = 2 };

// Transposition-table key. Only needs to be internally consistent (never
// compared cross-language), so plain string-building is fine -- correctness
// of solve_value's returned VALUE doesn't depend on this format at all.
std::string tt_key(const EuchreState& s) {
    std::ostringstream os;
    os << static_cast<int>(s.current_player) << "|";
    for (const auto& p : s.current_trick) os << static_cast<int>(p.player) << ":" << static_cast<int>(p.card) << ";";
    os << "|";
    for (int p = 0; p < 4; ++p) {
        auto ids = hand_to_vector(s.hands[p]);
        std::sort(ids.begin(), ids.end());
        for (CardId c : ids) os << static_cast<int>(c) << ",";
        os << "|";
    }
    os << s.tricks_won[0] << "," << s.tricks_won[1];
    return os.str();
}

// Move reduction + strongest-first ordering (mirrors solver.py's
// _ordered_plays). The exact tie-break order among strategically-equivalent
// alternatives doesn't affect solve_value's returned value (alpha-beta's
// result is move-order-invariant given a correct move SET, only pruning
// efficiency varies) -- only the kept representative SET matters for
// correctness, which is what this must get exactly right.
std::vector<CardId> ordered_plays(const EuchreState& state, int player) {
    auto plays = state.legal_plays(player);
    if (plays.size() <= 1) return plays;
    int trump = *state.trump;
    bool my[NUM_CARDS] = {false};
    for (CardId c : hand_to_vector(state.hands[player])) my[c] = true;
    bool legal[NUM_CARDS] = {false};
    for (CardId c : plays) legal[c] = true;

    // Bucket by effective suit (0..3): all 4 hands' cards + current-trick cards.
    std::array<std::vector<CardId>, NUM_SUITS> buckets;
    for (int p = 0; p < 4; ++p)
        for (CardId c : hand_to_vector(state.hands[p])) buckets[effective_suit(c, trump)].push_back(c);
    for (const auto& tp : state.current_trick) buckets[effective_suit(tp.card, trump)].push_back(tp.card);

    std::vector<CardId> keep;
    for (int suit = 0; suit < NUM_SUITS; ++suit) {
        auto& cards = buckets[suit];
        if (cards.empty()) continue;
        std::sort(cards.begin(), cards.end(), [&](CardId a, CardId b) {
            return card_strength(a, trump, suit) > card_strength(b, trump, suit);
        });
        size_t i = 0, n = cards.size();
        while (i < n) {
            if (my[cards[i]]) {
                std::vector<CardId> run;
                while (i < n && my[cards[i]]) { run.push_back(cards[i]); ++i; }
                CardId lowest_legal = -1;
                for (CardId c : run) if (legal[c]) lowest_legal = c;  // last legal in run
                if (lowest_legal >= 0) keep.push_back(lowest_legal);
            } else {
                ++i;
            }
        }
    }

    const std::vector<CardId>& reduced = keep.empty() ? plays : keep;
    int led = -1;
    bool has_led = !state.current_trick.empty();
    if (has_led) led = effective_suit(state.current_trick[0].card, trump);

    std::vector<CardId> out(reduced);
    std::sort(out.begin(), out.end(), [&](CardId a, CardId b) {
        int la = has_led ? led : card_suit(a);
        int lb = has_led ? led : card_suit(b);
        return card_strength(a, trump, la) > card_strength(b, trump, lb);
    });
    return out;
}

int ab(const EuchreState& state, int alpha, int beta,
      std::unordered_map<std::string, std::pair<int, int>>& memo) {
    if (state.is_terminal()) {
        auto [p0, p1] = state.returns();
        return p0 - p1;
    }

    int a0 = alpha, b0 = beta;
    std::string key = tt_key(state);
    auto it = memo.find(key);
    if (it != memo.end()) {
        auto [val, flag] = it->second;
        if (flag == EXACT) return val;
        if (flag == LOWER) alpha = std::max(alpha, val);
        else beta = std::min(beta, val);
        if (alpha >= beta) return val;
    }

    int player = state.current_player;
    bool maximizing = team_of(player) == 0;
    auto plays = ordered_plays(state, player);

    int value;
    if (maximizing) {
        value = -INF;
        for (CardId c : plays) {
            value = std::max(value, ab(state.apply(Action::play(c)), alpha, beta, memo));
            alpha = std::max(alpha, value);
            if (alpha >= beta) break;
        }
    } else {
        value = INF;
        for (CardId c : plays) {
            value = std::min(value, ab(state.apply(Action::play(c)), alpha, beta, memo));
            beta = std::min(beta, value);
            if (alpha >= beta) break;
        }
    }

    int flag = (value <= a0) ? UPPER : (value >= b0) ? LOWER : EXACT;
    memo[key] = {value, flag};
    return value;
}

}  // namespace

int solve_value(const EuchreState& state, std::unordered_map<std::string, std::pair<int, int>>* memo) {
    if (state.is_terminal()) {
        auto [p0, p1] = state.returns();
        return p0 - p1;
    }
    std::unordered_map<std::string, std::pair<int, int>> local;
    auto& m = memo ? *memo : local;
    return ab(state, -INF, INF, m);
}

}  // namespace mceuchre
