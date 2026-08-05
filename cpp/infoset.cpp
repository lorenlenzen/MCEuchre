#include "infoset.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace mceuchre {

namespace {

inline int rel(int seat, int me) { return ((seat - me) % 4 + 4) % 4; }

// Index into the 5-entry phase list used for the observation's phase one-hot
// (mirrors euchre/infoset.py's _PHASES -- Deal is deliberately absent;
// observation_tensor is never called on a Deal-phase state).
int phase_obs_index(Phase p) {
    switch (p) {
        case Phase::BidRound1: return 0;
        case Phase::BidRound2: return 1;
        case Phase::DealerDiscard: return 2;
        case Phase::Play: return 3;
        case Phase::Terminal: return 4;
        default: throw std::runtime_error("observation_tensor called on Deal phase");
    }
}

// Matches Python's Phase(enum.auto()) numbering exactly (declaration order,
// 1-based): Deal=1, BidRound1=2, BidRound2=3, DealerDiscard=4, Play=5,
// Terminal=6. infoset_key must byte-match the Python string, which embeds
// this value directly.
int phase_value(Phase p) {
    switch (p) {
        case Phase::Deal: return 1;
        case Phase::BidRound1: return 2;
        case Phase::BidRound2: return 3;
        case Phase::DealerDiscard: return 4;
        case Phase::Play: return 5;
        case Phase::Terminal: return 6;
    }
    throw std::runtime_error("unreachable");
}

// The suit everything is encoded relative to: trump once set, else the
// up-card's suit during bidding. -1 only in Deal/Terminal (no decision).
int reference_suit(const EuchreState& s) {
    if (s.trump.has_value()) return *s.trump;
    if (s.up_card.has_value()) return card_suit(*s.up_card);
    return -1;
}

int role_of(int suit, int ref) {
    if (ref < 0) return ROLE_GREEN;  // no reference; role is unused
    if (suit == ref) return ROLE_REF;
    if (suit == same_color_suit(ref)) return ROLE_NEXT;
    return ROLE_GREEN;
}

// _TRUMP_HOLDING_RANKS = [ACE, KING, QUEEN, TEN, NINE] -> rank indices
// (0=9,1=10,2=J,3=Q,4=K,5=A in this engine's numbering).
constexpr int TRUMP_HOLDING_RANK_IDX[5] = {5, 4, 3, 1, 0};

}  // namespace

void observation_tensor(const EuchreState& state, int player, float* out) {
    std::memset(out, 0, sizeof(float) * OBS_SIZE);
    Hand hand = state.hands[player];
    int trump = state.trump.has_value() ? *state.trump : -1;
    int ref = reference_suit(state);
    bool show_up = state.up_card.has_value() && !state.trump.has_value();

    // ---- global block ----
    int o = GLOBAL_OFF;
    out[o + phase_obs_index(state.phase)] = 1.0f;
    o += 5;
    out[o + rel(state.dealer, player)] = 1.0f;
    o += 4;
    out[o + (state.maker.has_value() ? rel(*state.maker, player) + 1 : 0)] = 1.0f;
    o += 5;
    out[o] = state.alone ? 1.0f : 0.0f;
    o += 1;
    int my_team = player % 2;
    out[o] = state.tricks_won[my_team] / 5.0f;
    out[o + 1] = state.tricks_won[1 - my_team] / 5.0f;
    o += 2;
    int my_score = my_team == 0 ? state.team0_score : state.team1_score;
    int their_score = my_team == 0 ? state.team1_score : state.team0_score;
    out[o] = static_cast<float>(my_score) / MATCH_TARGET;
    out[o + 1] = static_cast<float>(their_score) / MATCH_TARGET;
    o += 2;
    out[o] = (state.phase == Phase::Play && state.current_trick.empty()) ? 1.0f : 0.0f;
    o += 1;
    out[o] = static_cast<float>(state.bids_seen) / 7.0f;
    o += 1;
    out[o] = show_up ? 1.0f : 0.0f;
    o += 1;
    if (show_up) out[o + card_rank(*state.up_card)] = 1.0f;
    o += 6;
    if (!state.current_trick.empty()) {
        int led_role = role_of(effective_suit(state.current_trick[0].card, trump), ref);
        out[o + led_role + 1] = 1.0f;
    } else {
        out[o] = 1.0f;
    }
    o += 4;
    // assert(o == SUIT_OFF);

    // ---- per-suit blocks ----
    // Precompute public play info once: (rel_seat, card) for every card
    // played so far (completed tricks + current trick).
    std::vector<std::pair<int, CardId>> played_cards;
    for (const auto& t : state.completed_tricks)
        for (const auto& p : t.plays) played_cards.emplace_back(rel(p.player, player), p.card);
    for (const auto& p : state.current_trick)
        played_cards.emplace_back(rel(p.player, player), p.card);

    for (int s = 0; s < NUM_SUITS; ++s) {
        o = SUIT_OFF + s * SUIT_BLOCK_DIM;
        out[o + role_of(s, ref)] = 1.0f;
        o += N_ROLES;
        out[o] = (state.turned_down.has_value() && *state.turned_down == s) ? 1.0f : 0.0f;
        o += 1;
        out[o] = (trump == s) ? 1.0f : 0.0f;
        o += 1;
        // as-if-s-were-trump holdings: right bower, left bower, A, K, Q, 10, 9
        out[o] = hand_has(hand, make_card(s, 2)) ? 1.0f : 0.0f;              // right bower (Jack of s)
        out[o + 1] = hand_has(hand, make_card(same_color_suit(s), 2)) ? 1.0f : 0.0f;  // left bower
        for (int i = 0; i < 5; ++i)
            out[o + 2 + i] = hand_has(hand, make_card(s, TRUMP_HOLDING_RANK_IDX[i])) ? 1.0f : 0.0f;
        o += 7;
        int as_if_trump = 0;
        for (CardId c : hand_to_vector(hand)) if (is_trump(c, s)) as_if_trump++;
        out[o] = static_cast<float>(as_if_trump) / 5.0f;
        o += 1;
        // effective plain-suit holdings under ACTUAL trump (by rank)
        std::vector<CardId> eff_here;
        for (CardId c : hand_to_vector(hand))
            if (effective_suit(c, trump) == s) eff_here.push_back(c);
        for (CardId c : eff_here) out[o + card_rank(c)] = 1.0f;
        o += 6;
        out[o] = eff_here.empty() ? 1.0f : 0.0f;
        o += 1;
        // cards of this effective suit played, per relative seat (/5)
        for (const auto& [rel_seat, c] : played_cards)
            if (effective_suit(c, trump) == s) out[o + rel_seat] += 1.0f / 5.0f;
        o += 4;
        // cards of this (effective) suit seen so far. Divisor 7: effective
        // trump suit spans 7 cards (its own 6 + the left bower).
        int seen = static_cast<int>(eff_here.size());
        for (const auto& [_rs, c] : played_cards)
            if (effective_suit(c, trump) == s) seen++;
        if (show_up && effective_suit(*state.up_card, trump) == s) seen++;
        out[o] = static_cast<float>(seen) / 7.0f;
        o += 1;
    }

    // ---- per-card feature blocks (card-id order) ----
    std::vector<bool> played_set(NUM_CARDS, false);
    for (const auto& [_rs, c] : played_cards) played_set[c] = true;
    for (CardId cid = 0; cid < NUM_CARDS; ++cid) {
        o = CARD_OFF + cid * CARD_FEAT_DIM;
        out[o + card_rank(cid)] = 1.0f;
        o += 6;
        out[o] = hand_has(hand, cid) ? 1.0f : 0.0f;
        o += 1;
        out[o] = (state.trump.has_value() && is_right_bower(cid, *state.trump)) ? 1.0f : 0.0f;
        o += 1;
        out[o] = (state.trump.has_value() && is_left_bower(cid, *state.trump)) ? 1.0f : 0.0f;
        o += 1;
        out[o] = (state.trump.has_value() && is_trump(cid, *state.trump)) ? 1.0f : 0.0f;
        o += 1;
        out[o] = played_set[cid] ? 1.0f : 0.0f;
        o += 1;
    }
}

std::string infoset_key(const EuchreState& state, int player) {
    int my_score, their_score;
    if (team_of(player) == 0) {
        my_score = state.team0_score; their_score = state.team1_score;
    } else {
        my_score = state.team1_score; their_score = state.team0_score;
    }

    std::ostringstream os;
    os << "ph" << phase_value(state.phase)
       << "/d" << rel(state.dealer, player)
       << "/sc" << my_score << "," << their_score;

    // Private hand (sorted by id for canonical order).
    std::vector<CardId> hand_ids = hand_to_vector(state.hands[player]);
    os << "/h";
    for (size_t i = 0; i < hand_ids.size(); ++i) {
        if (i) os << ",";
        os << static_cast<int>(hand_ids[i]);
    }

    // Up-card is public during bidding.
    if ((state.phase == Phase::BidRound1 || state.phase == Phase::BidRound2 ||
        state.phase == Phase::DealerDiscard) && state.up_card.has_value()) {
        os << "/u" << static_cast<int>(*state.up_card);
    }
    if (state.turned_down.has_value()) os << "/td" << static_cast<int>(*state.turned_down);
    os << "/b" << state.bids_seen;

    if (state.trump.has_value()) {
        os << "/t" << static_cast<int>(*state.trump)
           << "/m" << rel(*state.maker, player)
           << "/a" << (state.alone ? 1 : 0);
    }

    for (const auto& t : state.completed_tricks) {
        os << "/T" << rel(t.winner, player) << "|";
        for (size_t i = 0; i < t.plays.size(); ++i) {
            if (i) os << ";";
            os << rel(t.plays[i].player, player) << ":" << static_cast<int>(t.plays[i].card);
        }
    }
    os << "/C";
    for (size_t i = 0; i < state.current_trick.size(); ++i) {
        if (i) os << ";";
        os << rel(state.current_trick[i].player, player) << ":"
           << static_cast<int>(state.current_trick[i].card);
    }
    os << "/w" << state.tricks_won[0] << "," << state.tricks_won[1];
    return os.str();
}

void legal_mask(const EuchreState& state, bool* out) {
    std::fill(out, out + NUM_ACTIONS, false);
    for (const auto& a : state.legal_actions()) {
        out[a.index()] = true;
    }
}

}  // namespace mceuchre
