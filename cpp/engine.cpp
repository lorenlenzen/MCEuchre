#include "engine.h"

#include <algorithm>

namespace mceuchre {

// --- trump tables ------------------------------------------------------

TrumpTables::TrumpTables() {
    for (int suit = 0; suit < NUM_SUITS; ++suit) {
        for (int rank = 0; rank < NUM_RANKS; ++rank) {
            CardId c = make_card(suit, rank);
            for (int trump = 0; trump < NUM_SUITS; ++trump) {
                bool rb = (rank == 2 /* Jack */) && (suit == trump);
                bool lb = (rank == 2) && (suit == same_color_suit(trump));
                is_right_bower[c][trump] = rb;
                is_left_bower[c][trump] = lb;
                is_trump[c][trump] = (suit == trump) || lb;
                effective_suit[c][trump] = lb ? static_cast<int8_t>(trump)
                                              : static_cast<int8_t>(suit);
            }
        }
    }
}

const TrumpTables TRUMP;

namespace {
// Rank index 0=9,1=10,2=J,3=Q,4=K,5=A (matches euchre/cards.py's RANKS order).
constexpr int TRUMP_RANK_STRENGTH[NUM_RANKS] = {2, 3, -1, 4, 5, 6};  // J unused (bower path)
constexpr int PLAIN_RANK_STRENGTH[NUM_RANKS] = {1, 2, 3, 4, 5, 6};
}  // namespace

int card_strength(CardId c, int trump, int led_suit) {
    if (is_right_bower(c, trump)) return 200 + 8;
    if (is_left_bower(c, trump)) return 200 + 7;
    int suit = card_suit(c), rank = card_rank(c);
    if (suit == trump) return 200 + TRUMP_RANK_STRENGTH[rank];
    if (effective_suit(c, trump) == led_suit) return 100 + PLAIN_RANK_STRENGTH[rank];
    return PLAIN_RANK_STRENGTH[rank];
}

static int trick_winner(const std::vector<TrickPlay>& plays, int trump) {
    int led = effective_suit(plays[0].card, trump);
    int best_player = plays[0].player;
    int best_strength = card_strength(plays[0].card, trump, led);
    for (size_t i = 1; i < plays.size(); ++i) {
        int s = card_strength(plays[i].card, trump, led);
        if (s > best_strength) {
            best_strength = s;
            best_player = plays[i].player;
        }
    }
    return best_player;
}

// --- hand utilities ------------------------------------------------------

int hand_count(Hand h) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt(h));
#else
    return __builtin_popcount(h);
#endif
}

std::vector<CardId> hand_to_vector(Hand h) {
    std::vector<CardId> out;
    for (CardId c = 0; c < NUM_CARDS; ++c) {
        if (hand_has(h, c)) out.push_back(c);
    }
    return out;
}

// --- action index mapping (mirrors euchre/actions.py exactly) -----------

int Action::index() const {
    switch (kind) {
        case ActionKind::Play: return ACT_PLAY_BASE + card;
        case ActionKind::Discard: return ACT_DISCARD_BASE + card;
        case ActionKind::Call: return (alone ? ACT_CALL_ALONE_BASE : ACT_CALL_BASE) + suit;
        case ActionKind::OrderUp: return alone ? ACT_ORDER_UP_ALONE : ACT_ORDER_UP;
        case ActionKind::Pass: return ACT_PASS;
    }
    throw std::runtime_error("unreachable");
}

Action Action::from_index(int idx) {
    if (idx >= ACT_PLAY_BASE && idx < ACT_DISCARD_BASE)
        return Action::play(static_cast<CardId>(idx - ACT_PLAY_BASE));
    if (idx >= ACT_DISCARD_BASE && idx < ACT_CALL_BASE)
        return Action::discard(static_cast<CardId>(idx - ACT_DISCARD_BASE));
    if (idx >= ACT_CALL_BASE && idx < ACT_CALL_ALONE_BASE)
        return Action::call(idx - ACT_CALL_BASE, false);
    if (idx >= ACT_CALL_ALONE_BASE && idx < ACT_ORDER_UP)
        return Action::call(idx - ACT_CALL_ALONE_BASE, true);
    if (idx == ACT_ORDER_UP) return Action::order_up(false);
    if (idx == ACT_ORDER_UP_ALONE) return Action::order_up(true);
    if (idx == ACT_PASS) return Action::pass_();
    throw std::runtime_error("action index out of range");
}

// --- EuchreState -----------------------------------------------------------

EuchreState EuchreState::new_hand(int dealer, bool stick_the_dealer,
                                  int team0_score, int team1_score) {
    EuchreState s;
    s.dealer = static_cast<int8_t>(dealer);
    s.phase = Phase::Deal;
    s.current_player = CHANCE;
    s.stick_the_dealer = stick_the_dealer;
    s.team0_score = team0_score;
    s.team1_score = team1_score;
    return s;
}

EuchreState EuchreState::deal_from_deck(const std::array<CardId, NUM_CARDS>& shuffled) const {
    EuchreState s = *this;
    for (int p = 0; p < 4; ++p) {
        Hand h = 0;
        for (int i = 0; i < 5; ++i) h = hand_add(h, shuffled[p * 5 + i]);
        s.hands[p] = h;
    }
    s.up_card = shuffled[20];
    s.kitty = {shuffled[21], shuffled[22], shuffled[23]};
    s.phase = Phase::BidRound1;
    s.current_player = (dealer + 1) % 4;
    s.bids_seen = 0;
    return s;
}

EuchreState EuchreState::deal_from(const std::array<Hand, 4>& hands_in, CardId up,
                                   const std::vector<CardId>& kitty_in) const {
    EuchreState s = *this;
    s.hands = hands_in;
    s.up_card = up;
    s.kitty = kitty_in;
    s.phase = Phase::BidRound1;
    s.current_player = (dealer + 1) % 4;
    s.bids_seen = 0;
    return s;
}

std::vector<CardId> EuchreState::legal_plays(int player) const {
    Hand hand = hands[player];
    if (current_trick.empty()) return hand_to_vector(hand);
    int trump_i = trump.has_value() ? *trump : -1;
    int led = effective_suit(current_trick[0].card, trump_i);
    std::vector<CardId> follow;
    for (CardId c : hand_to_vector(hand)) {
        if (effective_suit(c, trump_i) == led) follow.push_back(c);
    }
    return follow.empty() ? hand_to_vector(hand) : follow;
}

std::vector<Action> EuchreState::legal_actions() const {
    std::vector<Action> out;
    if (phase == Phase::BidRound1) {
        out = {Action::pass_(), Action::order_up(false), Action::order_up(true)};
        return out;
    }
    if (phase == Phase::BidRound2) {
        out.push_back(Action::pass_());
        for (int suit = 0; suit < NUM_SUITS; ++suit) {
            if (turned_down.has_value() && suit == *turned_down) continue;
            out.push_back(Action::call(suit, false));
            out.push_back(Action::call(suit, true));
        }
        if (stick_the_dealer && current_player == dealer && bids_seen == 3) {
            std::vector<Action> filtered;
            for (const auto& a : out) if (a.kind != ActionKind::Pass) filtered.push_back(a);
            out = filtered;
        }
        return out;
    }
    if (phase == Phase::DealerDiscard) {
        for (CardId c : hand_to_vector(hands[dealer])) out.push_back(Action::discard(c));
        return out;
    }
    if (phase == Phase::Play) {
        for (CardId c : legal_plays(current_player)) out.push_back(Action::play(c));
        return out;
    }
    return out;
}

EuchreState EuchreState::apply(const Action& a) const {
    switch (phase) {
        case Phase::BidRound1: return apply_bid1(a);
        case Phase::BidRound2: return apply_bid2(a);
        case Phase::DealerDiscard: return apply_discard(a);
        case Phase::Play: return apply_play(a);
        default: throw std::runtime_error("no actions from this phase");
    }
}

void EuchreState::set_alone(EuchreState& s, bool is_alone, int maker_player) const {
    s.alone = is_alone;
    if (is_alone) {
        s.lone_player = static_cast<int8_t>(maker_player);
        s.sitting = static_cast<int8_t>(partner_of(maker_player));
    } else {
        s.lone_player.reset();
        s.sitting.reset();
    }
}

int EuchreState::next_player(const EuchreState& s, int player) const {
    int nxt = (player + 1) % 4;
    if (s.sitting.has_value() && nxt == *s.sitting) nxt = (nxt + 1) % 4;
    return nxt;
}

void EuchreState::begin_play(EuchreState& s) const {
    s.phase = Phase::Play;
    int leader = (s.dealer + 1) % 4;
    if (s.sitting.has_value() && leader == *s.sitting) leader = next_player(s, leader);
    s.trick_leader = static_cast<int8_t>(leader);
    s.current_player = static_cast<int8_t>(leader);
    s.current_trick.clear();
}

EuchreState EuchreState::apply_bid1(const Action& a) const {
    EuchreState s = *this;
    if (a.kind == ActionKind::Pass) {
        s.bids_seen += 1;
        if (s.bids_seen == 4) {
            s.turned_down = static_cast<int8_t>(card_suit(*up_card));
            s.phase = Phase::BidRound2;
            s.current_player = (dealer + 1) % 4;
            s.bids_seen = 0;
        } else {
            s.current_player = (current_player + 1) % 4;
        }
        return s;
    }
    if (a.kind == ActionKind::OrderUp) {
        s.trump = static_cast<int8_t>(card_suit(*up_card));
        s.maker = current_player;
        set_alone(s, a.alone, current_player);
        s.hands[dealer] = hand_add(s.hands[dealer], *up_card);
        s.phase = Phase::DealerDiscard;
        s.current_player = dealer;
        return s;
    }
    throw std::runtime_error("illegal round-1 bid");
}

EuchreState EuchreState::apply_bid2(const Action& a) const {
    EuchreState s = *this;
    if (a.kind == ActionKind::Pass) {
        s.bids_seen += 1;
        if (s.bids_seen == 4) {
            s.phase = Phase::Terminal;
            s.reward = std::make_pair(0, 0);
            s.current_player = CHANCE;
            return s;
        }
        s.current_player = (current_player + 1) % 4;
        return s;
    }
    if (a.kind == ActionKind::Call) {
        if (turned_down.has_value() && a.suit == *turned_down)
            throw std::runtime_error("cannot call the turned-down suit");
        s.trump = a.suit;
        s.maker = current_player;
        set_alone(s, a.alone, current_player);
        begin_play(s);
        return s;
    }
    throw std::runtime_error("illegal round-2 bid");
}

EuchreState EuchreState::apply_discard(const Action& a) const {
    if (a.kind != ActionKind::Discard) throw std::runtime_error("expected a discard");
    EuchreState s = *this;
    s.hands[dealer] = hand_remove(s.hands[dealer], a.card);
    s.kitty.push_back(a.card);
    begin_play(s);
    return s;
}

void EuchreState::finish_hand(EuchreState& s) const {
    s.phase = Phase::Terminal;
    s.current_player = CHANCE;
    int maker_team = team_of(*s.maker);
    int maker_tricks = s.tricks_won[maker_team];
    int points[2] = {0, 0};
    if (maker_tricks >= 3) {
        points[maker_team] = (maker_tricks == 5) ? (s.alone ? 4 : 2) : 1;
    } else {
        points[1 - maker_team] = 2;
    }
    s.reward = std::make_pair(points[0], points[1]);
}

EuchreState EuchreState::apply_play(const Action& a) const {
    if (a.kind != ActionKind::Play) throw std::runtime_error("expected a play");
    auto legal = legal_plays(current_player);
    if (std::find(legal.begin(), legal.end(), a.card) == legal.end())
        throw std::runtime_error("illegal play");
    EuchreState s = *this;
    s.hands[current_player] = hand_remove(s.hands[current_player], a.card);
    s.current_trick.push_back({static_cast<int8_t>(current_player), a.card});

    int expected = alone ? 3 : 4;
    if (static_cast<int>(s.current_trick.size()) == expected) {
        int trump_i = *s.trump;
        int winner = trick_winner(s.current_trick, trump_i);
        s.completed_tricks.push_back({static_cast<int8_t>(winner), s.current_trick});
        s.tricks_won[team_of(winner)] += 1;
        s.current_trick.clear();
        if (static_cast<int>(s.completed_tricks.size()) == 5) {
            finish_hand(s);
        } else {
            s.trick_leader = static_cast<int8_t>(winner);
            s.current_player = static_cast<int8_t>(winner);
        }
    } else {
        s.current_player = static_cast<int8_t>(next_player(s, current_player));
    }
    return s;
}

std::pair<int, int> EuchreState::returns() const {
    if (!reward.has_value()) throw std::runtime_error("returns() called before terminal");
    return *reward;
}

}  // namespace mceuchre
