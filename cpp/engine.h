// Euchre game engine, ported from euchre/game.py + euchre/cards.py +
// euchre/actions.py. See those files for the authoritative rules/semantics
// this must match exactly -- differential tests (tests/test_cpp_equivalence.py)
// verify this against the Python engine, which stays the correctness oracle.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mceuchre {

// --- cards -------------------------------------------------------------
// Card id: suit*6 + rank_index, 0..23, matching euchre/cards.py's Card.id
// exactly (needed for observation/checkpoint continuity). Suit 0=Clubs,
// 1=Diamonds, 2=Hearts, 3=Spades. Rank index 0=9,1=10,2=J,3=Q,4=K,5=A.
using CardId = int8_t;
constexpr int NUM_SUITS = 4;
constexpr int NUM_RANKS = 6;
constexpr int NUM_CARDS = NUM_SUITS * NUM_RANKS;  // 24

inline int card_suit(CardId c) { return c / NUM_RANKS; }
inline int card_rank(CardId c) { return c % NUM_RANKS; }  // 0=9 ... 5=A
inline CardId make_card(int suit, int rank) { return static_cast<CardId>(suit * NUM_RANKS + rank); }

// Same-color partner: Clubs<->Spades, Diamonds<->Hearts (matches
// cards.py's _SAME_COLOR).
inline int same_color_suit(int suit) {
    static const int table[4] = {3, 2, 1, 0};  // C<->S, D<->H
    return table[suit];
}

// Precomputed (card, trump) -> bool/suit lookup tables, same idea as the
// Python lookup-table optimization in euchre/cards.py, built once at
// startup. trump in [0,4); is_trump/is_right_bower/is_left_bower assume a
// real trump (never called with "no trump" in this engine, mirroring the
// guarded call sites in the Python code).
struct TrumpTables {
    std::array<std::array<bool, NUM_SUITS>, NUM_CARDS> is_right_bower{};
    std::array<std::array<bool, NUM_SUITS>, NUM_CARDS> is_left_bower{};
    std::array<std::array<bool, NUM_SUITS>, NUM_CARDS> is_trump{};
    std::array<std::array<int8_t, NUM_SUITS>, NUM_CARDS> effective_suit{};  // trump given
    TrumpTables();
};
extern const TrumpTables TRUMP;

inline bool is_right_bower(CardId c, int trump) { return TRUMP.is_right_bower[c][trump]; }
inline bool is_left_bower(CardId c, int trump) { return TRUMP.is_left_bower[c][trump]; }
inline bool is_trump(CardId c, int trump) { return TRUMP.is_trump[c][trump]; }
// trump == -1 means "no trump set" (bidding phase): effective suit is just the card's own suit.
inline int effective_suit(CardId c, int trump) {
    return trump < 0 ? card_suit(c) : TRUMP.effective_suit[c][trump];
}

// Trick-resolution strength (mirrors cards.py's card_strength/trick_winner).
int card_strength(CardId c, int trump, int led_suit);

// --- hands as bitmasks ---------------------------------------------------
using Hand = uint32_t;  // bit i set <=> card i in hand (24 bits used)
inline bool hand_has(Hand h, CardId c) { return (h >> c) & 1u; }
inline Hand hand_add(Hand h, CardId c) { return h | (1u << c); }
inline Hand hand_remove(Hand h, CardId c) { return h & ~(1u << c); }
int hand_count(Hand h);
std::vector<CardId> hand_to_vector(Hand h);

// --- actions -------------------------------------------------------------
// Mirrors euchre/actions.py's flat action-index space exactly (NUM_ACTIONS=59):
//   [0,24)   Play card c            -> c
//   [24,48)  Discard card c         -> 24+c
//   [48,52)  Call suit s (not alone)-> 48+s
//   [52,56)  Call suit s alone      -> 52+s
//   56       OrderUp (not alone)
//   57       OrderUp alone
//   58       Pass
constexpr int NUM_ACTIONS = 59;
constexpr int ACT_PLAY_BASE = 0;
constexpr int ACT_DISCARD_BASE = 24;
constexpr int ACT_CALL_BASE = 48;
constexpr int ACT_CALL_ALONE_BASE = 52;
constexpr int ACT_ORDER_UP = 56;
constexpr int ACT_ORDER_UP_ALONE = 57;
constexpr int ACT_PASS = 58;

enum class ActionKind : uint8_t { Pass, OrderUp, Call, Discard, Play };

struct Action {
    ActionKind kind;
    bool alone = false;      // OrderUp, Call
    int8_t suit = -1;        // Call
    CardId card = -1;        // Discard, Play

    static Action pass_() { return {ActionKind::Pass}; }
    static Action order_up(bool alone) { return {ActionKind::OrderUp, alone}; }
    static Action call(int suit, bool alone) { return {ActionKind::Call, alone, static_cast<int8_t>(suit)}; }
    static Action discard(CardId c) { return {ActionKind::Discard, false, -1, c}; }
    static Action play(CardId c) { return {ActionKind::Play, false, -1, c}; }

    int index() const;
    static Action from_index(int idx);
    bool operator==(const Action& o) const {
        return kind == o.kind && alone == o.alone && suit == o.suit && card == o.card;
    }
};

// --- phases ----------------------------------------------------------------
enum class Phase : uint8_t { Deal, BidRound1, BidRound2, DealerDiscard, Play, Terminal };

constexpr int CHANCE = -1;

inline int team_of(int player) { return player % 2; }
inline int partner_of(int player) { return (player + 2) % 4; }

// (winner, plays) for one completed trick; plays is (player, card) in play order.
struct TrickPlay { int8_t player; CardId card; };
struct CompletedTrick { int8_t winner; std::vector<TrickPlay> plays; };

// --- state -----------------------------------------------------------------
// Value semantics throughout (mirrors EuchreState's clone-on-apply design).
// Deliberately NOT holding any RNG state -- dealing is done by the caller
// (deal_random/deal_from), mirroring EuchreState.deal()/deal_from().
class EuchreState {
public:
    int8_t dealer = 0;
    Phase phase = Phase::Deal;
    int8_t current_player = CHANCE;

    std::array<Hand, 4> hands{};
    std::optional<CardId> up_card;
    std::vector<CardId> kitty;

    std::optional<int8_t> trump;         // -1/nullopt = none
    std::optional<int8_t> maker;
    bool alone = false;
    std::optional<int8_t> lone_player;
    std::optional<int8_t> sitting;
    std::optional<int8_t> turned_down;

    int bids_seen = 0;

    int8_t trick_leader = 0;
    std::vector<TrickPlay> current_trick;
    std::vector<CompletedTrick> completed_tricks;
    std::array<int, 2> tricks_won{0, 0};

    bool stick_the_dealer = false;
    int team0_score = 0;
    int team1_score = 0;

    std::optional<std::pair<int, int>> reward;

    static EuchreState new_hand(int dealer, bool stick_the_dealer,
                                int team0_score, int team1_score);

    // Dealing: caller supplies a shuffled 24-card deck (matches DECK order
    // semantics -- caller's responsibility to shuffle however it wants, so
    // Python-side RNG can drive it identically for differential testing).
    EuchreState deal_from_deck(const std::array<CardId, NUM_CARDS>& shuffled) const;
    EuchreState deal_from(const std::array<Hand, 4>& hands_in, CardId up,
                          const std::vector<CardId>& kitty_in) const;

    std::vector<Action> legal_actions() const;
    EuchreState apply(const Action& a) const;

    bool is_terminal() const { return phase == Phase::Terminal; }
    bool is_chance() const { return phase == Phase::Deal; }
    std::pair<int, int> returns() const;

    // Public (unlike euchre/game.py's _legal_plays, this is used by the
    // solver, which needs it for players other than current_player too when
    // bucketing all 4 hands for move-reduction).
    std::vector<CardId> legal_plays(int player) const;

private:
    EuchreState apply_bid1(const Action& a) const;
    EuchreState apply_bid2(const Action& a) const;
    EuchreState apply_discard(const Action& a) const;
    EuchreState apply_play(const Action& a) const;
    void set_alone(EuchreState& s, bool alone, int maker) const;
    void begin_play(EuchreState& s) const;
    int next_player(const EuchreState& s, int player) const;
    void finish_hand(EuchreState& s) const;
};

}  // namespace mceuchre
