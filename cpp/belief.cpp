#include "belief.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <stdexcept>

namespace mceuchre {

std::array<uint8_t, 4> known_voids(const EuchreState& state) {
    std::array<uint8_t, 4> voids{0, 0, 0, 0};
    if (!state.trump.has_value()) return voids;
    int trump = *state.trump;

    auto process = [&](const std::vector<TrickPlay>& plays) {
        if (plays.empty()) return;
        int led = effective_suit(plays[0].card, trump);
        for (size_t i = 1; i < plays.size(); ++i) {
            if (effective_suit(plays[i].card, trump) != led)
                voids[plays[i].player] |= (1u << led);
        }
    };
    for (const auto& t : state.completed_tricks) process(t.plays);
    process(state.current_trick);
    return voids;
}

namespace {
bool ordered_up(const EuchreState& s) {
    return s.up_card.has_value() && s.trump.has_value() && s.maker.has_value()
        && !s.turned_down.has_value();
}
}  // namespace

EuchreState sample_determinization(const EuchreState& state, int player,
                                   std::mt19937_64& rng, int max_tries) {
    auto voids = known_voids(state);
    std::optional<int> trump = state.trump;
    std::optional<CardId> up = state.up_card;
    bool ord_up = ordered_up(state);
    bool discard_done = state.kitty.size() == 4;

    bool known[NUM_CARDS] = {false};
    for (CardId c : hand_to_vector(state.hands[player])) known[c] = true;
    for (const auto& t : state.completed_tricks)
        for (const auto& p : t.plays) known[p.card] = true;
    for (const auto& p : state.current_trick) known[p.card] = true;

    std::vector<CardId> kitty_fixed;
    struct UpConstraint { std::vector<int> seats; bool kitty_ok; };
    std::optional<UpConstraint> up_constraint;

    if (up.has_value()) {
        if (!ord_up) {
            known[*up] = true;  // case 1
        } else if (player == state.dealer) {  // case 2
            known[*up] = true;
            if (discard_done) {
                CardId discard = state.kitty.back();
                known[discard] = true;
                kitty_fixed.push_back(discard);
            }
        } else if (!discard_done) {  // case 3
            up_constraint = UpConstraint{{state.dealer}, false};
        } else {  // case 4
            up_constraint = UpConstraint{{state.dealer}, true};
        }
    }

    std::vector<CardId> unseen;
    for (CardId c = 0; c < NUM_CARDS; ++c) if (!known[c]) unseen.push_back(c);

    std::array<int, 4> need{};
    for (int p = 0; p < 4; ++p) need[p] = hand_count(state.hands[p]);
    need[player] = 0;
    int kitty_target = static_cast<int>(state.kitty.size());
    int kitty_slots = kitty_target - static_cast<int>(kitty_fixed.size());

    int need_sum = need[0] + need[1] + need[2] + need[3];
    if (need_sum + kitty_slots != static_cast<int>(unseen.size()))
        throw std::runtime_error("determinization slot count mismatch");

    for (int attempt = 0; attempt < max_tries; ++attempt) {
        std::vector<CardId> pool = unseen;
        std::shuffle(pool.begin(), pool.end(), rng);
        std::array<std::vector<CardId>, 4> assign;
        assign[player] = hand_to_vector(state.hands[player]);
        std::array<int, 4> cur_need = need;
        std::vector<CardId> kitty = kitty_fixed;

        auto destinations = [&](CardId card) -> std::vector<int> {
            if (up_constraint.has_value() && up.has_value() && card == *up) {
                std::vector<int> out;
                for (int p : up_constraint->seats) if (cur_need[p] > 0) out.push_back(p);
                return out;
            }
            std::vector<int> opts;
            for (int p = 0; p < 4; ++p) {
                if (p == player || cur_need[p] <= 0) continue;
                if (trump.has_value() && (voids[p] & (1u << effective_suit(card, *trump)))) continue;
                opts.push_back(p);
            }
            return opts;
        };
        auto kitty_allowed = [&](CardId card) -> bool {
            if (up_constraint.has_value() && up.has_value() && card == *up)
                return up_constraint->kitty_ok;
            return true;
        };

        std::function<bool(size_t)> place = [&](size_t i) -> bool {
            if (i == pool.size()) {
                for (int p = 0; p < 4; ++p) if (cur_need[p] != 0) return false;
                return static_cast<int>(kitty.size()) == kitty_target;
            }
            CardId card = pool[i];
            auto opts = destinations(card);
            std::shuffle(opts.begin(), opts.end(), rng);
            for (int p : opts) {
                assign[p].push_back(card);
                cur_need[p] -= 1;
                if (place(i + 1)) return true;
                assign[p].pop_back();
                cur_need[p] += 1;
            }
            if (kitty_allowed(card) && static_cast<int>(kitty.size()) < kitty_target) {
                kitty.push_back(card);
                if (place(i + 1)) return true;
                kitty.pop_back();
            }
            return false;
        };

        if (place(0)) {
            EuchreState s = state;
            std::array<Hand, 4> hands{};
            for (int p = 0; p < 4; ++p) {
                Hand h = 0;
                for (CardId c : assign[p]) h = hand_add(h, c);
                hands[p] = h;
            }
            s.hands = hands;
            s.kitty = kitty;
            return s;
        }
    }
    throw std::runtime_error("could not sample a consistent determinization");
}

}  // namespace mceuchre
