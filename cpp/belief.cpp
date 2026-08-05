#include "belief.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <stdexcept>

#include "infoset.h"

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

std::pair<std::vector<EuchreState>, std::vector<double>>
sample_weighted_worlds(const EuchreState& root, int actor, int num_worlds,
                       PolicyValueNetImpl& net, std::mt19937_64& rng,
                       double weight_floor) {
    if (root.phase != Phase::BidRound1 && root.phase != Phase::BidRound2)
        throw std::invalid_argument(
            "sample_weighted_worlds: root must be BidRound1 or BidRound2");

    std::vector<EuchreState> worlds;
    worlds.reserve(num_worlds);
    for (int i = 0; i < num_worlds; ++i)
        worlds.push_back(sample_determinization(root, actor, rng));

    // See belief.h's docstring: the prefix is fully determined by
    // (root.phase, root.bids_seen), never by hands.
    int n_round1_pass = (root.phase == Phase::BidRound2) ? 4 : root.bids_seen;
    int n_round2_pass = (root.phase == Phase::BidRound2) ? root.bids_seen : 0;
    int steps_per_world = n_round1_pass + n_round2_pass;

    std::vector<double> uniform(num_worlds, 1.0 / num_worlds);
    if (steps_per_world == 0) {
        // Nothing observed yet (first to act in round 1) -- no bids to
        // condition on.
        return {std::move(worlds), std::move(uniform)};
    }

    // Collect every (obs, legal_mask) row across ALL worlds' prefix steps,
    // then score them in ONE batched net.policy() call -- the whole point
    // being this costs one extra forward pass per solve, not one per world
    // or per step.
    int total_steps = num_worlds * steps_per_world;
    torch::NoGradGuard no_grad;
    torch::Tensor obs = torch::empty({total_steps, OBS_SIZE}, torch::kFloat32);
    torch::Tensor mask = torch::zeros({total_steps, NUM_ACTIONS}, torch::kBool);
    float* obs_data = obs.data_ptr<float>();
    bool* mask_data = mask.data_ptr<bool>();

    int row = 0;
    for (const auto& w : worlds) {
        EuchreState replay = EuchreState::new_hand(root.dealer, root.stick_the_dealer,
                                                    root.team0_score, root.team1_score)
                                 .deal_from(w.hands, *w.up_card, w.kitty);
        for (int i = 0; i < steps_per_world; ++i) {
            observation_tensor(replay, replay.current_player, obs_data + row * OBS_SIZE);
            legal_mask(replay, mask_data + row * NUM_ACTIONS);
            replay = replay.apply(Action::pass_());
            ++row;
        }
    }

    torch::Tensor probs = net.policy(obs, mask);          // (total_steps, NUM_ACTIONS)
    torch::Tensor pass_p = probs.select(1, ACT_PASS).contiguous();
    const float* pp = pass_p.data_ptr<float>();

    std::vector<double> log_w(num_worlds, 0.0);
    row = 0;
    for (int wi = 0; wi < num_worlds; ++wi) {
        for (int i = 0; i < steps_per_world; ++i) {
            float p = std::max(pp[row], 1e-6f);  // floor: avoid log(0)
            log_w[wi] += std::log(static_cast<double>(p));
            ++row;
        }
    }

    double m = *std::max_element(log_w.begin(), log_w.end());
    std::vector<double> weights(num_worlds);
    double total = 0.0;
    for (int wi = 0; wi < num_worlds; ++wi) {
        weights[wi] = std::exp(log_w[wi] - m);
        total += weights[wi];
    }
    // Floor relative to uniform, then renormalize -- bounds how hard an
    // early (noisy) policy can starve a world of support entirely.
    double total2 = 0.0;
    for (int wi = 0; wi < num_worlds; ++wi) {
        weights[wi] = std::max(weights[wi] / total, weight_floor * uniform[wi]);
        total2 += weights[wi];
    }
    for (auto& w : weights) w /= total2;

    return {std::move(worlds), std::move(weights)};
}

}  // namespace mceuchre
