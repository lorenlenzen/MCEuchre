#include "subgame.h"

#include <numeric>
#include <stdexcept>

#include "belief.h"
#include "infoset.h"

namespace mceuchre {

std::vector<double> Info::strategy() const {
    std::vector<double> pos(regret.size());
    double s = 0.0;
    for (size_t i = 0; i < regret.size(); ++i) {
        pos[i] = regret[i] > 0.0 ? regret[i] : 0.0;
        s += pos[i];
    }
    if (s > 0.0) {
        double inv = 1.0 / s;
        for (auto& x : pos) x *= inv;
        return pos;
    }
    double u = 1.0 / static_cast<double>(pos.size());
    return std::vector<double>(pos.size(), u);
}

std::vector<double> Info::average() const {
    double s = std::accumulate(strategy_sum.begin(), strategy_sum.end(), 0.0);
    if (s > 0.0) {
        double inv = 1.0 / s;
        std::vector<double> out(strategy_sum.size());
        for (size_t i = 0; i < out.size(); ++i) out[i] = strategy_sum[i] * inv;
        return out;
    }
    double u = 1.0 / static_cast<double>(strategy_sum.size());
    return std::vector<double>(strategy_sum.size(), u);
}

void SubgameSolver::init_common(const EuchreState& root, int actor, int iterations,
                                int depth_limit, BatchValueFn batch_value_fn,
                                const MatchEquityModel* equity_model) {
    if (root.is_terminal() || root.current_player != actor)
        throw std::invalid_argument("subgame root must be a decision node for actor");
    actor_ = actor;
    iterations_ = iterations;
    depth_limit_ = depth_limit;
    batch_value_fn_ = std::move(batch_value_fn);
    equity_model_ = equity_model;
    team0_score_ = root.team0_score;
    team1_score_ = root.team1_score;
    dealer_is_team0_ = team_of(root.dealer) == 0;
    root_phase_ = root.phase;
    root_key_ = infoset_key(root, actor);
}

SubgameSolver::SubgameSolver(const EuchreState& root, int actor, int num_worlds, int iterations,
                             int depth_limit, BatchValueFn batch_value_fn,
                             const MatchEquityModel* equity_model, uint64_t seed) {
    init_common(root, actor, iterations, depth_limit, std::move(batch_value_fn), equity_model);
    std::mt19937_64 rng(seed);
    worlds_.reserve(num_worlds);
    for (int i = 0; i < num_worlds; ++i)
        worlds_.push_back(sample_determinization(root, actor, rng));
    weights_.assign(num_worlds, 1.0 / num_worlds);
}

SubgameSolver::SubgameSolver(const EuchreState& root, int actor,
                             std::vector<EuchreState> worlds, std::vector<double> weights,
                             int iterations, int depth_limit, BatchValueFn batch_value_fn,
                             const MatchEquityModel* equity_model) {
    init_common(root, actor, iterations, depth_limit, std::move(batch_value_fn), equity_model);
    worlds_ = std::move(worlds);
    weights_ = std::move(weights);
}

TNode* SubgameSolver::build(const EuchreState& state, int depth) {
    node_storage_.emplace_back();
    TNode* node = &node_storage_.back();

    if (state.is_terminal()) {
        auto [p0, p1] = state.returns();
        double diff = equity_model_
            ? equity_model_->equity_delta(team0_score_, team1_score_, dealer_is_team0_, p0, p1)
            : static_cast<double>(p0 - p1);
        for (int p = 0; p < 4; ++p) node->util[p] = (team_of(p) == 0) ? diff : -diff;
        node->has_util = true;
        return node;
    }
    if (depth_limit_ >= 0 && depth >= depth_limit_) {
        if (batch_value_fn_) {
            pending_leaves_.emplace_back(node, state);  // resolved later, batched
            return node;
        }
        node->util = {0.0, 0.0, 0.0, 0.0};
        node->has_util = true;
        return node;
    }

    int player = state.current_player;
    std::string key = infoset_key(state, player);
    auto it = infosets_.find(key);
    if (it == infosets_.end()) {
        it = infosets_.emplace(key, Info(state.legal_actions())).first;
    }
    Info& info = it->second;
    node->info = &info;
    node->player = player;
    node->children.reserve(info.actions.size());

    // Subgame boundary = phase boundary (ported from rebel/subgame.py's
    // build() -- see its comment for the full rationale). BidRound1/2 nodes
    // expand FULLY regardless of depth_limit_ -- the auction is short and
    // bounded on its own (<=4 round-1 + <=4 round-2 decisions before trump
    // is set or a real misdeal terminal is hit, caught by the is_terminal()
    // check above) -- and the instant a child's phase becomes Play, that
    // child is cut immediately as a leaf instead of recursing into real card
    // play. This fixes the same structural asymmetry the Python side does
    // (OrderUp/Call reaching search-backed values while Pass fell back on an
    // unverified net guess) without the blowup a naive "just don't count
    // bidding plies" version has -- measured on the Python path at ~53x
    // slower; this version instead measured ~23x FASTER than the pre-fix
    // flat ply-count on BidRound1-rooted solves.
    //
    // DealerDiscard and BidRound2 are free ONLY when reached as an INTERNAL
    // node of a bidding-rooted solve, NOT when either is the solve's own
    // root (a real discard or round-2-call decision). Both have at least
    // one action that transitions DIRECTLY into Phase::Play (Discard;
    // Call -- unlike round 1's OrderUp, which always passes through
    // DealerDiscard first, euchre/game.py's _apply_bid2 calls _begin_play()
    // directly for Call). If either were always free, a root-level solve
    // over them would be nothing but the root plus its immediate leaves --
    // no real search at all, so regret-matching over them degenerates to
    // comparing unbacked value-net guesses. Measured for DealerDiscard:
    // exactly 1 infoset, exactly-uniform policy regardless of net quality.
    // BidRound2 had the identical gap for its Call actions specifically --
    // every alone/not-alone/suit option was an unbacked same-depth leaf,
    // letting any value-head bias between them train directly into the
    // policy uncorrected (surfaced as "every alone option outranks its
    // same-suit non-alone twin" on the quiz). BidRound1 has no such gap
    // (neither Pass nor OrderUp's child is ever directly Phase::Play), so
    // it stays free unconditionally, root or not. Falling through to the
    // ply-counted path below instead restores real card-play lookahead for
    // root-level decisions over either phase, same as Play.
    bool is_free = (state.phase == Phase::BidRound1
                    || ((state.phase == Phase::BidRound2 || state.phase == Phase::DealerDiscard)
                        && root_phase_ != state.phase));
    if (depth_limit_ >= 0 && is_free) {
        for (const auto& a : info.actions) {
            EuchreState child = state.apply(a);
            if (child.phase == Phase::Play && !child.is_terminal()) {
                node_storage_.emplace_back();
                TNode* leaf = &node_storage_.back();
                if (batch_value_fn_) {
                    pending_leaves_.emplace_back(leaf, child);
                } else {
                    leaf->util = {0.0, 0.0, 0.0, 0.0};
                    leaf->has_util = true;
                }
                node->children.push_back(leaf);
            } else {
                node->children.push_back(build(child, depth));
            }
        }
        return node;
    }

    int child_depth = is_free ? depth : depth + 1;
    for (const auto& a : info.actions) {
        node->children.push_back(build(state.apply(a), child_depth));
    }
    return node;
}

std::array<double, 4> SubgameSolver::cfr(TNode* node, std::array<double, 4> reach, double chance_reach) {
    if (node->info == nullptr) return node->util;  // terminal or depth-limit leaf

    Info& info = *node->info;
    int player = node->player;
    std::vector<double> strat = info.strategy();
    size_t n = node->children.size();

    std::array<double, 4> node_util{0, 0, 0, 0};
    std::vector<std::array<double, 4>> child_utils(n);
    for (size_t i = 0; i < n; ++i) {
        std::array<double, 4> new_reach = reach;
        new_reach[player] *= strat[i];
        child_utils[i] = cfr(node->children[i], new_reach, chance_reach);
        double si = strat[i];
        for (int q = 0; q < 4; ++q) node_util[q] += si * child_utils[i][q];
    }

    double cf = chance_reach;
    for (int q = 0; q < 4; ++q) if (q != player) cf *= reach[q];
    double rp = reach[player];
    double npu = node_util[player];
    for (size_t i = 0; i < n; ++i) {
        info.regret[i] += cf * (child_utils[i][player] - npu);
        info.strategy_sum[i] += rp * strat[i];
    }
    return node_util;
}

void SubgameSolver::build_trees() {
    if (built_) return;
    roots_.clear();
    roots_.reserve(worlds_.size());
    for (const auto& w : worlds_) roots_.push_back(build(w, 0));
    if (!pending_leaves_.empty()) {
        std::vector<EuchreState> states;
        states.reserve(pending_leaves_.size());
        for (auto& [node, st] : pending_leaves_) states.push_back(st);
        std::vector<float> values = batch_value_fn_(states);
        for (size_t i = 0; i < pending_leaves_.size(); ++i) {
            TNode* node = pending_leaves_[i].first;
            double v0 = values[i];
            for (int p = 0; p < 4; ++p) node->util[p] = (team_of(p) == 0) ? v0 : -v0;
            node->has_util = true;
        }
        pending_leaves_.clear();
    }
    built_ = true;
}

void SubgameSolver::run() {
    build_trees();
    for (int iter = 0; iter < iterations_; ++iter) {
        for (size_t i = 0; i < roots_.size(); ++i) {
            cfr(roots_[i], {1.0, 1.0, 1.0, 1.0}, weights_[i]);
        }
    }
}

std::unordered_map<int, double> SubgameSolver::root_policy() {
    if (!built_) run();
    Info& info = infosets_.at(root_key_);
    std::vector<double> avg = info.average();
    std::unordered_map<int, double> out;
    for (size_t i = 0; i < info.actions.size(); ++i) out[info.actions[i].index()] = avg[i];
    return out;
}

std::array<double, 4> SubgameSolver::expected_value(TNode* node) {
    if (node->info == nullptr) return node->util;
    std::vector<double> avg = node->info->average();
    std::array<double, 4> node_util{0, 0, 0, 0};
    for (size_t i = 0; i < node->children.size(); ++i) {
        auto cu = expected_value(node->children[i]);
        double a = avg[i];
        for (int q = 0; q < 4; ++q) node_util[q] += a * cu[q];
    }
    return node_util;
}

double SubgameSolver::root_value() {
    build_trees();
    double total = 0.0;
    for (size_t i = 0; i < roots_.size(); ++i) {
        total += weights_[i] * expected_value(roots_[i])[0];
    }
    return total;
}

}  // namespace mceuchre
