// Depth-limited CFR subgame solver, ported from rebel/subgame.py. This is
// the actual hot loop the whole C++ port exists for (69M _cfr calls in the
// profile that started this). See that file's module docstring for the
// algorithm description -- unchanged here, just compiled.
#pragma once

#include <deque>
#include <functional>
#include <random>
#include <unordered_map>

#include "engine.h"
#include "match_equity.h"

namespace mceuchre {

// Batched leaf-value callback: many states -> team0-team1 estimate per
// state. A Python callable can be passed here via pybind11 (calling the
// existing PyTorch net) so this is usable *before* the network is ported
// too -- the per-iteration CFR loop (the actual hot path) still runs
// entirely in C++ regardless, since leaf evaluation only happens once per
// _build_trees(), not once per iteration.
using BatchValueFn = std::function<std::vector<float>(const std::vector<EuchreState>&)>;

struct Info {
    std::vector<Action> actions;
    std::vector<double> regret;
    std::vector<double> strategy_sum;

    explicit Info(std::vector<Action> a)
        : actions(std::move(a)), regret(actions.size(), 0.0), strategy_sum(actions.size(), 0.0) {}

    std::vector<double> strategy() const;
    std::vector<double> average() const;
};

struct TNode {
    std::array<double, 4> util{};  // valid iff children.empty()
    bool has_util = false;
    Info* info = nullptr;          // non-null iff this is a decision node
    int player = -1;
    std::vector<TNode*> children;
};

class SubgameSolver {
public:
    // Normal (production) constructor: samples num_worlds determinizations
    // internally via sample_determinization.
    SubgameSolver(const EuchreState& root, int actor, int num_worlds, int iterations,
                  int depth_limit /* -1 = unlimited */, BatchValueFn batch_value_fn,
                  const MatchEquityModel* equity_model, uint64_t seed);

    // Testing constructor: takes explicit pre-built worlds (bypasses random
    // sampling entirely) so CFR's own math can be differential-tested
    // against Python bit-for-bit, independent of sample_determinization's
    // (deliberately not cross-language-identical) RNG sequence.
    SubgameSolver(const EuchreState& root, int actor,
                  std::vector<EuchreState> worlds, std::vector<double> weights,
                  int iterations, int depth_limit, BatchValueFn batch_value_fn,
                  const MatchEquityModel* equity_model);

    void run();
    // action_index -> probability, average strategy at the actor's root infoset.
    std::unordered_map<int, double> root_policy();
    double root_value();

private:
    void init_common(const EuchreState& root, int actor, int iterations, int depth_limit,
                     BatchValueFn batch_value_fn, const MatchEquityModel* equity_model);
    TNode* build(const EuchreState& state, int depth);
    std::array<double, 4> cfr(TNode* node, std::array<double, 4> reach, double chance_reach);
    void build_trees();
    std::array<double, 4> expected_value(TNode* node);

    int actor_ = 0;
    int iterations_ = 0;
    int depth_limit_ = -1;
    BatchValueFn batch_value_fn_;
    const MatchEquityModel* equity_model_ = nullptr;
    int team0_score_ = 0, team1_score_ = 0;
    bool dealer_is_team0_ = true;  // fixed for the whole subgame -- the deal doesn't change mid-hand

    std::vector<EuchreState> worlds_;
    std::vector<double> weights_;
    std::unordered_map<std::string, Info> infosets_;
    std::string root_key_;

    std::deque<TNode> node_storage_;  // stable references; owns all TNodes
    bool built_ = false;
    std::vector<TNode*> roots_;
    std::vector<std::pair<TNode*, EuchreState>> pending_leaves_;
};

}  // namespace mceuchre
