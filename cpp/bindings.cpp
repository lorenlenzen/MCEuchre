#include <torch/extension.h>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "belief.h"
#include "engine.h"
#include "infoset.h"
#include "match_equity.h"
#include "network.h"
#include "solver.h"
#include "subgame.h"

namespace py = pybind11;
using namespace mceuchre;

static py::array_t<float> py_observation_tensor(const EuchreState& state, int player) {
    py::array_t<float> arr(OBS_SIZE);
    observation_tensor(state, player, arr.mutable_data());
    return arr;
}

static int py_solve_value(const EuchreState& state) { return solve_value(state); }

static EuchreState py_sample_determinization(const EuchreState& state, int player, uint64_t seed) {
    std::mt19937_64 rng(seed);
    return sample_determinization(state, player, rng);
}

PYBIND11_MODULE(mceuchre_cpp, m) {
    m.doc() = "C++ hot-path Euchre engine (see cpp/README.md)";

    py::enum_<Phase>(m, "Phase")
        .value("Deal", Phase::Deal)
        .value("BidRound1", Phase::BidRound1)
        .value("BidRound2", Phase::BidRound2)
        .value("DealerDiscard", Phase::DealerDiscard)
        .value("Play", Phase::Play)
        .value("Terminal", Phase::Terminal);

    py::enum_<ActionKind>(m, "ActionKind")
        .value("Pass", ActionKind::Pass)
        .value("OrderUp", ActionKind::OrderUp)
        .value("Call", ActionKind::Call)
        .value("Discard", ActionKind::Discard)
        .value("Play", ActionKind::Play);

    py::class_<Action>(m, "Action")
        .def_static("pass_", &Action::pass_)
        .def_static("order_up", &Action::order_up, py::arg("alone"))
        .def_static("call", &Action::call, py::arg("suit"), py::arg("alone"))
        .def_static("discard", &Action::discard, py::arg("card"))
        .def_static("play", &Action::play, py::arg("card"))
        .def_static("from_index", &Action::from_index, py::arg("index"))
        .def("index", &Action::index)
        .def_readonly("kind", &Action::kind)
        .def_readonly("alone", &Action::alone)
        .def_readonly("suit", &Action::suit)
        .def_readonly("card", &Action::card)
        .def("__eq__", &Action::operator==)
        .def("__repr__", [](const Action& a) {
            return "<Action idx=" + std::to_string(a.index()) + ">";
        });

    py::class_<TrickPlay>(m, "TrickPlay")
        .def_readonly("player", &TrickPlay::player)
        .def_readonly("card", &TrickPlay::card);

    py::class_<CompletedTrick>(m, "CompletedTrick")
        .def_readonly("winner", &CompletedTrick::winner)
        .def_readonly("plays", &CompletedTrick::plays);

    py::class_<EuchreState>(m, "EuchreState")
        .def_static("new_hand", &EuchreState::new_hand,
                    py::arg("dealer") = 0, py::arg("stick_the_dealer") = false,
                    py::arg("team0_score") = 0, py::arg("team1_score") = 0)
        .def("deal_from_deck", &EuchreState::deal_from_deck, py::arg("shuffled"))
        .def("deal_from", &EuchreState::deal_from,
            py::arg("hands"), py::arg("up_card"), py::arg("kitty"))
        .def("legal_actions", &EuchreState::legal_actions)
        .def("apply", &EuchreState::apply, py::arg("action"))
        .def("is_terminal", &EuchreState::is_terminal)
        .def("is_chance", &EuchreState::is_chance)
        .def("returns", &EuchreState::returns)
        .def_readonly("dealer", &EuchreState::dealer)
        .def_readonly("phase", &EuchreState::phase)
        .def_readonly("current_player", &EuchreState::current_player)
        .def_readonly("hands", &EuchreState::hands)
        .def_readonly("up_card", &EuchreState::up_card)
        .def_readonly("kitty", &EuchreState::kitty)
        .def_readonly("trump", &EuchreState::trump)
        .def_readonly("maker", &EuchreState::maker)
        .def_readonly("alone", &EuchreState::alone)
        .def_readonly("lone_player", &EuchreState::lone_player)
        .def_readonly("sitting", &EuchreState::sitting)
        .def_readonly("turned_down", &EuchreState::turned_down)
        .def_readonly("bids_seen", &EuchreState::bids_seen)
        .def_readonly("trick_leader", &EuchreState::trick_leader)
        .def_readonly("current_trick", &EuchreState::current_trick)
        .def_readonly("completed_tricks", &EuchreState::completed_tricks)
        .def_readonly("tricks_won", &EuchreState::tricks_won)
        .def_readonly("stick_the_dealer", &EuchreState::stick_the_dealer)
        .def_readonly("team0_score", &EuchreState::team0_score)
        .def_readonly("team1_score", &EuchreState::team1_score);

    // Free functions useful for differential testing / infoset encoding.
    m.def("is_right_bower", &is_right_bower);
    m.def("is_left_bower", &is_left_bower);
    m.def("is_trump", &is_trump);
    m.def("effective_suit", &effective_suit);
    m.def("same_color_suit", &same_color_suit);
    m.attr("NUM_CARDS") = NUM_CARDS;
    m.attr("NUM_ACTIONS") = NUM_ACTIONS;

    m.def("observation_tensor", &py_observation_tensor, py::arg("state"), py::arg("player"));
    m.def("infoset_key", &infoset_key, py::arg("state"), py::arg("player"));
    m.attr("OBS_SIZE") = OBS_SIZE;
    m.attr("GLOBAL_DIM") = GLOBAL_DIM;
    m.attr("SUIT_BLOCK_DIM") = SUIT_BLOCK_DIM;
    m.attr("CARD_FEAT_DIM") = CARD_FEAT_DIM;

    m.def("solve_value", &py_solve_value, py::arg("state"));

    py::class_<MatchEquityModel>(m, "MatchEquityModel")
        .def(py::init<int, std::vector<double>>(), py::arg("target"), py::arg("table_flat"))
        .def("win_prob", &MatchEquityModel::win_prob, py::arg("team0_score"), py::arg("team1_score"))
        .def("equity_delta", &MatchEquityModel::equity_delta,
            py::arg("team0_score"), py::arg("team1_score"), py::arg("p0"), py::arg("p1"))
        .def("target", &MatchEquityModel::target);

    m.def("sample_determinization", &py_sample_determinization,
         py::arg("state"), py::arg("player"), py::arg("seed"));

    py::class_<SubgameSolver>(m, "SubgameSolver")
        .def(py::init<const EuchreState&, int, int, int, int, BatchValueFn,
                      const MatchEquityModel*, uint64_t>(),
            py::arg("root"), py::arg("actor"), py::arg("num_worlds"), py::arg("iterations"),
            py::arg("depth_limit") = -1, py::arg("batch_value_fn") = BatchValueFn(),
            py::arg("equity_model") = nullptr, py::arg("seed") = 0,
            py::keep_alive<1, 7>())  // keep equity_model alive as long as the solver is
        .def(py::init<const EuchreState&, int, std::vector<EuchreState>, std::vector<double>,
                      int, int, BatchValueFn, const MatchEquityModel*>(),
            py::arg("root"), py::arg("actor"), py::arg("worlds"), py::arg("weights"),
            py::arg("iterations"), py::arg("depth_limit") = -1,
            py::arg("batch_value_fn") = BatchValueFn(), py::arg("equity_model") = nullptr,
            py::keep_alive<1, 8>())
        .def("run", &SubgameSolver::run)
        .def("root_policy", &SubgameSolver::root_policy)
        .def("root_value", &SubgameSolver::root_value);

    // Full nn.Module interop (state_dict/load_state_dict/parameters/to/
    // train/eval all work on the Python side normally) via bind_module --
    // submodule names above match rebel/networks.py's attribute names
    // exactly, so a checkpoint saved by the Python PolicyValueNet loads here
    // directly via load_state_dict.
    torch::python::bind_module<PolicyValueNetImpl>(m, "PolicyValueNet")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t>(),
            py::arg("obs_size") = OBS_SIZE, py::arg("num_actions") = NUM_ACTIONS,
            py::arg("suit_emb") = 64, py::arg("hidden") = 128, py::arg("context") = 128)
        .def("forward", &PolicyValueNetImpl::forward)
        .def("policy", &PolicyValueNetImpl::policy,
            py::arg("obs"), py::arg("legal_mask") = py::none())
        .def("state_dict_", &PolicyValueNetImpl::state_dict_)
        .def("load_state_dict_", &PolicyValueNetImpl::load_state_dict_, py::arg("state_dict"));
}
