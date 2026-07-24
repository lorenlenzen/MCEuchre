// PolicyValueNet, ported from rebel/networks.py. A real torch::nn::Module
// (not a plain struct) so the Python binding (torch::python::bind_module, in
// bindings.cpp) makes this a first-class nn.Module on the Python side --
// .state_dict()/.load_state_dict()/.parameters()/.to()/.train()/.eval() all
// work normally, and a checkpoint saved by the existing Python PolicyValueNet
// loads directly (submodule names below are chosen to match
// rebel/networks.py's attribute names exactly, so state_dict keys agree).
#pragma once

#include <torch/torch.h>

#include <map>
#include <string>
#include <tuple>

namespace mceuchre {

struct MLPImpl : torch::nn::Module {
    torch::nn::Sequential trunk{nullptr};
    torch::nn::Linear head{nullptr};

    MLPImpl(int64_t in_dim, int64_t hidden, int64_t out_dim, int64_t depth = 3);
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(MLP);

struct PolicyValueNetImpl : torch::nn::Module {
    int64_t suit_emb_dim, context_dim;

    MLP suit_encoder{nullptr};
    MLP context{nullptr};
    MLP make_trump{nullptr};
    MLP play_scorer{nullptr};
    MLP discard_scorer{nullptr};
    torch::nn::Linear pass_head{nullptr};
    MLP value_head{nullptr};

    PolicyValueNetImpl(int64_t obs_size, int64_t num_actions,
                       int64_t suit_emb = 64, int64_t hidden = 128, int64_t context_dim_ = 128);

    // Returns (logits[b, NUM_ACTIONS], value[b]).
    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& obs);

    // Legal, normalized action distribution (legal_mask: bool tensor, same
    // shape as logits; illegal entries set to -inf before softmax).
    torch::Tensor policy(const torch::Tensor& obs,
                        const c10::optional<torch::Tensor>& legal_mask);

    // torch::python::bind_module only gives the TOP-level module full Python
    // nn.Module machinery; nested C++ submodules lack _load_from_state_dict,
    // so the generic (Python-side) load_state_dict()/state_dict() recursion
    // breaks on this module's children. These walk named_parameters()/
    // named_buffers() directly (pure C++, unaffected by that gap) instead --
    // state_dict_() returns a plain name->tensor dict Python's torch.save()
    // and a plain Python PolicyValueNet's own load_state_dict() both handle
    // normally, and load_state_dict_() accepts exactly that same shape back
    // (e.g. from an existing Python-trained checkpoint's state_dict()).
    std::map<std::string, torch::Tensor> state_dict_() const;
    void load_state_dict_(const std::map<std::string, torch::Tensor>& sd);
};
TORCH_MODULE(PolicyValueNet);

}  // namespace mceuchre
