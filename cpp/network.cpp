#include "network.h"

#include <limits>

#include "engine.h"
#include "infoset.h"

namespace mceuchre {

namespace {
constexpr int64_t CARDS_PER_SUIT = NUM_CARDS / NUM_SUITS;  // 6
}  // namespace

MLPImpl::MLPImpl(int64_t in_dim, int64_t hidden, int64_t out_dim, int64_t depth) {
    trunk = torch::nn::Sequential();
    trunk->push_back(torch::nn::Linear(in_dim, hidden));
    trunk->push_back(torch::nn::ReLU());
    for (int64_t i = 1; i < depth; ++i) {
        trunk->push_back(torch::nn::Linear(hidden, hidden));
        trunk->push_back(torch::nn::ReLU());
    }
    register_module("trunk", trunk);
    head = torch::nn::Linear(hidden, out_dim);
    register_module("head", head);
}

torch::Tensor MLPImpl::forward(const torch::Tensor& x) {
    return head(trunk->forward(x));
}

PolicyValueNetImpl::PolicyValueNetImpl(int64_t obs_size, int64_t num_actions,
                                       int64_t suit_emb, int64_t hidden, int64_t context_dim_) {
    TORCH_CHECK(obs_size == OBS_SIZE, "obs_size must equal OBS_SIZE");
    TORCH_CHECK(num_actions == NUM_ACTIONS, "num_actions must equal NUM_ACTIONS");
    suit_emb_dim = suit_emb;
    context_dim = context_dim_;

    suit_encoder = MLP(SUIT_BLOCK_DIM, hidden, suit_emb, 2);
    register_module("suit_encoder", suit_encoder);
    context = MLP(suit_emb + GLOBAL_DIM, hidden, context_dim_, 2);
    register_module("context", context);
    make_trump = MLP(suit_emb + context_dim_, hidden, 2, 2);
    register_module("make_trump", make_trump);
    play_scorer = MLP(suit_emb + CARD_FEAT_DIM + context_dim_, hidden, 1, 2);
    register_module("play_scorer", play_scorer);
    discard_scorer = MLP(suit_emb + CARD_FEAT_DIM + context_dim_, hidden, 1, 2);
    register_module("discard_scorer", discard_scorer);
    pass_head = torch::nn::Linear(context_dim_, 1);
    register_module("pass_head", pass_head);
    value_head = MLP(context_dim_, hidden, 1, 2);
    register_module("value_head", value_head);
}

std::tuple<torch::Tensor, torch::Tensor> PolicyValueNetImpl::forward(const torch::Tensor& obs) {
    int64_t b = obs.size(0);
    auto glob = obs.slice(1, GLOBAL_OFF, GLOBAL_DIM);
    auto suit_blocks = obs.slice(1, SUIT_OFF, SUIT_OFF + NUM_SUITS * SUIT_BLOCK_DIM)
                          .reshape({b, NUM_SUITS, SUIT_BLOCK_DIM});
    auto card_feats = obs.slice(1, CARD_OFF, CARD_OFF + NUM_CARDS * CARD_FEAT_DIM)
                          .reshape({b, NUM_CARDS, CARD_FEAT_DIM});

    // Role one-hot's index 0 (reference role) selects the up-card/trump suit,
    // used to route the make-trump score to OrderUp.
    auto ref_ind = suit_blocks.select(2, 0);  // (b, NUM_SUITS)

    auto suit_e = suit_encoder->forward(suit_blocks);  // (b, NUM_SUITS, E)
    auto pooled = suit_e.mean(1);                      // (b, E) -- symmetric pool
    auto ctx = context->forward(torch::cat({pooled, glob}, -1));  // (b, C)

    auto ctx_suit = ctx.unsqueeze(1).expand({b, NUM_SUITS, context_dim});
    auto make = make_trump->forward(torch::cat({suit_e, ctx_suit}, -1));  // (b, NUM_SUITS, 2)
    auto call_na = make.select(2, 0);  // (b, NUM_SUITS)
    auto call_al = make.select(2, 1);  // (b, NUM_SUITS)
    auto orderup_na = (ref_ind * call_na).sum(1, /*keepdim=*/true);  // (b, 1)
    auto orderup_al = (ref_ind * call_al).sum(1, /*keepdim=*/true);  // (b, 1)

    // Cards are id-ordered (suit*6 + rank): repeat each suit embedding for
    // its 6 ranks to get a per-card suit embedding.
    auto card_suit_e = suit_e.repeat_interleave(CARDS_PER_SUIT, /*dim=*/1);  // (b, 24, E)
    auto ctx_card = ctx.unsqueeze(1).expand({b, NUM_CARDS, context_dim});
    auto card_in = torch::cat({card_suit_e, card_feats, ctx_card}, -1);
    auto play = play_scorer->forward(card_in).squeeze(-1);         // (b, 24)
    auto discard = discard_scorer->forward(card_in).squeeze(-1);   // (b, 24)
    auto pass_l = pass_head->forward(ctx);                         // (b, 1)

    // Exact order of euchre/actions.py's flat index space: play[0:24],
    // discard[24:48], call[48:52], call_alone[52:56], orderup[56],
    // orderup_alone[57], pass[58].
    auto logits = torch::cat({play, discard, call_na, call_al,
                              orderup_na, orderup_al, pass_l}, -1);
    auto value = value_head->forward(ctx).squeeze(-1);
    return {logits, value};
}

torch::Tensor PolicyValueNetImpl::policy(const torch::Tensor& obs,
                                         const c10::optional<torch::Tensor>& legal_mask) {
    auto [logits, value] = forward(obs);
    (void)value;
    if (legal_mask.has_value()) {
        logits = logits.masked_fill(~(*legal_mask), -std::numeric_limits<float>::infinity());
    }
    return torch::softmax(logits, -1);
}

std::map<std::string, torch::Tensor> PolicyValueNetImpl::state_dict_() const {
    std::map<std::string, torch::Tensor> out;
    for (const auto& p : named_parameters()) out[p.key()] = p.value();
    for (const auto& b : named_buffers()) out[b.key()] = b.value();
    return out;
}

void PolicyValueNetImpl::load_state_dict_(const std::map<std::string, torch::Tensor>& sd) {
    torch::NoGradGuard guard;
    for (auto& p : named_parameters()) {
        auto it = sd.find(p.key());
        TORCH_CHECK(it != sd.end(), "PolicyValueNet::load_state_dict_ missing key: ", p.key());
        p.value().copy_(it->second);
    }
    for (auto& b : named_buffers()) {
        auto it = sd.find(b.key());
        TORCH_CHECK(it != sd.end(), "PolicyValueNet::load_state_dict_ missing key: ", b.key());
        b.value().copy_(it->second);
    }
}

}  // namespace mceuchre
