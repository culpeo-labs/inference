#include <string_view>

#include "model/model_family.h"
#include <model/model.h>

using namespace std::string_view_literals;
using namespace culpeo::inference;

template<model::model_family family>
struct tensor_names
{
    static_assert("Unsupported family");
};

template<>
struct tensor_names<model::model_family::llama>
{
    static constexpr auto embed_tokens{ "model.token.weights"sv };
    static constexpr auto norm{ "model.norm.weights"sv };
    static constexpr auto input_layer_norm{ "model.layers.{}.input_layernorm.weight"sv };
    static constexpr auto q_proj{ "model.layers.{}.self_attn.q_proj.weight"sv };
    static constexpr auto k_proj{ "model.layers.{}.self_attn.k_proj.weight"sv };
    static constexpr auto v_proj{ "model.layers.{}.self_attn.v_proj.weight"sv };
    static constexpr auto o_proj{ "model.layers.{}.self_attn.o_proj.weight"sv };
    static constexpr auto gate_proj{ "model.layers.{N}.mlp.gate_proj.weight"sv };
    static constexpr auto up_proj{ "model.layers.{N}.mlp.up_proj.weight"sv };
    static constexpr auto down_proj{ "model.layers.{N}.mlp.down_proj.weight"sv };
    static constexpr auto lm_head{ "lm_head.weight" };
};
