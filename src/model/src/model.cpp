#include "model/hf/llama_config.h"
#include "model/safetensors.h"
#include "util/types.h"
#include <array>
#include <expected>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <type_traits>

#include <model/config.h>
#include <model/model_family.h>
#include <model/model.h>
#include <model/hf/llama_config.h>

using namespace std::string_view_literals;
using namespace culpeo::inference;

template<model::model_family family>
struct tensor_names
{
    static_assert(std::false_type::value, "Unsupported family");
};

template<>
struct tensor_names<model::model_family::llama>
{
    static constexpr auto embed_tokens{ "model.embed_tokens.weight"sv };
    static constexpr auto norm{ "model.norm.weight"sv };
    static constexpr auto input_layer_norm{ "model.layers.{}.input_layernorm.weight"sv };
    static constexpr auto q_proj{ "model.layers.{}.self_attn.q_proj.weight"sv };
    static constexpr auto k_proj{ "model.layers.{}.self_attn.k_proj.weight"sv };
    static constexpr auto v_proj{ "model.layers.{}.self_attn.v_proj.weight"sv };
    static constexpr auto o_proj{ "model.layers.{}.self_attn.o_proj.weight"sv };
    static constexpr auto gate_proj{ "model.layers.{}.mlp.gate_proj.weight"sv };
    static constexpr auto up_proj{ "model.layers.{}.mlp.up_proj.weight"sv };
    static constexpr auto down_proj{ "model.layers.{}.mlp.down_proj.weight"sv };
    static constexpr auto post_attention_layernorm{ "model.layers.{}.post_attention_layernorm.weight"sv };
    static constexpr auto lm_head{ "lm_head.weight" };
};

template<size_t Rank>
std::expected<util::matrix<util::data_type::BF16>::const_type<Rank>, std::string> load_tensor(const model::safetensors & safetensors, const std::string_view name, const std::array<std::size_t, Rank> dimensions)
{
    auto tensor = safetensors.get_tensor(name);
    if (!tensor)
    {
        return std::unexpected{ std::move(tensor).error() };
    }
    if constexpr (Rank == 2) {
        auto m = tensor->as_mat<util::data_type::BF16>();
        if (!m)
        {
            return std::unexpected{ std::move(m).error() };
        }
        const auto [rows, cols] = dimensions;
        if (m->extent(0) != rows || m->extent(1) != cols)
        {
            return std::unexpected{ "Tensor has the wrong dimensions." };
        }
        return *m;
    }
    else
    {
        auto v = tensor->as_vec<util::data_type::BF16>();
        if (!v)
        {
            return std::unexpected{ std::move(v).error() };
        }
        if (v->extent(0) != dimensions[0])
        {
            return std::unexpected{ "Tensor has the wrong dimesnions." };
        }
        return *v;
    }
}

std::expected<model::layer_weights, std::string> load_layer(const model::config & config, const model::safetensors & safetensors, std::size_t layer)
{
    using names = tensor_names<model::model_family::llama>;
    auto input_layer_norm = load_tensor(
        safetensors,
        std::format(names::input_layer_norm, layer),
        std::array{config.hidden_size}
    );
    if (!input_layer_norm)
    {
        return std::unexpected{ std::move(input_layer_norm).error() };
    }
    auto q_proj = load_tensor(
        safetensors,
        std::format(names::q_proj, layer),
        std::array{config.num_attention_heads * config.head_dim, config.hidden_size}
    );
    if (!q_proj)
    {
        return std::unexpected{ std::move(q_proj).error() };
    }
    auto k_proj = load_tensor(
        safetensors,
        std::format(names::k_proj, layer),
        std::array{ config.num_key_value_heads * config.head_dim, config.hidden_size }
    );
    if (!k_proj)
    {
        return std::unexpected{ std::move(k_proj).error() };
    }
    auto v_proj = load_tensor(
        safetensors,
        std::format(names::v_proj, layer),
        std::array{ config.num_key_value_heads * config.head_dim, config.hidden_size }
    );
    if (!v_proj)
    {
        return std::unexpected{ std::move(v_proj).error() };
    }
    auto o_proj = load_tensor(
        safetensors,
        std::format(names::o_proj, layer),
        std::array{ config.hidden_size, config.num_attention_heads * config.head_dim }
    );
    if (!o_proj)
    {
        return std::unexpected{ std::move(o_proj).error() };
    }
    auto gate_proj = load_tensor(
        safetensors,
        std::format(names::gate_proj, layer),
        std::array{ config.intermediate_size, config.hidden_size }
    );
    if (!gate_proj)
    {
        return std::unexpected{ std::move(gate_proj).error() };
    }
    auto up_proj = load_tensor(
        safetensors,
        std::format(names::up_proj, layer),
        std::array{ config.intermediate_size, config.hidden_size }
    );
    if (!up_proj)
    {
        return std::unexpected{ std::move(up_proj).error() };
    }
    auto down_proj = load_tensor(
        safetensors,
        std::format(names::down_proj, layer),
        std::array{ config.hidden_size, config.intermediate_size }
    );
    if (!down_proj)
    {
        return std::unexpected{ std::move(down_proj).error() };
    }
    auto post_attention_layernorm = load_tensor(
        safetensors,
        std::format(names::post_attention_layernorm, layer),
        std::array{ config.hidden_size }
    );
    if (!post_attention_layernorm)
    {
        return std::unexpected{ std::move(post_attention_layernorm).error() };
    }
    return model::layer_weights
    {
        .k_proj = std::move(k_proj).value(),
        .v_proj = std::move(v_proj).value(),
        .q_proj = std::move(q_proj).value(),
        .o_proj = std::move(o_proj).value(),
        .input_layernorm = std::move(input_layer_norm).value(),
        .up_proj = std::move(up_proj).value(),
        .down_proj = std::move(down_proj).value(),
        .post_attention_layernorm = std::move(post_attention_layernorm).value(),
        .gate_proj = std::move(gate_proj).value(),
    };
}

std::expected<model::model, std::string> model::model::load(const std::filesystem::path & dir)
{
    using names = tensor_names<model_family::llama>;
    if (!std::filesystem::exists(dir))
    {
        return std::unexpected("Provided checkpoint path does not exist.");
    }
    if (!std::filesystem::is_directory(dir))
    {
        return std::unexpected("Provided checkpoint path is not a directory.");
    }
    constexpr auto config_file_name{ "config.json"sv };
    constexpr auto safetensors_file_name{ "model.safetensors"sv };
    auto config = config::load<hf::llama_config>(dir / config_file_name);
    if (!config.has_value())
    {
        return std::unexpected(std::move(config).error());
    }
    auto safetensors = safetensors::load(dir / safetensors_file_name);
    if (!safetensors.has_value())
    {
        return std::unexpected(std::move(safetensors).error());
    }
    auto embed_tokens = load_tensor(
        *safetensors,
        names::embed_tokens, std::array{  config->vocab_size, config->hidden_size });
    if (!embed_tokens.has_value())
    {
        return std::unexpected(std::move(embed_tokens).error());
    }
    auto norm = load_tensor(
        *safetensors,
        names::norm,
        std::array{ config->hidden_size }
    );
    if (!norm.has_value())
    {
        return std::unexpected(std::move(norm).error());
    }
    auto lm_head = config->tie_word_embeddings ?
        embed_tokens :
        load_tensor(
            *safetensors,
            names::lm_head,
            std::array{config->vocab_size, config->hidden_size}
        );
    if (!lm_head.has_value())
    {
        return std::unexpected(std::move(lm_head).error());
    }
    std::vector<layer_weights> layers{};
    for (std::size_t i{0 }; i < config->num_hidden_layers; i++)
    {
        auto layer = load_layer(*config, *safetensors, i);
        if (!layer)
        {
            return std::unexpected{ std::move(layer).error() };
        }
        layers.push_back(std::move(layer).value());
    }
    return model
    {
        .config = std::move(config).value(),
        .safetensors = std::move(safetensors).value(),
        .layers = std::move(layers),
        .embed_tokens  = std::move(embed_tokens).value(),
        .norm = std::move(norm).value(),
        .lm_head = std::move(lm_head).value(),
    };
}

