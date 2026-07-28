#include <algorithm>
#include <expected>
#include <filesystem>
#include <glaze/core/opts.hpp>
#include <glaze/glaze.hpp>

#include <glaze/json/read.hpp>
#include <model/config.h>
#include <model/hf/llama_config.h>

using namespace culpeo::inference;

template<>
std::expected<model::config, std::string> model::config::load<model::hf::llama_config>(const std::filesystem::path & path)
{
    if (!std::filesystem::exists(path))
    {
        return std::unexpected("File does not exist.");
    }
    hf::llama_config config{};

    const auto error = glz::read_file_json<glz::opts{ .error_on_unknown_keys = false, .error_on_missing_keys = true }>(config, path.c_str(), std::string{});
    if (error)
    {
        return std::unexpected("Error parsing config.");
    }

    if (std::find(config.architectures.begin(), config.architectures.end(), "LlamaForCausalLM") == config.architectures.end())
    {
        return std::unexpected("architectures must contain LlamaForCausalLM");
    }

    if (config.num_attention_heads == 0)
    {
        return std::unexpected("num_attention_heads must be nonzero");
    }

    if (config.num_key_value_heads.has_value())
    {
        const auto kvh = *config.num_key_value_heads;
        if (kvh == 0)
        {
            return std::unexpected("num_key_value_heads must be nonzero");
        }
        if (config.num_attention_heads % kvh != 0)
        {
            return std::unexpected("num_attention_heads must be divisible by num_key_value_heads");
        }
    }

    return model::config{
        .family = model::model_family::llama,
        .head_dim = config.head_dim.value_or(
            config.hidden_size / config.num_attention_heads
        ),
        .hidden_size = config.hidden_size,
        .intermediate_size = config.intermediate_size,
        .num_attention_heads = config.num_attention_heads,
        .num_hidden_layers = config.num_hidden_layers,
        .num_key_value_heads = config.num_key_value_heads.value_or(config.num_attention_heads),
        .rms_norm_eps = config.rms_norm_eps,
        .rope_theta = config.rope_theta,
        .tie_word_embeddings = config.tie_word_embeddings,
        .vocab_size = config.vocab_size,
        .original_max_position_embeddings = config.rope_scaling.transform([](auto& r) { return r.original_max_position_embeddings; })
    };
}